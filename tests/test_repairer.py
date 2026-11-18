"""修复请求发送层的自测（需要框架可导入，不联网）。

用一个假的 ``create_llm_request`` 冒充模型，验证：

- 任务解析与回退（``tool_use`` → ``utils`` → …）；
- 温度/最大输出覆盖只作用在本次请求的 model_set 上；
- 只发 SYSTEM + USER 两条 payload；
- 回复解析、超时与异常都 fail-open 成 ``invalid``。

    python plugins/all_toolcall_failures_go_die/tests/test_repairer.py
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
_FRAMEWORK_ROOT = _PLUGIN_ROOT.parents[1]
for _path in (str(_FRAMEWORK_ROOT), str(_PLUGIN_ROOT.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.app.plugin_system.api import llm_api  # noqa: E402
from src.kernel.llm import ROLE, Text  # noqa: E402

_schemas_mod = importlib.import_module("all_toolcall_failures_go_die.core.schemas")
_prompts_mod = importlib.import_module("all_toolcall_failures_go_die.core.prompts")
_repairer_mod = importlib.import_module("all_toolcall_failures_go_die.core.repairer")

RepairLLM = _repairer_mod.RepairLLM
RepairLLMSettings = _repairer_mod.RepairLLMSettings
ToolCatalog = _schemas_mod.ToolCatalog
ToolSpec = _schemas_mod.ToolSpec
KIND_CALLS = _prompts_mod.KIND_CALLS
KIND_INVALID = _prompts_mod.KIND_INVALID
KIND_NONE = _prompts_mod.KIND_NONE

SEND_TEXT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "action-send_text",
        "description": "发送一段文本",
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
    },
}

CATALOG = ToolCatalog({"action-send_text": ToolSpec.from_schema(SEND_TEXT_SCHEMA)})

# 打桩前的原始函数（模块对象是共享的，必须在导入时保存）
_ORIGINAL_GET_MODEL_SET = llm_api.get_model_set_by_task
_ORIGINAL_CREATE_REQUEST = llm_api.create_llm_request


def _model_entry() -> dict[str, Any]:
    """一份符合 kernel 校验要求的最小 model_set 元素。"""
    return {
        "api_provider": "fake",
        "base_url": "http://localhost",
        "model_identifier": "fake-model",
        "api_key": "k",
        "client_type": "openai",
        "max_retry": 0,
        "timeout": 1.0,
        "retry_interval": 0.0,
        "price_in": 0.0,
        "price_out": 0.0,
        "temperature": 0.7,
        "max_tokens": 800,
        "extra_params": {},
    }


class FakeResponse:
    """模仿 ``LLMResponse``：await 之后得到纯文本。"""

    def __init__(self, text: str) -> None:
        self.text = text

    def __await__(self):
        async def _collect() -> str:
            return self.text

        return _collect().__await__()


class FakeRequest:
    """记录 payload，并按预设脚本回复/抛错。"""

    def __init__(self, reply: str | Exception) -> None:
        self.payloads: list[Any] = []
        self.reply = reply
        self.model_set: Any = None
        self.request_name: str = ""

    def add_payload(self, payload: Any) -> None:
        self.payloads.append(payload)

    async def send(self, *, stream: bool = True, auto_append_response: bool = True):
        if isinstance(self.reply, Exception):
            raise self.reply
        return FakeResponse(self.reply)


def _patch(
    *,
    reply: str | Exception = '{"tool_calls": []}',
    task_failures: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """替换 llm_api 的两个入口，返回 (调用记录, 可用 model_set)。"""
    calls: list[dict[str, Any]] = []
    created: dict[str, Any] = {}

    def fake_get_model_set_by_task(name: str) -> list[dict[str, Any]]:
        calls.append({"op": "task", "name": name})
        if name in task_failures:
            raise ValueError(f"任务 '{name}' 未找到对应的配置")
        return [_model_entry()]

    def fake_create_llm_request(model_set: Any, request_name: str = "") -> FakeRequest:
        request = FakeRequest(reply)
        request.request_name = request_name
        created["request"] = request
        created["model_set"] = model_set
        calls.append({"op": "create", "request_name": request_name})
        return request

    llm_api.get_model_set_by_task = fake_get_model_set_by_task  # type: ignore[assignment]
    llm_api.create_llm_request = fake_create_llm_request  # type: ignore[assignment]
    return calls, created


def _restore() -> None:
    """还原被打桩的 llm_api 入口。"""
    llm_api.get_model_set_by_task = _ORIGINAL_GET_MODEL_SET  # type: ignore[assignment]
    llm_api.create_llm_request = _ORIGINAL_CREATE_REQUEST  # type: ignore[assignment]


async def test_plan_sends_only_system_and_user() -> None:
    """重建模式能从回复里解析出调用。"""
    _patch(
        reply='{"tool_calls": [{"name": "action-send_text", "args": {"content": "你好"}}]}'
    )
    try:
        repairer = RepairLLM(RepairLLMSettings(temperature=0.0, max_tokens=256))
        decision = await repairer.plan(raw_text="发送你好", catalog=CATALOG)
    finally:
        _restore()

    assert decision.kind == KIND_CALLS, decision
    assert decision.candidates[0].args == {"content": "你好"}


async def test_payload_shape_and_overrides() -> None:
    """payload 只有 SYSTEM/USER；model_set 上的覆盖只影响本次。"""
    calls, created = _patch(reply='{"no_tool_call": true, "reason": "闲聊"}')
    try:
        settings = RepairLLMSettings(temperature=0.1, max_tokens=128)
        repairer = RepairLLM(settings)
        decision = await repairer.plan(raw_text="今天天气不错", catalog=CATALOG)
        request = created["request"]
        model_set = created["model_set"]
    finally:
        _restore()

    assert decision.kind == KIND_NONE
    assert [p.role for p in request.payloads] == [ROLE.SYSTEM, ROLE.USER]
    assert all(isinstance(p.content[0], Text) for p in request.payloads)
    assert request.request_name == "toolcall_repair"
    assert model_set[0]["temperature"] == 0.1
    assert model_set[0]["max_tokens"] == 128
    # 回退链只在需要时才多问一次
    assert [c["name"] for c in calls if c["op"] == "task"] == ["tool_use"]


async def test_fill_args_uses_fallback_name() -> None:
    """只给 args 的偷懒回复，用目标工具名补全。"""
    _patch(reply='{"args": {"content": "在的"}}')
    try:
        repairer = RepairLLM(RepairLLMSettings())
        spec = CATALOG.get("action-send_text")
        assert spec is not None
        decision = await repairer.fill_args(raw_text="在的", spec=spec)
    finally:
        _restore()

    assert decision.kind == KIND_CALLS
    assert decision.candidates[0].name == "action-send_text"


async def test_timeout_is_fail_open() -> None:
    """请求超时不得抛出，只返回 invalid。"""
    _patch(reply=asyncio.TimeoutError())
    try:
        repairer = RepairLLM(RepairLLMSettings(timeout_seconds=0.01))
        decision = await repairer.plan(raw_text="x", catalog=CATALOG)
    finally:
        _restore()
    assert decision.kind == KIND_INVALID


async def test_exception_is_fail_open() -> None:
    """请求异常不得抛出，只返回 invalid。"""
    _patch(reply=RuntimeError("boom"))
    try:
        repairer = RepairLLM(RepairLLMSettings())
        decision = await repairer.plan(raw_text="x", catalog=CATALOG)
    finally:
        _restore()
    assert decision.kind == KIND_INVALID


async def test_task_fallback() -> None:
    """默认任务取不到时应回退到下一个可用任务。"""
    calls, _created = _patch(
        reply='{"no_tool_call": true}', task_failures=("tool_use",)
    )
    try:
        repairer = RepairLLM(RepairLLMSettings())
        decision = await repairer.plan(raw_text="x", catalog=CATALOG)
    finally:
        _restore()

    assert decision.kind == KIND_NONE
    assert [c["name"] for c in calls if c["op"] == "task"] == ["tool_use", "utils"]


async def test_all_tasks_missing_is_fail_open() -> None:
    """所有任务都取不到时也只能返回 invalid，不能崩。"""
    _patch(
        reply='{"no_tool_call": true}',
        task_failures=("tool_use", "utils", "utils_small", "actor"),
    )
    try:
        repairer = RepairLLM(RepairLLMSettings())
        decision = await repairer.plan(raw_text="x", catalog=CATALOG)
    finally:
        _restore()
    assert decision.kind == KIND_INVALID


async def _main() -> int:
    """运行全部用例。"""
    cases = [
        ("plan_sends_only_system_and_user", test_plan_sends_only_system_and_user),
        ("payload_shape_and_overrides", test_payload_shape_and_overrides),
        ("fill_args_uses_fallback_name", test_fill_args_uses_fallback_name),
        ("timeout_is_fail_open", test_timeout_is_fail_open),
        ("exception_is_fail_open", test_exception_is_fail_open),
        ("task_fallback", test_task_fallback),
        ("all_tasks_missing_is_fail_open", test_all_tasks_missing_is_fail_open),
    ]
    failed = 0
    for name, case in cases:
        try:
            await case()
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
        else:
            print(f"OK   {name}")
    print(f"\n{len(cases) - failed}/{len(cases)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
