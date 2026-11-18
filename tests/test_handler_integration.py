"""事件处理器的集成自测（需要框架运行时，不联网）。

验证「BEFORE 缓存目录 → AFTER 校验/本地修复/LLM 修复 → 写回 params」整条链路，
用假的组件类与假的修复器，避免真的调用模型。

    python plugins/all_toolcall_failures_go_die/tests/test_handler_integration.py
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

from src.kernel.event import EventDecision  # noqa: E402

# 以「脚本」方式运行时插件目录不一定在 sys.path 上，用 importlib 动态加载，
# 这样既能在插件目录内使用相对导入，也能满足「插件内部不得绝对导入」的规范。
_config_mod = importlib.import_module("all_toolcall_failures_go_die.config")
_extraction_mod = importlib.import_module(
    "all_toolcall_failures_go_die.core.extraction"
)
_prompts_mod = importlib.import_module("all_toolcall_failures_go_die.core.prompts")
_handler_mod = importlib.import_module("all_toolcall_failures_go_die.handler")

ToolCallRepairConfig = _config_mod.ToolCallRepairConfig
RawCallCandidate = _extraction_mod.RawCallCandidate
RepairDecision = _prompts_mod.RepairDecision
ToolCallRepairHandler = _handler_mod.ToolCallRepairHandler

SEND_TEXT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "action-send_text",
        "description": "发送一段文本消息给用户",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要发送的文本"}
            },
            "required": ["content"],
        },
    },
}


class FakeSendText:
    """假的 send_text 组件（只要 to_schema / execute 两个接口）。"""

    @classmethod
    def to_schema(cls) -> dict[str, Any]:
        return SEND_TEXT_SCHEMA

    async def execute(self, content: str) -> tuple[bool, str]:
        return True, f"sent:{content}"


class FakePlugin:
    """假插件：只暴露 config。"""

    def __init__(self, config: ToolCallRepairConfig) -> None:
        self.config = config


class FakeRepairer:
    """假修复器：记录调用并返回预置结论。"""

    request_name = "toolcall_repair"

    def __init__(self, decision: RepairDecision) -> None:
        self.decision = decision
        self.calls: list[dict[str, Any]] = []
        self.wrap_calls: list[dict[str, Any]] = []

    async def plan(self, **kwargs: Any) -> RepairDecision:
        self.calls.append(kwargs)
        return self.decision

    async def fill_args(self, **kwargs: Any) -> RepairDecision:
        self.calls.append(kwargs)
        return self.decision

    async def wrap(self, **kwargs: Any) -> RepairDecision:
        self.wrap_calls.append(kwargs)
        return self.decision


def _before_params(request_name: str = "neo_default_chatter") -> dict[str, Any]:
    """构造 BEFORE_LLM_REQUEST 的事件参数。"""
    return {
        "request_name": request_name,
        "model_identifier": "fake-model",
        "stream": False,
        "tool_choice": "auto",
        "tools": [FakeSendText, {"name": "x"}],
        "meta_data": {"stream_id": "stream-1"},
    }


def _after_params(
    message: str,
    tool_calls: list[Any] | None = None,
    request_name: str = "neo_default_chatter",
) -> dict[str, Any]:
    """构造 AFTER_LLM_REQUEST 的事件参数（键集合与框架一致）。"""
    return {
        "request_name": request_name,
        "model_identifier": "fake-model",
        "stream": False,
        "success": True,
        "latency": 0.1,
        "retry_count": 0,
        "message": message,
        "reasoning_content": None,
        "reasoning_parts": [],
        "tool_calls": tool_calls or [],
        "usage": {},
        "meta_data": {"stream_id": "stream-1"},
    }


async def _prepare(config: ToolCallRepairConfig) -> ToolCallRepairHandler:
    """创建处理器并写入 BEFORE 缓存。"""
    handler = ToolCallRepairHandler(FakePlugin(config))
    decision, _ = await handler.execute("before_llm_request", _before_params())
    assert decision is EventDecision.SUCCESS
    return handler


def _field(call: Any, key: str) -> Any:
    """兼容 ToolCall 对象与 dict 的字段读取。"""
    if isinstance(call, dict):
        return call.get(key)
    return getattr(call, key, None)


async def test_local_recovery_from_plain_text() -> None:
    """纯文本里能本地抢救出的调用应直接写回，且不调用修复模型。"""
    config = ToolCallRepairConfig()
    handler = await _prepare(config)
    handler._get_repairer = lambda _cfg: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("不该调用修复模型")
    )

    original = '{"name": "send_text", "args": {"content": "你好"}}'
    params = _after_params(original)
    decision, result = await handler.execute("after_llm_request", params)

    assert decision is EventDecision.SUCCESS
    calls = result["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["name"] == "action-send_text"
    assert calls[0]["args"]["content"] == "你好"
    assert calls[0]["id"], "修复出的调用必须带 id，保证 TOOL_RESULT 能配对"
    # 默认 clear_message_on_repair=false：模型原文保留为 assistant 独白
    assert result["message"] == original


async def test_clear_message_switch() -> None:
    """把 clear_message_on_repair 打开后，救回来的原文应被清空。"""
    config = ToolCallRepairConfig()
    config.plain_text.clear_message_on_repair = True
    handler = await _prepare(config)
    handler._get_repairer = lambda _cfg: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("不该调用修复模型")
    )

    params = _after_params('{"name": "send_text", "args": {"content": "你好"}}')
    _, result = await handler.execute("after_llm_request", params)

    assert result["tool_calls"][0]["name"] == "action-send_text"
    assert result["message"] == ""


async def test_local_fix_for_missing_prefix_and_bad_args() -> None:
    """工具名缺前缀 + args 是 JSON 字符串应本地修好。"""
    config = ToolCallRepairConfig()
    handler = await _prepare(config)
    handler._get_repairer = lambda _cfg: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("不该调用修复模型")
    )

    class FakeCall:
        id = "call_1"
        name = "send_text"
        args = '{"content": "在吗"}'

    params = _after_params("", [FakeCall()])
    _, result = await handler.execute("after_llm_request", params)

    assert _field(result["tool_calls"][0], "name") == "action-send_text"
    assert _field(result["tool_calls"][0], "args")["content"] == "在吗"
    # 这条不是「纯文本转调用」，message 本来就空，不应被改动语义
    assert result["message"] == ""


async def test_llm_repair_for_missing_required() -> None:
    """缺必填参数应请求修复模型，并写回它给出的调用。"""
    config = ToolCallRepairConfig()
    handler = await _prepare(config)
    fake = FakeRepairer(
        RepairDecision(
            "calls",
            candidates=[
                RawCallCandidate(
                    name="action-send_text",
                    args={"content": "你好呀"},
                )
            ],
        )
    )
    handler._get_repairer = lambda _cfg: fake  # type: ignore[assignment]

    class FakeCall:
        id = "call_2"
        name = "action-send_text"
        args = {}

    params = _after_params("用户说：你好呀，请回复", [FakeCall()])
    _, result = await handler.execute("after_llm_request", params)

    assert fake.calls, "应发起一次补参修复"
    assert result["tool_calls"][0]["args"]["content"] == "你好呀"
    # 原 message 是用户内容，不应被清空
    assert result["message"] == "用户说：你好呀，请回复"


async def test_llm_says_no_tool_call_keeps_response() -> None:
    """修复模型判定「不是工具调用」时应保持原响应不变。"""
    config = ToolCallRepairConfig()
    config.plain_text.mode = "always"
    handler = await _prepare(config)
    fake = FakeRepairer(RepairDecision("none", reason="只是普通回复"))
    handler._get_repairer = lambda _cfg: fake  # type: ignore[assignment]

    params = _after_params("今天天气不错呀")
    _, result = await handler.execute("after_llm_request", params)

    assert fake.calls
    assert result["tool_calls"] == []
    assert result["message"] == "今天天气不错呀"


async def test_wrap_plain_text_locally() -> None:
    """最核心的用例：不带任何格式的纯文本应被本地套壳成发送调用（零 token）。"""
    config = ToolCallRepairConfig()
    assert config.plain_text.mode == "wrap", "默认模式应是 wrap"
    handler = await _prepare(config)
    handler._get_repairer = lambda _cfg: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("能本地套壳时不该调用修复模型")
    )

    text = "你好，我在"
    params = _after_params(text)
    _, result = await handler.execute("after_llm_request", params)

    calls = result["tool_calls"]
    assert len(calls) == 1
    call = calls[0]
    name = call["name"] if isinstance(call, dict) else _field(call, "name")
    args = call["args"] if isinstance(call, dict) else _field(call, "args")
    assert name == "action-send_text", "应套成当前 chatter 的发送工具"
    assert args["content"] == text, "正文必须原样搬过去"
    cid = call["id"] if isinstance(call, dict) else _field(call, "id")
    assert cid, "套壳出来的调用必须补 id"
    assert result["message"] == text, "默认保留原文"


class FakeReplyAction:
    """另一个 chatter 的发送动作（名字不同，带 chatter_allow）。"""

    chatter_allow = ["another_chatter"]

    @classmethod
    def to_schema(cls) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "action-reply_message",
                "description": "向用户回复一段消息",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }


class FakeEmojiAction:
    """看着也像发送、但不是文本的干扰项。"""

    @classmethod
    def to_schema(cls) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "action-send_emoji_meme",
                "description": "发送一个表情包",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "emotion_tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["description"],
                },
            },
        }


async def test_wrap_picks_current_chatter_tool() -> None:
    """多个发送动作同时存在时，应按 chatter 归属选中当前 chatter 的那个。"""
    config = ToolCallRepairConfig()
    handler = ToolCallRepairHandler(FakePlugin(config))
    await handler.execute(
        "before_llm_request",
        {
            "request_name": "another_chatter",
            "model_identifier": "fake",
            "stream": False,
            "tools": [FakeSendText, FakeReplyAction, FakeEmojiAction],
            "payloads": [],
            "meta_data": {"stream_id": "stream-2"},
        },
    )
    handler._get_repairer = lambda _cfg: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("不该调用修复模型")
    )

    params = _after_params("好的我明白了", request_name="another_chatter")
    _, result = await handler.execute("after_llm_request", params)

    assert len(result["tool_calls"]) == 1
    call = result["tool_calls"][0]
    name = call["name"] if isinstance(call, dict) else _field(call, "name")
    args = call["args"] if isinstance(call, dict) else _field(call, "args")
    assert name == "action-reply_message", f"应选中本 chatter 的发送工具，实际 {name}"
    assert args["text"] == "好的我明白了"


async def test_wrap_skipped_without_send_tool() -> None:
    """目录里没有发送类工具时，普通纯文本应原样放行（不发请求）。"""
    config = ToolCallRepairConfig()
    handler = ToolCallRepairHandler(FakePlugin(config))
    await handler.execute(
        "before_llm_request",
        {
            "request_name": "neo_default_chatter",
            "model_identifier": "fake",
            "stream": False,
            "tools": [FakeEmojiAction],
            "payloads": [],
            "meta_data": {"stream_id": "stream-3"},
        },
    )
    handler._get_repairer = lambda _cfg: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("不该调用修复模型")
    )

    params = _after_params("今天天气不错", request_name="neo_default_chatter")
    _, result = await handler.execute("after_llm_request", params)
    assert result["tool_calls"] == []
    assert result["message"] == "今天天气不错"


async def test_wrap_via_always_llm() -> None:
    """wrap_via=always-llm 时应交给模型套壳，并使用它给出的工具与参数。"""
    config = ToolCallRepairConfig()
    config.plain_text.wrap_via = "always-llm"
    handler = await _prepare(config)
    fake = FakeRepairer(
        RepairDecision(
            "calls",
            candidates=[
                RawCallCandidate(
                    name="action-send_text",
                    args={"content": "你好，我在"},
                )
            ],
        )
    )
    handler._get_repairer = lambda _cfg: fake  # type: ignore[assignment]

    params = _after_params("你好，我在")
    _, result = await handler.execute("after_llm_request", params)

    assert fake.wrap_calls, "always-llm 模式下应发起一次套壳请求"
    assert result["tool_calls"][0]["name"] == "action-send_text"
    assert result["tool_calls"][0]["args"]["content"] == "你好，我在"


async def test_skip_repair_request_itself() -> None:
    """修复请求自身的事件必须被跳过（防递归）。"""
    config = ToolCallRepairConfig()
    handler = await _prepare(config)
    handler._get_repairer = lambda _cfg: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("不该调用修复模型")
    )
    params = _after_params(
        '{"name":"send_text","args":{}}', request_name="toolcall_repair"
    )
    _, result = await handler.execute("after_llm_request", params)
    assert result["tool_calls"] == []


async def test_valid_calls_are_untouched() -> None:
    """本来就合法的调用应原样保留。"""
    config = ToolCallRepairConfig()
    handler = await _prepare(config)
    handler._get_repairer = lambda _cfg: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("不该调用修复模型")
    )

    class GoodCall:
        id = "call_3"
        name = "action-send_text"
        args = {"content": "hi", "reason": "想回复"}

    params = _after_params("", [GoodCall()])
    _, result = await handler.execute("after_llm_request", params)
    assert result["tool_calls"][0] is params["tool_calls"][0]
    assert _field(result["tool_calls"][0], "args") == {
        "content": "hi",
        "reason": "想回复",
    }


async def test_streaming_is_skipped() -> None:
    """流式响应在 AFTER 阶段没有 tool_calls，应跳过。"""
    config = ToolCallRepairConfig()
    config.plain_text.mode = "always"
    handler = await _prepare(config)
    handler._get_repairer = lambda _cfg: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("不该调用修复模型")
    )
    params = _after_params("")
    params["stream"] = True
    _, result = await handler.execute("after_llm_request", params)
    assert result["tool_calls"] == []


async def test_no_second_request_when_first_fails() -> None:
    """补参请求失败后（max_attempts=1）不得再追加第二次请求。"""
    config = ToolCallRepairConfig()
    config.repair.max_attempts = 1
    handler = await _prepare(config)

    sent: list[str] = []

    class PickyRepairer:
        async def plan(self, **_kwargs: Any) -> RepairDecision:
            sent.append("plan")
            return RepairDecision("invalid", raw="嗯嗯")

        async def fill_args(self, **_kwargs: Any) -> RepairDecision:
            sent.append("fill")
            return RepairDecision("invalid", raw="嗯嗯")

    handler._get_repairer = lambda _cfg: PickyRepairer()  # type: ignore[assignment]

    class FakeCall:
        id = "call_4"
        name = "action-send_text"
        args = {}

    params = _after_params("发送：你好", [FakeCall()])
    _, result = await handler.execute("after_llm_request", params)

    assert sent == ["fill"], f"只应发一次请求，实际: {sent}"
    assert result["tool_calls"] == params["tool_calls"]


async def test_catalog_miss_skips_repair_by_default() -> None:
    """BEFORE 未缓存到目录时默认放弃修复（不猜全局注册表里的工具）。"""
    config = ToolCallRepairConfig()
    handler = ToolCallRepairHandler(FakePlugin(config))
    handler._get_repairer = lambda _cfg: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("不该调用修复模型")
    )

    # 故意不发 BEFORE 事件 → 目录为空
    params = _after_params('{"name": "send_text", "args": {"content": "你好"}}')
    decision, result = await handler.execute("after_llm_request", params)

    assert decision is EventDecision.SUCCESS
    assert result["tool_calls"] == []
    assert result["message"] == params["message"]


async def test_catalog_fallback_switch() -> None:
    """打开 catalog_fallback 后，目录未命中时才会退回全局注册表（这里只验不崩）。"""
    config = ToolCallRepairConfig()
    config.main.catalog_fallback = True
    handler = ToolCallRepairHandler(FakePlugin(config))
    handler._get_repairer = lambda _cfg: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("不该调用修复模型")
    )

    params = _after_params("今天天气不错")
    decision, result = await handler.execute("after_llm_request", params)
    assert decision is EventDecision.SUCCESS
    assert result["message"] == "今天天气不错"


async def _main() -> int:
    """运行全部集成用例。"""
    cases = [
        ("local_recovery_from_plain_text", test_local_recovery_from_plain_text),
        ("clear_message_switch", test_clear_message_switch),
        (
            "local_fix_for_missing_prefix_and_bad_args",
            test_local_fix_for_missing_prefix_and_bad_args,
        ),
        ("llm_repair_for_missing_required", test_llm_repair_for_missing_required),
        (
            "llm_says_no_tool_call_keeps_response",
            test_llm_says_no_tool_call_keeps_response,
        ),
        ("skip_repair_request_itself", test_skip_repair_request_itself),
        ("valid_calls_are_untouched", test_valid_calls_are_untouched),
        ("streaming_is_skipped", test_streaming_is_skipped),
        ("no_second_request_when_first_fails", test_no_second_request_when_first_fails),
        (
            "catalog_miss_skips_repair_by_default",
            test_catalog_miss_skips_repair_by_default,
        ),
        ("catalog_fallback_switch", test_catalog_fallback_switch),
        ("wrap_plain_text_locally", test_wrap_plain_text_locally),
        ("wrap_picks_current_chatter_tool", test_wrap_picks_current_chatter_tool),
        ("wrap_skipped_without_send_tool", test_wrap_skipped_without_send_tool),
        ("wrap_via_always_llm", test_wrap_via_always_llm),
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
