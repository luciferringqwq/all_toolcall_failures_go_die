"""真实模型冒烟测试：把一段「纯文本响应」按插件真实运行流程走一遍。

流程与线上完全一致：

1. 加载框架配置（core / model）与插件的真实 `config.toml`；
2. 创建一个真实的事件总线，按 `weight` / `timeout` 订阅 BEFORE / AFTER 事件；
3. 发布 `BEFORE_LLM_REQUEST`（带真实组件类）→ 处理器缓存本次工具目录；
4. 发布 `AFTER_LLM_REQUEST`（message=样本语句、tool_calls 为空）→
   处理器判定、必要时**真的调用修复模型**；
5. 打印修复请求（模型/temperature/max_tokens/提示词）、模型原始回复、写回结果。

用法：

    python plugins/all_toolcall_failures_go_die/tests/real_llm_probe.py
    python plugins/all_toolcall_failures_go_die/tests/real_llm_probe.py "你的样本语句"
    python plugins/all_toolcall_failures_go_die/tests/real_llm_probe.py --text-file sample.txt
    python plugins/all_toolcall_failures_go_die/tests/real_llm_probe.py --show-prompts
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
_FRAMEWORK_ROOT = _PLUGIN_ROOT.parents[1]
for _path in (str(_FRAMEWORK_ROOT), str(_PLUGIN_ROOT.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.core.config import init_core_config, init_model_config  # noqa: E402
from src.core.components.base.action import BaseAction  # noqa: E402
from src.core.components.types import EventType  # noqa: E402
from src.kernel.event import EventBus  # noqa: E402

# 默认样本：模型把工具调用写成了「半截 JSON 散文」，本地解析不出，只能请模型修
DEFAULT_TEXT = (
    "好的，我这就发出去："
    '{"name": "send_text", "args": {"content": "在吗？在的话回我一下'
)


# ----------------------------------------------------------------------
# 真实组件（用框架自己的 to_schema 生成 schema）
# ----------------------------------------------------------------------


class SendTextAction(BaseAction):
    """发送文本。"""

    name: str = "send_text"
    description: str = "向当前聊天流发送一段文本消息"

    async def execute(
        self, content: str, reply_to: str | None = None
    ) -> tuple[bool, str]:
        return True, content


class CheckWeatherAction(BaseAction):
    """查询天气（凑数，让目录里不止一个工具）。"""

    name: str = "check_weather"
    description: str = "查询指定城市的天气"

    async def execute(self, city: str, days: int = 1) -> tuple[bool, str]:
        return True, f"{city}/{days}"


class FakePlugin:
    """只暴露 config，供处理器读取配置。"""

    def __init__(self, config: Any) -> None:
        self.config = config


# ----------------------------------------------------------------------
# 可选场景：KFC（kokoro_flow_chatter）的回复工具
#
# schema 与 plugins/kokoro_flow_chatter/actions/reply.py 保持一致：
#   · 工具名 kfc_reply（框架加前缀后是 action-kfc_reply）
#   · content 是 list[str]（一次响应里分段发送）
#   · thought / expected_reaction / max_wait_seconds / mood 在 schema 里被强制必填
#   · chatter_allow 只允许 kokoro_flow_chatter
# ----------------------------------------------------------------------

_KFC_METADATA_FIELDS = ("thought", "expected_reaction", "max_wait_seconds", "mood")


class KfcReplyAction(BaseAction):
    """KFC 的回复动作（复刻真实 schema）。"""

    name: str = "kfc_reply"
    description: str = (
        "发送文本消息给对方。content 为消息段落列表，每个元素是一条独立消息，"
        "系统会依次发出。可选的 reply_to 参数允许你引用消息。"
        "严禁在一次响应中多次调用此工具；若有多段内容，请通过 content 列表一次性传入。"
        "调用时必须明确给出 thought / expected_reaction / max_wait_seconds / mood 这四个字段。"
    )
    chatter_allow: list[str] = ["kokoro_flow_chatter"]

    @classmethod
    def to_schema(cls) -> dict:  # type: ignore[override]
        """与真实插件一致：把元数据字段标成必填。"""
        schema = super().to_schema()
        parameters = schema.get("function", {}).get("parameters", {})
        properties = parameters.get("properties", {}) or {}
        required = list(parameters.get("required", []) or [])
        for field_name in _KFC_METADATA_FIELDS:
            if field_name not in properties:
                continue
            properties[field_name].pop("default", None)
            if field_name not in required:
                required.append(field_name)
        parameters["required"] = required
        return schema

    async def execute(
        self,
        content: list[str],
        thought: str = "",
        expected_reaction: str = "",
        max_wait_seconds: float = 0.0,
        mood: str = "",
        reply_to: str = "",
    ) -> tuple[bool, str]:
        return True, " ".join(content)


class KfcPassAndWaitAction(BaseAction):
    """KFC 的等待动作（凑数，让目录里不止一个工具）。"""

    name: str = "pass_and_wait"
    description: str = "等待对方说话，不发送任何消息"
    chatter_allow: list[str] = ["kokoro_flow_chatter"]

    async def execute(self, thought: str = "") -> tuple[bool, str]:
        return True, thought


#: 场景表：request_name / 工具列表 / 默认样本。
SCENARIOS: dict[str, dict[str, Any]] = {
    "default": {
        "request_name": "neo_default_chatter",
        "tools": [SendTextAction, CheckWeatherAction],
    },
    "kfc": {
        "request_name": "kokoro_flow_chatter",
        "tools": [KfcReplyAction, KfcPassAndWaitAction],
    },
}


def _load_plugin_config() -> Any:
    """优先加载真实配置文件，缺失时退回默认值。"""
    config_mod = importlib.import_module("all_toolcall_failures_go_die.config")
    config_cls = config_mod.ToolCallRepairConfig
    path = (
        _FRAMEWORK_ROOT
        / "config"
        / "plugins"
        / "all_toolcall_failures_go_die"
        / "config.toml"
    )
    if path.exists():
        return config_cls.load(str(path))
    return config_cls()


def _spy(repairer_mod: Any) -> dict[str, Any]:
    """给修复器装上间谍：记录请求参数、提示词与模型原始回复。"""
    captured: dict[str, Any] = {}

    real_create = repairer_mod.llm_api.create_llm_request
    real_parse = repairer_mod.parse_repair_reply
    captured["real_create"] = real_create
    captured["real_parse"] = real_parse

    def spy_create(model_set: Any, *args: Any, **kwargs: Any) -> Any:
        request = real_create(model_set, *args, **kwargs)
        captured["models"] = [
            {
                "model_identifier": entry.get("model_identifier"),
                "temperature": entry.get("temperature"),
                "max_tokens": entry.get("max_tokens"),
            }
            for entry in model_set
        ]
        captured["request_name"] = kwargs.get("request_name", "")
        # payload 要等调用方 add_payload 之后再读，这里只记对象引用。
        captured["request"] = request
        return request

    def spy_parse(text: str, *args: Any, **kwargs: Any) -> Any:
        captured["reply"] = text
        request = captured.get("request")
        if request is not None and "payloads" not in captured:
            captured["payloads"] = [
                {
                    "role": str(getattr(payload.role, "value", payload.role)),
                    "text": "".join(
                        str(getattr(part, "text", part)) for part in payload.content
                    ),
                }
                for payload in request.payloads
            ]
        return real_parse(text, *args, **kwargs)

    repairer_mod.llm_api.create_llm_request = spy_create
    repairer_mod.parse_repair_reply = spy_parse
    return captured


def _restore(repairer_mod: Any, captured: dict[str, Any]) -> None:
    """还原间谍挂载。"""
    if "real_create" in captured:
        repairer_mod.llm_api.create_llm_request = captured["real_create"]
    if "real_parse" in captured:
        repairer_mod.parse_repair_reply = captured["real_parse"]


async def _run(args: argparse.Namespace) -> int:
    """跑一次完整流程。"""
    text = args.text
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()
    if not text:
        text = DEFAULT_TEXT

    print("=" * 72)
    print("阶段 0 · 加载框架配置")
    init_core_config(str(_FRAMEWORK_ROOT / "config" / "core.toml"))
    init_model_config(str(_FRAMEWORK_ROOT / "config" / "model.toml"))
    print(f"  core.toml  : {_FRAMEWORK_ROOT / 'config' / 'core.toml'}")
    print(f"  model.toml : {_FRAMEWORK_ROOT / 'config' / 'model.toml'}")

    config = _load_plugin_config()
    config.main.debug_log = True
    if args.mode:
        config.plain_text.mode = args.mode
    if args.wrap_via:
        config.plain_text.wrap_via = args.wrap_via
    print(
        f"  插件配置   : task={config.repair.task_name} "
        f"mode={config.plain_text.mode} "
        f"fallback={config.repair.fallback_task_names}"
    )

    handler_mod = importlib.import_module("all_toolcall_failures_go_die.handler")
    repairer_mod = importlib.import_module("all_toolcall_failures_go_die.core.repairer")
    handler = handler_mod.ToolCallRepairHandler(FakePlugin(config))

    print("\n阶段 1 · 注册事件订阅（真实 EventBus）")
    bus = EventBus(name="probe")
    bus.subscribe(
        EventType.BEFORE_LLM_REQUEST,
        handler.execute,
        priority=handler.weight,
        timeout=handler.timeout,
    )
    bus.subscribe(
        EventType.AFTER_LLM_REQUEST,
        handler.execute,
        priority=handler.weight,
        timeout=handler.timeout,
    )
    print(
        f"  BEFORE/AFTER 均已订阅 priority={handler.weight} timeout={handler.timeout}"
    )

    print("\n阶段 2 · BEFORE_LLM_REQUEST：缓存本次工具目录")
    scenario = SCENARIOS[args.chatter]
    request_name = scenario["request_name"]
    before_params: dict[str, Any] = {
        "request_name": request_name,
        "model_identifier": "deepseek-v4-flash",
        "stream": False,
        "tools": scenario["tools"],
        "payloads": [],
        "meta_data": {"stream_id": "probe-stream-1"},
    }
    await bus.publish(EventType.BEFORE_LLM_REQUEST, before_params)
    catalog = None
    for (name, sid), entry in handler._catalogs.items():
        catalog = entry.catalog
        print(f"  缓存键=({name!r}, {sid!r}) 工具={entry.catalog.names()}")

    print(
        f"\n阶段 3 · AFTER_LLM_REQUEST：{('纯文本响应' if not args.broken else args.broken)}"
    )
    broken_calls: list[dict[str, Any]] = []
    if args.broken:
        broken_calls = _broken_calls(args.broken, scenario)
    after_params: dict[str, Any] = {
        "request_name": request_name,
        "model_identifier": "deepseek-v4-flash",
        "stream": False,
        "success": True,
        "latency": 0.42,
        "retry_count": 0,
        "message": text,
        "reasoning_content": None,
        "reasoning_parts": [],
        "tool_calls": broken_calls,
        "usage": {},
        "meta_data": {"stream_id": "probe-stream-1"},
    }
    print(f"  样本语句: {text!r}")
    if broken_calls:
        print(f"  构造的错误调用: {json.dumps(broken_calls, ensure_ascii=False)}")
    extraction_mod = importlib.import_module(
        "all_toolcall_failures_go_die.core.extraction"
    )
    if catalog is not None:
        looks = extraction_mod.looks_like_tool_call(text, catalog)
        candidates = extraction_mod.extract_candidates(text)
        print(f"  启发式判定像工具调用: {looks}")
        print(f"  本地可提取候选: {[c.name for c in candidates] or '无'}")

    captured = _spy(repairer_mod)
    started = time.perf_counter()
    decision, final_params = await bus.publish(
        EventType.AFTER_LLM_REQUEST, dict(after_params)
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    print("\n阶段 4 · 修复请求（真实模型调用）")
    if captured.get("models"):
        for entry in captured["models"]:
            print(f"  model_set: {entry}")
        print(f"  request_name={captured.get('request_name')!r}")
        for payload in captured.get("payloads", []):
            body = payload["text"]
            if args.show_prompts:
                print(f"  --- {payload['role']} ---\n{body}\n  --- end ---")
            else:
                print(f"  {payload['role']}: {len(body)} 字符")
    else:
        print("  未发起模型调用（本地已解决或未判定为需要修复）")

    print("\n阶段 5 · 模型原始回复")
    print(f"  {captured.get('reply')!r}")

    print("\n阶段 6 · 写回结果")
    print(f"  事件结论: {decision}")
    calls = final_params.get("tool_calls") or []
    print(f"  tool_calls({len(calls)}):")
    for call in calls:
        if isinstance(call, dict):
            print(
                f"    id={call.get('id')!r} name={call.get('name')!r} "
                f"args={json.dumps(call.get('args'), ensure_ascii=False)}"
            )
        else:
            print(f"    {call!r}")
    kept = final_params.get("message") == text
    print(f"  message 保留原文: {kept}  长度={len(final_params.get('message') or '')}")
    print(f"  耗时: {elapsed_ms:.0f} ms")

    # 关键判定：写回结果必须能直接交给 chatter 执行（再跑一遍本地校验）
    schemas_mod = importlib.import_module("all_toolcall_failures_go_die.core.schemas")
    options = schemas_mod.AuditOptions(
        require_known_tool=True,
        resolve_partial_names=True,
        drop_unknown_args=True,
        coerce_scalar_types=True,
        inject_reason=True,
    )
    print("\n阶段 6.5 · 写回结果的可执行性校验")
    executable = True
    for call in calls:
        audit = schemas_mod.audit_call(call, catalog, options)
        flag = "OK" if audit.ok else "问题"
        print(
            f"  [{flag}] name={audit.call.get('name')!r} "
            f"args={json.dumps(audit.call.get('args'), ensure_ascii=False)}"
        )
        for issue in audit.issues:
            print(f"        · [{issue.code}] {issue.message}")
        if not audit.ok or audit.needs_llm:
            executable = False

    print("\n阶段 7 · 判定")
    llm_called = bool(captured.get("models"))
    mode = str(config.plain_text.mode)
    if calls and executable:
        print("  ✅ 通过：坏响应被修成可执行的工具调用")
        ok = True
    elif calls:
        print("  ❌ 异常：写回的调用仍然不能执行")
        ok = False
    elif llm_called:
        print("  ✅ 通过：模型判定「不是工具调用」→ 原样放行（fail-open）")
        ok = True
    elif mode == "always":
        print("  ❌ 异常：always 模式下竟然没有发起修复请求")
        ok = False
    else:
        print("  ✅ 通过：本地启发式闸门直接挡下，零成本跳过（未发起模型调用）")
        ok = True
    if decision is None:
        print("  事件结论为空，异常路径")
    _restore(repairer_mod, captured)
    return 0 if ok else 1


def _broken_calls(kind: str, scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """构造「模型返回了坏掉的工具调用」样本。

    Args:
        kind: ``bad-name`` / ``missing-meta`` / ``string-args`` / ``no-prefix``。
        scenario: 场景字典（决定真实工具名）。

    Returns:
        ``params["tool_calls"]`` 形态的坏调用列表。
    """
    is_kfc = scenario["request_name"] == "kokoro_flow_chatter"
    if kind == "bad-name":
        # 工具名根本不存在（模型瞎编了一个）
        return [{"name": "reply", "args": {"content": ["在的，怎么了？"]}}]
    if kind == "missing-meta" and is_kfc:
        # 只有 content，缺 KFC 强制必填的四个元数据字段
        return [{"name": "kfc_reply", "args": {"content": ["在的，怎么了？"]}}]
    if kind == "missing-meta":
        # 通用场景：缺必填 content
        return [{"name": "send_text", "args": {}}]
    if kind == "string-args":
        return [
            {
                "name": "action-kfc_reply" if is_kfc else "action-send_text",
                "args": '{"content": ["在的，怎么了？"]}',
            }
        ]
    if kind == "no-prefix":
        name = "kfc_reply" if is_kfc else "send_text"
        return [{"name": name, "args": {"content": ["在的，怎么了？"]}}]
    return []


def _main() -> int:
    """解析参数并执行。"""
    parser = argparse.ArgumentParser(description="工具调用修复插件真实模型冒烟测试")
    parser.add_argument("text", nargs="*", help="样本语句（不传则用内置样本）")
    parser.add_argument("--text-file", help="从文件读取样本语句")
    parser.add_argument("--show-prompts", action="store_true", help="打印完整提示词")
    parser.add_argument(
        "--mode",
        choices=["off", "wrap", "suspicious", "always"],
        help="覆盖 plain_text.mode（默认用配置文件里的值）",
    )
    parser.add_argument(
        "--wrap-via",
        choices=["local-first", "always-llm"],
        help="覆盖 plain_text.wrap_via（always-llm 会真的调用模型套壳）",
    )
    parser.add_argument(
        "--chatter",
        choices=sorted(SCENARIOS),
        default="default",
        help="用哪个 chatter 的工具集跑（default=send_text，kfc=kfc_reply）",
    )
    parser.add_argument(
        "--broken",
        choices=["bad-name", "missing-meta", "string-args", "no-prefix"],
        help="构造一份「坏掉的工具调用」当输入（不传则走纯文本路径）",
    )
    args = parser.parse_args()
    args.text = " ".join(args.text) if args.text else ""
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(_main())
