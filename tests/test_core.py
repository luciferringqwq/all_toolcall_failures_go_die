"""``core`` 纯标准库部分的离线自测。

不依赖 bot 运行时、不联网、不需要 pytest：

    python plugins/all_toolcall_failures_go_die/tests/test_core.py

若环境里有 pytest，也可以直接 ``pytest tools/...``（函数名兼容）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from core.extraction import extract_candidates, looks_like_tool_call  # noqa: E402
from core.prompts import (  # noqa: E402
    KIND_CALLS,
    KIND_INVALID,
    KIND_NONE,
    build_args_user_prompt,
    build_plan_user_prompt,
    build_system_prompt,
    parse_repair_reply,
)
from core.schemas import (  # noqa: E402
    AuditOptions,
    ToolCatalog,
    ToolSpec,
    audit_call,
    coerce_args_object,
    parse_json_text,
)
from core.send_tool import (  # noqa: E402
    SendToolSettings,
    build_wrap_call,
    find_send_tool,
)

SEND_TEXT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "action-send_text",
        "description": "发送一段文本消息给用户",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要发送的文本"},
                "reply_to": {"type": "string", "description": "引用消息 ID"},
            },
            "required": ["content"],
        },
    },
}

WAIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "action-pass_and_wait",
        "description": "等待新消息",
        "parameters": {
            "type": "object",
            "properties": {"seconds": {"type": "number", "description": "等待秒数"}},
            "required": [],
        },
    },
}

CATALOG = ToolCatalog(
    {
        spec.name: spec
        for spec in (
            ToolSpec.from_schema(SEND_TEXT_SCHEMA),
            ToolSpec.from_schema(WAIT_SCHEMA),
        )
        if spec is not None
    }
)


def test_tool_spec_from_schema() -> None:
    """schema 应被正确摘要。"""
    spec = ToolSpec.from_schema(SEND_TEXT_SCHEMA)
    assert spec is not None
    assert spec.name == "action-send_text"
    assert spec.required == ("content",)
    assert "必填: content" in spec.short_line()


def test_catalog_resolve_partial_names() -> None:
    """缺类型前缀的工具名应被本地纠正。"""
    assert CATALOG.resolve("action-send_text") == "action-send_text"
    assert CATALOG.resolve("send_text") == "action-send_text"
    assert CATALOG.resolve("Send_Text") == "action-send_text"
    assert CATALOG.resolve("nope") is None


def test_parse_json_text_variants() -> None:
    """JSON 解析应容错围栏与前后杂质。"""
    value, source = parse_json_text('```json\n{"a": 1}\n```')
    assert value == {"a": 1}
    assert source == "fence"

    value, source = parse_json_text('好的：{"a": {"b": 2}} 完毕')
    assert value == {"a": {"b": 2}}
    assert source == "slice"

    assert parse_json_text("完全不是 JSON")[0] is None


def test_coerce_args_object() -> None:
    """字符串形式的 arguments 应被解析成 dict。"""
    args, error = coerce_args_object('{"content": "你好"}')
    assert error is None
    assert args == {"content": "你好"}

    args, error = coerce_args_object("{不是 json}")
    assert args == {}
    assert error is not None and "JSON" in error


def test_audit_missing_required() -> None:
    """缺必填参数应产生需要 LLM 修复的问题。"""
    audit = audit_call(
        {"name": "action-send_text", "args": {"reply_to": "m1"}},
        CATALOG,
        AuditOptions(),
    )
    assert not audit.ok
    assert audit.needs_llm
    codes = {issue.code for issue in audit.issues}
    assert "missing_required" in codes
    assert audit.call["args"]["reason"] == "修正工具调用格式"


def test_audit_unknown_tool() -> None:
    """未知工具名应被标记。"""
    audit = audit_call(
        {"name": "action-whatever", "args": {}},
        CATALOG,
        AuditOptions(),
    )
    assert {issue.code for issue in audit.issues} == {"unknown_tool"}


def test_audit_local_fixes_without_llm() -> None:
    """名字、args 字符串、类型、多余参数都应本地修好且不需要 LLM。"""
    audit = audit_call(
        {"name": "pass_and_wait", "args": '{"seconds": "30", "extra": 1}'},
        CATALOG,
        AuditOptions(),
    )
    assert audit.ok
    assert audit.changed
    assert audit.call["name"] == "action-pass_and_wait"
    assert audit.call["args"]["seconds"] == 30
    assert "extra" not in audit.call["args"]
    assert audit.call["args"]["reason"]


def test_audit_var_kwargs_keeps_extra_args() -> None:
    """宜接受 **kwargs 的工具不应被删参数。"""
    spec = ToolSpec.from_schema(SEND_TEXT_SCHEMA)
    assert spec is not None
    loose = ToolSpec(
        name=spec.name,
        description=spec.description,
        parameters=spec.parameters,
        accepts_var_kwargs=True,
    )
    catalog = ToolCatalog({loose.name: loose})
    audit = audit_call(
        {"name": "action-send_text", "args": {"content": "hi", "extra": 1}},
        catalog,
        AuditOptions(),
    )
    assert audit.call["args"]["extra"] == 1


def test_audit_drops_args_for_no_param_tool() -> None:
    """schema 无参数的工具，多传的参数应被清掉（否则执行会报 unexpected kwarg）。"""
    no_param_schema = {
        "type": "function",
        "function": {
            "name": "action-do_nothing",
            "description": "什么也不做",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    spec = ToolSpec.from_schema(no_param_schema)
    assert spec is not None
    audit = audit_call(
        {"name": "action-do_nothing", "args": {"stray": "x", "reason": "选择不回复"}},
        ToolCatalog({spec.name: spec}),
        AuditOptions(),
    )
    assert audit.ok
    assert "stray" not in audit.call["args"]
    assert audit.call["args"]["reason"] == "选择不回复"


def test_extract_candidates_shapes() -> None:
    """各种包裹形态都应能提取出候选调用。"""
    shapes = [
        '{"name": "send_text", "args": {"content": "hi"}}',
        '{"tool_calls": [{"name": "send_text", "arguments": {"content": "hi"}}]}',
        '{"tool_call": {"function": {"name": "send_text", "arguments": "{\\"content\\": \\"hi\\"}"}}}',
        '```json\n{"name": "send_text", "parameters": {"content": "hi"}}\n```',
        '我这就发送：send_text(content="hi", at=None)',
    ]
    for text in shapes:
        candidates = extract_candidates(text)
        assert candidates, text
        assert candidates[0].name == "send_text", text


def test_looks_like_tool_call() -> None:
    """工具调用特征应被识别，普通回复不应误判。"""
    assert looks_like_tool_call('{"name": "send_text", "args": {}}', CATALOG)
    assert looks_like_tool_call("我想用 send_text 发句话", CATALOG)
    assert not looks_like_tool_call("今天天气不错，我们出去走走吧。", CATALOG)


def test_plan_prompt_and_reply() -> None:
    """目录渲染与回复解析应闭环。"""
    prompt = build_plan_user_prompt(
        raw_text='{"name": "send_text", "args": {}}',
        catalog=CATALOG,
        current_calls=[{"name": "action-send_text", "args": {"reply_to": "m1"}}],
        errors=["调用 action-send_text 缺少必填参数 content"],
    )
    assert "action-send_text" in prompt
    assert "缺少必填参数 content" in prompt
    assert "【可用工具】" in prompt

    decision = parse_repair_reply(
        '{"tool_calls": [{"name": "action-send_text", "args": {"content": "你好"}}]}'
    )
    assert decision.kind == KIND_CALLS
    assert decision.candidates[0].args == {"content": "你好"}


def test_parse_repair_reply_none_and_args() -> None:
    """``no_tool_call`` 与只给 args 的回复都应被正确处理。"""
    decision = parse_repair_reply('{"no_tool_call": true, "reason": "只是闲聊"}')
    assert decision.kind == KIND_NONE
    assert "闲聊" in decision.reason

    decision = parse_repair_reply(
        '{"args": {"content": "hi"}}', fallback_name="action-send_text"
    )
    assert decision.kind == KIND_CALLS
    assert decision.candidates[0].name == "action-send_text"

    assert parse_repair_reply("嗯嗯好的呀").kind == KIND_INVALID


def test_args_prompt_is_compact() -> None:
    """补参提示词应包含目标工具与报错。"""
    spec = CATALOG.get("action-send_text")
    assert spec is not None
    prompt = build_args_user_prompt(
        raw_text="发送：你好",
        spec=spec,
        current_args={},
        errors=["缺少必填参数 content"],
    )
    assert "action-send_text" in prompt
    assert "缺少必填参数 content" in prompt
    assert "工具调用修复器" in build_system_prompt()


def test_find_send_tool_by_name() -> None:
    """按名字模式识别发送工具，并排除「像发送但不是文本」的工具。"""
    emoji = ToolSpec.from_schema(
        {
            "type": "function",
            "function": {
                "name": "action-send_emoji_meme",
                "description": "发送一个表情包",
                "parameters": {
                    "type": "object",
                    "properties": {"description": {"type": "string"}},
                    "required": ["description"],
                },
            },
        }
    )
    send = ToolSpec.from_schema(SEND_TEXT_SCHEMA)
    assert emoji is not None and send is not None
    catalog = ToolCatalog({emoji.name: emoji, send.name: send})

    match = find_send_tool(catalog, "neo_default_chatter", SendToolSettings())
    assert match.resolved
    assert match.tool is not None and match.tool.name == "action-send_text"
    assert match.body_param == "content"


def test_find_send_tool_prefers_current_chatter() -> None:
    """两个 chatter 各有一个发送工具时，应按 chatter_allow 选中当前 chat。"""
    left = ToolSpec(
        name="action-send_text",
        description="发送文本",
        parameters={
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
        chatter_allow=("chatter_a",),
    )
    right = ToolSpec(
        name="action-reply_message",
        description="回复消息",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        chatter_allow=("chatter_b",),
    )
    catalog = ToolCatalog({left.name: left, right.name: right})

    match_a = find_send_tool(catalog, "chatter_a", SendToolSettings())
    assert match_a.resolved and match_a.tool is not None
    assert match_a.tool.name == "action-send_text"

    match_b = find_send_tool(catalog, "chatter_b", SendToolSettings())
    assert match_b.resolved and match_b.tool is not None
    assert match_b.tool.name == "action-reply_message"
    assert match_b.body_param == "text"


def test_find_send_tool_structural_and_missing() -> None:
    """名字不含 send/reply 时走结构特征；完全没有发送工具时应识别为空。"""
    odd = ToolSpec(
        name="action-post_out",
        description="把内容推送到当前会话",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "dry_run": {"type": "boolean"},
            },
            "required": ["message"],
        },
    )
    weather = ToolSpec(
        name="action-check_weather",
        description="查询天气",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    match = find_send_tool(
        ToolCatalog({odd.name: odd, weather.name: weather}),
        "neo_default_chatter",
        SendToolSettings(),
    )
    assert match.resolved and match.tool is not None
    assert match.tool.name == "action-post_out"

    empty = find_send_tool(
        ToolCatalog({weather.name: weather}), "neo_default_chatter", SendToolSettings()
    )
    assert not empty.resolved
    assert empty.candidates == []


def test_find_send_tool_pinned_name() -> None:
    """配置显式指定工具名时应直接采用。"""
    send = ToolSpec.from_schema(SEND_TEXT_SCHEMA)
    assert send is not None
    settings = SendToolSettings(send_tool_name="send_text")
    match = find_send_tool(ToolCatalog({send.name: send}), "x", settings)
    assert match.resolved and match.tool is not None
    assert match.tool.name == "action-send_text"
    assert build_wrap_call(match.tool, match.body_param, "你好") == {
        "name": "action-send_text",
        "args": {"content": "你好"},
    }


def _run_all() -> int:
    """依次运行所有 ``test_*`` 函数。"""
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed = 0
    for name, func in tests:
        try:
            func()
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
        else:
            print(f"OK   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
