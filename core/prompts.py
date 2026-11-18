"""修复请求的提示词构建与回复解析（纯标准库，可离线单测）。

设计原则是「token 花在刀刃上」：

- 只发一条 system + 一条 user，不带聊天历史、不带图片、不带工具 schema 全集；
- 工具目录压成一行一个的短目录，只有「被点名的工具」才附完整参数 schema；
- 强制模型只回一个 JSON，回复解析失败就放弃（绝不猜测、绝不再多轮）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .extraction import extract_candidates
from .schemas import (
    ToolCatalog,
    ToolSpec,
    parse_json_text,
)

#: 修复结果是「一批工具调用」。
KIND_CALLS = "calls"
#: 修复模型判定「原输出不是工具调用」。
KIND_NONE = "none"
#: 回复不可解析 / 不可用。
KIND_INVALID = "invalid"

_SYSTEM_PROMPT = """\
你是「工具调用修复器」。你唯一的任务：把一条无法被机器人框架解析的助手输出，\
改写成语义等价、可被框架正确解析的工具调用。

硬性规则：
1. 只输出一个 JSON 对象；不要解释、不要思考过程、不要 markdown 代码块。
2. 正常输出格式：{"tool_calls":[{"name":"<工具名>","args":{...}}]}
3. 若原输出确实不是工具调用（只是一句普通回复，或没有任何可用工具能表达它），\
输出：{"no_tool_call":true,"reason":"<不超过30字的理由>"}
4. name 必须与「可用工具」里的名字逐字一致，禁止发明新工具、禁止改大小写。
5. args 必须是 JSON 对象，且必须包含该工具的全部必填参数；不要给出参数表之外的名字。
6. 严格保留原输出中的实际内容（例如要发送的文本、要查询的关键词），\
不要改写语义、不要扩写、不要编造原输出没有的内容。
7. 参数值的类型必须符合 schema（integer 就是数字，不要写成字符串）。
8. `reason` 由框架自动补充，可以省略。\
"""


@dataclass(frozen=True, slots=True)
class RepairDecision:
    """修复模型给出的结论。"""

    kind: str
    """``calls`` / ``none`` / ``invalid``。"""

    candidates: list[Any] = field(default_factory=list)
    """``kind == "calls"`` 时的候选调用（:class:`~..extraction.RawCallCandidate`）。"""

    reason: str = ""
    """``kind == "none"`` 时模型给出的理由。"""

    raw: str = ""
    """修复模型的原始回复（截断后），用于日志。"""

    @property
    def ok(self) -> bool:
        """是否产出了可用调用。"""
        return self.kind == KIND_CALLS and bool(self.candidates)


def build_system_prompt() -> str:
    """返回修复请求的 system 提示词。"""
    return _SYSTEM_PROMPT


_WRAP_SYSTEM_PROMPT = """\
你是「消息发送套壳器」。机器人框架只认工具调用，模型却直接把要说的话当成了普通文本，\
你要把这段文本套进正确的发送工具里，让框架能正常发出去。

硬性规则：
1. 只输出一个 JSON 对象；不要解释、不要思考过程、不要 markdown 代码块。
2. 输出格式：{"tool_calls":[{"name":"<发送工具名>","args":{...}}]}
3. 必须选「可用工具」里真正负责把文本发给用户的那个工具；名字逐字照抄，禁止发明。
4. 正文必须原样放进它的正文参数，一个字都不要改写、删除、补全或拆句。
5. 该工具 schema 里**所有必填参数都必须给**。除了正文参数，其余必填参数（常见于
   thought / expected_reaction / max_wait_seconds / mood 这类决策元数据）按原文语气
   给简短、合理、类型正确的值：字符串写一小句，数字给 0，布尔给 false，数组给一项。
   不要留空、不要省略，否则框架会拒绝这次调用。
6. 非必填参数不要填（例如引用 reply_to、@、表情包），除非原文明确要求。
7. 只有在「可用工具」里确实没有任何能发送文本的工具时，才输出：\
{"no_tool_call":true,"reason":"<不超过30字的理由>"}\
"""

#: 正文参数名候选（套壳提示词里提示模型用）。
WRAP_BODY_HINT = ("content", "text", "message", "msg", "reply", "body")


def build_wrap_system_prompt() -> str:
    """返回「纯文本套壳」请求的 system 提示词。"""
    return _WRAP_SYSTEM_PROMPT


def build_wrap_user_prompt(
    *,
    raw_text: str,
    catalog: ToolCatalog,
    candidates: Sequence[ToolSpec] = (),
    body_params: Sequence[str] = WRAP_BODY_HINT,
    max_chars: int = 8000,
    max_tools: int = 60,
    max_description: int = 120,
) -> str:
    """构建「把纯文本套壳成发送调用」的 user 提示词。

    Args:
        raw_text: 模型直接吐出的纯文本（即要发给用户的内容）。
        catalog: 本次请求可用工具集。
        candidates: 本地识别出的发送工具候选（用于重点展示）。
        body_params: 正文参数名候选。
        max_chars: 原文最大字符数。
        max_tools: 目录里最多列多少个工具。
        max_description: 每个工具描述最大字符数。

    Returns:
        完整的 user 提示词文本。
    """
    focus = [spec.name for spec in candidates]
    lines: list[str] = ["【可用工具】（name：描述 | 必填/可选参数）"]
    lines.extend(_render_catalog(catalog, focus, max_tools, max_description))
    lines.append("")

    lines.append("【发送类工具候选】")
    if candidates:
        for spec in candidates:
            params = ", ".join(repr(key) for key in spec.properties)
            required = ", ".join(repr(key) for key in spec.required) or "无"
            lines.append(
                f"- {spec.name}｜必填: {required}（全部都要给）｜全部参数: {params or '无'}"
            )
    else:
        lines.append("- （未预判出候选，请从上面的工具里自行选择，并补齐它的必填参数）")
    lines.append("")

    details = _render_details(catalog, focus, max_details=2)
    if details:
        lines.append("【候选工具的完整参数 schema】")
        lines.extend(details)
        lines.append("")

    lines.append("【正文参数名候选】")
    lines.append(", ".join(repr(key) for key in body_params))
    lines.append("")

    lines.append("【模型直接输出的文本】（这就是要发给用户的正文，必须原样使用）")
    lines.append(_truncate(raw_text.strip() or "（空）", max_chars))
    lines.append("")

    lines.append(
        "【任务】把这段文本套成一次发送工具调用，输出 "
        '{"tool_calls":[{"name":"<发送工具名>","args":{...}}]}；'
        "正文参数放原文且一字不改，该工具的全部必填参数都要填上。"
    )
    return "\n".join(lines)


def build_plan_user_prompt(
    *,
    raw_text: str,
    catalog: ToolCatalog,
    current_calls: Sequence[Mapping[str, Any]] = (),
    errors: Sequence[str] = (),
    max_chars: int = 8000,
    max_tools: int = 60,
    max_description: int = 120,
    max_details: int = 5,
) -> str:
    """构建「从输出重建工具调用」的 user 提示词。

    Args:
        raw_text: 助手原始输出（可含说明文字/坏 JSON）。
        catalog: 本次请求可用工具集。
        current_calls: 框架已经解析出的调用（可能为空或部分损坏）。
        errors: 框架视角的报错文本列表。
        max_chars: 原始输出最大字符数。
        max_tools: 目录里最多列多少个工具。
        max_description: 每个工具描述最大字符数。
        max_details: 最多给几个工具附完整参数 schema。

    Returns:
        完整的 user 提示词文本。
    """
    focus = _focus_names(raw_text, current_calls, catalog)
    lines: list[str] = ["【可用工具】（name：描述 | 必填/可选参数）"]
    lines.extend(_render_catalog(catalog, focus, max_tools, max_description))
    lines.append("")

    details = _render_details(catalog, focus, max_details)
    if details:
        lines.append("【重点工具的完整参数 schema】")
        lines.extend(details)
        lines.append("")

    lines.append("【原始助手输出】（这可能是一段坏掉的工具调用）")
    lines.append(_truncate(raw_text.strip() or "（空）", max_chars))
    lines.append("")

    lines.append("【框架已经解析出的工具调用】")
    lines.append(_render_current_calls(current_calls))
    lines.append("")

    if errors:
        lines.append("【框架报错】")
        lines.extend(f"- {error}" for error in errors)
        lines.append("")

    lines.append(
        "【任务】请基于原始助手输出的真实意图，输出一个 JSON 对象修正上述问题；"
        '若原输出根本不是工具调用，输出 {"no_tool_call":true,"reason":"..."}。'
    )
    return "\n".join(lines)


def build_args_user_prompt(
    *,
    raw_text: str,
    spec: ToolSpec,
    current_args: Mapping[str, Any] | None = None,
    errors: Sequence[str] = (),
    max_chars: int = 4000,
) -> str:
    """构建「只补参数」的 user 提示词（单工具、最省 token 的形态）。

    Args:
        raw_text: 助手原始输出。
        spec: 目标工具。
        current_args: 已解析出的参数（可能缺字段/类型不对）。
        errors: 框架视角的报错文本列表。
        max_chars: 原始输出最大字符数。

    Returns:
        完整的 user 提示词文本。
    """
    schema_text = json.dumps(
        {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    lines = [
        f"【目标工具】{spec.name}",
        f"【参数 schema】{schema_text}",
        "",
        "【原始助手输出】",
        _truncate(raw_text.strip() or "（空）", max_chars),
        "",
        "【框架已经解析出的参数】",
        json.dumps(dict(current_args or {}), ensure_ascii=False),
        "",
    ]
    if errors:
        lines.append("【框架报错】")
        lines.extend(f"- {error}" for error in errors)
        lines.append("")
    lines.append(
        '【任务】只补齐/纠正这个工具的参数，输出 {"tool_calls":'
        '[{"name":"' + spec.name + '","args":{...}}]}；'
        "参数值必须来自原始助手输出，不要凭空编造。"
    )
    return "\n".join(lines)


def parse_repair_reply(text: str, *, fallback_name: str = "") -> RepairDecision:
    """解析修复模型的回复。

    Args:
        text: 修复模型返回的文本。
        fallback_name: 仅解析出参数（如 ``{"args": {...}}``）时使用的工具名。

    Returns:
        :class:`RepairDecision`。
    """
    raw = (text or "").strip()
    if not raw:
        return RepairDecision(KIND_INVALID, raw=raw)

    parsed, _ = parse_json_text(raw)

    if isinstance(parsed, Mapping) and _truthy(parsed.get("no_tool_call")):
        reason = parsed.get("reason")
        return RepairDecision(
            KIND_NONE,
            reason=str(reason)[:200] if reason is not None else "",
            raw=_truncate(raw, 400),
        )

    candidates = extract_candidates(raw)
    if candidates:
        return RepairDecision(
            KIND_CALLS,
            candidates=candidates,
            raw=_truncate(raw, 400),
        )

    if fallback_name and isinstance(parsed, Mapping):
        for key in ("args", "arguments", "parameters", "input"):
            value = parsed.get(key)
            if isinstance(value, Mapping):
                from .extraction import RawCallCandidate

                return RepairDecision(
                    KIND_CALLS,
                    candidates=[RawCallCandidate(name=fallback_name, args=dict(value))],
                    raw=_truncate(raw, 400),
                )

    return RepairDecision(KIND_INVALID, raw=_truncate(raw, 400))


def _focus_names(
    raw_text: str,
    current_calls: Sequence[Mapping[str, Any]],
    catalog: ToolCatalog,
) -> list[str]:
    """挑出需要「重点展示」的工具名：调用里出现的 > 文本里提到的。"""
    focus: list[str] = []
    for call in current_calls:
        name = call.get("name")
        resolved = catalog.resolve(name, allow_partial=True)
        if resolved and resolved not in focus:
            focus.append(resolved)
    for spec in catalog.mentioned(raw_text):
        if spec.name not in focus:
            focus.append(spec.name)
    return focus


def _render_catalog(
    catalog: ToolCatalog,
    focus: Iterable[str],
    max_tools: int,
    max_description: int,
) -> list[str]:
    """渲染紧凑工具目录，重点工具排在最前面。"""
    focus_set = set(focus)
    ordered = [name for name in focus if name in catalog.specs]
    ordered.extend(name for name in catalog.names() if name not in focus_set)
    if max_tools > 0:
        visible = ordered[:max_tools]
    else:
        visible = ordered
    lines = [
        catalog.specs[name].short_line(max_description=max_description)
        for name in visible
    ]
    hidden = len(ordered) - len(visible)
    if hidden > 0:
        lines.append(f"- （另有 {hidden} 个工具未列出，原则上不要使用它们）")
    return lines


def _render_details(
    catalog: ToolCatalog,
    focus: Iterable[str],
    max_details: int,
) -> list[str]:
    """渲染重点工具的完整参数 schema。"""
    lines: list[str] = []
    for name in list(focus)[: max(max_details, 0)]:
        spec = catalog.get(name)
        if spec is None:
            continue
        payload = {
            "name": spec.name,
            "parameters": spec.parameters,
        }
        lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return lines


def _render_current_calls(calls: Sequence[Mapping[str, Any]]) -> str:
    """渲染框架已解析出的调用（无则给出空数组）。"""
    if not calls:
        return "[]"
    normalized = [
        {
            "name": str(call.get("name") or ""),
            "args": call.get("args") if isinstance(call.get("args"), Mapping) else {},
        }
        for call in calls
    ]
    return json.dumps(normalized, ensure_ascii=False)


def _truthy(value: Any) -> bool:
    """宽松判断 JSON 里的布尔值。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _truncate(text: str, limit: int) -> str:
    """按字符数截断文本。"""
    if limit <= 0 or len(text) <= limit:
        return text
    head = text[: max(limit - 20, 0)]
    return head + "\n…（原始输出过长，已截断）"
