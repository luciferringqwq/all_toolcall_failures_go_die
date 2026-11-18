"""从「纯文本 / 坏掉的输出」中抢救工具调用（纯标准库，可离线单测）。

模型把工具调用写进正文而不是 ``tool_calls`` 字段时，形态非常多：

- 合法 JSON：``{"name": "action-send_text", "args": {"content": "hi"}}``
- 各种外层包裹：``tool_calls`` / ``tool_call`` / ``function_call`` / ``function``
- markdown 围栏、正文夹杂说明文字、参数被二次编码成字符串
- 函数调用语法：``send_text(content="hi")``

本模块只产出「候选调用」，合法性由 :mod:`core.schemas` 负责校验。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .schemas import ToolCatalog, iter_balanced_json, parse_json_text

#: 常见的外层容器键，值可能是单个对象或数组。
_CONTAINER_KEYS = (
    "tool_calls",
    "tool_call",
    "tools_call",
    "function_call",
    "functionCalls",
    "calls",
    "invoke",
    "actions",
)

#: ``args`` 字段的常见别名。
_ARGS_KEYS = ("args", "arguments", "parameters", "params", "input", "arguments_json")

#: ``name`` 字段的常见别名。
_NAME_KEYS = ("name", "tool", "tool_name", "action", "function_name", "callee")

#: 函数调用语法：identifier( ... )
_CALL_RE = re.compile(r"(?<![\w.\-])([A-Za-z_][\w\-]*(?:[\w\-])?)\s*\(")


@dataclass(frozen=True, slots=True)
class RawCallCandidate:
    """从文本中抢救出的一个候选工具调用。"""

    name: str
    args: Any = field(default_factory=dict)
    call_id: str | None = None
    source: str = "json"
    """来源：``json`` / ``syntax``。"""


def extract_candidates(text: str) -> list[RawCallCandidate]:
    """从文本中提取所有候选工具调用。

    Args:
        text: 助手原始输出。

    Returns:
        候选列表（按出现顺序，可能为空）。参数值不做校验。
    """
    if not text or not text.strip():
        return []

    candidates: list[RawCallCandidate] = []
    seen: set[tuple[str, str]] = set()

    def add(candidate: RawCallCandidate) -> None:
        key = (candidate.name, repr(candidate.args))
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    parsed, _ = parse_json_text(text)
    if parsed is not None:
        for candidate in _candidates_from_value(parsed):
            add(candidate)

    if not candidates:
        for chunk in iter_balanced_json(text):
            value, _ = parse_json_text(chunk)
            if value is None:
                continue
            for candidate in _candidates_from_value(value):
                add(candidate)
            if candidates:
                break

    for candidate in _candidates_from_syntax(text):
        add(candidate)

    return candidates


def _candidates_from_value(value: Any, depth: int = 0) -> list[RawCallCandidate]:
    """把解析后的 JSON 值转换成候选调用（支持常见包裹形态）。"""
    if depth > 4:
        return []

    if isinstance(value, list):
        out: list[RawCallCandidate] = []
        for item in value:
            out.extend(_candidates_from_value(item, depth + 1))
        return out

    if not isinstance(value, Mapping):
        return []

    for key in _CONTAINER_KEYS:
        if key in value:
            nested = _candidates_from_value(value[key], depth + 1)
            if nested:
                return nested

    function = value.get("function")
    if isinstance(function, Mapping):
        merged = dict(function)
        if "id" in value and "id" not in merged:
            merged["id"] = value.get("id")
        nested = _candidates_from_value(merged, depth + 1)
        if nested:
            return nested

    name = _pick(value, _NAME_KEYS)
    if isinstance(name, str) and name.strip():
        args: Any = {}
        for key in _ARGS_KEYS:
            if key in value:
                args = value[key]
                break
        else:
            if "function" in value:
                args = value["function"]
        call_id = value.get("id")
        return [
            RawCallCandidate(
                name=name.strip(),
                args=args if args is not None else {},
                call_id=call_id if isinstance(call_id, str) else None,
                source="json",
            )
        ]

    # 常见误写：{"tool_calls": ...} 之外还套了一层 {"result": {...}} 之类
    for nested_value in value.values():
        if isinstance(nested_value, (Mapping, list)):
            nested = _candidates_from_value(nested_value, depth + 1)
            if nested:
                return nested
    return []


def _pick(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    """按键优先级取第一个存在的值。"""
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _candidates_from_syntax(text: str) -> list[RawCallCandidate]:
    """解析 ``name(k=v, ...)`` 形态的伪函数调用。"""
    out: list[RawCallCandidate] = []
    for match in _CALL_RE.finditer(text):
        name = match.group(1)
        call_text = _slice_call(text, match.start())
        if call_text is None:
            continue
        candidate = _parse_call_expression(call_text, name)
        if candidate is not None:
            out.append(candidate)
    return out


def _slice_call(text: str, start: int) -> str | None:
    """从 ``start`` 开始截取平衡括号内的完整调用表达式。"""
    open_index = text.find("(", start)
    if open_index < 0:
        return None
    depth = 0
    in_string: str | None = None
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if in_string is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            continue
        if char in {"'", '"'}:
            in_string = char
            continue
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _parse_call_expression(call_text: str, name: str) -> RawCallCandidate | None:
    """用 ``ast`` 安全解析伪函数调用（不执行任何代码）。"""
    try:
        node = ast.parse(call_text, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call):
        return None
    if node.args or not node.keywords:
        return None

    args: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            return None
        try:
            args[keyword.arg] = ast.literal_eval(keyword.value)
        except Exception:
            return None
    return RawCallCandidate(name=name.strip(), args=args, source="syntax")


#: 纯文本「像工具调用」的强特征词。
_SUSPICIOUS_TOKENS = (
    "tool_calls",
    "tool_call",
    "function_call",
    "arguments",
    '"args"',
    "tool_use",
)


def looks_like_tool_call(text: str, catalog: ToolCatalog) -> bool:
    """判断纯文本是否像「被写坏的工具调用」。

    宁可漏判也不要误判：普通聊天回复不应被这里拦下，否则会把正常回复
    改写成工具调用。

    Args:
        text: 助手原始输出。
        catalog: 本次可用工具集（用于识别工具名）。

    Returns:
        像工具调用时返回 ``True``。
    """
    stripped = text.strip()
    if not stripped:
        return False

    # 1) 整段就是 JSON 对象/数组
    if stripped[0] in "{[":
        parsed, _ = parse_json_text(stripped)
        if parsed is not None and _candidates_from_value(parsed):
            return True

    lowered = stripped.lower()

    # 2) 出现明确的工具调用字段名
    if any(token in lowered for token in _SUSPICIOUS_TOKENS):
        return True

    # 3) 出现 XML 风格的工具调用标签
    if "<tool_call" in lowered or "<function=" in lowered or "<antml:" in lowered:
        return True

    # 4) 文本里出现了已知工具名（完整名或去掉前缀后的短名）
    for name in catalog.names():
        if name.lower() in lowered:
            return True
        short = name.split("-", 1)[-1].lower()
        if len(short) >= 4 and re.search(
            rf"(?<![\w]){re.escape(short)}(?![\w])", lowered
        ):
            return True

    return False
