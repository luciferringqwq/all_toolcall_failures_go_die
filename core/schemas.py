"""工具 Schema 目录与工具调用校验（纯标准库，可离线单测）。

本模块刻意不导入框架代码，只处理**普通 dict**：

- :class:`ToolSpec`：单个可用工具的 schema 摘要。
- :class:`ToolCatalog`：本次请求可用的工具集合，提供名字解析。
- :func:`audit_call`：把一个 ``{id, name, args}`` 调用与 schema 对照，
  尽量本地修复，修不掉的产出一条 :class:`CallIssue`。

「框架视角」的报错文本由这里生成，用于喂给修复模型，因此文案要贴近框架
真实行为：未知工具、``TypeError: missing argument``、``unexpected keyword`` 等。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: 框架会自动为几乎每个工具注入 ``reason`` 参数（见 openai_client 的
#: ``_to_openai_tool``），因此它既不该被当成「缺失必填」，也不该被当成多余参数。
REASON_KEY = "reason"

#: 允许参与标量类型纠正的 JSON Schema 类型。
_SCALAR_TYPES = ("string", "integer", "number", "boolean")


def _schema_types(prop: Mapping[str, Any]) -> tuple[str, ...]:
    """读取属性 schema 声明的类型集合（兼容 ``type`` 为 list 的写法）。"""
    raw = prop.get("type")
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list):
        return tuple(item for item in raw if isinstance(item, str))
    return ()


def _truncate(text: str, limit: int) -> str:
    """按字符数截断文本并追加省略号。"""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)] + "…"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """一个可用工具的 schema 摘要。

    Attributes:
        name: 工具在 schema 中的完整名字（框架会自动加类型前缀，
            例如 ``action-send_text``），与 tool call 的 ``name`` 字段同源。
        description: 工具描述（可能被截断）。
        parameters: JSON Schema 的 ``parameters`` 片段。
        accepts_var_kwargs: 该工具的 ``execute`` 是否接受 ``**kwargs``。
            为 ``True`` 时不能随意删除 schema 之外的参数。
        chatter_allow: 组件声明的所属 chatter 列表（``chatter_allow`` 属性）。
            空元组表示对该属性不作限制。用于判断「这个工具属于哪个 chatter」。
    """

    name: str
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    accepts_var_kwargs: bool = False
    chatter_allow: tuple[str, ...] = ()

    @property
    def properties(self) -> Mapping[str, Any]:
        """返回 ``parameters.properties``（缺失时为空 dict）。"""
        props = self.parameters.get("properties")
        return props if isinstance(props, Mapping) else {}

    @property
    def required(self) -> tuple[str, ...]:
        """返回必填参数名（不含 ``reason``）。"""
        raw = self.parameters.get("required")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return ()
        return tuple(
            item
            for item in raw
            if isinstance(item, str) and item and item != REASON_KEY
        )

    def param_schema(self, key: str) -> Mapping[str, Any]:
        """取单个参数的 schema 片段。"""
        value = self.properties.get(key)
        return value if isinstance(value, Mapping) else {}

    def short_line(self, *, max_description: int = 120) -> str:
        """渲染成便宜的一行目录文本，用于修复提示词。"""
        description = _truncate(
            self.description.replace("\n", " ").strip(), max_description
        )
        parts = [f"- {self.name}"]
        if description:
            parts.append(f"：{description}")
        required = self.required
        optional = [
            key for key in self.properties if key != REASON_KEY and key not in required
        ]
        if required:
            parts.append(f" | 必填: {', '.join(required)}")
        if optional:
            parts.append(f" | 可选: {', '.join(optional)}")
        return "".join(parts)

    @classmethod
    def from_schema(
        cls,
        schema: Any,
        *,
        accepts_var_kwargs: bool = False,
    ) -> "ToolSpec | None":
        """从 ``type:function`` 形式（或裸 function 形式）的 schema 构造。

        Args:
            schema: 组件 ``to_schema()`` 的返回值。
            accepts_var_kwargs: 该组件 ``execute`` 是否接受 ``**kwargs``。

        Returns:
            解析成功时返回 :class:`ToolSpec`，否则返回 ``None``。
        """
        if not isinstance(schema, Mapping):
            return None
        function = schema.get("function")
        block = function if isinstance(function, Mapping) else schema
        name = block.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        parameters = block.get("parameters")
        description = block.get("description")
        return cls(
            name=name.strip(),
            description=description if isinstance(description, str) else "",
            parameters=parameters if isinstance(parameters, Mapping) else {},
            accepts_var_kwargs=accepts_var_kwargs,
        )


@dataclass(frozen=True, slots=True)
class ToolCatalog:
    """一次 LLM 请求可用的工具集合。"""

    specs: Mapping[str, ToolSpec] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """空目录视为不可用。"""
        return bool(self.specs)

    def __len__(self) -> int:
        """返回工具数量。"""
        return len(self.specs)

    def get(self, name: str) -> ToolSpec | None:
        """按完整名字取工具。"""
        return self.specs.get(name)

    def names(self) -> tuple[str, ...]:
        """返回全部工具名（有序）。"""
        return tuple(sorted(self.specs))

    def tools(self) -> tuple[ToolSpec, ...]:
        """返回全部工具（按名字排序，便于稳定决策）。"""
        return tuple(self.specs[name] for name in self.names())

    def resolve(self, raw_name: Any, *, allow_partial: bool = True) -> str | None:
        """把模型给出的工具名解析成目录里的真实工具名。

        依次尝试：完全一致 → 大小写不敏感 → 去掉类型前缀后的后缀匹配
        （``send_text`` → ``action-send_text``）。

        Args:
            raw_name: 模型给出的工具名。
            allow_partial: 是否允许做后缀/大小写等宽松匹配。

        Returns:
            解析成功返回真实工具名，否则返回 ``None``。
        """
        if not isinstance(raw_name, str):
            return None
        candidate = raw_name.strip().strip("\"'` ")
        if not candidate:
            return None
        if candidate in self.specs:
            return candidate
        if not allow_partial:
            return None

        lowered = candidate.lower()
        by_lower = [name for name in self.specs if name.lower() == lowered]
        if len(by_lower) == 1:
            return by_lower[0]

        suffix_hits = [
            name
            for name in self.specs
            if name.split("-", 1)[-1].lower() == lowered
            or name.lower().endswith("-" + lowered)
        ]
        if len(suffix_hits) == 1:
            return suffix_hits[0]
        return None

    def mentioned(self, text: str) -> list[ToolSpec]:
        """返回文本中提到过的工具（用于挑「参数详情」子集）。"""
        if not text:
            return []
        lowered = text.lower()
        hits: list[ToolSpec] = []
        for name in self.names():
            short = name.split("-", 1)[-1].lower()
            if name.lower() in lowered or (len(short) >= 3 and short in lowered):
                hits.append(self.specs[name])
        return hits


@dataclass(frozen=True, slots=True)
class CallIssue:
    """一条「框架会报错」的调用问题。"""

    code: str
    """问题代码：``empty_name`` / ``unknown_tool`` / ``args_not_object`` /
    ``missing_required`` / ``invalid_enum``。"""

    message: str
    """面向框架语义的报错文本，会直接喂给修复模型。"""

    needs_llm: bool = True
    """是否必须交给修复模型（``False`` 表示已在本地修复）。"""


@dataclass(slots=True)
class CallAudit:
    """单条工具调用的校验结果。"""

    call: dict[str, Any]
    issues: list[CallIssue] = field(default_factory=list)
    changed: bool = False
    """是否发生了「语义层面」的本地修复（改名字/解析 args/纠类型/删多余参数）。"""

    reason_added: bool = False
    """是否只是补了 ``reason``。单独标记，避免为了补 reason 而改写正常调用。"""

    @property
    def ok(self) -> bool:
        """没有任何问题时为 ``True``。"""
        return not self.issues

    @property
    def needs_llm(self) -> bool:
        """存在必须交给 LLM 的问题时为 ``True``。"""
        return any(issue.needs_llm for issue in self.issues)


@dataclass(frozen=True, slots=True)
class AuditOptions:
    """本地校验/本地修复的开关集合。"""

    require_known_tool: bool = True
    resolve_partial_names: bool = True
    drop_unknown_args: bool = True
    coerce_scalar_types: bool = True
    inject_reason: bool = True
    reason_text: str = "修正工具调用格式"


def coerce_args_object(raw: Any) -> tuple[dict[str, Any], str | None]:
    """把 ``args`` 字段规范化成 dict。

    Args:
        raw: 可能是 dict、JSON 字符串（或带 ``` 围栏 / 前后杂质的字符串）、
            其它非法类型。

    Returns:
        ``(args, error)``：成功时 error 为 ``None``；失败时 args 为空 dict，
        error 是贴近框架行为的报错文本。
    """
    if isinstance(raw, Mapping):
        return dict(raw), None
    if raw is None:
        return {}, None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}, None
        parsed, _ = parse_json_text(text)
        if isinstance(parsed, Mapping):
            return dict(parsed), None
        return {}, (
            "arguments 不是合法 JSON 对象（框架收到的是字符串，"
            f"无法解析为参数：{_truncate(text, 80)!r}）"
        )
    return {}, f"arguments 类型非法（{type(raw).__name__}），必须是 JSON 对象"


def parse_json_text(text: str) -> tuple[Any, str]:
    """容错解析文本中的 JSON 值。

    依次尝试：原文 → 去掉 markdown 围栏 → 第一个平衡的 ``{...}`` / ``[...]``。

    Args:
        text: 原始文本。

    Returns:
        ``(value, source)``：value 解析失败时为 ``None``；source 为 ``raw`` /
        ``fence`` / ``slice`` / ``none``，便于日志区分。
    """
    stripped = text.strip()
    if not stripped:
        return None, "none"

    try:
        return json.loads(stripped), "raw"
    except Exception:
        pass

    fenced = strip_code_fence(stripped)
    if fenced != stripped:
        try:
            return json.loads(fenced), "fence"
        except Exception:
            pass

    for candidate in iter_balanced_json(fenced or stripped):
        try:
            return json.loads(candidate), "slice"
        except Exception:
            continue
    return None, "none"


def strip_code_fence(text: str) -> str:
    """去掉 markdown 代码围栏（```` ```json ... ``` ````）。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped[3:]
    newline = body.find("\n")
    if newline >= 0:
        first_line = body[:newline].strip().lower()
        if not first_line or first_line in {"json", "json5", "javascript", "js"}:
            body = body[newline + 1 :]
        else:
            body = body[newline + 1 :]
    closing = body.rfind("```")
    if closing >= 0:
        body = body[:closing]
    return body.strip()


def iter_balanced_json(text: str) -> Iterable[str]:
    """产出文本中所有平衡的 ``{...}`` / ``[...]`` 片段（由外向内）。"""
    length = len(text)
    index = 0
    while index < length:
        char = text[index]
        if char not in "{[":
            index += 1
            continue
        end = _match_balanced(text, index)
        if end is None:
            index += 1
            continue
        yield text[index : end + 1]
        index += 1


def _match_balanced(text: str, start: int) -> int | None:
    """返回从 start 开始的平衡片段结束下标（含），失败返回 ``None``。"""
    pairs = {"{": "}", "[": "]"}
    opening = text[start]
    closing = pairs[opening]
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opening:
            depth += 1
            continue
        if char == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def audit_call(
    raw_call: Mapping[str, Any],
    catalog: ToolCatalog,
    options: AuditOptions,
) -> CallAudit:
    """校验并尽量本地修复一条工具调用。

    Args:
        raw_call: ``{"id": ..., "name": ..., "args": ...}``。
        catalog: 本次请求的可用工具集。
        options: 本地修复开关。

    Returns:
        :class:`CallAudit`：``call`` 是规范化后的调用（可能已本地修复）。
    """
    call_id = raw_call.get("id")
    call: dict[str, Any] = {
        "id": call_id if isinstance(call_id, str) else None,
        "name": raw_call.get("name") if isinstance(raw_call.get("name"), str) else "",
        "args": {},
    }
    audit = CallAudit(call=call)
    raw_name = call["name"].strip()

    if not raw_name:
        audit.issues.append(
            CallIssue("empty_name", "工具调用缺少 name 字段，框架无法路由该调用")
        )
        args, error = coerce_args_object(raw_call.get("args"))
        call["args"] = args
        if error:
            audit.issues.append(CallIssue("args_not_object", error))
        return audit

    resolved = catalog.resolve(raw_name, allow_partial=options.resolve_partial_names)
    if resolved is None:
        call["name"] = raw_name
        if options.require_known_tool and catalog:
            known = ", ".join(catalog.names()[:20])
            audit.issues.append(
                CallIssue(
                    "unknown_tool",
                    f"未知的工具: {raw_name}（框架会返回『未知的工具』且不执行；"
                    f"可用工具: {known}）",
                )
            )
    elif resolved != raw_name:
        call["name"] = resolved
        audit.changed = True

    args, error = coerce_args_object(raw_call.get("args"))
    call["args"] = args
    if error:
        audit.issues.append(CallIssue("args_not_object", error))

    spec = catalog.get(call["name"])
    if spec is None:
        return audit

    # schema 里没有参数（例如 ``execute(self)`` 无参）时也清理多余参数：
    # 这类组件多传一个 kwarg 就会 ``unexpected keyword argument``。
    if options.drop_unknown_args and not spec.accepts_var_kwargs:
        removed = [
            key
            for key in list(call["args"])
            if key != REASON_KEY and key not in spec.properties
        ]
        for key in removed:
            call["args"].pop(key, None)
        if removed:
            audit.changed = True

    if options.coerce_scalar_types:
        for key, value in list(call["args"].items()):
            if key == REASON_KEY:
                continue
            coerced, did_change = coerce_scalar(value, spec.param_schema(key))
            if did_change:
                call["args"][key] = coerced
                audit.changed = True

    for key in spec.required:
        if _is_missing(call["args"], key, spec):
            audit.issues.append(
                CallIssue(
                    "missing_required",
                    f"调用 {spec.name} 缺少必填参数 {key}；"
                    f"框架执行时会抛 "
                    f"TypeError: execute() missing 1 required argument: {key!r}",
                )
            )

    for key, value in list(call["args"].items()):
        if key == REASON_KEY:
            continue
        prop = spec.param_schema(key)
        if not prop:
            continue
        if _is_missing(call["args"], key, spec):
            continue
        enum = prop.get("enum")
        if isinstance(enum, list) and enum and value not in enum:
            audit.issues.append(
                CallIssue(
                    "invalid_enum",
                    f"调用 {spec.name} 的参数 {key} 值 {value!r} 不在允许范围内"
                    f"（允许: {', '.join(str(item) for item in enum)}）",
                )
            )

    if options.inject_reason and REASON_KEY not in call["args"]:
        call["args"][REASON_KEY] = options.reason_text
        audit.reason_added = True

    return audit


def _is_missing(args: Mapping[str, Any], key: str, spec: ToolSpec) -> bool:
    """判断必填参数是否缺失（``None`` 且 schema 不允许 null 时也算缺失）。"""
    if key not in args:
        return True
    if args[key] is not None:
        return False
    return "null" not in _schema_types(spec.param_schema(key))


def coerce_scalar(value: Any, prop: Mapping[str, Any]) -> tuple[Any, bool]:
    """按 schema 的类型声明纠正标量值。

    Args:
        value: 当前值。
        prop: 该参数的 JSON Schema 片段。

    Returns:
        ``(new_value, changed)``。
    """
    types = _schema_types(prop)
    if not types:
        return value, False
    if "null" in types and value is None:
        return value, False

    if isinstance(value, str):
        text = value.strip()
        if "integer" in types:
            parsed = _try_int(text)
            if parsed is not None:
                return parsed, True
        if "number" in types and "integer" not in types:
            parsed_float = _try_float(text)
            if parsed_float is not None:
                return parsed_float, True
        if "boolean" in types:
            lowered = text.lower()
            if lowered in {"true", "yes", "1"}:
                return True, True
            if lowered in {"false", "no", "0"}:
                return False, True
        return value, False

    if (
        "string" in types
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return str(value), True
    return value, False


def _try_int(text: str) -> int | None:
    """尝试把字符串转成 int，失败返回 ``None``。"""
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _try_float(text: str) -> float | None:
    """尝试把字符串转成 float，失败返回 ``None``。"""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None
