"""识别「当前请求里哪个工具负责把文本发给用户」。

纯文本套壳的关键是**不要写死工具名**：不同 chatter 提供的发送动作可能叫
``send_text`` / ``send_message`` / ``reply`` ……，而且同一个请求里还可能同时存在
``send_emoji_meme`` 这类「看起来也是发送、其实不是文本」的动作。

识别顺序（越靠前越可信）：

1. 配置里显式指定的工具名（``plain_text.send_tool_name``）；
2. 名字命中 ``send_tool_patterns``（glob）且存在正文参数；
3. 唯一的「只有一个必填字符串参数、参数名属于正文参数名」的工具；
4. 描述里含发送/回复关键词且存在正文参数。

若一次命中多个，优先取 ``chatter_allow`` 与本次请求名一致的（chatter 归属），
再优先 ``chatter_allow`` 为空的（对所有 chatter 通用）；仍然无法唯一确定时，
返回多个候选交由调用方决定（可以交给 LLM 选）。
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Sequence

from .schemas import REASON_KEY, ToolCatalog, ToolSpec

#: 默认的发送类工具名模式（glob，不区分大小写）。
DEFAULT_SEND_PATTERNS = (
    "*send_text*",
    "*send_message*",
    "*send_msg*",
    "*reply_text*",
    "*reply_message*",
    "send*",
    "reply*",
)

#: 默认的「正文参数」名，按优先级排列。
DEFAULT_BODY_PARAMS = ("content", "text", "message", "msg", "reply", "body")

#: 描述里出现这些词时，认为该工具「像是在发消息」。
_SEND_WORDS = ("发送", "回复", "发给", "send", "reply")


@dataclass(frozen=True, slots=True)
class SendToolSettings:
    """发送工具识别配置。"""

    patterns: tuple[str, ...] = DEFAULT_SEND_PATTERNS
    body_params: tuple[str, ...] = DEFAULT_BODY_PARAMS
    send_tool_name: str = ""
    """显式指定发送工具名（留空表示自动识别）。"""


@dataclass(slots=True)
class SendToolMatch:
    """识别结果。"""

    tool: ToolSpec | None = None
    """唯一确定时的发送工具。"""
    body_param: str = ""
    """正文参数名（``tool`` 非空时有效）。"""
    candidates: list[ToolSpec] = field(default_factory=list)
    """无法唯一确定时的候选列表。"""
    reason: str = ""
    """面向日志的说明。"""

    @property
    def resolved(self) -> bool:
        """是否已唯一定位到发送工具。"""
        return self.tool is not None and bool(self.body_param)


def body_param_of(spec: ToolSpec, settings: SendToolSettings) -> str:
    """返回该工具的正文参数名；找不到时返回空字符串。"""
    required = spec.required
    for key in settings.body_params:
        if key in required or key in spec.properties:
            return key
    return ""


def _is_string_param(spec: ToolSpec, key: str) -> bool:
    """判断参数是否为字符串类型（缺省 ``type`` 时按字符串处理）。"""
    prop = spec.param_schema(key)
    if not prop:
        return False
    declared = prop.get("type")
    if declared is None:
        return True
    return declared == "string" or (isinstance(declared, list) and "string" in declared)


def _name_matches(spec: ToolSpec, patterns: Sequence[str]) -> bool:
    """工具名（含去掉前缀后的短名）是否命中任一模式。"""
    full = spec.name.lower()
    short = full.split("-", 1)[-1]
    for pattern in patterns:
        lowered = pattern.lower()
        if fnmatch.fnmatch(full, lowered) or fnmatch.fnmatch(short, lowered):
            return True
    return False


def _description_suggests_send(spec: ToolSpec) -> bool:
    """描述里是否含发送类关键词。"""
    lowered = spec.description.lower()
    return any(word in lowered for word in _SEND_WORDS)


def _prefer_chatter(
    candidates: list[ToolSpec],
    request_name: str,
) -> list[ToolSpec]:
    """优先保留与本次请求（chatter）匹配的候选。"""
    if not candidates:
        return candidates
    if request_name:
        matched = [spec for spec in candidates if request_name in spec.chatter_allow]
        if matched:
            return matched
    generic = [spec for spec in candidates if not spec.chatter_allow]
    return generic or candidates


def find_send_tool(
    catalog: ToolCatalog,
    request_name: str,
    settings: SendToolSettings,
) -> SendToolMatch:
    """在本次请求的工具目录里定位发送工具。

    Args:
        catalog: 本次请求真正提供给模型的工具目录。
        request_name: 本次 LLM 请求名（框架默认等于 chatter 名），用于
            按 ``chatter_allow`` 消除歧义。
        settings: 识别配置。

    Returns:
        :class:`SendToolMatch`：唯一确定时 ``tool``/``body_param`` 有值，
        否则 ``candidates`` 给出可能项供上位逻辑（例如让 LLM 选）使用。
    """
    if not catalog:
        return SendToolMatch(reason="工具目录为空")

    if settings.send_tool_name:
        pinned_name = catalog.resolve(settings.send_tool_name)
        spec = catalog.get(pinned_name) if pinned_name else None
        if spec is None:
            return SendToolMatch(
                reason=f"配置的发送工具 {settings.send_tool_name!r} 不在本次工具目录中"
            )
        body = body_param_of(spec, settings)
        if not body:
            return SendToolMatch(
                reason=f"指定的发送工具 {spec.name} 没有可用的正文参数"
            )
        return SendToolMatch(
            tool=spec, body_param=body, candidates=[spec], reason="配置显式指定"
        )

    # 第 2 层：名字模式
    by_name = [
        spec
        for spec in catalog.tools()
        if _name_matches(spec, settings.patterns) and body_param_of(spec, settings)
    ]
    by_name = _prefer_chatter(by_name, request_name)
    if len(by_name) == 1:
        spec = by_name[0]
        return SendToolMatch(
            tool=spec,
            body_param=body_param_of(spec, settings),
            candidates=[spec],
            reason="名字模式唯一命中",
        )

    # 第 3 层：只有一个必填字符串参数，且该参数是正文参数名
    structural: list[ToolSpec] = []
    for spec in catalog.tools():
        body = body_param_of(spec, settings)
        if not body:
            continue
        required = [key for key in spec.required if key != REASON_KEY]
        if len(required) != 1 or required[0] != body:
            continue
        if not _is_string_param(spec, body):
            continue
        statement = spec.chatter_allow
        if statement and request_name and request_name not in statement:
            continue
        structural.append(spec)
    structural = _prefer_chatter(structural, request_name)
    if len(structural) == 1:
        spec = structural[0]
        return SendToolMatch(
            tool=spec,
            body_param=body_param_of(spec, settings),
            candidates=[spec],
            reason="结构特征唯一命中",
        )

    # 第 4 层：描述关键词
    by_desc = [
        spec
        for spec in catalog.tools()
        if _description_suggests_send(spec) and body_param_of(spec, settings)
    ]
    by_desc = _prefer_chatter(by_desc, request_name)
    if len(by_desc) == 1:
        spec = by_desc[0]
        return SendToolMatch(
            tool=spec,
            body_param=body_param_of(spec, settings),
            candidates=[spec],
            reason="描述关键词唯一命中",
        )

    pool = by_name or structural or by_desc
    if not pool:
        return SendToolMatch(reason="未找到任何发送类工具")
    if len(pool) == 1:
        spec = pool[0]
        return SendToolMatch(
            tool=spec,
            body_param=body_param_of(spec, settings),
            candidates=[spec],
            reason="仅一个候选",
        )
    return SendToolMatch(
        candidates=list(pool),
        reason=f"存在 {len(pool)} 个发送类候选: {[s.name for s in pool]}",
    )


def build_wrap_call(spec: ToolSpec, body_param: str, text: str) -> dict[str, object]:
    """把纯文本套成一次工具调用。

    Args:
        spec: 发送工具。
        body_param: 正文参数名。
        text: 待发送的原始文本（原样放入，不做任何改写）。

    Returns:
        ``{"name": ..., "args": {...}}``。
    """
    return {"name": spec.name, "args": {body_param: text}}
