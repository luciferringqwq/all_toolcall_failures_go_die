"""修复内核：纯标准库实现的 schema 校验与容错提取。

本包内的模块都不导入框架代码，因此可以脱离 bot 进程离线自测：

- :mod:`core.schemas`：工具目录、工具调用校验、本地修复。
- :mod:`core.extraction`：从纯文本里抢救工具调用。
- :mod:`core.prompts`：修复提示词构建与回复解析。

依赖框架的部分（真正发起第二个 LLM 请求）在 ``core.repairer`` 中，
它属于插件运行时，不进本包的 :data:`__all__`。
"""

from __future__ import annotations

from .extraction import RawCallCandidate, extract_candidates, looks_like_tool_call
from .prompts import (
    KIND_CALLS,
    KIND_INVALID,
    KIND_NONE,
    RepairDecision,
    build_args_user_prompt,
    build_plan_user_prompt,
    build_system_prompt,
    parse_repair_reply,
)
from .schemas import (
    REASON_KEY,
    AuditOptions,
    CallAudit,
    CallIssue,
    ToolCatalog,
    ToolSpec,
    audit_call,
    coerce_args_object,
    parse_json_text,
)

__all__ = [
    "REASON_KEY",
    "AuditOptions",
    "CallAudit",
    "CallIssue",
    "KIND_CALLS",
    "KIND_INVALID",
    "KIND_NONE",
    "RawCallCandidate",
    "RepairDecision",
    "ToolCatalog",
    "ToolSpec",
    "audit_call",
    "build_args_user_prompt",
    "build_plan_user_prompt",
    "build_system_prompt",
    "coerce_args_object",
    "extract_candidates",
    "looks_like_tool_call",
    "parse_json_text",
    "parse_repair_reply",
]
