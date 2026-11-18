"""工具调用修复服务。

事件处理器负责「拦截并就地修好」，本服务负责「把修复能力开放出去」：
其它插件可以在自己的流程里主动调用，把一个坏掉的工具调用或一段纯文本
变成合法调用。返回值统一是纯 ``dict`` 列表，不依赖框架类型。

Examples:
    >>> svc = service_manager.get_service("all_toolcall_failures_go_die:toolcall_repair")
    >>> calls = await svc.repair_text('{"name":"send_text","args":{"content":"你好"}}')
    >>> calls[0]["name"]
    'action-send_text'
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import BaseService

from .config import ToolCallRepairConfig
from .core.catalog import build_catalog, fallback_catalog
from .core.extraction import extract_candidates
from .core.prompts import KIND_CALLS
from .core.repairer import RepairLLM, RepairLLMSettings
from .core.schemas import AuditOptions, ToolCatalog, audit_call

logger = get_logger("all_toolcall_failures_go_die.service")


class ToolCallRepairService(BaseService):
    """对外暴露的工具调用修复服务。"""

    name: str = "toolcall_repair"
    description: str = "校验 / 本地修复 / LLM 重写工具调用"
    version: str = "1.0.0"

    service_name: str = "toolcall_repair"
    service_description: str = "校验 / 本地修复 / LLM 重写工具调用"

    def __init__(self, plugin: Any) -> None:
        """初始化服务。

        Args:
            plugin: 所属插件实例。
        """
        super().__init__(plugin)
        self._repairer: RepairLLM | None = None
        self._settings: RepairLLMSettings | None = None

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def catalog(self, tools: Sequence[Any] | None = None) -> ToolCatalog:
        """返回工具目录。

        Args:
            tools: 可选的组件类集合；为 ``None`` 时使用全局注册表。

        Returns:
            :class:`ToolCatalog`。
        """
        if tools is None:
            return fallback_catalog()
        return build_catalog(tools)

    def audit_calls(
        self,
        calls: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Any] | None = None,
        catalog: ToolCatalog | None = None,
    ) -> list[dict[str, Any]]:
        """校验并做本地修复，不发起任何 LLM 请求。

        Args:
            calls: ``{"name": ..., "args": ...}`` 形态的调用列表。
            tools: 可选组件类集合（用于构建目录）。
            catalog: 可选现成目录，优先于 ``tools``。

        Returns:
            规范化后的调用列表；仍然不合法的调用按原样返回（附注入的 ``reason``）。
        """
        target = catalog if catalog is not None else self.catalog(tools)
        options = self._options()
        results: list[dict[str, Any]] = []
        for call in calls:
            audit = audit_call(dict(call), target, options)
            if audit.issues:
                logger.debug(
                    f"[audit] {audit.call.get('name')} 仍有问题: "
                    f"{[issue.code for issue in audit.issues]}"
                )
            results.append(dict(audit.call))
        return results

    async def repair_text(
        self,
        raw_text: str,
        *,
        tools: Sequence[Any] | None = None,
        catalog: ToolCatalog | None = None,
        max_calls: int | None = None,
    ) -> list[dict[str, Any]]:
        """把一段（可能是坏的）输出修复成合法工具调用。

        先尝试零成本的本地抢救，失败后再请求一次修复模型。

        Args:
            raw_text: 助手输出文本。
            tools: 可选组件类集合。
            catalog: 可选现成目录。
            max_calls: 最多返回几个调用；``None`` 时使用配置。

        Returns:
            合法调用列表（``{"id": None, "name": ..., "args": {...}}``）；
            无法修复时返回空列表。
        """
        target = catalog if catalog is not None else self.catalog(tools)
        if not target:
            logger.warning("[repair_text] 当前没有任何可用工具，无法修复")
            return []

        config = self._config()
        limit = max_calls or (int(config.plain_text.max_calls) if config else 4)
        options = self._options()

        local = self._verify(extract_candidates(raw_text), target, options, limit)
        if local:
            return local

        if config is None:
            return []

        repairer = self._get_repairer(config)
        decision = await repairer.plan(
            raw_text=raw_text,
            catalog=target,
            current_calls=[],
            errors=[],
            max_chars=int(config.plain_text.max_chars),
        )
        if decision.kind != KIND_CALLS:
            logger.debug(f"[repair_text] 修复模型未给出调用: {decision.kind}")
            return []
        return self._verify(decision.candidates, target, options, limit) or []

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _verify(
        self,
        candidates: Sequence[Any],
        catalog: ToolCatalog,
        options: AuditOptions,
        limit: int,
    ) -> list[dict[str, Any]] | None:
        """校验候选调用，全部合法才返回。"""
        if not candidates:
            return None
        verified: list[dict[str, Any]] = []
        for candidate in list(candidates)[: max(limit, 1)]:
            raw = {
                "id": getattr(candidate, "call_id", None),
                "name": getattr(candidate, "name", ""),
                "args": getattr(candidate, "args", {}),
            }
            audit = audit_call(raw, catalog, options)
            if not audit.ok and audit.needs_llm:
                return None
            if catalog.get(str(audit.call.get("name")) or "") is None:
                return None
            verified.append(dict(audit.call))
        return verified or None

    def _options(self) -> AuditOptions:
        """从配置生成校验开关。"""
        config = self._config()
        if config is None:
            return AuditOptions()
        return AuditOptions(
            require_known_tool=bool(config.validation.require_known_tool),
            resolve_partial_names=bool(config.validation.resolve_partial_names),
            drop_unknown_args=bool(config.validation.drop_unknown_args),
            coerce_scalar_types=bool(config.validation.coerce_scalar_types),
            inject_reason=bool(config.validation.inject_reason),
            reason_text=str(config.validation.reason_text or "修正工具调用格式"),
        )

    def _config(self) -> ToolCallRepairConfig | None:
        """读取插件配置。"""
        config = getattr(self.plugin, "config", None)
        return config if isinstance(config, ToolCallRepairConfig) else None

    def _get_repairer(self, config: ToolCallRepairConfig) -> RepairLLM:
        """按配置懒加载修复器。"""
        settings = RepairLLMSettings(
            task_name=str(config.repair.task_name),
            request_name=str(config.repair.request_name),
            temperature=float(config.repair.temperature),
            max_tokens=int(config.repair.max_tokens),
            timeout_seconds=float(config.repair.timeout_seconds),
            max_attempts=int(config.repair.max_attempts),
            fallback_tasks=tuple(
                str(name) for name in (config.repair.fallback_task_names or ())
            ),
        )
        if self._repairer is None or self._settings != settings:
            self._repairer = RepairLLM(settings)
            self._settings = settings
        return self._repairer
