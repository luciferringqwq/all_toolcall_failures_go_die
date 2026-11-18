"""``AFTER_LLM_REQUEST`` 事件处理器：拦截并修复工具调用故障。

工作链路（全程不改动框架与其它插件）：

1. ``BEFORE_LLM_REQUEST``：把本次请求真正发给模型的工具集缓存成
   :class:`~.core.schemas.ToolCatalog`（这是修复时唯一可信的「可用工具」来源）。
2. ``AFTER_LLM_REQUEST``：拿到框架解析后的 ``message`` / ``tool_calls``，
   按 schema 做一次校验：

   - 一切合法 → 立刻返回，零额外开销；
   - 只有「本地就能修」的问题（工具名缺前缀、args 是 JSON 字符串、
     参数类型不对、多余参数）→ 本地改好写回，不花一个 token；
   - 本地修不掉（缺必填参数、工具名未知、纯文本冒充工具调用）→
     把「原始输出 + 已解析调用 + 框架报错 + 精简工具目录」发给第二个
     LLM，要求它只回一个 JSON，然后把结果写回 ``params["tool_calls"]``。

写回后 chatter 拿到的就是合法调用，可以直接执行，不需要再发起一轮
「模型自我纠错」的完整对话，因而节省的是**整轮上下文**的 token。
"""

from __future__ import annotations

import fnmatch
import time
from collections import OrderedDict, deque
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from uuid import uuid4

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import BaseEventHandler
from src.core.components.types import EventType
from src.kernel.event import EventDecision

from .config import ToolCallRepairConfig
from .core.catalog import build_catalog, fallback_catalog
from .core.extraction import RawCallCandidate, extract_candidates, looks_like_tool_call
from .core.prompts import KIND_CALLS, KIND_NONE
from .core.repairer import RepairLLM, RepairLLMSettings
from .core.schemas import (
    AuditOptions,
    ToolCatalog,
    ToolSpec,
    audit_call,
    coerce_args_object,
)
from .core.send_tool import (
    SendToolMatch,
    SendToolSettings,
    build_wrap_call,
    find_send_tool,
)

logger = get_logger("all_toolcall_failures_go_die")

#: 工具目录缓存的有效期（秒）。缓存键是 ``(request_name, stream_id)``。
_CATALOG_TTL = 300.0

#: 工具目录缓存的最大条目数。
_CATALOG_MAX_ENTRIES = 128

#: 递归保护：修复请求自身触发的 ``AFTER_LLM_REQUEST`` 不再处理。
_IN_REPAIR: ContextVar[bool] = ContextVar("toolcall_repair_in_progress", default=False)


class _NotSent:
    """标记「修复请求因限流而没有真正发出」。"""


_UNSENT = _NotSent()


@dataclass(slots=True)
class _CatalogEntry:
    """一次请求的工具目录快照。"""

    catalog: ToolCatalog
    created_at: float


@dataclass(slots=True)
class _Outcome:
    """一次修复尝试的结论。"""

    calls: list[dict[str, Any]] | None = None
    """``None`` 表示「不改动原响应」；否则是要写回的工具调用列表。"""

    local: bool = False
    """是否由本地零成本修复得到。"""

    plain_text: bool = False
    """是否由「纯文本 → 工具调用」的修复得到。"""

    note: str = ""
    """原因/说明，用于日志。"""


class ToolCallRepairHandler(BaseEventHandler):
    """《所有的工具调用失败都给我tm去死吧插件》的处理器。

    Class Attributes:
        handler_name: 处理器名（旧版属性名，工具链仍然读取）
        handler_description: 处理器描述（旧版属性名）
        name: 处理器名（新版属性名）
        description: 处理器描述（新版属性名）
        weight: 200 —— 高于 ``plain_text_kicker``（100），保证「先把调用修好」
            再让其它处理器判断是否还需要处理纯文本。
        timeout: 0 —— 禁用订阅者级超时，修复请求自带超时控制。
        init_subscribe: 订阅请求生命周期中的两个事件。
    """

    handler_name = "toolcall_repair_handler"
    handler_description = (
        "拦截并修复无法解析的工具调用（纯文本冒充、缺参、名字错误、args 非法）"
    )

    name = "toolcall_repair_handler"
    description = (
        "拦截并修复无法解析的工具调用（纯文本冒充、缺参、名字错误、args 非法）"
    )
    weight = 200
    timeout: float | None = 0.0
    init_subscribe: list[EventType | str] = [
        EventType.BEFORE_LLM_REQUEST,
        EventType.AFTER_LLM_REQUEST,
    ]

    def __init__(self, plugin: Any) -> None:
        """初始化处理器。

        Args:
            plugin: 所属插件实例。
        """
        super().__init__(plugin)
        self._catalogs: OrderedDict[tuple[str, str], _CatalogEntry] = OrderedDict()
        self._repairer: RepairLLM | None = None
        self._repairer_settings: RepairLLMSettings | None = None
        self._recent_repairs: deque[float] = deque()

    # ------------------------------------------------------------------
    # 事件入口
    # ------------------------------------------------------------------

    async def execute(
        self,
        event_name: str,
        params: dict[str, Any],
    ) -> tuple[EventDecision, dict[str, Any]]:
        """处理 LLM 请求生命周期事件。

        Args:
            event_name: 事件名（由 EventBus 传入的字符串）。
            params: 事件参数；``AFTER_LLM_REQUEST`` 时可就地修改
                ``tool_calls`` / ``message``。

        Returns:
            ``(EventDecision.SUCCESS, params)``。任何内部异常都只记录日志，
            绝不改变原响应。
        """
        name = (
            event_name.value if isinstance(event_name, EventType) else str(event_name)
        )
        try:
            if name == EventType.BEFORE_LLM_REQUEST.value:
                self._remember_tools(params)
                return EventDecision.SUCCESS, params
            if name == EventType.AFTER_LLM_REQUEST.value:
                return await self._handle_response(params)
        except Exception as error:  # noqa: BLE001 - 修复失败必须 fail-open
            logger.warning(f"工具调用修复处理器异常，已忽略: {error}", exc_info=True)
        return EventDecision.SUCCESS, params

    # ------------------------------------------------------------------
    # BEFORE：缓存本次请求真实的工具目录
    # ------------------------------------------------------------------

    def _remember_tools(self, params: Mapping[str, Any]) -> None:
        """把本次请求的工具集缓存起来。"""
        tools = params.get("tools") or []
        if not isinstance(tools, (list, tuple)) or not tools:
            return

        catalog = build_catalog(tools)
        if not catalog:
            return

        request_name = str(params.get("request_name") or "")
        meta_data = params.get("meta_data") or {}
        stream_id = ""
        if isinstance(meta_data, Mapping):
            stream_id = str(meta_data.get("stream_id") or "")

        now = time.monotonic()
        self._catalogs[(request_name, stream_id)] = _CatalogEntry(catalog, now)
        self._catalogs.move_to_end((request_name, stream_id))
        self._prune_catalogs(now)

    def _prune_catalogs(self, now: float) -> None:
        """清理过期与超量缓存。"""
        expired = [
            key
            for key, entry in self._catalogs.items()
            if now - entry.created_at > _CATALOG_TTL
        ]
        for key in expired:
            self._catalogs.pop(key, None)
        while len(self._catalogs) > _CATALOG_MAX_ENTRIES:
            self._catalogs.popitem(last=False)

    def _catalog_for(
        self,
        request_name: str,
        params: Mapping[str, Any],
        config: ToolCallRepairConfig,
    ) -> ToolCatalog:
        """取本次请求的工具目录。

        顺序：``(request_name, stream_id)`` 精确命中 → 同 request_name 最近一次
        →（仅在 ``main.catalog_fallback=true`` 时）全局注册表兜底。

        默认**不做**全局兜底：全局注册表里有、但本次请求没提供的工具，一旦被
        写回就会在调度层变成「未知的工具」，比不修还糟。
        """
        meta_data = params.get("meta_data") or {}
        stream_id = ""
        if isinstance(meta_data, Mapping):
            stream_id = str(meta_data.get("stream_id") or "")

        now = time.monotonic()

        def fresh(entry: _CatalogEntry | None) -> bool:
            return entry is not None and now - entry.created_at <= _CATALOG_TTL

        entry = self._catalogs.get((request_name, stream_id))
        if fresh(entry):
            return entry.catalog  # type: ignore[union-attr]

        for (cached_name, _sid), cached in reversed(list(self._catalogs.items())):
            if cached_name == request_name and fresh(cached):
                return cached.catalog

        if not config.main.catalog_fallback:
            return ToolCatalog({})

        catalog = fallback_catalog()
        if catalog:
            self._catalogs[(request_name, stream_id)] = _CatalogEntry(catalog, now)
            self._prune_catalogs(now)
        return catalog

    # ------------------------------------------------------------------
    # AFTER：校验 + 本地修复 + （必要时）LLM 修复
    # ------------------------------------------------------------------

    async def _handle_response(
        self,
        params: dict[str, Any],
    ) -> tuple[EventDecision, dict[str, Any]]:
        """处理一次 LLM 响应。"""
        config = self._config()
        if config is None or not config.main.enabled:
            return EventDecision.SUCCESS, params
        if _IN_REPAIR.get():
            return EventDecision.SUCCESS, params

        request_name = str(params.get("request_name") or "")
        if not self._match_request(request_name, config):
            return EventDecision.SUCCESS, params
        if bool(params.get("stream")):
            # 流式响应在 AFTER 阶段还没有 tool_calls，无法在此修复。
            return EventDecision.SUCCESS, params

        catalog = self._catalog_for(request_name, params, config)
        if not catalog:
            if config.main.debug_log:
                logger.info(
                    f"[toolcall_repair] 跳过：本次请求的工具目录未命中 "
                    f"req={request_name or '<空>'}（BEFORE 事件未缓存到工具集）"
                )
            return EventDecision.SUCCESS, params

        raw_text = self._raw_text(params, config)
        calls = [self._as_call_dict(item) for item in (params.get("tool_calls") or [])]
        calls = [item for item in calls if item is not None]

        if config.main.debug_log:
            logger.info(
                f"[toolcall_repair] 响应检查 req={request_name or '<空>'} "
                f"calls={len(calls)} text={len(raw_text)} tools={len(catalog)}"
            )

        outcome = await self._decide(config, catalog, raw_text, calls, request_name)
        if outcome.calls is None:
            return EventDecision.SUCCESS, params

        params["tool_calls"] = self._ensure_ids(outcome.calls)
        if outcome.plain_text and config.plain_text.clear_message_on_repair:
            params["message"] = ""

        logger.info(
            f"[toolcall_repair] 已修复工具调用 "
            f"req={request_name or '<空>'} "
            f"mode={'local' if outcome.local else 'llm'} "
            f"from={len(calls)} to={len(outcome.calls)} "
            f"names={[item.get('name') for item in outcome.calls]} "
            f"note={outcome.note}"
        )
        return EventDecision.SUCCESS, params

    async def _decide(
        self,
        config: ToolCallRepairConfig,
        catalog: ToolCatalog,
        raw_text: str,
        calls: Sequence[dict[str, Any]],
        request_name: str = "",
    ) -> _Outcome:
        """判断是否需要修复，并返回修复结果。"""
        options = self._audit_options(config)
        audits = [audit_call(call, catalog, options) for call in calls]
        broken = [audit for audit in audits if not audit.ok]

        if calls and not broken:
            if not any(audit.changed for audit in audits):
                return _Outcome(note="无需修复")
            return _Outcome(
                calls=[dict(audit.call) for audit in audits],
                local=True,
                note="本地规范化（名字前缀/args 解析/类型/多余参数）",
            )

        if calls and broken and not any(audit.needs_llm for audit in broken):
            return _Outcome(
                calls=[dict(audit.call) for audit in audits],
                local=True,
                note="本地修复（名字/类型/多余参数）",
            )

        if not calls:
            return await self._decide_plain_text(
                config, catalog, raw_text, options, request_name
            )

        errors = [issue.message for audit in audits for issue in audit.issues]
        broken_names = [audit.call.get("name") for audit in broken]

        if len(broken) <= config.validation.max_broken_calls:
            single = len(calls) == 1 and calls[0].get("name")
            spec = catalog.get(str(calls[0].get("name"))) if single else None
            if spec is not None:
                decision = await self._call_llm_args(
                    config,
                    raw_text,
                    spec,
                    calls[0].get("args"),
                    errors,
                )
                if decision is _UNSENT:
                    return _Outcome(note="触发限流，跳过修复")
                verified = self._verify_candidates(
                    decision.candidates if decision else [],
                    catalog,
                    options,
                    config.plain_text.max_calls,
                )
                if verified:
                    return _Outcome(calls=verified, note="LLM 补参成功")
                if decision is not None and int(config.repair.max_attempts) <= 1:
                    # 已经花了一次请求还没修好：不再为同一故障追加第二次请求。
                    return _Outcome(
                        note=f"LLM 补参结果不合法（未重试）: {decision.raw[:120]}"
                    )

        decision = await self._call_llm_plan(
            config,
            raw_text,
            catalog,
            calls,
            errors,
        )
        if decision is _UNSENT:
            return _Outcome(note="触发限流，跳过修复")
        if decision is None:
            return _Outcome(note="修复请求失败（超时/异常）")
        if decision.kind == KIND_NONE:
            return _Outcome(note=f"LLM 判定非工具调用: {decision.reason}")
        verified = self._verify_candidates(
            decision.candidates,
            catalog,
            options,
            config.plain_text.max_calls,
        )
        if not verified:
            return _Outcome(note=f"LLM 修复结果仍不合法: {decision.raw[:120]}")
        return _Outcome(calls=verified, note=f"LLM 重建调用（坏调用={broken_names}）")

    async def _decide_plain_text(
        self,
        config: ToolCallRepairConfig,
        catalog: ToolCatalog,
        raw_text: str,
        options: AuditOptions,
        request_name: str = "",
    ) -> _Outcome:
        """处理「模型只吐纯文本」的情况。

        顺序：
        1. 本地抢救（文本里藏着坏掉的调用时，直接抽出来，零 token）；
        2. 套壳（把纯文本包成「发送消息」工具调用，这需要先定位发送工具）；
        3. 剩下的文本只有在「看起来像坏调用」时才交给修复模型。
        """
        mode = str(config.plain_text.mode or "wrap").lower()
        if mode == "off" or not raw_text.strip():
            return _Outcome(note="纯文本不处理")

        recovered = self._recover_from_text(raw_text, catalog, options, config)
        if recovered:
            return _Outcome(
                calls=recovered,
                local=True,
                plain_text=True,
                note="本地从纯文本中抢救出调用（零 token）",
            )

        # 文本「像坏掉的工具调用」时不能套壳——那会把半截 JSON 一起发给用户，
        # 应该走修复路径把它还原成真正的调用。
        suspicious = looks_like_tool_call(raw_text, catalog)

        if mode == "wrap" and not suspicious:
            wrapped = await self._wrap_plain_text(
                config, catalog, raw_text, options, request_name
            )
            if wrapped is not None:
                return wrapped

        if mode != "always" and not suspicious:
            reason = (
                "纯文本不像工具调用，且目录里没有可用发送工具"
                if mode == "wrap"
                else "纯文本不像工具调用"
            )
            return _Outcome(note=reason)

        decision = await self._call_llm_plan(config, raw_text, catalog, [], [])
        if decision is _UNSENT:
            return _Outcome(note="触发限流，跳过修复")
        if decision is None:
            return _Outcome(note="修复请求失败（超时/异常）")
        if decision.kind == KIND_NONE:
            return _Outcome(note=f"LLM 判定非工具调用: {decision.reason}")
        if decision.kind != KIND_CALLS:
            return _Outcome(note=f"修复回复不可解析: {decision.raw[:120]}")
        verified = self._verify_candidates(
            decision.candidates,
            catalog,
            options,
            config.plain_text.max_calls,
        )
        if not verified:
            return _Outcome(note=f"LLM 修复结果仍不合法: {decision.raw[:120]}")
        return _Outcome(
            calls=verified,
            plain_text=True,
            note="LLM 把纯文本转成工具调用",
        )

    async def _wrap_plain_text(
        self,
        config: ToolCallRepairConfig,
        catalog: ToolCatalog,
        raw_text: str,
        options: AuditOptions,
        request_name: str,
    ) -> _Outcome | None:
        """把纯文本套壳成「发送消息」工具调用。

        Returns:
            :class:`_Outcome`；``None`` 表示本次目录里没有可用的发送工具，
            应交给上层按「坏调用」逻辑继续判断。
        """
        settings = self._send_tool_settings(config)
        match: SendToolMatch = find_send_tool(catalog, request_name, settings)
        via = str(config.plain_text.wrap_via or "local-first")

        if match.resolved and via == "local-first":
            tool = match.tool
            assert tool is not None
            call = build_wrap_call(tool, match.body_param, raw_text)
            verified = self._verify_candidates(
                [
                    RawCallCandidate(
                        name=str(call["name"]),
                        args=call["args"],
                    )
                ],
                catalog,
                options,
                1,
            )
            if verified:
                return _Outcome(
                    calls=verified,
                    local=True,
                    plain_text=True,
                    note=f"本地套壳 → {tool.name}（{match.reason}，零 token）",
                )

        if not match.resolved and not match.candidates:
            return None

        decision = await self._call_llm_wrap(
            config,
            raw_text,
            catalog,
            match,
            config.plain_text.body_params,
        )
        if decision is _UNSENT:
            return _Outcome(note="触发限流，跳过套壳")
        if decision is None:
            return _Outcome(note="套壳请求失败（超时/异常）")
        if decision.kind == KIND_NONE:
            return _Outcome(note=f"LLM 判定没有可用发送工具: {decision.reason}")
        if decision.kind != KIND_CALLS:
            return _Outcome(note=f"套壳回复不可解析: {decision.raw[:120]}")
        verified = self._verify_candidates(
            decision.candidates,
            catalog,
            options,
            config.plain_text.max_calls,
        )
        if not verified:
            return _Outcome(note=f"套壳结果仍不合法: {decision.raw[:120]}")
        return _Outcome(
            calls=verified,
            plain_text=True,
            note="LLM 把纯文本套壳成发送调用",
        )

    # ------------------------------------------------------------------
    # 本地抢救与结果校验
    # ------------------------------------------------------------------

    def _recover_from_text(
        self,
        raw_text: str,
        catalog: ToolCatalog,
        options: AuditOptions,
        config: ToolCallRepairConfig,
    ) -> list[dict[str, Any]] | None:
        """尝试完全在本地从文本里恢复出合法调用。"""
        candidates = extract_candidates(raw_text)
        if not candidates:
            return None
        return self._verify_candidates(
            candidates,
            catalog,
            options,
            config.plain_text.max_calls,
        )

    def _verify_candidates(
        self,
        candidates: Sequence[Any],
        catalog: ToolCatalog,
        options: AuditOptions,
        max_calls: int,
    ) -> list[dict[str, Any]] | None:
        """校验候选调用，全部合法才返回规范化结果。"""
        if not candidates:
            return None
        limit = max(max_calls, 1)
        verified: list[dict[str, Any]] = []
        for candidate in list(candidates)[:limit]:
            raw = {
                "id": getattr(candidate, "call_id", None),
                "name": getattr(candidate, "name", ""),
                "args": getattr(candidate, "args", {}),
            }
            if not isinstance(raw["args"], (Mapping, str)) and raw["args"] is not None:
                raw["args"] = {}
            audit = audit_call(raw, catalog, options)
            if not audit.ok and audit.needs_llm:
                return None
            if catalog.get(str(audit.call.get("name")) or "") is None:
                return None
            verified.append(dict(audit.call))
        return verified or None

    # ------------------------------------------------------------------
    # LLM 调用包装
    # ------------------------------------------------------------------

    async def _call_llm_plan(
        self,
        config: ToolCallRepairConfig,
        raw_text: str,
        catalog: ToolCatalog,
        calls: Sequence[Mapping[str, Any]],
        errors: Sequence[str],
    ) -> Any:
        """发起「重建调用」修复请求。

        Returns:
            :class:`RepairDecision`；请求失败/超时为 ``None``；
            因限流没有发出时为 :data:`_UNSENT`。
        """
        if not self._allow_repair(config):
            return _UNSENT
        repairer = self._get_repairer(config)
        token = _IN_REPAIR.set(True)
        try:
            return await repairer.plan(
                raw_text=raw_text,
                catalog=catalog,
                current_calls=calls,
                errors=errors,
                max_chars=config.plain_text.max_chars,
            )
        finally:
            _IN_REPAIR.reset(token)

    async def _call_llm_wrap(
        self,
        config: ToolCallRepairConfig,
        raw_text: str,
        catalog: ToolCatalog,
        match: SendToolMatch,
        body_params: Sequence[str],
    ) -> Any:
        """发起「纯文本套壳」请求。

        Returns:
            :class:`RepairDecision`；请求失败/超时为 ``None``；
            因限流没有发出时为 :data:`_UNSENT`。
        """
        if not self._allow_repair(config):
            return _UNSENT
        repairer = self._get_repairer(config)
        token = _IN_REPAIR.set(True)
        try:
            return await repairer.wrap(
                raw_text=raw_text,
                catalog=catalog,
                candidates=tuple(match.candidates),
                body_params=tuple(body_params),
                max_chars=config.plain_text.max_chars,
            )
        finally:
            _IN_REPAIR.reset(token)

    async def _call_llm_args(
        self,
        config: ToolCallRepairConfig,
        raw_text: str,
        spec: ToolSpec,
        current_args: Any,
        errors: Sequence[str],
    ) -> Any:
        """发起「只补参数」修复请求。

        Returns:
            :class:`RepairDecision`；请求失败/超时为 ``None``；
            因限流没有发出时为 :data:`_UNSENT`。
        """
        if not self._allow_repair(config):
            return _UNSENT
        repairer = self._get_repairer(config)
        args, _error = coerce_args_object(current_args)
        token = _IN_REPAIR.set(True)
        try:
            return await repairer.fill_args(
                raw_text=raw_text,
                spec=spec,
                current_args=args,
                errors=errors,
                max_chars=config.plain_text.max_chars,
            )
        finally:
            _IN_REPAIR.reset(token)

    def _allow_repair(self, config: ToolCallRepairConfig) -> bool:
        """按配置做每分钟限流。"""
        limit = int(config.main.max_repairs_per_minute)
        if limit <= 0:
            return True
        now = time.monotonic()
        while self._recent_repairs and now - self._recent_repairs[0] > 60.0:
            self._recent_repairs.popleft()
        if len(self._recent_repairs) >= limit:
            logger.warning(
                f"[toolcall_repair] 触发限流（>{limit} 次/分钟），本次跳过修复"
            )
            return False
        self._recent_repairs.append(now)
        return True

    def _get_repairer(self, config: ToolCallRepairConfig) -> RepairLLM:
        """按当前配置懒加载修复器。"""
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
        if self._repairer is None or self._repairer_settings != settings:
            self._repairer = RepairLLM(settings)
            self._repairer_settings = settings
        return self._repairer

    # ------------------------------------------------------------------
    # 小工具
    # ------------------------------------------------------------------

    @staticmethod
    def _send_tool_settings(config: ToolCallRepairConfig) -> SendToolSettings:
        """把配置转换成发送工具识别参数。"""
        return SendToolSettings(
            patterns=tuple(
                str(item) for item in (config.plain_text.send_tool_patterns or ())
            ),
            body_params=tuple(
                str(item) for item in (config.plain_text.body_params or ())
            ),
            send_tool_name=str(config.plain_text.send_tool_name or ""),
        )

    def _config(self) -> ToolCallRepairConfig | None:
        """读取插件配置（缺失或类型不符时返回 ``None``）。"""
        config = getattr(self.plugin, "config", None)
        return config if isinstance(config, ToolCallRepairConfig) else None

    @staticmethod
    def _audit_options(config: ToolCallRepairConfig) -> AuditOptions:
        """把配置映射成校验开关。"""
        return AuditOptions(
            require_known_tool=bool(config.validation.require_known_tool),
            resolve_partial_names=bool(config.validation.resolve_partial_names),
            drop_unknown_args=bool(config.validation.drop_unknown_args),
            coerce_scalar_types=bool(config.validation.coerce_scalar_types),
            inject_reason=bool(config.validation.inject_reason),
            reason_text=str(config.validation.reason_text or "修正工具调用格式"),
        )

    @staticmethod
    def _match_request(request_name: str, config: ToolCallRepairConfig) -> bool:
        """判断该请求是否参与修复。"""
        cfg_request_name = str(config.repair.request_name or "")
        if cfg_request_name and request_name == cfg_request_name:
            return False
        for pattern in config.main.skip_request_name_patterns or []:
            if fnmatch.fnmatch(request_name, pattern):
                return False
        patterns = config.main.request_name_patterns or []
        if not patterns:
            return True
        return any(fnmatch.fnmatch(request_name, pattern) for pattern in patterns)

    @staticmethod
    def _raw_text(params: Mapping[str, Any], config: ToolCallRepairConfig) -> str:
        """取原始输出文本；必要时兜底使用 reasoning。"""
        message = params.get("message")
        if isinstance(message, str) and message.strip():
            return message
        reasoning = params.get("reasoning_content")
        if config.repair.include_reasoning and isinstance(reasoning, str):
            return reasoning
        return message if isinstance(message, str) else ""

    @staticmethod
    def _ensure_ids(calls: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """给缺少 id 的调用补一个稳定唯一 id。

        框架在回写 ``TOOL_RESULT`` 时依赖 ``call_id`` 与 assistant 的 tool_call
        配对；修复出来的调用原本没有 id（纯文本里没有），补一个可以避免
        provider 侧出现「tool 消息缺少 tool_call_id」的错配。
        """
        out: list[dict[str, Any]] = []
        for index, call in enumerate(calls):
            item = dict(call)
            if not item.get("id"):
                item["id"] = f"toolcall_repair_{index}_{uuid4().hex[:8]}"
            out.append(item)
        return out

    @staticmethod
    def _as_call_dict(call: Any) -> dict[str, Any] | None:
        """把 ToolCall / dict 统一成 ``{id, name, args}``。"""
        if isinstance(call, Mapping):
            return {
                "id": call.get("id"),
                "name": call.get("name"),
                "args": call.get("args"),
            }
        name = getattr(call, "name", None)
        if name is None:
            return None
        return {
            "id": getattr(call, "id", None),
            "name": name,
            "args": getattr(call, "args", {}),
        }

    def _get_config(self) -> ToolCallRepairConfig | None:
        """向后兼容的配置访问方法。"""
        return self._config()
