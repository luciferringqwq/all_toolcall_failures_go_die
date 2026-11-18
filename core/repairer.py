"""调用第二个 LLM 执行工具调用修复（框架相关）。

只做三件事：

1. 按任务名取模型集合（``tool_use`` → 失败回退 ``utils`` → ``actor``）。
2. 发送「system + user」两条 payload 的单轮请求，带超时与有限重试。
3. 把回复交给 :func:`core.prompts.parse_repair_reply` 解析。

请求自身也会触发 ``AFTER_LLM_REQUEST``，因此 request_name 默认带
``toolcall_repair`` 前缀，并被事件处理器跳过，不会递归。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.app.plugin_system.api import llm_api
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.types import LLMPayload, ROLE, Text

from .prompts import (
    KIND_INVALID,
    RepairDecision,
    WRAP_BODY_HINT,
    build_args_user_prompt,
    build_plan_user_prompt,
    build_system_prompt,
    build_wrap_system_prompt,
    build_wrap_user_prompt,
    parse_repair_reply,
)
from .schemas import ToolCatalog, ToolSpec

logger = get_logger("all_toolcall_failures_go_die.repair")

#: 依次尝试的模型任务名：前一个取不到就用后一个。
_FALLBACK_TASKS = ("tool_use", "utils", "utils_small", "actor")


def _remaining(deadline: float, loop: asyncio.AbstractEventLoop) -> float:
    """返回距离 deadline 还剩下的秒数（至少给一点余量，避免 0）。"""
    return max(deadline - loop.time(), 0.001)


@dataclass(frozen=True, slots=True)
class RepairLLMSettings:
    """修复请求的调用参数。"""

    task_name: str = "tool_use"
    request_name: str = "toolcall_repair"
    temperature: float = 0.0
    max_tokens: int = 1200
    timeout_seconds: float = 45.0
    max_attempts: int = 1
    fallback_tasks: tuple[str, ...] = _FALLBACK_TASKS
    """``task_name`` 取不到时依次尝试的任务名。"""


class RepairLLM:
    """执行单轮修复请求。"""

    def __init__(self, settings: RepairLLMSettings) -> None:
        """初始化修复器。

        Args:
            settings: 模型任务、超时、温度等设置。
        """
        self._settings = settings

    @property
    def request_name(self) -> str:
        """修复请求使用的 request_name。"""
        return self._settings.request_name

    async def plan(
        self,
        *,
        raw_text: str,
        catalog: ToolCatalog,
        current_calls: Sequence[Mapping[str, Any]] = (),
        errors: Sequence[str] = (),
        max_chars: int = 8000,
    ) -> RepairDecision:
        """让修复模型基于原始输出重建工具调用。

        Args:
            raw_text: 助手原始输出。
            catalog: 本次请求可用工具集。
            current_calls: 框架已解析出的调用。
            errors: 框架视角的报错。
            max_chars: 原始输出最大字符数。

        Returns:
            :class:`RepairDecision`；请求失败时为 ``invalid``。
        """
        user_prompt = build_plan_user_prompt(
            raw_text=raw_text,
            catalog=catalog,
            current_calls=current_calls,
            errors=errors,
            max_chars=max_chars,
        )
        return await self._request(user_prompt)

    async def fill_args(
        self,
        *,
        raw_text: str,
        spec: ToolSpec,
        current_args: Mapping[str, Any] | None = None,
        errors: Sequence[str] = (),
        max_chars: int = 4000,
    ) -> RepairDecision:
        """只让修复模型补齐某个工具的参数。

        Args:
            raw_text: 助手原始输出。
            spec: 目标工具。
            current_args: 已解析出的参数。
            errors: 框架视角的报错。
            max_chars: 原始输出最大字符数。

        Returns:
            :class:`RepairDecision`；请求失败时为 ``invalid``。
        """
        user_prompt = build_args_user_prompt(
            raw_text=raw_text,
            spec=spec,
            current_args=current_args,
            errors=errors,
            max_chars=max_chars,
        )
        return await self._request(user_prompt, fallback_name=spec.name)

    async def wrap(
        self,
        *,
        raw_text: str,
        catalog: ToolCatalog,
        candidates: Sequence[ToolSpec] = (),
        body_params: Sequence[str] = (),
        max_chars: int = 8000,
    ) -> RepairDecision:
        """让修复模型把纯文本套壳成「发送消息」工具调用。

        Args:
            raw_text: 模型直接输出的纯文本（待发送正文）。
            catalog: 本次请求可用工具集。
            candidates: 本地识别出的发送工具候选。
            body_params: 正文参数名候选。
            max_chars: 原文最大字符数。

        Returns:
            :class:`RepairDecision`；请求失败时为 ``invalid``。
        """
        user_prompt = build_wrap_user_prompt(
            raw_text=raw_text,
            catalog=catalog,
            candidates=candidates,
            body_params=body_params or WRAP_BODY_HINT,
            max_chars=max_chars,
        )
        return await self._request(
            user_prompt, system_prompt=build_wrap_system_prompt()
        )

    async def _request(
        self,
        user_prompt: str,
        *,
        fallback_name: str = "",
        system_prompt: str | None = None,
    ) -> RepairDecision:
        """发送修复请求并解析回复。"""
        attempts = max(self._settings.max_attempts, 1)
        last: RepairDecision = RepairDecision(KIND_INVALID)
        for attempt in range(attempts):
            try:
                reply = await self._complete(user_prompt, system_prompt=system_prompt)
            except asyncio.TimeoutError:
                logger.warning(
                    f"工具调用修复请求超时（{self._settings.timeout_seconds}s）"
                )
                return RepairDecision(KIND_INVALID)
            except Exception as error:  # noqa: BLE001 - 修复失败必须 fail-open
                logger.warning(f"工具调用修复请求失败: {error}", exc_info=True)
                return RepairDecision(KIND_INVALID)

            decision = parse_repair_reply(reply, fallback_name=fallback_name)
            if decision.kind != KIND_INVALID or attempt + 1 >= attempts:
                return decision
            last = decision
        return last

    async def _complete(
        self, user_prompt: str, *, system_prompt: str | None = None
    ) -> str:
        """构造并发送一次单轮请求，返回纯文本回复。

        发送与等待回复共用同一个超时预算（不是各给一份），避免一次修复
        最坏阻塞 ``2 × timeout``。
        """
        model_set = self._resolve_model_set()
        request = llm_api.create_llm_request(
            model_set,
            request_name=self._settings.request_name,
        )
        request.add_payload(
            LLMPayload(ROLE.SYSTEM, Text(system_prompt or build_system_prompt()))
        )
        request.add_payload(LLMPayload(ROLE.USER, Text(user_prompt)))

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._settings.timeout_seconds

        response = await asyncio.wait_for(
            request.send(stream=False),
            timeout=_remaining(deadline, loop),
        )
        text = await asyncio.wait_for(
            response,
            timeout=_remaining(deadline, loop),
        )
        return text if isinstance(text, str) else str(text or "")

    def _resolve_model_set(self) -> Any:
        """按任务名解析模型集合，并覆盖温度与最大输出。"""
        candidates = [self._settings.task_name, *self._settings.fallback_tasks]
        seen: set[str] = set()
        last_error: Exception | None = None
        for task_name in candidates:
            if not task_name or task_name in seen:
                continue
            seen.add(task_name)
            try:
                model_set = llm_api.get_model_set_by_task(task_name)
            except Exception as error:  # noqa: BLE001 - 任务缺失时继续回退
                last_error = error
                continue
            if not model_set:
                continue
            for entry in model_set:
                entry["temperature"] = self._settings.temperature
                entry["max_tokens"] = self._settings.max_tokens
            return model_set
        raise RuntimeError(
            f"没有可用的修复模型任务（尝试过 {sorted(seen)}）: {last_error}"
        )
