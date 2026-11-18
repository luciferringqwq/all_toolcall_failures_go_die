"""把框架组件类转换成工具目录（需要框架运行时）。

只负责「拿到可用工具并读 schema」这一件事，方便事件处理器与 Service 复用。
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from src.app.plugin_system.api import action_api, agent_api, tool_api

from .schemas import ToolCatalog, ToolSpec


def accepts_var_kwargs(cls: type) -> bool:
    """判断组件的 ``execute`` 是否接受 ``**kwargs``。

    Args:
        cls: 组件类。

    Returns:
        接受 ``**kwargs`` 时返回 ``True``。
    """
    execute = getattr(cls, "execute", None)
    if execute is None:
        return False
    try:
        signature = inspect.signature(execute)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def chatter_allow_of(cls: type) -> tuple[str, ...]:
    """读取组件的 ``chatter_allow``（组件属于哪些 chatter）。

    Args:
        cls: 组件类。

    Returns:
        声明的 chatter 名称元组；未声明或格式不对时为空元组。
    """
    raw = getattr(cls, "chatter_allow", ())
    if isinstance(raw, str):
        return (raw,) if raw else ()
    if isinstance(raw, Sequence):
        return tuple(item for item in raw if isinstance(item, str) and item)
    return ()


def build_catalog(tools: Iterable[Any]) -> ToolCatalog:
    """把组件类（或实例）集合转换成工具目录。

    Args:
        tools: 组件类/实例的可迭代对象。

    Returns:
        :class:`ToolCatalog`；单个组件 schema 异常会被跳过。
    """
    specs: dict[str, ToolSpec] = {}
    for item in tools:
        cls = item if isinstance(item, type) else type(item)
        to_schema = getattr(cls, "to_schema", None)
        if not callable(to_schema):
            continue
        try:
            schema = to_schema()
        except Exception:  # noqa: BLE001 - 单个组件异常不影响其它工具
            continue
        spec = ToolSpec.from_schema(
            schema,
            accepts_var_kwargs=accepts_var_kwargs(cls),
        )
        if spec is not None:
            specs[spec.name] = replace(spec, chatter_allow=chatter_allow_of(cls))
    return ToolCatalog(specs)


def registered_tool_classes() -> list[Any]:
    """从全局注册表收集所有 Action / Tool / Agent 组件类。"""
    classes: list[Any] = []
    for getter in (
        action_api.get_all_actions,
        tool_api.get_all_tools,
        agent_api.get_all_agents,
    ):
        try:
            mapping = getter()
        except Exception:  # noqa: BLE001 - 管理器尚未就绪时静默降级
            continue
        if isinstance(mapping, Mapping):
            classes.extend(mapping.values())
    return classes


def fallback_catalog() -> ToolCatalog:
    """用全局注册表构建兜底工具目录（缓存未命中时使用）。"""
    classes = registered_tool_classes()
    return build_catalog(classes) if classes else ToolCatalog({})


def schemas_for(names: Sequence[str]) -> list[dict[str, Any]]:
    """按工具名取 schema（供排查/调试使用）。"""
    catalog = fallback_catalog()
    out: list[dict[str, Any]] = []
    for name in names:
        spec = catalog.get(name)
        if spec is None:
            continue
        out.append(
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": dict(spec.parameters),
            }
        )
    return out
