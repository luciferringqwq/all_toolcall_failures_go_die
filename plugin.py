"""《所有的工具调用失败都给我tm去死吧插件》插件入口。

技术名（目录 / 模块 / 配置路径）保持 ``all_toolcall_failures_go_die``，
``display_name`` 才是给人看的名字。

在 ``AFTER_LLM_REQUEST`` 处拦截无法解析的工具调用，先用零 token 的本地
修复，再按需请求一次轻量 LLM 重写，然后把结果写回本次响应。
"""

from __future__ import annotations

from src.app.plugin_system.base import BasePlugin, register_plugin

from .config import ToolCallRepairConfig
from .handler import ToolCallRepairHandler
from .service import ToolCallRepairService


@register_plugin
class ToolCallRepairPlugin(BasePlugin):
    """所有的工具调用失败都给我tm去死吧插件。"""

    plugin_name: str = "all_toolcall_failures_go_die"

    configs = [ToolCallRepairConfig]

    def get_components(self) -> list[type]:
        """返回本插件提供的组件类。

        Returns:
            list[type]: Service 与 EventHandler 组件类。
        """
        return [ToolCallRepairService, ToolCallRepairHandler]
