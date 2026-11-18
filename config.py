"""all_toolcall_failures_go_die 插件配置。

配置文件默认路径：``config/plugins/all_toolcall_failures_go_die/config.toml``。

配置按职责分成四节：

- ``main``：总开关、请求名过滤、调试日志。
- ``repair``：第二个 LLM（修复模型）的调用参数。
- ``plain_text``：模型只吐纯文本（没有 tool_calls）时的处理策略。
- ``validation``：本地校验/本地修复的宽严程度。
"""

from __future__ import annotations

from typing import ClassVar

from src.app.plugin_system.base import BaseConfig, Field, SectionBase, config_section


class ToolCallRepairConfig(BaseConfig):
    """《所有的工具调用失败都给我tm去死吧插件》配置。"""

    name: ClassVar[str] = "config"
    description: ClassVar[str] = (
        "所有的工具调用失败都给我tm去死吧插件：拦截并重写无法解析的工具调用"
    )

    @config_section("main")
    class MainSection(SectionBase):
        """插件主配置节。"""

        enabled: bool = Field(
            default=True,
            description="是否启用工具调用自修复",
        )

        request_name_patterns: list[str] = Field(
            default=["*"],
            description=(
                "需要参与修复的 LLM 请求名（glob）。默认 `*` 匹配全部。"
                "建议至少排除掉你自己插件的轻量判定请求，避免无意义开销。"
            ),
        )

        skip_request_name_patterns: list[str] = Field(
            default=["toolcall_repair*"],
            description=(
                "不参与修复的 LLM 请求名（glob），优先级高于 request_name_patterns。"
                "默认排除本插件自身的修复请求，防止递归。"
            ),
        )

        max_repairs_per_minute: int = Field(
            default=30,
            ge=0,
            description=(
                "每分钟最多触发多少次修复（保护性限流，0 表示不限制）。"
                "只统计真正发出修复请求的次数，本地零成本修复不计入。"
            ),
        )

        debug_log: bool = Field(
            default=False,
            description="是否输出详细调试日志",
        )

        catalog_fallback: bool = Field(
            default=False,
            description=(
                "当 `BEFORE_LLM_REQUEST` 缓存未命中时，是否用全局组件注册表兜底。"
                "默认关闭：全局注册表里存在、但本次请求实际没有提供的工具，一旦被"
                "修复写回，调度层会判定「未知的工具」，比不修更糟。"
                "只有当你的框架版本不发布 BEFORE 事件时才建议打开。"
            ),
        )

    main: MainSection = Field(default_factory=MainSection)

    @config_section("repair")
    class RepairSection(SectionBase):
        """修复模型调用配置节。"""

        task_name: str = Field(
            default="tool_use",
            description=(
                "修复请求使用的模型任务名（对应 config/model.toml 的 model_tasks）。"
                "建议使用轻量、便宜、指令遵循能力强的模型。"
                "留空或任务不存在时会按 fallback_task_names 依次回退。"
            ),
        )

        fallback_task_names: list[str] = Field(
            default=["utils", "utils_small", "actor"],
            description=(
                "`task_name` 取不到时依次尝试的任务名列表（留空则只用 task_name）。"
                "顺序即优先级，第一个能取到 model_set 的任务会被使用。"
            ),
        )

        request_name: str = Field(
            default="toolcall_repair",
            description="修复请求的 request_name（用于日志/统计，也会被 skip 规则排除）",
        )

        temperature: float = Field(
            default=0.0,
            ge=0.0,
            le=2.0,
            description="修复请求温度，建议 0 以保证稳定复现",
        )

        max_tokens: int = Field(
            default=1200,
            ge=64,
            description="修复请求最大输出 token（限制成本，输出只是一个 JSON）",
        )

        timeout_seconds: float = Field(
            default=45.0,
            gt=0.0,
            description="单次修复请求超时秒数，超时则放弃修复并保持原响应",
        )

        max_attempts: int = Field(
            default=1,
            ge=1,
            le=3,
            description="同一次故障最多发起几次修复请求（默认 1 次，避免 token 浪费）",
        )

        include_reasoning: bool = Field(
            default=False,
            description="是否把模型的 reasoning_content 一并交给修复模型（会增加 token）",
        )

    repair: RepairSection = Field(default_factory=RepairSection)

    @config_section("plain_text")
    class PlainTextSection(SectionBase):
        """纯文本（无 tool_calls）处理策略配置节。"""

        mode: str = Field(
            default="wrap",
            choices=["off", "wrap", "suspicious", "always"],
            description=(
                "纯文本处理策略：`off` 不处理；`wrap`（默认）把纯文本套壳成"
                "本次 chatter 的「发送消息」工具调用（先本地套壳，无法唯一确定"
                "发送工具时才问模型）；`suspicious` 仅当文本看起来像（被破坏的）"
                "工具调用时才修复；`always` 任何纯文本都交给修复模型处理。"
            ),
        )

        wrap_via: str = Field(
            default="local-first",
            choices=["local-first", "always-llm"],
            description=(
                "套壳方式：`local-first`（默认）能唯一定位发送工具时本地套壳"
                "（零 token，结果等价），否则才请求模型；`always-llm` 每次都让"
                "模型决定用哪个工具、正文放进哪个参数（更贵）。"
            ),
        )

        send_tool_patterns: list[str] = Field(
            default=["*send_text*", "*send_message*", "*send_msg*", "*reply*", "send*"],
            description=(
                "发送类工具的名字匹配模式（glob，不区分大小写，同时匹配完整名与"
                "去前缀后的短名）。用于在不同 chatter 的发送工具名字不同时自动识别。"
            ),
        )

        send_tool_name: str = Field(
            default="",
            description=(
                "显式指定发送工具名（留空=自动识别）。当自动识别有歧义、或你的"
                "chatter 用了很特别的名字时填这里，例如 `action-send_text`。"
            ),
        )

        body_params: list[str] = Field(
            default=["content", "text", "message", "msg", "reply", "body"],
            description=(
                "「正文参数」的候选名字，按优先级排列。套壳时纯文本会被原样放进"
                "第一个匹配到的参数里。"
            ),
        )

        max_chars: int = Field(
            default=8000,
            ge=200,
            description="送入修复模型的原始文本最大字符数，超出部分截断",
        )

        max_calls: int = Field(
            default=4,
            ge=1,
            description="一次修复最多接受几个工具调用（超出部分丢弃）",
        )

        clear_message_on_repair: bool = Field(
            default=False,
            description=(
                "纯文本被成功转成工具调用后，是否清空本次响应的 message。"
                "默认保留（false）：模型原文会作为 assistant 独白与调用一起写回上下文，"
                "对依赖文本推进上下文的 chatter 更友好；设为 true 则清空，"
                "避免把坏掉的原始文本（例如半截 JSON）写回上下文。"
            ),
        )

    plain_text: PlainTextSection = Field(default_factory=PlainTextSection)

    @config_section("validation")
    class ValidationSection(SectionBase):
        """本地校验与本地修复配置节。"""

        require_known_tool: bool = Field(
            default=True,
            description="工具名不在本次可用工具集中时视为故障（会尝试修复）",
        )

        resolve_partial_names: bool = Field(
            default=True,
            description=(
                "允许本地补全工具名前缀：例如模型给出 `send_text` 而实际工具是"
                "`action-send_text` 时本地直接改正，不消耗 token"
            ),
        )

        drop_unknown_args: bool = Field(
            default=True,
            description=(
                "本地删除 schema 之外的多余参数（仅当该工具不接受 **kwargs 时执行），"
                "避免 `unexpected keyword argument` 类执行异常"
            ),
        )

        coerce_scalar_types: bool = Field(
            default=True,
            description='本地尝试把 `"3"` 之类的标量纠正成 schema 要求的类型',
        )

        inject_reason: bool = Field(
            default=True,
            description=(
                "修复后的调用若缺少 `reason` 则自动补一个（框架发给模型的 schema 里"
                "`reason` 通常是必填；不支持它的组件由框架自动剥离）"
            ),
        )

        reason_text: str = Field(
            default="修正工具调用格式",
            description="自动补充的 `reason` 文本",
        )

        max_broken_calls: int = Field(
            default=6,
            ge=1,
            description=(
                "坏调用数量超过该值时不再尝试「单工具补参」，直接走「整体重建」"
                "（一次性把整批调用交给修复模型）"
            ),
        )

    validation: ValidationSection = Field(default_factory=ValidationSection)
