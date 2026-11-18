# 所有的工具调用失败都给我tm去死吧插件

> 技术名（目录 / 模块 / 配置路径 / 服务名）：`all_toolcall_failures_go_die`；
> `display_name` 是这个中文名，框架与插件市场里显示的就是它。
>
> 拦截「模型返回坏掉的工具调用」或「只回一句纯文本」，在框架执行之前把它修成
> 可执行的工具调用。纯文本套壳默认**零 token 本地完成**，其余情况也只请求一次
> 轻量 LLM。

## 它解决什么问题

Neo-MoFox 的发言机制是「LLM 返回 JSON → 框架解析出工具调用」。实际运行时
模型经常会返回下面这些「框架解析不了」的东西：

| 现象 | 框架侧表现 |
| --- | --- |
| 只回纯文本，没有 `tool_calls`（例如「你好，我在」） | chatter 记一条「LLM 返回纯文本而非工具调用」，直接等待，**本轮不发言** |
| 工具名写成 `send_text`（真实工具是 `action-send_text`） | 「未知的工具」 |
| `arguments` 是字符串（`"{\"content\":\"hi\"}"`） | 参数解析失败，`args` 退化成 `{}` |
| 必填参数缺失 | 执行时 `TypeError: missing required argument` |
| 多带了 schema 里没有的参数 | `unexpected keyword argument` |

普通纯文本这一条尤其常见（相当于 `plain_text_kicker` 负责的场景）：消息已经
写好了，只是没包成工具调用。本插件的默认行为就是**把它套壳成当前 chatter 的
「发送消息」工具调用**，让框架照常发出去。

## 工作流程

```
LLM 响应
   │
   ├─ AFTER_LLM_REQUEST 事件（本插件 weight=200，先于 plain_text_kicker 等执行）
   │
   ├─ ① 有 tool_calls：用「本次请求真实发给模型的工具 schema」做校验
   │      └─ 全部合法 → 立刻放行，零开销
   │
   ├─ ② 本地能修就本地修（0 token）
   │      · 工具名缺前缀 → 补成 action-xxx
   │      · args 是 JSON 字符串 → 解析成对象
   │      · "30" → 30（按 schema 纠标量类型）
   │      · 删除 schema 之外的多余参数（除非工具接受 **kwargs）
   │      · 补 reason（框架要求）
   │      · 纯文本里直接能抠出合法调用 → 直接抠（fenced JSON / tool_calls 容器 /
   │        函数调用式语法 / 平衡括号切片）
   │
   ├─ ③ 纯文本（mode=wrap，默认）：套壳成「发送消息」工具调用
   │      · 先在本次工具目录里定位当前 chatter 的发送工具（见下节）
   │      · 定位到唯一工具 → 本地把原文塞进正文参数（0 token）
   │      · 定位到多个候选 / wrap_via=always-llm → 让模型选工具并套壳
   │      · 目录里没有发送工具 → 不做套壳，退回「坏调用」判断
   │
   └─ ④ 本地修不掉且看起来像坏调用 → 一次轻量修复请求
          · 输入：原始输出（截断）+ 已解析调用 + 框架报错 + 精简工具目录
          · 输出：只允许一个 JSON（{"tool_calls":[...]} 或 {"no_tool_call":true}）
          · 拿到结果后再校验一遍：只有完全合法才写回，否则保持原样
```

修复结果写回 `params["tool_calls"]`（`[{"id","name","args"}]`），框架会自动
重建 `ToolCall` 对象，chatter 拿到的就是可直接执行的合法调用。

### 怎么找到「当前 chatter 的发送工具」

工具名不能写死（不同 chatter 可能叫 `send_text` / `reply_message` / ……），
所以从**本次请求的工具目录**里按优先级识别：

1. `plain_text.send_tool_name` 显式指定（最省事，推荐给名字特别的 chatter）；
2. 名字命中 `send_tool_patterns`（glob，默认可匹配 `send_*` / `*reply*`），且有正文参数；
3. 唯一的「只有一个必填字符串参数、参数名属于 `body_params`」的工具；
4. 描述里含发送/回复关键词，且有正文参数。

命中多个时，优先取 `chatter_allow` 与本次请求名一致的（即**当前生效的 chatter**），
再优先没有 `chatter_allow` 限制的；仍然不唯一就交给模型选。
`send_emoji_meme` 这类「也在发送、但不是文本」的工具会被名字与结构规则排除。

### 纯文本处理的四种模式

| `plain_text.mode` | 行为 |
| --- | --- |
| `off` | 完全不处理 |
| `wrap`（默认） | 先本地抢救坏调用；再套壳成发送调用（本地优先）；都不行才不动 |
| `suspicious` | 只在文本看起来像坏掉的调用时才处理（旧行为） |
| `always` | 任何纯文本都直接交给修复模型（不套壳，调试用） |

### 各路路径的成本

| 路径 | 触发条件 | token 成本 |
| --- | --- | --- |
| 直接放行 | 调用合法 | 0 |
| 本地修复 | 名字/类型/多余参数/args 字符串 | **0** |
| 纯文本套壳（本地） | 目录里能唯一定位发送工具 | **0** |
| 纯文本套壳（模型） | 候选不唯一 / `wrap_via=always-llm` | 小（精简目录 + 原文） |
| 单工具补参 | 只有一个调用坏了，且工具名已确定 | 极小（工具 schema + 报错） |
| 整体重建 | 多个调用坏了 / 工具名未知 | 小（精简目录 + 原始输出） |

## 安装

把整个 `all_toolcall_failures_go_die/` 目录放进 `plugins/`，重启 Neo-MoFox 即可，
首次加载会自动生成 `config/plugins/all_toolcall_failures_go_die/config.toml`。

不需要修改框架，也不需要动其它插件；本插件不依赖任何第三方库。

## 配置速查

完整注释见 `config/plugins/all_toolcall_failures_go_die/config.toml`。

```toml
[main]
enabled = true                      # 总开关
request_name_patterns = ["*"]       # 参与修复的请求名（glob）
skip_request_name_patterns = ["toolcall_repair*"]
max_repairs_per_minute = 30         # 修复请求限流（本地修复不计入）
debug_log = false
catalog_fallback = false            # 工具目录未命中时是否用全局注册表兜底（不建议）

[repair]
task_name = "tool_use"              # 用哪个模型任务做修复（可自行改）
fallback_task_names = ["utils", "utils_small", "actor"]  # 取不到时按序回退
temperature = 0.0
max_tokens = 1200
timeout_seconds = 45.0
max_attempts = 1                    # 一次故障最多修一次，绝不追求「修到对为止」

[plain_text]
mode = "wrap"                       # off / wrap / suspicious / always
wrap_via = "local-first"            # local-first（零 token 优先） / always-llm
send_tool_patterns = ["*send_text*", "*send_message*", "*send_msg*", "*reply*", "send*"]
send_tool_name = ""                 # 显式指定发送工具（留空=自动识别）
body_params = ["content", "text", "message", "msg", "reply", "body"]
max_calls = 4
clear_message_on_repair = false     # 默认保留模型原文作为 assistant 独白

[validation]
require_known_tool = true
resolve_partial_names = true
drop_unknown_args = true
coerce_scalar_types = true
inject_reason = true
```

### 选哪个修复模型

`repair.task_name` 默认 `tool_use`，**完全由配置决定**：想省钱就改成你
`config/model.toml` 里更便宜的小模型任务（例如 `utils_small`），修复任务只需
「读 schema + 补参数」，对智能要求很低。`task_name` 取不到时按
`repair.fallback_task_names` 依次回退（默认 `utils → utils_small → actor`），
所以写错任务名也不会崩，只会降级或放弃这次修复。

### 纯文本修复后要不要清空原文

`plain_text.clear_message_on_repair` 默认 **`false`（保留）**：模型原来的文本会
作为 assistant 独白，和修好的工具调用一起写回上下文，对依赖文本推进上下文的
chatter 更友好。如果你发现「半截 JSON 原文」留在上下文里反而干扰模型，
把它改成 `true` 即可在修复成功时清空 `message`。

### `plain_text.mode` 怎么选

- `wrap`（默认）：把纯文本**套壳**成当前 chatter 的发送工具调用。
  顺序是「本地抢救坏调用 → 本地套壳 → 必要时问模型 → 都不行就不动」。
  普通聊天（「你好，我在」）会在 1ms 内变成 `action-send_text(content="你好，我在")`，
  零 token。
- `off`：完全不处理纯文本（只用本地修复调用类故障）。
- `suspicious`：只有当文本「看起来像坏掉的工具调用」时才处理
  （JSON 片段、`tool_calls`/`function_call` 等关键词、`工具名(...)` 语法、
  提到了本次可用工具名）。不套壳，普通聊天完全不受影响。
- `always`：任何纯文本都问一次修复模型（不套壳）。最激进也最费钱，
  只在排查问题时用。

### `plain_text.wrap_via` 怎么选

- `local-first`（默认）：能在本次工具目录里唯一定位发送工具时，直接把原文塞进
  正文参数（零 token，结果和模型套壳完全等价）；只有在候选不唯一（例如同时
  存在多个 `send_*`）时才请求一次模型，让模型按语义挑。
- `always-llm`：每次都让模型决定用哪个工具、正文放进哪个参数。更贵，
  适合发送工具参数结构复杂（需要一并填引用、@、话题等）的场景。

### 套壳不会做的事

- 不把原文拆成多条消息（一次纯文本 → 一次发送调用）；
- 不改写、不补全、不删除原文；
- 目录里没有发送工具时不做套壳（宁可不动，也不伪造一个本次请求没提供的工具）。

### 发送工具还有别的必填参数怎么办（以 KFC 为例）

有些 chatter 的发送工具除了正文之外还有必填字段。例如
`kokoro_flow_chatter`（KFC）的 `kfc_reply`：`content` 是 `list[str]`，另外强制要求
`thought` / `expected_reaction` / `max_wait_seconds` / `mood` 四个决策元数据。

这种情况下本地套壳只填得动 `content`，会被本地校验判「缺必填」，
于是自动升级成一次 LLM 套壳：提示词里带上该工具的**完整 schema 与必填清单**，
要求「正文原样 + 其余必填项按语气给合理值」。实测产出：

```json
{"name": "action-kfc_reply",
 "args": {"content": ["在的，怎么了？"], "thought": "对方叫我，我先应一声。",
          "expected_reaction": "对方说出找我什么事。", "max_wait_seconds": 60,
          "mood": "平静"}}
```

比 KFC 自己的兜底（拿到纯文本后带着**整轮上下文**再请求一次）便宜得多。
实测 KFC 的几种坏响应都能修好，明细见 `tests/README`（见下方「自测」）。

## 安全与兜底

- **fail-open**：任何异常、超时、修复结果不合法、限流命中，都保持原响应不变；
  本插件永远不会让一次修复失败变成一次崩溃。
- **防递归**：修复请求自身的响应（`request_name` 命中 skip 规则）不会被再次修复；
  另外还有 `ContextVar` 级别的递归保护。
- **不流式处理**：`stream=true` 的响应在 `AFTER_LLM_REQUEST` 阶段还没有
  `tool_calls`，直接跳过，避免误判。
- **工具集来源可信**：用 `BEFORE_LLM_REQUEST` 缓存的「本次真的发给模型的工具列表」
  （框架从 `ROLE.TOOL` payload 里抽出的 `LLMUsable`，就是交给 provider 的那一份），
  而不是全局注册表。缓存未命中时默认**放弃修复**而不是用全局注册表猜
  （`main.catalog_fallback=false`）：全局有、本次没提供的工具一旦写回，调度层会判
  「未知的工具」，比不修更糟。
- **补 id**：修复出来的调用如果没有 `id`，会补一个唯一 id，保证 `TOOL_RESULT`
  与 assistant 的 tool_call 正确配对。

## 与 `plain_text_kicker` 的关系

本插件是它的升级版：纯文本套壳这条路径本身就覆盖了
`plain_text_kicker` 的职责，而且走的是「正常工具调用」，chatter 的工具流、
冷却、发送后处理全都照常生效，不是旁路直发。

`plain_text_kicker`（weight=100）在本插件（weight=200）之后跑，并且它看到
`tool_calls` 非空就跳过。所以：

- 套壳/修复成功 → `plain_text_kicker` 自动让路，不会重复发送；
- 本插件判定「没有发送工具 / 不值得动」→ 仍由 `plain_text_kicker` 按自己的
  策略兜底。

两者可以同时启用，也可以只用本插件（把 `plain_text.mode` 设为 `wrap` 即完全
接管纯文本）。

## 其它插件怎么用

服务 `all_toolcall_failures_go_die:toolcall_repair` 也对外开放：

```python
svc = service_manager.get_service("all_toolcall_failures_go_die:toolcall_repair")

# 只做本地校验/修复，不发请求
calls = svc.audit_calls([{"name": "send_text", "args": {"content": "hi"}}])

# 尽力把一段文本变成合法调用（本地抢救 → 必要时一次 LLM）
calls = await svc.repair_text('{"name": "send_text", "args": {}}')
```

## 目录结构

```
all_toolcall_failures_go_die/
├── manifest.json
├── plugin.py            # 插件入口与组件注册
├── config.py            # 配置定义（main / repair / plain_text / validation）
├── handler.py           # AFTER_LLM_REQUEST 事件处理器（核心编排）
├── service.py           # 对外服务
├── core/
│   ├── schemas.py       # 工具目录、schema 校验、本地修复、JSON 抢救（纯标准库）
│   ├── extraction.py    # 从文本里抽取候选调用（纯标准库）
│   ├── send_tool.py     # 识别「当前 chatter 的发送工具」并套壳（纯标准库）
│   ├── prompts.py       # 修复/套壳提示词与回复解析（纯标准库）
│   ├── catalog.py       # 组件类 → 工具目录
│   └── repairer.py      # 修复请求的发送与容错
├── tests/
│   ├── test_core.py               # 纯标准库离线自测
│   ├── test_handler_integration.py # 事件链路集成自测
│   ├── test_repairer.py           # 修复请求的容错自测
│   └── real_llm_probe.py          # 真实模型冒烟（可选 --chatter kfc / --broken ...）
└── docs/设计说明.md
```

## 自测

```bash
# 只测纯逻辑，不需要框架、不联网
python plugins/all_toolcall_failures_go_die/tests/test_core.py

# 测事件链路（需要框架可导入，不联网：修复器被打桩）
python plugins/all_toolcall_failures_go_die/tests/test_handler_integration.py

# 真实模型冒烟：默认场景 / KFC 场景 / 构造坏响应
python plugins/all_toolcall_failures_go_die/tests/real_llm_probe.py "你好，我在"
python plugins/all_toolcall_failures_go_die/tests/real_llm_probe.py --chatter kfc "在的，怎么了？"
python plugins/all_toolcall_failures_go_die/tests/real_llm_probe.py --chatter kfc --broken missing-meta "在的"
python plugins/all_toolcall_failures_go_die/tests/real_llm_probe.py --broken bad-name --show-prompts "你好"
```

`real_llm_probe.py` 的 `--broken` 可选：`bad-name`（瞎编工具名）、
`missing-meta`（缺必填）、`string-args`（args 是 JSON 字符串）、`no-prefix`（缺类型前缀）。
退出码 0 表示通过。

## 已知限制

- 流式响应的 `tool_calls` 在 `AFTER_LLM_REQUEST` 阶段还拿不到，不做修复。
- 「模型该调用工具但回了一段人话」这种语义层判断，本插件只能按
  `plain_text.mode` 的启发式规则处理；`suspicious` 模式下明显是闲聊的内容不会被打扰。
- 发送工具有多个必填参数（如 KFC 的四个决策元数据）时，本地套壳只能填正文，
  一定升级成一次 LLM 套壳（正确性优先，成本仍远低于重发整轮上下文）。
- 修复请求本身也会失败（小模型不听话、被截断、偶发返回空内容）。此时保持原样，
  不重试——这比「为了修对再烧几轮」更符合「省 token」的初衷。
