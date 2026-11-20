# APIProxy

把国产 AI 平台（WorkBuddy / CodeBuddy、Qoder）封装成本地 **OpenAI Chat Completions** 兼容接口的 Rust 网关。单二进制，自带网页 Console 和 CLI；面向个人本机使用。

## 功能

- `POST /v1/chat/completions`：支持 `stream=true/false`、tools、图片输入、`reasoning_content` 透传、`stream_options.include_usage`
- `GET /v1/models`：自动合并所有提供商/账号的上游模型
- 多账号池：轮询、失败自动切号、401/403 自动停用、429 冷却、额度耗尽冷却
- 浏览器授权登录：生成授权链接、用户打开授权、网关自动轮询并添加账号（WorkBuddy / CodeBuddy、Qoder）
- 自动签到：可配置每日时间（默认 `08:00`），按提供商开关
- 网页 Console：账号 / 模型 / 提供商 / 统计 / 设置
- 统计页：累计请求/错误/Token 用量、各账号上游积分余额、最近请求日志
- CLI：授权登录、账号管理、模型刷新、签到、直接对话测试
- 模块化 Provider：新增一个 crate 即可接入新平台，自动获得账号池、模型合并、签到调度、Console 管理

## 快速开始

```bash
cargo build --release
./target/release/apiproxy serve
```

启动后终端会打印：

```
Console : http://127.0.0.1:8787/
API     : http://127.0.0.1:8787/v1
API Key : sk-...
数据目录: ~/.local/share/apiproxy
```

首次启动自动生成 API Key，保存在 `<数据目录>/data.json`（Unix 下权限 `600`）。端口、监听地址、签到时间可在 Console「设置」修改后重启生效；也可用 `--port` 临时覆盖。

### 授权登录（推荐）

Console「账号 → 授权登录」选择提供商与区域，点「获取授权链接」；浏览器完成授权后，页面会自动轮询并添加账号。

CLI：

```bash
apiproxy login --provider workbuddy --region cn
apiproxy login --provider qoder --region global
# --no-open 只打印链接；--name 可指定账号名称
```

流程：

1. 获取官方设备授权链接（Qoder 使用 PKCE S256；WorkBuddy 使用 CodeBuddy 设备授权）
2. 自动打开浏览器（CLI 可关），用户完成授权
3. 网关每 2 秒轮询一次，成功后自动保存 access token / refresh token / 过期时间、添加账号并刷新模型
4. 链接 10 分钟有效；Console 与 CLI 都显示等待/失败/过期状态

重新登录同一上游账号（稳定 user id / email）会更新凭证；不同账号会保留为多个账号。

仅支持浏览器授权登录；不再提供手动粘贴 token / PAT 的入口。已有账号仍可继续管理（启停、测试、签到、删除）。

## 客户端接入

任意 OpenAI SDK / 客户端：

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-Max",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="sk-...")
print(client.chat.completions.create(
    model="DeepSeek-V4.1-Flash",
    messages=[{"role": "user", "content": "你好"}],
).choices[0].message.content)
```

模型列表：

```bash
curl http://127.0.0.1:8787/v1/models -H "Authorization: Bearer sk-..."
```

### 模型合并

- 所有启用提供商的上游模型自动合并；`/v1/models` 返回官方 `id`、`display_name`、`providers`、`aliases`
- 先按官方模型表重映射别名，例如 Qoder `DeepSeek-Flash` 与 WorkBuddy `deepseek-v4.1-flash` 合并为 `DeepSeek-V4.1-Flash`
- 未在官方别名表中的模型，按大小写、空格、下划线、连字符忽略的方式归一化：`DeepSeek-V4-Pro` 与 `deepseek-v4-pro`、`Qwen-3.8-Max` 与 `qwen3.8 max`、`GLM-5.3` 与 `glm-5.3` 视为同一模型
- 合并后的模型只列出一次；**下游应使用官方 ID 调用**，原 id、显示名、`provider/model` 仍作为兼容别名保留
- 同一模型在多个平台可用时，网关按提供商顺序尝试，某平台无账号/报错会自动切到下一个平台
- 上游模型列表拉取失败时回退到内置模型表（Console 会标注），不影响已有模型路由
- 提供商页显示模型来源：`动态` / `缓存` / `回退` / `停用`，并给出失败原因，避免静默 0 模型
- 最近一次成功获取的完整模型目录会持久化；上游限流返回最小配置时优先使用缓存

## 统计与请求日志

Console「统计」页显示：

- 累计请求数、错误数、提示/补全/总 Token
- 每个启用账号的可用余额、月额度、累计额度与单位（60 秒缓存；WorkBuddy 使用 `billing/meter/get-user-resource`，Qoder 使用 `quota/usage`）
  - WorkBuddy 可用余额 = 各套餐 `CycleCapacityRemain` 之和（当前周期真实可用）
  - WorkBuddy 月额度 = `Status=0` 套餐的 `CapacitySize` 之和
  - WorkBuddy 累计额度 = 所有套餐 `CapacitySize` 之和
  - `CapacityRemain` 是套餐总量/历史余额，不作为当前可用余额
  - 不同单位（credits / tokens）不混合求和，单位一致时才显示总额
- 最近 1000 条请求日志：ID、时间、模型/上游模型、提供商/账号、HTTP 状态、耗时、Token、错误

日志写入数据目录的 `requests.jsonl`（权限 `600`），仅记录元数据，不保存提示词/回复正文；重启后仍可查看。内存与页面均保留最近 1000 条（未认证请求只进内存，不落盘）。

## CLI

```bash
apiproxy serve [--port 8787]     # 启动服务（默认命令）
apiproxy key                     # 打印本机 API Key
apiproxy login --provider qoder --region cn
apiproxy login --provider workbuddy --region global --no-open
apiproxy accounts list
apiproxy accounts remove <id>
apiproxy accounts enable|disable <id>
apiproxy accounts test <id>
apiproxy models [--refresh]
apiproxy providers
apiproxy checkin [--account <id>]
apiproxy settings
apiproxy chat <model> "你好"
```

CLI 通过 HTTP 调用本机服务；若 `serve --port` 使用了非默认端口，请加 `--server http://127.0.0.1:<port>`，或用全局 `--data-dir` 指向同一数据目录。

## 架构

```text
crates/
  core/                 gateway-core
    provider.rs         Provider trait、ChatDelta、错误分类
    registry.rs         官方模型 ID 归一化、跨提供商合并/重名路由
    store.rs            data.json（设置、账号、签到记录、冷却状态）
    openai.rs           OpenAI 响应/SSE 辅助
  provider-workbuddy/   WorkBuddy / CodeBuddy
  provider-qoder/       Qoder（COSY 签名、RSA/AES 会话、自定义 base64）
  gateway/              apiproxy 单二进制
    server.rs           OpenAI API + 管理 API + 调度器
    console.html        内嵌网页 Console
    main.rs             CLI + 启动
```

请求路径：

```text
client → /v1/chat/completions
       → 合并模型表解析 model
       → 账号池选号（轮询 + 冷却/停用过滤）
       → Provider::chat 打开上游流
       → 统一 ChatDelta → OpenAI SSE / 非流式聚合
```

签到：后台每 30 秒检查本地时间；到达设定时间且当天未执行时，对启用且开启自动签到的提供商账号逐个签到，结果写入 `data.json` 并显示在 Console。手动签到不受提供商自动签到开关限制。

### 新增 Provider

1. `crates/provider-foo`，依赖 `gateway-core`
2. 实现 `Provider`：`models`、`chat`、`checkin`、`fallback_models`，可选覆盖 `validate`
   以及 `supports_login` / `start_login` / `poll_login`（浏览器授权）
3. `crates/gateway/src/main.rs` 的 `providers()` 注册
4. `crates/core` 无需改动；账号池、模型合并、Console、CLI、签到调度自动生效

`chat` 始终返回增量流；非流式由网关聚合，因此上游只支持 SSE 也没问题。

## 数据与安全

- 默认监听 `127.0.0.1:8787`，只适合本机个人使用
- 账号凭证（access token / refresh token）明文保存在 `data.json`（权限 `600`）；请求日志在 `requests.jsonl`（`600`）
- 不要把数据目录或 Console 暴露到公网
- 若确需从其他设备访问，请自行加反向代理、TLS 和访问控制；本项目没有多用户/权限体系
- 上游协议（尤其 Qoder 的 COSY 签名与审核策略）可能随时变化；请遵守各平台服务条款，仅用于个人自用

## 已知边界

- WorkBuddy 签到固定调用官方 `workbuddy.cn` 接口，仅 OAuth 登录账号可签；`ck_` Key 会显示跳过
- Qoder 签到依赖官方活动接口；无活动时返回“暂无可签到活动”
- OAuth 的 refresh token 会保存，但自动续期尚未接入，凭证过期后需重新授权
- 请求日志仅记录元数据（不含提示词/回复正文），页面展示最近 1000 条
- Qoder 自定义 Base URL 同时用于 OAuth/签到 OpenAPI；一般建议留空
- 上游非流式请求也不做本地模型参数校验，错误按上游状态码回传

## 验证

```bash
cargo test
cargo clippy --all-targets
```

本地端到端验证可用任意 OpenAI 客户端指向 `serve` 打印的地址，或在 Console 中先「测试」账号、再在模型页刷新模型。
