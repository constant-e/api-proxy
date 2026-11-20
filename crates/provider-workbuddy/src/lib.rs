//! WorkBuddy / CodeBuddy (Tencent) provider.
//!
//! Upstream is OpenAI-compatible (`POST /v2/chat/completions`), authenticated
//! with a `ck_` API key or OAuth Bearer token. It only supports streaming, so
//! the gateway aggregates for `stream=false`. Daily check-in is a separate
//! `workbuddy.cn` endpoint.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use base64::Engine;
use futures::StreamExt;
use gateway_core::provider::{LoginAccount, LoginPoll, LoginStart, Provider, ProviderError};
use gateway_core::{AccountView, Balance, ChatDelta, ChatStream, CheckinOutcome, RemoteModel};
use serde_json::{json, Map, Value};

const CHAT_PATH: &str = "/v2/chat/completions";
const MODELS_PATH: &str = "/v2/models";
const CONFIG_PATH: &str = "/v3/config";
const CHECKIN_URL: &str = "https://www.workbuddy.cn/billing/meter/daily-checkin";
const UA: &str = "CLI/unknown CodeBuddy/2.136.0";

#[derive(Clone)]
struct PendingLogin {
    state: String,
    base: String,
    region: String,
    expires_at: i64,
}

pub struct WorkBuddyProvider {
    logins: Mutex<HashMap<String, PendingLogin>>,
}

impl WorkBuddyProvider {
    pub fn new() -> Self {
        Self {
            logins: Mutex::new(HashMap::new()),
        }
    }

    fn base(account: &AccountView<'_>) -> String {
        if let Some(base) = account.base_url.filter(|s| !s.trim().is_empty()) {
            return base.trim_end_matches('/').to_string();
        }
        let global = matches!(
            account.region.map(|r| {
                let r = r.to_ascii_lowercase();
                r == "global" || r == "intl" || r == "international" || r == "saas"
            }),
            Some(true)
        );
        if global {
            "https://www.codebuddy.ai".into()
        } else {
            "https://copilot.tencent.com".into()
        }
    }

    fn login_base(region: Option<&str>, base: Option<&str>) -> (String, String) {
        let global = matches!(
            region.map(|r| r.to_ascii_lowercase()),
            Some(r) if r == "global" || r == "intl" || r == "international" || r == "saas"
        );
        let base = base
            .filter(|b| !b.trim().is_empty())
            .map(|b| b.trim_end_matches('/').to_string())
            .unwrap_or_else(|| {
                if global {
                    "https://www.codebuddy.ai".into()
                } else {
                    "https://copilot.tencent.com".into()
                }
            });
        let tag = if base.contains("codebuddy.ai") {
            "global"
        } else {
            "cn"
        };
        (base, tag.to_string())
    }

    fn login_headers(base: &str, extra_b3: bool) -> Vec<(&'static str, String)> {
        let host = reqwest::Url::parse(base)
            .ok()
            .and_then(|u| u.host_str().map(str::to_string))
            .unwrap_or_default();
        let mut headers = vec![
            ("Host", host.clone()),
            ("X-Domain", host),
            ("User-Agent", "CLI/1.0.8 CodeBuddy/1.0.8".to_string()),
            ("X-Product", "SaaS".to_string()),
            ("X-No-Authorization", "true".to_string()),
            ("X-No-User-Id", "true".to_string()),
            ("Accept", "application/json".to_string()),
        ];
        if extra_b3 {
            headers.push(("X-B3-TraceId", rand_hex(16)));
            headers.push(("X-B3-SpanId", rand_hex(16)));
            headers.push(("X-B3-Sampled", "1".to_string()));
        }
        headers
    }

    fn apply_auth(
        req: reqwest::RequestBuilder,
        token: &str,
        account: &AccountView<'_>,
        base: &str,
    ) -> reqwest::RequestBuilder {
        let mut req = req
            .header("Authorization", format!("Bearer {token}"))
            .header("Accept", "text/event-stream")
            .header("Content-Type", "application/json")
            .header("Accept-Encoding", "identity")
            .header("User-Agent", UA);
        // X-API-Key is required by /v3/config but rejected by chat/billing for
        // OAuth JWTs (401 not_found), so it is added only on model requests.
        if token.starts_with("ck_") {
            req = req.header("X-Api-Key", token);
        }
        if base.contains("codebuddy.ai") {
            req = req
                .header("X-Domain", "www.codebuddy.ai")
                .header("X-Product", "SaaS")
                .header("X-IDE-Type", "CLI")
                .header("X-Agent-Intent", "craft");
        } else if base.contains("copilot.tencent.com") && !token.starts_with("ck_") {
            req = req.header("X-Domain", "www.codebuddy.cn");
            if let Some(uid) = jwt_subject(token) {
                req = req.header("X-User-Id", uid);
            }
        }
        let _ = account;
        req
    }
}

impl Default for WorkBuddyProvider {
    fn default() -> Self {
        Self::new()
    }
}

fn classify(status: u16, body: &str) -> ProviderError {
    let msg = format!(
        "HTTP {status}: {}",
        body.chars().take(400).collect::<String>()
    );
    match status {
        401 | 403 => ProviderError::Auth(msg),
        429 => ProviderError::RateLimited {
            message: msg,
            retry_after_ms: 60_000,
        },
        _ if (400..500).contains(&status) => ProviderError::BadRequest(msg),
        _ => ProviderError::Unavailable(msg),
    }
}

#[async_trait]
impl Provider for WorkBuddyProvider {
    fn id(&self) -> &'static str {
        "workbuddy"
    }

    fn name(&self) -> &'static str {
        "WorkBuddy / CodeBuddy"
    }

    fn default_base_url(&self) -> &'static str {
        "https://copilot.tencent.com"
    }

    fn credential_identity(&self, credential: &str) -> Option<String> {
        jwt_remote_id(credential)
    }

    fn supports_login(&self) -> bool {
        true
    }

    async fn start_login(
        &self,
        client: &reqwest::Client,
        region: Option<&str>,
        base: Option<&str>,
    ) -> Result<LoginStart, ProviderError> {
        let (base, tag) = Self::login_base(region, base);
        let nonce = rand_hex(8);
        let url = format!("{base}/v2/plugin/auth/state?platform=CLI&nonce={nonce}");
        let mut req = client
            .post(url)
            .timeout(std::time::Duration::from_secs(20))
            .json(&json!({ "nonce": nonce }));
        for (k, v) in Self::login_headers(&base, false) {
            req = req.header(k, v);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| ProviderError::Unavailable(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        if status >= 400 {
            return Err(classify(status, &text));
        }
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| ProviderError::Unavailable(format!("登录响应解析失败: {e}")))?;
        let auth_url = value
            .pointer("/data/authUrl")
            .and_then(Value::as_str)
            .unwrap_or("");
        let state = value
            .pointer("/data/state")
            .and_then(Value::as_str)
            .unwrap_or("");
        if auth_url.is_empty() || state.is_empty() {
            return Err(ProviderError::Unavailable(format!(
                "上游未返回授权链接: {}",
                text.chars().take(300).collect::<String>()
            )));
        }
        let flow_id = uuid::Uuid::new_v4().to_string();
        let now = gateway_core::now_ms();
        self.logins.lock().unwrap().insert(
            flow_id.clone(),
            PendingLogin {
                state: state.to_string(),
                base,
                region: tag,
                expires_at: now + 10 * 60 * 1000,
            },
        );
        Ok(LoginStart {
            flow_id,
            url: auth_url.to_string(),
            message: "在浏览器中完成 WorkBuddy / CodeBuddy 登录".into(),
            expires_at: Some(now + 10 * 60 * 1000),
        })
    }

    async fn poll_login(
        &self,
        client: &reqwest::Client,
        flow_id: &str,
    ) -> Result<LoginPoll, ProviderError> {
        let pending = self.logins.lock().unwrap().get(flow_id).cloned();
        let Some(pending) = pending else {
            return Ok(LoginPoll::Expired {
                message: "登录会话不存在或已结束".into(),
            });
        };
        if gateway_core::now_ms() > pending.expires_at {
            self.logins.lock().unwrap().remove(flow_id);
            return Ok(LoginPoll::Expired {
                message: "登录链接已过期，请重新获取".into(),
            });
        }
        let url = format!(
            "{}/v2/plugin/auth/token?state={}",
            pending.base,
            urlencode(&pending.state)
        );
        let mut req = client.get(url).timeout(std::time::Duration::from_secs(20));
        for (k, v) in Self::login_headers(&pending.base, true) {
            req = req.header(k, v);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| ProviderError::Unavailable(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        if status >= 400 {
            return Err(classify(status, &text));
        }
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| ProviderError::Unavailable(format!("授权轮询解析失败: {e}")))?;
        let code = value.get("code").and_then(|c| {
            c.as_i64()
                .or_else(|| c.as_str().and_then(|s| s.parse().ok()))
        });
        if code == Some(11217) {
            return Ok(LoginPoll::Pending {
                message: "等待浏览器完成授权".into(),
            });
        }
        if code != Some(0) {
            return Ok(LoginPoll::Failed {
                message: format!(
                    "授权失败 code={}: {}",
                    code.unwrap_or(-1),
                    text.chars().take(300).collect::<String>()
                ),
            });
        }
        let data = value.get("data").cloned().unwrap_or(Value::Null);
        let credential = ["accessToken", "token"]
            .iter()
            .find_map(|k| data.get(*k).and_then(Value::as_str))
            .unwrap_or("")
            .to_string();
        if credential.is_empty() {
            return Ok(LoginPoll::Pending {
                message: "等待授权令牌".into(),
            });
        }
        let name = jwt_name(&credential).unwrap_or_else(|| {
            let tail: String = credential
                .chars()
                .rev()
                .take(6)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            format!("WorkBuddy {} {tail}", pending.region)
        });
        let remote_id = jwt_remote_id(&credential);
        let refresh_token = data
            .get("refreshToken")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        let expires_at = data
            .get("expiresIn")
            .and_then(Value::as_i64)
            .filter(|n| *n > 0)
            .map(|secs| gateway_core::now_ms() + secs * 1000);
        self.logins.lock().unwrap().remove(flow_id);
        Ok(LoginPoll::Success(LoginAccount {
            name,
            remote_id,
            credential,
            refresh_token,
            expires_at,
            region: Some(pending.region),
            base_url: None,
        }))
    }

    async fn cancel_login(&self, flow_id: &str) {
        self.logins.lock().unwrap().remove(flow_id);
    }

    fn fallback_models(&self) -> Vec<RemoteModel> {
        // Current CodeBuddy CLI catalog (captured from /v3/config
        // data.agents[].models + data.models). Keep this in sync with the
        // live list; it is only used when model discovery is throttled.
        [
            ("hy4std", "Hy4", 960_000u32, true, true),
            ("hy4flash-free", "Hy4 flash", 960_000, true, true),
            ("hy4flash", "Hy4 flash", 960_000, true, true),
            ("default", "Default", 200_000, false, false),
            (
                "deepseek-v4.1-flash",
                "DeepSeek-V4.1-Flash",
                1_000_000,
                true,
                true,
            ),
            ("glm-5.3", "GLM-5.3", 1_000_000, true, true),
            ("glm-5.3-flash", "GLM-5.3-Flash", 1_000_000, true, true),
            ("glm-5.2", "GLM-5.2", 1_000_000, true, true),
            ("space-bunny", "Space-Bunny", 1_000_000, true, true),
            (
                "minimax-m3.1-flash",
                "MiniMax-M3.1-Flash",
                1_000_000,
                true,
                true,
            ),
            ("minimax-m2.7", "MiniMax-M2.7", 200_000, true, true),
            ("kimi-k3.1-pro", "Kimi-K3.1-Pro", 1_000_000, true, true),
            ("kimi-k3-1", "Kimi-K3", 1_000_000, true, true),
            (
                "kimi-k2.8-preview",
                "Kimi-K2.8-Preview",
                1_000_000,
                true,
                true,
            ),
            ("deepseek-v4-pro", "DeepSeek-V4-Pro", 1_000_000, true, true),
            ("step-5-preview", "Step-5-Preview", 1_000_000, true, true),
            (
                "mimo-v2.6-pro",
                "Xiaomi MiMo-V2.6-Pro",
                1_000_000,
                true,
                true,
            ),
        ]
        .into_iter()
        .map(|(id, name, context, vision, reasoning)| RemoteModel {
            id: id.to_string(),
            upstream_id: id.to_string(),
            name: name.to_string(),
            context_length: Some(context),
            reasoning,
            vision,
        })
        .collect()
    }

    async fn models(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
    ) -> Result<Vec<RemoteModel>, ProviderError> {
        let base = Self::base(account);
        let mut errors: Vec<String> = Vec::new();
        for path in [MODELS_PATH, CONFIG_PATH] {
            let attempts = if path == CONFIG_PATH { 2 } else { 1 };
            for attempt in 0..attempts {
                let req = Self::apply_auth(
                    client.get(format!("{base}{path}")),
                    account.credential,
                    account,
                    &base,
                )
                .timeout(std::time::Duration::from_secs(20))
                // /v3/config requires the key header for OAuth JWTs too.
                .header("X-Api-Key", account.credential)
                .header("Accept", "application/json");
                let resp = match req.send().await {
                    Ok(resp) => resp,
                    Err(e) => {
                        // A transport failure on /v2/models must not prevent
                        // trying /v3/config; otherwise a transient outage skips
                        // the only endpoint that currently serves the catalog.
                        errors.push(format!("{path} request failed: {e}"));
                        break;
                    }
                };
                let status = resp.status().as_u16();
                let text = resp.text().await.unwrap_or_default();
                if status >= 400 {
                    errors.push(format!("{path} HTTP {status}"));
                    break;
                }
                let value: Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        errors.push(format!("{path} JSON: {e}"));
                        break;
                    }
                };
                let models = if path == CONFIG_PATH {
                    parse_config_models(&value)
                } else {
                    parse_models(&value)
                };
                if !models.is_empty() {
                    tracing::info!(
                        provider = "workbuddy",
                        path,
                        count = models.len(),
                        "workbuddy models loaded"
                    );
                    return Ok(models);
                }
                let code = value.get("code").and_then(|c| c.as_i64());
                errors.push(format!(
                    "{path} returned no models (code={})",
                    code.map(|c| c.to_string()).unwrap_or_else(|| "?".into())
                ));
                if attempt + 1 < attempts {
                    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
                }
            }
        }
        Err(ProviderError::Unavailable(errors.join("; ")))
    }

    async fn chat(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
        request: &Value,
        upstream_model: &str,
    ) -> Result<ChatStream, ProviderError> {
        let base = Self::base(account);
        let payload = build_payload(request, upstream_model);
        let req = Self::apply_auth(
            client.post(format!("{base}{CHAT_PATH}")).json(&payload),
            account.credential,
            account,
            &base,
        );
        let resp = req
            .send()
            .await
            .map_err(|e| ProviderError::Unavailable(e.to_string()))?;
        let status = resp.status().as_u16();
        if status >= 400 {
            let text = resp.text().await.unwrap_or_default();
            return Err(classify(status, &text));
        }
        let is_sse = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.to_ascii_lowercase().contains("text/event-stream"))
            .unwrap_or(false);
        if !is_sse {
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Unavailable(format!(
                "上游未返回 SSE: {}",
                text.chars().take(300).collect::<String>()
            )));
        }

        let state = (resp.bytes_stream(), Vec::<u8>::new(), false);
        let stream = futures::stream::unfold(state, |(mut bytes, mut buf, mut done)| async move {
            loop {
                if let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    let line = String::from_utf8_lossy(&line);
                    let line = line.trim();
                    let Some(data) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let data = data.trim();
                    if data == "[DONE]" {
                        return None;
                    }
                    if let Ok(value) = serde_json::from_str::<Value>(data) {
                        if let Some(err) = value.get("error") {
                            return Some((
                                Err(ProviderError::Unavailable(err.to_string())),
                                (bytes, buf, done),
                            ));
                        }
                        let delta = parse_chunk(&value);
                        if !delta.is_empty() {
                            return Some((Ok(delta), (bytes, buf, done)));
                        }
                    }
                    continue;
                }
                if done {
                    if buf.is_empty() {
                        return None;
                    }
                    buf.push(b'\n');
                    continue;
                }
                match bytes.next().await {
                    Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
                    Some(Err(e)) => {
                        done = true;
                        return Some((
                            Err(ProviderError::Unavailable(e.to_string())),
                            (bytes, buf, done),
                        ));
                    }
                    None => done = true,
                }
            }
        });
        Ok(Box::pin(stream))
    }

    async fn checkin(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
    ) -> Result<CheckinOutcome, ProviderError> {
        if account.credential.starts_with("ck_") {
            return Ok(CheckinOutcome {
                status: "skipped".into(),
                message: "ck_ 静态 Key 不支持签到".into(),
                reward: None,
            });
        }
        let resp = client
            .post(CHECKIN_URL)
            .timeout(std::time::Duration::from_secs(20))
            .bearer_auth(account.credential)
            .header("Content-Type", "application/json")
            .body("{}")
            .send()
            .await
            .map_err(|e| ProviderError::Unavailable(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        parse_checkin_response(status, &text)
    }

    async fn balance(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
    ) -> Result<Option<Balance>, ProviderError> {
        let base = Self::base(account);
        let req = Self::apply_auth(
            client.post(format!("{base}/v2/billing/meter/get-user-resource")),
            account.credential,
            account,
            &base,
        )
        .timeout(std::time::Duration::from_secs(15))
        .header("Accept", "application/json")
        .json(&json!({}));
        let resp = req
            .send()
            .await
            .map_err(|e| ProviderError::Unavailable(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        if status >= 400 {
            return Err(classify(status, &text));
        }
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| ProviderError::Unavailable(format!("额度响应解析失败: {e}")))?;
        let code = value.get("code").and_then(|c| {
            c.as_i64()
                .or_else(|| c.as_str().and_then(|s| s.parse().ok()))
        });
        if code != Some(0) {
            return Err(ProviderError::BadRequest(format!(
                "额度查询失败 code={}: {}",
                code.map(|c| c.to_string()).unwrap_or_else(|| "?".into()),
                value
                    .get("msg")
                    .or_else(|| value.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
            )));
        }
        parse_resource_balance(&value).map_err(ProviderError::BadRequest)
    }

    async fn validate(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
    ) -> Result<String, ProviderError> {
        let base = Self::base(account);
        let req = Self::apply_auth(
            client.get(format!("{base}/v2/accounts")),
            account.credential,
            account,
            &base,
        )
        .timeout(std::time::Duration::from_secs(20))
        .header("Accept", "application/json");
        let resp = req
            .send()
            .await
            .map_err(|e| ProviderError::Unavailable(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        if status >= 400 {
            return Err(classify(status, &text));
        }
        let value: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        let nick = value
            .pointer("/data/accounts/0/nickname")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let models = self
            .models(client, account)
            .await
            .map(|m| m.len())
            .unwrap_or(0);
        if nick.is_empty() && models == 0 {
            return Err(ProviderError::Auth(
                "上游未返回账号信息或模型，凭证可能无效".into(),
            ));
        }
        Ok(if nick.is_empty() {
            format!("凭证有效，发现 {models} 个模型")
        } else {
            format!("凭证有效（{nick}），发现 {models} 个模型")
        })
    }
}

/// Build the upstream body from the client request. Keeps known chat fields
/// only; upstream rejects several OpenAI extras.
fn numeric(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|s| s.parse().ok()))
}

fn parse_resource_balance(value: &Value) -> Result<Option<Balance>, String> {
    let accounts = value
        .pointer("/data/Response/Data/Accounts")
        .and_then(Value::as_array);
    if let Some(items) = accounts {
        if !items.is_empty() {
            let mut available = 0.0;
            let mut monthly = 0.0;
            let mut total = 0.0;
            let mut found = false;
            let mut units = std::collections::BTreeSet::new();
            for item in items {
                // Current usable balance is the cycle remaining. CapacityRemain
                // is the package's lifetime/trial remainder and must not be
                // presented as the current balance.
                let remain = ["CycleCapacityRemain", "CapacityRemain"]
                    .iter()
                    .find_map(|k| item.get(*k).and_then(numeric));
                if let Some(n) = remain {
                    available += n;
                    found = true;
                }
                let size = item.get("CapacitySize").and_then(numeric).unwrap_or(0.0);
                total += size;
                let status = item.get("Status").and_then(numeric).unwrap_or(0.0);
                if status == 0.0 {
                    monthly += size;
                }
                if let Some(unit) = item
                    .get("CapacityUnit")
                    .and_then(Value::as_str)
                    .filter(|s| !s.trim().is_empty())
                {
                    units.insert(unit.to_string());
                }
            }
            if found {
                return finish_balance(available, monthly, total, units);
            }
        }
    }
    let direct = [
        "/data/Response/Data/CycleCapacityRemain",
        "/data/Response/Data/CapacityRemain",
    ]
    .iter()
    .find_map(|path| value.pointer(path).and_then(numeric));
    if let Some(n) = direct {
        let mut units = std::collections::BTreeSet::new();
        if let Some(unit) = value
            .pointer("/data/Response/Data/CapacityUnit")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
        {
            units.insert(unit.to_string());
        }
        let size = value
            .pointer("/data/Response/Data/CapacitySize")
            .and_then(numeric);
        return finish_balance(n, size.unwrap_or(0.0), size.unwrap_or(0.0), units);
    }
    Ok(None)
}

fn finish_balance(
    amount: f64,
    monthly_quota: f64,
    total_quota: f64,
    units: std::collections::BTreeSet<String>,
) -> Result<Option<Balance>, String> {
    if units.len() > 1 {
        return Err(format!(
            "额度包含多种单位，无法相加: {}",
            units.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    Ok(Some(Balance {
        amount,
        unit: units.into_iter().next().unwrap_or_else(|| "credits".into()),
        monthly_quota: if monthly_quota > 0.0 {
            Some(monthly_quota)
        } else {
            None
        },
        total_quota: if total_quota > 0.0 {
            Some(total_quota)
        } else {
            None
        },
    }))
}

fn parse_checkin_response(status: u16, text: &str) -> Result<CheckinOutcome, ProviderError> {
    // Parse the body even on HTTP 4xx: "already checked in" often arrives as
    // 400/409 (or code 10001), and must not be reported as an error.
    let value: Value = serde_json::from_str(text).unwrap_or(Value::Null);
    let code = value.get("code").and_then(|c| {
        c.as_i64()
            .or_else(|| c.as_str().and_then(|s| s.parse().ok()))
    });
    let data = value.get("data").cloned().unwrap_or(Value::Null);
    let credit = data
        .get("credit")
        .or_else(|| data.get("credits"))
        .and_then(Value::as_f64);
    let streak = data.get("streak_days").and_then(Value::as_i64);
    let msg = ["message", "msg", "errorMessage"]
        .iter()
        .find_map(|k| value.get(*k).and_then(Value::as_str))
        .or_else(|| data.get("message").and_then(Value::as_str))
        .unwrap_or("")
        .to_string();
    let lowered = text.to_ascii_lowercase();
    let already_text = lowered.contains("already")
        || lowered.contains("checked in")
        || text.contains("已签到")
        || text.contains("重复签到")
        || text.contains("今日已")
        || text.contains("已完成签到")
        || text.contains("已领取");
    let already_flag = [
        "signed",
        "checked_in",
        "checkedIn",
        "isSigned",
        "signedToday",
    ]
    .iter()
    .find_map(|k| data.get(*k).and_then(Value::as_bool))
    .unwrap_or(false);

    if status == 401 || status == 403 {
        if already_text {
            return Ok(CheckinOutcome {
                status: "already".into(),
                message: if msg.is_empty() {
                    "今日已签到".into()
                } else {
                    msg
                },
                reward: None,
            });
        }
        return Err(classify(status, text));
    }
    if status >= 400 && !already_text && !already_flag {
        return Err(classify(status, text));
    }
    match code {
        Some(0) => Ok(CheckinOutcome {
            status: "success".into(),
            message: match (credit, streak) {
                (Some(c), Some(s)) => format!("签到成功 +{c} 积分，连续 {s} 天"),
                (Some(c), None) => format!("签到成功 +{c} 积分"),
                _ => "签到成功".into(),
            },
            reward: credit,
        }),
        Some(10001) => Ok(CheckinOutcome {
            status: "already".into(),
            message: if msg.is_empty() {
                "今日已签到".into()
            } else {
                msg
            },
            reward: None,
        }),
        _ if already_flag || already_text => Ok(CheckinOutcome {
            status: "already".into(),
            message: if msg.is_empty() {
                "今日已签到".into()
            } else {
                msg
            },
            reward: None,
        }),
        Some(other) => Err(ProviderError::BadRequest(format!(
            "签到失败 code={other}: {}",
            text.chars().take(300).collect::<String>()
        ))),
        None => Err(ProviderError::Unavailable(format!(
            "签到响应无法识别 (HTTP {status}): {}",
            text.chars().take(300).collect::<String>()
        ))),
    }
}

pub fn build_payload(request: &Value, model: &str) -> Value {
    const ALLOWED: &[&str] = &[
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "n",
        "response_format",
        "reasoning_effort",
        "parallel_tool_calls",
    ];
    let mut out = Map::new();
    out.insert("model".into(), Value::String(model.into()));
    out.insert("stream".into(), Value::Bool(true));
    for key in ALLOWED {
        if let Some(v) = request.get(*key) {
            out.insert((*key).into(), v.clone());
        }
    }
    if let Some(msgs) = request.get("messages").and_then(Value::as_array) {
        let mut messages: Vec<Value> = msgs.iter().map(normalize_message).collect();
        let first_is_system = messages
            .first()
            .and_then(|m| m.get("role"))
            .and_then(Value::as_str)
            == Some("system");
        if !first_is_system {
            messages.insert(
                0,
                json!({"role": "system", "content": "You are a helpful assistant."}),
            );
        }
        out.insert("messages".into(), Value::Array(messages));
    }
    if let Some(tools) = request.get("tools").and_then(Value::as_array) {
        let cleaned: Vec<Value> = tools.iter().filter_map(clean_tool).collect();
        if cleaned.is_empty() {
            out.remove("tool_choice");
        } else {
            out.insert("tools".into(), Value::Array(cleaned));
            if let Some(tc) = request.get("tool_choice") {
                out.insert("tool_choice".into(), tc.clone());
            }
        }
    }
    Value::Object(out)
}

fn normalize_message(msg: &Value) -> Value {
    let mut m = msg.clone();
    if let Some(content) = m.get("content").cloned() {
        if let Some(parts) = content.as_array() {
            let has_image = parts.iter().any(|p| {
                p.get("type").and_then(Value::as_str) == Some("image_url")
                    || p.get("image_url").is_some()
                    || p.get("input_image").is_some()
            });
            if !has_image {
                let text = parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str).or_else(|| p.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                m["content"] = Value::String(text);
            }
        }
    }
    m
}

fn clean_tool(tool: &Value) -> Option<Value> {
    if tool
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("function")
        != "function"
    {
        return None;
    }
    let mut tool = tool.clone();
    let function = tool.get_mut("function")?.as_object_mut()?;
    function.remove("strict");
    match function.get_mut("parameters") {
        Some(params @ Value::Object(_)) => {
            let valid = params
                .as_object()
                .map(|p| !p.is_empty() && p.contains_key("type"))
                .unwrap_or(false);
            if !valid {
                return None;
            }
            strip_schema(params);
        }
        Some(_) => return None,
        None => {}
    }
    Some(tool)
}

fn strip_schema(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("additionalProperties");
            map.remove("$schema");
            for v in map.values_mut() {
                strip_schema(v);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(strip_schema),
        _ => {}
    }
}

fn parse_chunk(value: &Value) -> ChatDelta {
    let mut delta = ChatDelta::default();
    if let Some(usage) = value.get("usage").filter(|u| !u.is_null()) {
        delta.usage = Some(usage.clone());
    }
    let Some(choice) = value.get("choices").and_then(|c| c.get(0)) else {
        return delta;
    };
    if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
        if !reason.is_empty() {
            delta.finish_reason = Some(reason.into());
        }
    }
    if let Some(d) = choice.get("delta") {
        delta.role = d.get("role").and_then(Value::as_str).map(str::to_string);
        delta.content = d.get("content").and_then(Value::as_str).map(str::to_string);
        delta.reasoning_content = d
            .get("reasoning_content")
            .and_then(Value::as_str)
            .map(str::to_string);
        if let Some(calls) = d.get("tool_calls").and_then(Value::as_array) {
            if !calls.is_empty() {
                delta.tool_calls = Some(calls.clone());
            }
        }
    }
    delta
}

/// Loose parser for `/v2/models` and `/v3/config` shapes.
/// Parse `/v3/config`: `data.agents[].models` lists the CLI-visible model ids;
/// `data.models[]` carries display names and capabilities. A throttled upstream
/// may return only `enterpriseId/productFeatures` (no models) — return empty so
/// the caller can report/fallback instead of inventing entries.
pub fn parse_config_models(value: &Value) -> Vec<RemoteModel> {
    let data = value.get("data").unwrap_or(value);
    let mut allowed: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Some(agents) = data.get("agents").and_then(Value::as_array) {
        for agent in agents {
            if let Some(models) = agent.get("models").and_then(Value::as_array) {
                for id in models {
                    if let Some(id) = id.as_str() {
                        if !id.is_empty() {
                            allowed.insert(id.to_string());
                        }
                    }
                }
            }
        }
    }

    // `default` is present in data.models and accepted by chat, but the CLI
    // agent list omits it. Keep it available instead of silently dropping it.
    if data
        .get("models")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .any(|m| m.get("id").and_then(Value::as_str) == Some("default"))
        })
        .unwrap_or(false)
    {
        allowed.insert("default".into());
    }

    let mut out: Vec<RemoteModel> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut push = |id: &str, meta: Option<&Value>| {
        if id.is_empty() || !seen.insert(id.to_string()) {
            return;
        }
        if !allowed.is_empty() && !allowed.contains(id) {
            return;
        }
        let pick = |keys: &[&str]| meta.and_then(|m| keys.iter().find_map(|k| m.get(*k)));
        let name = pick(&["display_name", "displayName", "name"])
            .and_then(Value::as_str)
            .unwrap_or(id);
        let context_length = pick(&["maxInputTokens", "max_input_tokens", "context_length"])
            .and_then(Value::as_u64)
            .map(|n| n.min(u32::MAX as u64) as u32);
        let vision = pick(&["supportsImages", "supports_images", "is_vl", "vision"])
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let reasoning = pick(&["onlyReasoning", "only_reasoning", "is_reasoning"])
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || match pick(&["reasoning"]) {
                Some(Value::Bool(v)) => *v,
                Some(Value::Object(_)) => true,
                _ => false,
            };
        out.push(RemoteModel {
            id: id.to_string(),
            upstream_id: id.to_string(),
            name: name.to_string(),
            context_length,
            reasoning,
            vision,
        });
    };

    if let Some(items) = data.get("models").and_then(Value::as_array) {
        for item in items {
            let id = ["id", "model", "key"]
                .iter()
                .find_map(|k| item.get(*k).and_then(Value::as_str))
                .unwrap_or("");
            push(id, Some(item));
        }
    } else if let Some(map) = data.get("models").and_then(Value::as_object) {
        for (id, item) in map {
            push(id, Some(item));
        }
    }

    // Minimal config can contain agents[].models without data.models metadata.
    for id in &allowed {
        push(id, None);
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

fn pick_meta<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    let map = value.as_object()?;
    keys.iter().find_map(|k| map.get(*k))
}

pub fn parse_models(value: &Value) -> Vec<RemoteModel> {
    const RESERVED: &[&str] = &[
        "data",
        "models",
        "chat",
        "app",
        "id",
        "name",
        "display_name",
        "displayName",
        "owned_by",
        "object",
        "error",
        "code",
        "message",
        "msg",
        "status",
        "version",
        "enable",
        "enabled",
        "is_vl",
        "is_reasoning",
        "reasoning",
        "vision",
        "params",
        "parameters",
        "temperature",
        "top_p",
        "max_tokens",
        "context_config",
        "thinking_config",
        "created",
        "type",
        "source",
        "key",
        "metadata",
        "config",
        "settings",
        "extra",
        "billing",
        "quota",
    ];
    fn is_model_key(key: &str) -> bool {
        !key.is_empty()
            && key.len() <= 64
            && !RESERVED.contains(&key)
            && key
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
    }
    fn walk(
        value: &Value,
        in_models: bool,
        out: &mut Vec<RemoteModel>,
        seen: &mut std::collections::HashSet<String>,
    ) {
        match value {
            Value::Array(items) => items.iter().for_each(|v| walk(v, in_models, out, seen)),
            Value::Object(map) => {
                if let Some(id) = map.get("id").and_then(Value::as_str) {
                    if !id.is_empty() && seen.insert(id.to_string()) {
                        let name = map
                            .get("display_name")
                            .or_else(|| map.get("displayName"))
                            .or_else(|| map.get("name"))
                            .and_then(Value::as_str)
                            .unwrap_or(id);
                        out.push(RemoteModel {
                            id: id.to_string(),
                            upstream_id: id.to_string(),
                            name: name.to_string(),
                            context_length: map
                                .get("context_length")
                                .and_then(Value::as_u64)
                                .map(|n| n as u32),
                            reasoning: map
                                .get("reasoning")
                                .and_then(Value::as_bool)
                                .unwrap_or(false),
                            vision: map.get("vision").and_then(Value::as_bool).unwrap_or(false),
                        });
                    }
                }
                for (key, child) in map {
                    let child_in_models =
                        in_models || matches!(key.as_str(), "models" | "chat" | "app" | "byok");
                    let child_has_id = child.as_object().and_then(|m| m.get("id")).is_some();
                    if in_models && is_model_key(key) && !child_has_id {
                        // map form: {"models": {"deepseek-v3": {}, "hy4-preview": {...}}}
                        if seen.insert(key.clone()) {
                            let name = pick_meta(child, &["display_name", "displayName", "name"])
                                .and_then(Value::as_str)
                                .unwrap_or(key);
                            out.push(RemoteModel {
                                id: key.to_string(),
                                upstream_id: key.to_string(),
                                name: name.to_string(),
                                context_length: pick_meta(child, &["context_length"])
                                    .and_then(Value::as_u64)
                                    .map(|n| n as u32),
                                reasoning: pick_meta(child, &["reasoning"])
                                    .and_then(Value::as_bool)
                                    .unwrap_or(false),
                                vision: pick_meta(child, &["vision"])
                                    .and_then(Value::as_bool)
                                    .unwrap_or(false),
                            });
                        }
                        continue;
                    }
                    if child.is_object() || child.is_array() {
                        walk(child, child_in_models, out, seen);
                    }
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    walk(value, false, &mut out, &mut seen);
    out
}

fn rand_hex(bytes: usize) -> String {
    use rand::RngCore;
    let mut buf = vec![0u8; bytes];
    rand::thread_rng().fill_bytes(&mut buf);
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

fn urlencode(value: &str) -> String {
    value
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn jwt_claims(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let padded = match payload.len() % 4 {
        2 => format!("{payload}=="),
        3 => format!("{payload}="),
        _ => payload.to_string(),
    };
    let raw = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload.trim_end_matches('='))
        .or_else(|_| base64::engine::general_purpose::STANDARD.decode(&padded))
        .ok()?;
    serde_json::from_slice(&raw).ok()
}

fn first_claim(claims: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|k| {
        claims
            .get(*k)
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .map(str::to_string)
    })
}

/// Extract a display name from an OAuth access token's JWT payload.
fn jwt_name(token: &str) -> Option<String> {
    let claims = jwt_claims(token)?;
    first_claim(&claims, &["email", "preferred_username", "sub"])
}

/// JWT subject, used by the CN gateway's `X-User-Id` header.
fn jwt_subject(token: &str) -> Option<String> {
    let claims = jwt_claims(token)?;
    first_claim(&claims, &["sub", "email", "preferred_username"])
}

/// Stable account identity for login dedupe. Prefer email/username so two
/// accounts never collapse onto a shared `sub`; empty claims are skipped.
fn jwt_remote_id(token: &str) -> Option<String> {
    let claims = jwt_claims(token)?;
    first_claim(&claims, &["email", "preferred_username", "sub"])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injects_system_and_forces_stream() {
        let req = json!({
            "model": "x",
            "stream": false,
            "stream_options": {"include_usage": true},
            "messages": [{"role": "user", "content": [{"type":"text","text":"hi"}]}],
            "unknown": 1
        });
        let payload = build_payload(&req, "hy4-preview");
        assert_eq!(payload["stream"], true);
        assert!(payload.get("stream_options").is_none());
        assert!(payload.get("unknown").is_none());
        assert_eq!(payload["messages"][0]["role"], "system");
        assert_eq!(payload["messages"][1]["content"], "hi");
    }

    #[test]
    fn parses_resource_balance_with_unit() {
        let v = json!({"code":0,"data":{"Response":{"Data":{"Accounts":[
            {"CapacityUnit":"credits","CapacitySize":1500,"CapacityRemain":0,"CycleCapacityRemain":0,"Status":3},
            {"CapacityUnit":"credits","CapacitySize":500,"CapacityRemain":500,"CycleCapacityRemain":0,"Status":0}
        ]}}}});
        let b = parse_resource_balance(&v).unwrap().unwrap();
        assert_eq!(b.amount, 0.0);
        assert_eq!(b.monthly_quota, Some(500.0));
        assert_eq!(b.total_quota, Some(2000.0));
        assert_eq!(b.unit, "credits");
        let v =
            json!({"data":{"Response":{"Data":{"CapacityUnit":"tokens","CapacityRemain":"12.5"}}}});
        assert_eq!(parse_resource_balance(&v).unwrap().unwrap().unit, "tokens");
        assert!(parse_resource_balance(&json!({"code":0}))
            .unwrap()
            .is_none());
        let mixed = json!({"data":{"Response":{"Data":{"Accounts":[
            {"CapacityUnit":"credits","CapacityRemain":1},
            {"CapacityUnit":"tokens","CapacityRemain":2}
        ]}}}});
        assert!(parse_resource_balance(&mixed).is_err());
    }

    #[test]
    fn checkin_already_is_not_an_error() {
        let out = parse_checkin_response(200, r#"{"code":10001,"msg":"今日已签到"}"#).unwrap();
        assert_eq!(out.status, "already");
        let out =
            parse_checkin_response(400, r#"{"code":11109,"msg":"already checked in"}"#).unwrap();
        assert_eq!(out.status, "already");
        let out = parse_checkin_response(409, "already signed today").unwrap();
        assert_eq!(out.status, "already");
        let out =
            parse_checkin_response(200, r#"{"code":0,"data":{"credit":100,"streak_days":7}}"#)
                .unwrap();
        assert_eq!(out.status, "success");
        assert_eq!(out.reward, Some(100.0));
        assert!(parse_checkin_response(400, r#"{"code":123,"msg":"bad request"}"#).is_err());
    }

    #[test]
    fn parses_current_config_catalog() {
        let v = json!({
            "code": 0,
            "data": {
                "agents": [{"models": ["hy4std", "deepseek-v4.1-flash"]}],
                "models": [
                    {"id": "hy4std", "name": "Hy4", "maxInputTokens": 960000, "supportsImages": true, "onlyReasoning": true},
                    {"id": "deepseek-v4.1-flash", "name": "DeepSeek-V4.1-Flash", "maxInputTokens": 1000000, "supportsImages": true, "onlyReasoning": true},
                    {"id": "default", "name": "Default", "maxInputTokens": 200000, "supportsToolCall": true},
                    {"id": "minimax-video-h3", "name": "Video", "maxInputTokens": 1000}
                ]
            }
        });
        let models = parse_config_models(&v);
        assert_eq!(models.len(), 3);
        assert!(models.iter().any(|m| m.id == "default"));
        let ds = models
            .iter()
            .find(|m| m.id == "deepseek-v4.1-flash")
            .unwrap();
        assert_eq!(ds.name, "DeepSeek-V4.1-Flash");
        assert_eq!(ds.context_length, Some(1_000_000));
        assert!(ds.reasoning && ds.vision);
        assert!(models.iter().all(|m| m.id != "minimax-video-h3"));
        assert!(parse_config_models(&json!({"data":{"enterpriseId":"x"}})).is_empty());
    }

    #[test]
    fn fallback_catalog_is_current_and_unique() {
        let models = WorkBuddyProvider::new().fallback_models();
        assert_eq!(models.len(), 17);
        let mut ids = std::collections::HashSet::new();
        for m in &models {
            assert!(ids.insert(m.id.clone()), "duplicate {}", m.id);
            if m.id != "default" {
                assert!(m.vision && m.reasoning);
            }
        }
        assert!(ids.contains("hy4std"));
        assert!(ids.contains("space-bunny"));
        assert!(ids.contains("mimo-v2.6-pro"));
        assert!(ids.contains("default"));
        assert!(!ids.contains("hy3"));
    }

    #[test]
    fn parses_map_form_models() {
        let v = json!({
            "data": {"models": {
                "deepseek-v3": {},
                "auto": {},
                "hy4-preview": {"display_name": "Hy4", "reasoning": true},
                "metadata": {"version": 3}
            }}
        });
        let models = parse_models(&v);
        let ids: Vec<_> = models.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"deepseek-v3"));
        assert!(ids.contains(&"auto"));
        assert!(ids.contains(&"hy4-preview"));
        assert!(!ids.contains(&"metadata"));
        let hy4 = models.iter().find(|m| m.id == "hy4-preview").unwrap();
        assert!(hy4.reasoning);
    }

    #[test]
    fn decodes_jwt_display_name() {
        use base64::Engine;
        let claims = json!({"email": "me@example.com"}).to_string();
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(claims);
        assert_eq!(
            jwt_name(&format!("h.{payload}.s")).as_deref(),
            Some("me@example.com")
        );
    }

    #[test]
    fn parses_openai_model_list() {
        let v = json!({"data": {"models": [{"id": "hy4", "display_name": "Hy4"}]}});
        let models = parse_models(&v);
        assert!(models.iter().any(|m| m.id == "hy4" && m.name == "Hy4"));
    }
}
