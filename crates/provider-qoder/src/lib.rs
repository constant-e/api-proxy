//! Qoder (CN / international) provider.
//!
//! Qoder's gateway is not OpenAI-compatible on the wire: requests use the COSY
//! signing scheme, a custom base64 body encoding, and an RSA/AES-sealed
//! session. This module implements the minimal subset needed to expose chat,
//! model listing and the daily credit check-in as OpenAI endpoints.

mod crypto;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use base64::Engine;
use futures::StreamExt;
use gateway_core::provider::{LoginAccount, LoginPoll, LoginStart, Provider, ProviderError};
use gateway_core::{AccountView, Balance, ChatDelta, ChatStream, CheckinOutcome, RemoteModel};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crypto::Session;

const DEFAULT_CN_CHAT: &str = "https://gateway.qoder.com.cn";
const DEFAULT_GLOBAL_CHAT: &str = "https://api1.qoder.sh";
const CHAT_PATH: &str = "/api/v2/service/pro/sse/agent_chat_generation";
const MODEL_LIST_PATH: &str = "/api/v2/model/list";
const JOB_TOKEN_PATH: &str = "/api/v3/user/jobToken";
const USER_STATUS_PATH: &str = "/api/v3/user/status";
const OPENAPI_USER_AGENT: &str = "qoder/1.1.32";

const FALLBACK_MODELS: &[(&str, &str)] = &[
    ("Qwen3.8-Max", "qmodel_38max"),
    ("Qwen3.8-Flash", "qfmodel"),
    ("Qwen3.7-Max", "qmodel_latest"),
    ("Qwen3.7-Plus", "qmodel"),
    ("Qwen3.7-Flash", "q37fmodel"),
    ("DeepSeek-V4-Pro", "dmodel"),
    ("DeepSeek-Flash", "dfmodel"),
    ("GLM-5.3", "gmodel"),
    ("GLM-5.3-Flash", "gfmodel"),
    ("GLM-5.2", "gm51model"),
    ("Kimi-K3", "kmodel_latest"),
    ("Kimi-K2.8-Preview", "kmodel"),
    ("MiniMax-M2.7", "mmodel"),
];

#[derive(Clone)]
struct PendingLogin {
    verifier: String,
    nonce: String,
    region: String,
    openapi: String,
    expires_at: i64,
}

pub struct QoderProvider {
    /// account fingerprint -> sealed session
    sessions: Arc<tokio::sync::Mutex<HashMap<String, Session>>>,
    logins: Mutex<HashMap<String, PendingLogin>>,
}

impl QoderProvider {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            logins: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for QoderProvider {
    fn default() -> Self {
        Self::new()
    }
}

fn is_global(account: &AccountView<'_>) -> bool {
    matches!(
        account.region.map(|r| r.to_ascii_lowercase()),
        Some(r) if r == "global" || r == "intl" || r == "international"
    )
}

fn chat_base(account: &AccountView<'_>) -> String {
    account
        .base_url
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim_end_matches('/').to_string())
        .unwrap_or_else(|| {
            if is_global(account) {
                DEFAULT_GLOBAL_CHAT
            } else {
                DEFAULT_CN_CHAT
            }
            .into()
        })
}

fn openapi_base(account: &AccountView<'_>) -> &'static str {
    if is_global(account) {
        "https://openapi.qoder.sh"
    } else {
        "https://openapi.qoder.com.cn"
    }
}

/// OpenAPI base, honoring a configured per-provider base (self-hosted/test).
fn openapi_for(account: &AccountView<'_>) -> String {
    account
        .base_url
        .filter(|b| !b.trim().is_empty())
        .map(|b| b.trim_end_matches('/').to_string())
        .unwrap_or_else(|| openapi_base(account).to_string())
}

struct LoginRegion {
    openapi: String,
    website: &'static str,
    client_id: &'static str,
    redirect_uri: &'static str,
    dashed_nonce: bool,
    send_redirect: bool,
    tag: &'static str,
}

fn login_region(region: Option<&str>, base: Option<&str>) -> LoginRegion {
    let global = matches!(
        region.map(|r| r.to_ascii_lowercase()),
        Some(r) if r == "global" || r == "intl" || r == "international"
    );
    let openapi = base
        .filter(|b| !b.trim().is_empty())
        .map(|b| b.trim_end_matches('/').to_string())
        .unwrap_or_else(|| {
            if global {
                "https://openapi.qoder.sh".into()
            } else {
                "https://openapi.qoder.com.cn".into()
            }
        });
    if global {
        LoginRegion {
            openapi,
            website: "https://qoder.com",
            client_id: "e883ade2-e6e3-4d6d-adf7-f92ceff5fdcb",
            redirect_uri: "qoder://aicoding.aicoding-agent/login-success",
            dashed_nonce: false,
            send_redirect: false,
            tag: "global",
        }
    } else {
        LoginRegion {
            openapi,
            website: "https://qoder.com.cn",
            client_id: "1c5e33e1-364d-4ce6-b02c-acaa81274a5c",
            redirect_uri: "qoder-work-cn://",
            dashed_nonce: true,
            send_redirect: true,
            tag: "cn",
        }
    }
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

fn make_pkce() -> (String, String) {
    use rand::RngCore;
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
    let mut raw = [0u8; 64];
    rand::thread_rng().fill_bytes(&mut raw);
    let verifier: String = raw
        .iter()
        .map(|b| ALPHABET[*b as usize % ALPHABET.len()] as char)
        .collect();
    let digest = Sha256::digest(verifier.as_bytes());
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);
    (verifier, challenge)
}

fn device_expiry_ms(value: &Value) -> i64 {
    if let Some(ms) = value.get("expires_in").and_then(Value::as_i64) {
        if ms > 0 {
            return gateway_core::now_ms() + ms;
        }
    }
    if let Some(at) = value.get("expires_at") {
        if let Some(ms) = at.as_i64() {
            if ms > 0 {
                return ms;
            }
        }
        if let Some(text) = at.as_str() {
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(text) {
                return dt.timestamp_millis();
            }
        }
    }
    gateway_core::now_ms() + 30 * 24 * 3600 * 1000
}

fn login_user_agent() -> &'static str {
    "QoderWork/1.1.75"
}

fn session_key(account: &AccountView<'_>) -> String {
    format!(
        "{}:{}:{}",
        account.id,
        crypto::md5_hex(account.credential),
        chat_base(account)
    )
}

fn classify_stream_status(code: &str, detail: &str) -> ProviderError {
    match code {
        "401" => ProviderError::Auth(detail.chars().take(400).collect()),
        "429" => ProviderError::RateLimited {
            message: detail.chars().take(400).collect(),
            retry_after_ms: 5_000,
        },
        "403" => {
            // code 112 / pricingUrl is a model/plan gate, not an invalid
            // credential. Treating it as account quota would put every account
            // into a 12h cooldown and make other models unusable.
            let plan_gate = detail.contains("pricingUrl") || detail.contains("\"112\"");
            if plan_gate {
                ProviderError::BadRequest(format!(
                    "该模型当前不可用或需要订阅: {}",
                    detail.chars().take(300).collect::<String>()
                ))
            } else if detail.contains("10605") {
                ProviderError::RateLimited {
                    message: detail.chars().take(400).collect(),
                    retry_after_ms: 2_000,
                }
            } else {
                ProviderError::Auth(detail.chars().take(400).collect())
            }
        }
        _ if code.starts_with('5') => {
            ProviderError::Unavailable(detail.chars().take(400).collect())
        }
        _ => ProviderError::Unavailable(format!(
            "Qoder 上游状态 {code}: {}",
            detail.chars().take(400).collect::<String>()
        )),
    }
}

fn classify(status: u16, body: &str) -> ProviderError {
    let text: String = body.chars().take(400).collect();
    // Qoder admission-control rejections use code 10605 over 401/403.
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        let code = v
            .get("code")
            .map(|c| {
                c.as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| c.to_string())
            })
            .unwrap_or_default();
        if code == "10605" {
            return ProviderError::RateLimited {
                message: format!("网关繁忙: {text}"),
                retry_after_ms: 2_000,
            };
        }
    }
    let msg = format!("HTTP {status}: {text}");
    match status {
        401 | 403 => ProviderError::Auth(msg),
        429 => ProviderError::RateLimited {
            message: msg,
            retry_after_ms: 5_000,
        },
        _ if (400..500).contains(&status) => ProviderError::BadRequest(msg),
        _ => ProviderError::Unavailable(msg),
    }
}

/// Extract a field from the response, accepting both flat and `data`-wrapped
/// shapes seen across Qoder regions.
fn field<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.get(key).or_else(|| value.get("data")?.get(key))
}

fn str_field(value: &Value, key: &str) -> String {
    field(value, key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

impl QoderProvider {
    async fn signed_post(
        &self,
        client: &reqwest::Client,
        base: &str,
        path: &str,
        inner: &Value,
        account: &AccountView<'_>,
    ) -> Result<Value, ProviderError> {
        let (machine_id, machine_token, machine_type) = crypto::machine_identity(account.id);
        let body = crypto::sealed_outer(inner);
        let url = format!("{base}/algo{path}?Encode=1");
        let mut req = client
            .post(url)
            .timeout(std::time::Duration::from_secs(20))
            .body(body);
        for (k, v) in crypto::signed_post_headers(&machine_id, &machine_token, &machine_type) {
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
            .map_err(|e| ProviderError::Unavailable(format!("响应解析失败: {e}")))?;
        Ok(crypto::unseal(&value))
    }

    async fn bootstrap(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
    ) -> Result<Session, ProviderError> {
        if account.credential.starts_with("dt-") || account.credential.starts_with("jt-") {
            return self.device_session(client, account).await;
        }
        let base = chat_base(account);
        let inner = json!({
            "personalToken": account.credential,
            "securityOauthToken": "",
            "refreshToken": "",
            "needRefresh": false,
            "authInfo": {}
        });
        let jt = self
            .signed_post(client, &base, JOB_TOKEN_PATH, &inner, account)
            .await?;
        let uid = str_field(&jt, "id");
        if uid.is_empty() {
            return Err(ProviderError::Auth(format!(
                "PAT 换取会话失败: {}",
                jt.to_string().chars().take(300).collect::<String>()
            )));
        }
        let name = {
            let n = str_field(&jt, "name");
            if n.is_empty() {
                account.name.to_string()
            } else {
                n
            }
        };
        // The job token carries no authoritative tier; /user/status provides it.
        let user_type = if uid.is_empty() {
            String::new()
        } else {
            let status_inner = json!({
                "userId": uid,
                "personalToken": "",
                "securityOauthToken": "",
                "refreshToken": "",
                "needRefresh": false,
                "authInfo": {}
            });
            match self
                .signed_post(client, &base, USER_STATUS_PATH, &status_inner, account)
                .await
            {
                Ok(v) => str_field(&v, "userType"),
                Err(_) => String::new(),
            }
        };
        let identity = crypto::Identity {
            name,
            uid,
            user_type: if user_type.is_empty() {
                "personal_standard".into()
            } else {
                user_type
            },
            security_oauth_token: str_field(&jt, "securityOauthToken"),
            refresh_token: str_field(&jt, "refreshToken"),
        };
        let expire_ms = field(&jt, "expireTime")
            .and_then(|v| {
                v.as_i64()
                    .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
            })
            .unwrap_or(0);
        let (machine_id, machine_token, machine_type) = crypto::machine_identity(account.id);
        Session::new(identity, machine_id, machine_token, machine_type, expire_ms)
            .map_err(ProviderError::Auth)
    }

    async fn session(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
    ) -> Result<Session, ProviderError> {
        let key = session_key(account);
        {
            let map = self.sessions.lock().await;
            if let Some(s) = map.get(&key) {
                if !s.needs_refresh() {
                    return Ok(s.clone());
                }
            }
        }
        let session = self.bootstrap(client, account).await?;
        self.sessions.lock().await.insert(key, session.clone());
        Ok(session)
    }

    async fn invalidate(&self, account: &AccountView<'_>) {
        self.sessions.lock().await.remove(&session_key(account));
    }

    async fn openapi_token(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
    ) -> Result<String, ProviderError> {
        if account.credential.starts_with("dt-") || account.credential.starts_with("jt-") {
            return Ok(account.credential.to_string());
        }
        let base = openapi_for(account);
        let resp = client
            .post(format!("{base}/api/v1/jobToken/exchange"))
            .timeout(std::time::Duration::from_secs(15))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("User-Agent", OPENAPI_USER_AGENT)
            .json(&json!({ "personal_token": account.credential }))
            .send()
            .await
            .map_err(|e| ProviderError::Unavailable(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        if status >= 400 {
            return Err(classify(status, &text));
        }
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| ProviderError::Unavailable(format!("OpenAPI token 解析失败: {e}")))?;
        let token = ["token", "access_token", "device_token"]
            .iter()
            .find_map(|k| value.get(*k).and_then(Value::as_str))
            .unwrap_or("")
            .to_string();
        if token.is_empty() {
            return Err(ProviderError::Auth("未能获取 OpenAPI token".into()));
        }
        Ok(token)
    }

    /// OAuth device tokens (`dt-`) authenticate directly: resolve identity via
    /// OpenAPI userinfo and seal it into a COSY session, no PAT exchange.
    async fn device_session(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
    ) -> Result<Session, ProviderError> {
        let openapi = openapi_for(account);
        let resp = client
            .get(format!("{openapi}/api/v1/userinfo"))
            .timeout(std::time::Duration::from_secs(20))
            .bearer_auth(account.credential)
            .header("Accept", "application/json")
            .header("User-Agent", login_user_agent())
            .send()
            .await
            .map_err(|e| ProviderError::Unavailable(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        if status >= 400 {
            return Err(classify(status, &text));
        }
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| ProviderError::Unavailable(format!("userinfo 解析失败: {e}")))?;
        let uid = str_field(&value, "id");
        if uid.is_empty() {
            return Err(ProviderError::Auth("OAuth 凭证未返回用户 ID".into()));
        }
        let identity = crypto::Identity {
            name: {
                let n = str_field(&value, "name");
                if n.is_empty() {
                    account.name.to_string()
                } else {
                    n
                }
            },
            uid,
            user_type: {
                let t = str_field(&value, "user_type");
                if t.is_empty() {
                    "personal_standard".into()
                } else {
                    t
                }
            },
            security_oauth_token: account.credential.to_string(),
            refresh_token: account.refresh_token.unwrap_or("").to_string(),
        };
        let expire_ms = account
            .expires_at
            .unwrap_or_else(|| gateway_core::now_ms() + 30 * 24 * 3600 * 1000);
        let (machine_id, machine_token, machine_type) = crypto::machine_identity(account.id);
        Session::new(identity, machine_id, machine_token, machine_type, expire_ms)
            .map_err(ProviderError::Auth)
    }

    /// Signed GET used by model listing.
    async fn signed_get(
        &self,
        client: &reqwest::Client,
        session: &Session,
        url: &str,
        path_without_algo: &str,
    ) -> Result<Value, ProviderError> {
        // The gateway signs and verifies the request body even for GET:
        // model/list expects qoder_encode("{}") as the body. A bodyless GET is
        // rejected or returns a degraded catalog (observed: one model).
        let body = crypto::encode(b"{}");
        let (date, bearer) = crypto::signed_body(session, &body, path_without_algo);
        let mut req = client
            .get(url)
            .timeout(std::time::Duration::from_secs(20))
            .body(body);
        for (k, v) in crypto::bearer_headers(session, &date, &bearer, "application/json") {
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
        serde_json::from_str(&text)
            .map_err(|e| ProviderError::Unavailable(format!("模型列表解析失败: {e}")))
    }
}

#[async_trait]
impl Provider for QoderProvider {
    fn id(&self) -> &'static str {
        "qoder"
    }

    fn name(&self) -> &'static str {
        "Qoder"
    }

    fn default_base_url(&self) -> &'static str {
        DEFAULT_CN_CHAT
    }

    fn supports_login(&self) -> bool {
        true
    }

    async fn start_login(
        &self,
        _client: &reqwest::Client,
        region: Option<&str>,
        base: Option<&str>,
    ) -> Result<LoginStart, ProviderError> {
        let cfg = login_region(region, base);
        let (verifier, challenge) = make_pkce();
        let nonce = if cfg.dashed_nonce {
            uuid::Uuid::new_v4().to_string()
        } else {
            uuid::Uuid::new_v4().simple().to_string()
        };
        let machine_id = crypto::machine_identity("qoder-login").0;
        let mut query = format!(
            "challenge={}&challenge_method=S256&nonce={}&client_id={}&machine_id={}",
            urlencode(&challenge),
            urlencode(&nonce),
            urlencode(cfg.client_id),
            urlencode(&machine_id)
        );
        if cfg.send_redirect {
            query.push_str(&format!("&redirect_uri={}", urlencode(cfg.redirect_uri)));
        }
        let url = format!("{}/device/selectAccounts?{query}", cfg.website);
        let flow_id = format!("qd-{}", uuid::Uuid::new_v4().simple());
        let now = gateway_core::now_ms();
        self.logins.lock().unwrap().insert(
            flow_id.clone(),
            PendingLogin {
                verifier,
                nonce,
                region: cfg.tag.to_string(),
                openapi: cfg.openapi,
                expires_at: now + 10 * 60 * 1000,
            },
        );
        Ok(LoginStart {
            flow_id,
            url,
            message: "在浏览器中完成 Qoder 授权".into(),
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
            "{}/api/v1/deviceToken/poll?nonce={}&verifier={}&challenge_method=S256",
            pending.openapi,
            urlencode(&pending.nonce),
            urlencode(&pending.verifier)
        );
        let resp = client
            .get(url)
            .timeout(std::time::Duration::from_secs(20))
            .header("Accept", "application/json")
            .header("User-Agent", login_user_agent())
            .send()
            .await
            .map_err(|e| ProviderError::Unavailable(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        if status == 202 || status == 404 {
            return Ok(LoginPoll::Pending {
                message: "等待浏览器完成 Qoder 授权".into(),
            });
        }
        if status >= 400 {
            return Ok(LoginPoll::Failed {
                message: format!(
                    "授权轮询失败 HTTP {status}: {}",
                    text.chars().take(300).collect::<String>()
                ),
            });
        }
        let value: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => {
                return Ok(LoginPoll::Pending {
                    message: "等待授权令牌".into(),
                })
            }
        };
        let credential = ["token", "device_token"]
            .iter()
            .find_map(|k| value.get(*k).and_then(Value::as_str))
            .unwrap_or("")
            .to_string();
        if credential.is_empty() {
            return Ok(LoginPoll::Pending {
                message: "等待授权令牌".into(),
            });
        }
        let mut uid = value
            .get("user_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let mut name = String::new();
        if let Ok(resp) = client
            .get(format!("{}/api/v1/userinfo", pending.openapi))
            .timeout(std::time::Duration::from_secs(15))
            .bearer_auth(&credential)
            .header("Accept", "application/json")
            .header("User-Agent", login_user_agent())
            .send()
            .await
        {
            if resp.status().is_success() {
                if let Ok(ui) = resp.json::<Value>().await {
                    uid = str_field(&ui, "id");
                    name = str_field(&ui, "name");
                }
            }
        }
        self.logins.lock().unwrap().remove(flow_id);
        let remote_id = if uid.is_empty() {
            None
        } else {
            Some(uid.clone())
        };
        Ok(LoginPoll::Success(LoginAccount {
            name: if name.is_empty() {
                if uid.is_empty() {
                    "Qoder 账号".into()
                } else {
                    format!("Qoder {uid}")
                }
            } else {
                name
            },
            remote_id,
            credential,
            refresh_token: value
                .get("refresh_token")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string),
            expires_at: Some(device_expiry_ms(&value)),
            region: Some(pending.region),
            base_url: None,
        }))
    }

    async fn cancel_login(&self, flow_id: &str) {
        self.logins.lock().unwrap().remove(flow_id);
    }

    fn fallback_models(&self) -> Vec<RemoteModel> {
        FALLBACK_MODELS
            .iter()
            .map(|(display, key)| RemoteModel {
                id: display.to_string(),
                upstream_id: key.to_string(),
                name: display.to_string(),
                context_length: None,
                reasoning: true,
                vision: true,
            })
            .collect()
    }

    async fn models(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
    ) -> Result<Vec<RemoteModel>, ProviderError> {
        let session = self.session(client, account).await?;
        let base = chat_base(account);
        let url = format!("{base}/algo{MODEL_LIST_PATH}?Encode=1");
        let value = self
            .signed_get(client, &session, &url, MODEL_LIST_PATH)
            .await?;
        let mut models = parse_catalog(&crypto::unseal(&value));
        if models.is_empty() {
            return Err(ProviderError::Unavailable("模型列表为空".into()));
        }
        // A single-entry catalog usually means the gateway returned a
        // degraded/region-filtered response; top it up with the known static
        // catalog so clients are not limited to one model.
        if models.len() <= 1 {
            for fallback in self.fallback_models() {
                if !models
                    .iter()
                    .any(|m| m.id.eq_ignore_ascii_case(&fallback.id))
                {
                    models.push(fallback);
                }
            }
        }
        Ok(models)
    }

    async fn chat(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
        request: &Value,
        upstream_model: &str,
    ) -> Result<ChatStream, ProviderError> {
        let base = chat_base(account);
        let url = format!(
            "{base}/algo{CHAT_PATH}?FetchKeys=llm_model_result&AgentId=agent_common&Encode=1"
        );
        let mut attempt = 0;
        let resp = loop {
            let session = self.session(client, account).await?;
            let body = build_chat_body(request, upstream_model, &session.identity.user_type);
            let encoded = crypto::encode(body.to_string().as_bytes());
            let (date, bearer) = crypto::signed_body(&session, &encoded, CHAT_PATH);
            let mut req = client.post(&url).body(encoded);
            for (k, v) in crypto::bearer_headers(&session, &date, &bearer, "text/event-stream") {
                req = req.header(k, v);
            }
            let resp = req
                .send()
                .await
                .map_err(|e| ProviderError::Unavailable(e.to_string()))?;
            let status = resp.status().as_u16();
            if status == 401 || status == 403 {
                // Admission-control rejections (code 10605) also ride 401/403;
                // refreshing the session cannot help and just adds latency.
                let text = resp.text().await.unwrap_or_default();
                if matches!(classify(status, &text), ProviderError::RateLimited { .. }) {
                    return Err(classify(status, &text));
                }
                self.invalidate(account).await;
                if attempt == 0 {
                    attempt += 1;
                    continue;
                }
                return Err(classify(status, &text));
            }
            if status >= 400 {
                let retry_after = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<i64>().ok())
                    .map(|s| s * 1000)
                    .unwrap_or(0);
                let text = resp.text().await.unwrap_or_default();
                let mut err = classify(status, &text);
                if let ProviderError::RateLimited { retry_after_ms, .. } = &mut err {
                    if retry_after > 0 {
                        *retry_after_ms = retry_after;
                    }
                }
                return Err(err);
            }
            break resp;
        };
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
                    let Ok(wrapper) = serde_json::from_str::<Value>(data) else {
                        continue;
                    };
                    let status_code = wrapper.get("statusCodeValue").map(|v| {
                        v.as_i64()
                            .map(|n| n.to_string())
                            .unwrap_or_else(|| v.to_string())
                    });
                    let body_raw = wrapper.get("body").and_then(Value::as_str);
                    tracing::debug!(
                        target: "qoder.sse",
                        status = ?status_code,
                        body_len = body_raw.map(str::len),
                        keys = ?wrapper.as_object().map(|m| m.keys().cloned().collect::<Vec<_>>()),
                        "qoder envelope"
                    );
                    if let Some(code) = status_code.as_deref() {
                        if code != "200" {
                            let detail = body_raw.unwrap_or(data);
                            return Some((
                                Err(classify_stream_status(code, detail)),
                                (bytes, buf, done),
                            ));
                        }
                    }
                    let inner = match body_raw {
                        Some("[DONE]") => return None,
                        Some(s) => match serde_json::from_str::<Value>(s) {
                            Ok(v) => v,
                            Err(_) => continue,
                        },
                        None => wrapper,
                    };
                    if let Some(err) = inner.get("error") {
                        return Some((
                            Err(ProviderError::Unavailable(err.to_string())),
                            (bytes, buf, done),
                        ));
                    }
                    let delta = parse_chunk(&inner);
                    if !delta.is_empty() {
                        return Some((Ok(delta), (bytes, buf, done)));
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
        let base = openapi_for(account);
        let token = self.openapi_token(client, account).await?;

        let origin = if is_global(account) {
            "https://qoder.sh"
        } else {
            "https://qoder.com.cn"
        };
        let campaigns_url = format!("{base}/sash/api/v1/me/campaigns");
        let http = |method: reqwest::Method, url: String| {
            client
                .request(method, url)
                .bearer_auth(&token)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .header("User-Agent", "Qoder")
                .header("Cosy-ClientType", "10")
                .header("Cosy-Version", "0.3.4")
                .header("Origin", origin)
                .header("Referer", format!("{base}/growth-page/activity-iframe"))
        };
        let resp = http(reqwest::Method::GET, campaigns_url.clone())
            .send()
            .await
            .map_err(|e| ProviderError::Unavailable(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        if status == 404 {
            return Ok(CheckinOutcome {
                status: "skipped".into(),
                message: "暂无可签到活动".into(),
                reward: None,
            });
        }
        if status >= 400 {
            return Err(classify(status, &text));
        }
        let list: Value = serde_json::from_str(&text)
            .map_err(|e| ProviderError::Unavailable(format!("签到活动解析失败: {e}")))?;
        let campaigns = collect_campaigns(&list);
        let benefits: Vec<&Value> = campaigns
            .iter()
            .filter(|c| {
                let credit = c
                    .pointer("/benefit/kind")
                    .and_then(Value::as_str)
                    .map(|k| k == "CREDITS")
                    .unwrap_or(true);
                credit
                    && c.get("actionType").and_then(Value::as_str) == Some("CLAIM_BENEFIT")
                    && !c
                        .get("campaignId")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .is_empty()
            })
            .collect();
        if benefits.is_empty() {
            return Ok(CheckinOutcome {
                status: "skipped".into(),
                message: "暂无可签到活动".into(),
                reward: None,
            });
        }
        let claimable: Vec<&&Value> = benefits
            .iter()
            .filter(|c| c.get("claimStatus").and_then(Value::as_str) == Some("CLAIMABLE"))
            .collect();
        if claimable.is_empty() {
            let claimed = benefits.iter().any(|c| {
                c.get("claimStatus")
                    .and_then(Value::as_str)
                    .map(is_claim_done)
                    .unwrap_or(false)
            });
            return Ok(CheckinOutcome {
                status: if claimed { "already" } else { "skipped" }.into(),
                message: if claimed {
                    "今日已签到"
                } else {
                    "暂无可签到活动"
                }
                .into(),
                reward: None,
            });
        }
        let mut reward = 0.0;
        let mut claimed = 0;
        for campaign in claimable {
            let id = campaign
                .get("campaignId")
                .and_then(Value::as_str)
                .unwrap_or("");
            let url = format!("{base}/sash/api/v1/me/campaigns/{id}/claim");
            let resp = http(reqwest::Method::POST, url)
                .send()
                .await
                .map_err(|e| ProviderError::Unavailable(e.to_string()))?;
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            let done_text = text_says_done(&text);
            if status >= 400 {
                if done_text {
                    return Ok(CheckinOutcome {
                        status: "already".into(),
                        message: "今日已签到".into(),
                        reward: None,
                    });
                }
                return Err(classify(status, &text));
            }
            if let Ok(v) = serde_json::from_str::<Value>(&text) {
                let claim_status = ["claimStatus", "status"]
                    .iter()
                    .find_map(|k| v.get(*k).and_then(Value::as_str))
                    .or_else(|| {
                        ["claimStatus", "status"]
                            .iter()
                            .find_map(|k| v.pointer(&format!("/data/{k}")).and_then(Value::as_str))
                    });
                let confirmed = claim_status.map(is_claim_done).unwrap_or(true)
                    || v.get("replayed").and_then(Value::as_bool).unwrap_or(false)
                    || done_text;
                if !confirmed {
                    return Err(ProviderError::Unavailable("签到结果未确认".into()));
                }
            }
            claimed += 1;
            if let Some(amount) = campaign.pointer("/benefit/amount").and_then(Value::as_f64) {
                reward += amount;
            }
        }
        Ok(CheckinOutcome {
            status: "success".into(),
            message: if reward > 0.0 {
                format!("签到成功 +{reward} 积分")
            } else {
                format!("签到成功（{claimed} 项）")
            },
            reward: if reward > 0.0 { Some(reward) } else { None },
        })
    }

    async fn balance(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
    ) -> Result<Option<Balance>, ProviderError> {
        let base = openapi_for(account);
        let token = self.openapi_token(client, account).await?;
        let origin = if is_global(account) {
            "https://qoder.sh"
        } else {
            "https://qoder.com.cn"
        };
        let resp = client
            .get(format!("{base}/api/v2/quota/usage"))
            .timeout(std::time::Duration::from_secs(15))
            .bearer_auth(token)
            .header("Accept", "application/json")
            .header("User-Agent", "Qoder")
            .header("Cosy-ClientType", "10")
            .header("Cosy-Version", "0.4.8")
            .header("Origin", origin)
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
        parse_qoder_balance(&value).map_err(ProviderError::BadRequest)
    }

    async fn validate(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
    ) -> Result<String, ProviderError> {
        let session = self.session(client, account).await?;
        let models = self
            .models(client, account)
            .await
            .map(|m| m.len())
            .unwrap_or(0);
        Ok(format!(
            "凭证有效（{}），发现 {models} 个模型",
            session.identity.name
        ))
    }
}

/// Parse `{chat: [...], app: [...]}` model catalogs.
pub fn parse_catalog(value: &Value) -> Vec<RemoteModel> {
    fn scene_items<'a>(value: &'a Value, scene: &str) -> Option<&'a Vec<Value>> {
        value
            .get(scene)
            .and_then(Value::as_array)
            .or_else(|| value.get("data")?.get(scene)?.as_array())
            .or_else(|| value.get("data")?.get("data")?.get(scene)?.as_array())
    }
    fn is_chat_model(key: &str) -> bool {
        key != "lite"
            && !key.starts_with("codewise-")
            && !key.starts_with("completion-")
            && !key.ends_with("-image-alpha")
            && !key.ends_with("-image-alpha-edit")
            && !key.ends_with("-taco-completion")
    }
    let mut out = Vec::new();
    let mut seen_keys = std::collections::HashSet::new();
    for scene in ["chat", "app"] {
        let Some(items) = scene_items(value, scene) else {
            continue;
        };
        for item in items {
            let key = item.get("key").and_then(Value::as_str).unwrap_or("");
            let name = item
                .get("display_name")
                .and_then(Value::as_str)
                .or_else(|| item.get("name").and_then(Value::as_str))
                .unwrap_or("");
            if key.is_empty()
                || key == "auto"
                || !is_chat_model(key)
                || !seen_keys.insert(key.to_string())
            {
                continue;
            }
            let display = if name.is_empty() { key } else { name };
            out.push(RemoteModel {
                id: display.to_string(),
                upstream_id: key.to_string(),
                name: display.to_string(),
                context_length: item
                    .get("max_output_tokens")
                    .and_then(Value::as_u64)
                    .map(|n| n as u32),
                reasoning: item
                    .get("is_reasoning")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                vision: item.get("is_vl").and_then(Value::as_bool).unwrap_or(false),
            });
        }
    }
    out
}

fn qnum(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|s| s.parse().ok()))
}

fn parse_qoder_balance(value: &Value) -> Result<Option<Balance>, String> {
    let root = value.get("data").unwrap_or(value);
    let mut available = 0.0;
    let mut monthly = 0.0;
    let mut addon_total = 0.0;
    let mut shared_total = 0.0;
    let mut monthly_found = false;
    let mut found = false;
    let mut units = std::collections::BTreeSet::new();

    if let Some(q) = root.get("userQuota") {
        if let Some(n) = q.get("remaining").and_then(qnum) {
            available += n;
            found = true;
        }
        if let Some(n) = q.get("total").and_then(qnum) {
            monthly += n;
            monthly_found = true;
        }
        if let Some(unit) = q
            .get("unit")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            units.insert(unit.to_string());
        }
    }
    if let Some(q) = root.get("addOnQuota") {
        if let Some(n) = q.get("remaining").and_then(qnum) {
            available += n;
            found = true;
        }
        if let Some(n) = q.get("total").and_then(qnum) {
            addon_total += n;
        }
        if let Some(unit) = q
            .get("unit")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            units.insert(unit.to_string());
        }
    }
    for key in [
        "orgResourcePackage",
        "sharedQuota",
        "teamQuota",
        "poolQuota",
    ] {
        if let Some(q) = root.get(key) {
            let cap = q.get("cap").or_else(|| q.get("total")).and_then(qnum);
            // cap=-1 is the upstream "unlimited" sentinel; its remaining value
            // is a placeholder and must not be treated as real balance.
            let unlimited = cap.map(|n| n < 0.0).unwrap_or(false);
            let usable = q.get("available").and_then(Value::as_bool) != Some(false) && !unlimited;
            if usable {
                if let Some(n) = q.get("remaining").and_then(qnum) {
                    available += n;
                    found = true;
                }
            }
            // The shared package cap belongs to the cumulative quota even when
            // the package is exhausted (available=false, remaining=0).
            if let Some(cap) = cap.filter(|n| *n > 0.0) {
                shared_total += cap;
            }
            if let Some(unit) = q
                .get("unit")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                units.insert(unit.to_string());
            }
        }
    }
    if !found {
        if let Some(n) = root.get("remaining").and_then(qnum) {
            available += n;
            found = true;
        }
    }
    if !found {
        return Ok(None);
    }
    if units.len() > 1 {
        return Err(format!(
            "额度包含多种单位，无法相加: {}",
            units.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    let total_quota = monthly + addon_total + shared_total;
    Ok(Some(Balance {
        amount: available,
        unit: units.into_iter().next().unwrap_or_else(|| "credits".into()),
        // Monthly plan quota excludes purchased add-on packages. Shared/org
        // package balance is part of the cumulative total, not the monthly cap.
        monthly_quota: if monthly_found { Some(monthly) } else { None },
        total_quota: if total_quota > 0.0 {
            Some(total_quota)
        } else {
            None
        },
    }))
}

fn is_claim_done(status: &str) -> bool {
    matches!(
        status,
        "CLAIMED" | "COMPLETED" | "ALREADY_CLAIMED" | "SUCCESS" | "DONE"
    )
}

fn text_says_done(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("already")
        || lower.contains("claimed")
        || lower.contains("checked in")
        || text.contains("已签到")
        || text.contains("已领取")
        || text.contains("重复签到")
}

fn collect_campaigns(value: &Value) -> Vec<Value> {
    if let Some(items) = value.get("campaigns").and_then(Value::as_array) {
        return items.clone();
    }
    if let Some(items) = value.pointer("/data/campaigns").and_then(Value::as_array) {
        return items.clone();
    }
    if let Some(items) = value.get("data").and_then(Value::as_array) {
        return items.clone();
    }
    Vec::new()
}

fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str).or_else(|| p.as_str()))
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

fn image_urls(content: &Value) -> Vec<String> {
    let Some(parts) = content.as_array() else {
        return Vec::new();
    };
    parts
        .iter()
        .filter_map(|p| {
            p.pointer("/image_url/url")
                .and_then(Value::as_str)
                .or_else(|| p.get("url").and_then(Value::as_str))
        })
        .map(str::to_string)
        .collect()
}

/// Convert OpenAI messages to the Qoder message envelope.
pub fn build_messages(request: &Value) -> Vec<Value> {
    let messages = request
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let tools_enabled = request
        .get("tools")
        .and_then(Value::as_array)
        .map(|t| !t.is_empty())
        .unwrap_or(false);
    let mut out = Vec::new();
    for msg in messages {
        let mut role = msg
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user")
            .to_string();
        if role == "developer" {
            role = "system".into();
        }
        let content = msg.get("content").cloned().unwrap_or(Value::Null);
        let text = content_text(&content);
        let images = if role == "user" {
            image_urls(&content)
        } else {
            Vec::new()
        };
        let tool_calls = msg.get("tool_calls").and_then(Value::as_array).cloned();
        if role == "tool" && !tools_enabled {
            role = "user".into();
        }
        if text.trim().is_empty() && images.is_empty() && tool_calls.is_none() {
            continue;
        }
        let mut base = Map::new();
        base.insert("role".into(), Value::String(role.clone()));
        base.insert(
            "response_meta".into(),
            json!({
                "id": "",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                    "prompt_tokens_details": {"cached_tokens": 0}
                }
            }),
        );
        base.insert(
            "reasoning_content_signature".into(),
            Value::String(String::new()),
        );
        if role == "user" {
            let mut contents = Vec::new();
            for url in images {
                contents.push(json!({"type": "image_url", "image_url": {"url": url}}));
            }
            contents.push(json!({"type": "text", "text": text}));
            base.insert("content".into(), Value::String(String::new()));
            base.insert("contents".into(), Value::Array(contents));
        } else {
            base.insert("content".into(), Value::String(text));
        }
        if let Some(calls) = tool_calls {
            base.insert("tool_calls".into(), Value::Array(calls));
        }
        if role == "tool" {
            if let Some(name) = msg.get("name").and_then(Value::as_str) {
                base.insert("name".into(), Value::String(name.into()));
            }
            if let Some(id) = msg.get("tool_call_id").and_then(Value::as_str) {
                base.insert("tool_call_id".into(), Value::String(id.into()));
            }
        }
        out.push(Value::Object(base));
    }
    out
}

/// Build the full COSY chat request body.
pub fn build_chat_body(request: &Value, upstream_model: &str, user_type: &str) -> Value {
    let request_id = uuid::Uuid::new_v4().to_string();
    let mut parameters = Map::new();
    let max_tokens = request
        .get("max_tokens")
        .or_else(|| request.get("max_completion_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(32000));
    parameters.insert("max_tokens".into(), max_tokens);
    for key in [
        "temperature",
        "top_p",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "n",
        "response_format",
        "tool_choice",
        "reasoning_effort",
        "max_thinking_tokens",
    ] {
        if let Some(v) = request.get(key) {
            parameters.insert(key.into(), v.clone());
        }
    }
    if parameters.get("reasoning_effort").and_then(Value::as_str) == Some("none")
        && !parameters.contains_key("max_thinking_tokens")
    {
        parameters.insert("max_thinking_tokens".into(), json!(0));
    }

    let mut body = json!({
        "request_id": request_id,
        "request_set_id": uuid::Uuid::new_v4().to_string(),
        "chat_record_id": request_id,
        "stream": true,
        "chat_task": "FREE_INPUT",
        "chat_context": {
            "extra": {
                "modelConfig": {"is_reasoning": true, "key": upstream_model},
                "originalContent": {"type": "text", "text": ""}
            },
            "text": {"type": "text", "text": ""}
        },
        "session_id": uuid::Uuid::new_v4().to_string(),
        "source": 1,
        "version": "3",
        "aliyun_user_type": user_type,
        "session_type": "qodercli",
        "agent_id": "agent_common",
        "task_id": "common",
        "model_config": {
            "key": upstream_model,
            "is_vl": false,
            "is_reasoning": true,
            "source": "system"
        },
        "messages": build_messages(request),
        "business": {"id": "", "name": "hi", "begin_at": Value::Null},
        "parameters": Value::Object(parameters),
    });
    if let Some(tools) = request.get("tools").and_then(Value::as_array) {
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools.clone());
            if let Some(p) = request.get("parallel_tool_calls") {
                body["parallel_tool_calls"] = p.clone();
            }
        }
    }
    body
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_body_with_messages_and_tools() {
        let req = json!({
            "model": "Qwen3.8-Max",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "name": "f", "content": "ok"}
            ],
            "tools": [{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
            "temperature": 0.2
        });
        let body = build_chat_body(&req, "qmodel_38max", "personal_standard");
        assert_eq!(body["stream"], true);
        assert_eq!(body["model_config"]["key"], "qmodel_38max");
        assert_eq!(body["messages"].as_array().unwrap().len(), 4);
        assert_eq!(body["messages"][1]["contents"][0]["text"], "hi");
        assert_eq!(body["parameters"]["temperature"], 0.2);
        assert_eq!(body["aliyun_user_type"], "personal_standard");
    }

    #[test]
    fn parses_quota_usage_credits() {
        // Personal: add-on packages must not inflate the monthly plan quota.
        let v = json!({"userQuota":{"remaining":0,"total":300,"unit":"credits"},"addOnQuota":{"remaining":0,"total":200,"unit":"credits"}});
        let b = parse_qoder_balance(&v).unwrap().unwrap();
        assert_eq!(b.amount, 0.0);
        assert_eq!(b.monthly_quota, Some(300.0));
        assert_eq!(b.total_quota, Some(500.0));
        assert_eq!(b.unit, "credits");

        // Enterprise: shared package cap belongs to the cumulative total.
        let v = json!({"userQuota":{"remaining":0,"total":300,"unit":"credits"},"addOnQuota":{"remaining":0,"total":200,"unit":"credits"},"orgResourcePackage":{"remaining":1000,"cap":2000,"available":true}});
        let b = parse_qoder_balance(&v).unwrap().unwrap();
        assert_eq!(b.amount, 1000.0);
        assert_eq!(b.monthly_quota, Some(300.0));
        assert_eq!(b.total_quota, Some(2500.0));

        let v = json!({"userQuota":{"remaining":0,"total":3000,"unit":"credits"},"orgResourcePackage":{"remaining":0,"cap":11000,"available":false}});
        let b = parse_qoder_balance(&v).unwrap().unwrap();
        assert_eq!(b.amount, 0.0);
        assert_eq!(b.monthly_quota, Some(3000.0));
        assert_eq!(b.total_quota, Some(14000.0));

        let v = json!({"userQuota":{"remaining":0,"total":300,"unit":"credits"},"orgResourcePackage":{"remaining":999,"cap":-1,"available":true}});
        let b = parse_qoder_balance(&v).unwrap().unwrap();
        assert_eq!(b.amount, 0.0);
        assert_eq!(b.total_quota, Some(300.0));

        assert!(parse_qoder_balance(&json!({"foo":1})).unwrap().is_none());
    }

    #[test]
    fn pkce_challenge_is_s256_base64url() {
        let (verifier, challenge) = make_pkce();
        assert_eq!(verifier.len(), 64);
        let expected = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(Sha256::digest(verifier.as_bytes()));
        assert_eq!(challenge, expected);
        assert!(!challenge.contains('='));
    }

    #[test]
    fn login_regions_use_expected_clients() {
        let cn = login_region(Some("cn"), None);
        assert_eq!(cn.tag, "cn");
        assert!(cn.send_redirect);
        assert!(cn.dashed_nonce);
        let global = login_region(Some("global"), None);
        assert_eq!(global.tag, "global");
        assert!(!global.send_redirect);
        let custom = login_region(Some("cn"), Some("http://127.0.0.1:9/x/"));
        assert_eq!(custom.openapi, "http://127.0.0.1:9/x");
    }

    #[test]
    fn parses_data_wrapped_catalog() {
        let v = json!({"data": {"chat": [
            {"key": "a", "display_name": "Model A", "enable": false},
            {"key": "lite", "display_name": "Lite"},
            {"key": "b", "name": "Model B"}
        ]}});
        let models = parse_catalog(&v);
        let names: Vec<_> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(names, vec!["Model A", "Model B"]);
    }

    #[test]
    fn parses_catalog_and_dedupes() {
        let v = json!({
            "chat": [
                {"key":"a","display_name":"Model A","enable":true},
                {"key":"b","display_name":"Model B","enable":false}
            ],
            "app": [
                {"key":"a","display_name":"Model A"},
                {"key":"c","display_name":"Model C"}
            ]
        });
        let models = parse_catalog(&v);
        let names: Vec<_> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(names, vec!["Model A", "Model B", "Model C"]);
    }
}
