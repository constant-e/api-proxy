//! Provider abstraction: every upstream module implements [`Provider`].
//!
//! Providers are intentionally small: list models, open a chat stream, and
//! (optionally) check in. Account rotation, failover, model merging and the
//! OpenAI wire format all live in the gateway/core layer.

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

use crate::store::Store;

pub type ChatStream = BoxStream<'static, Result<ChatDelta, ProviderError>>;

#[derive(Debug, Clone)]
pub struct AccountView<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub credential: &'a str,
    pub region: Option<&'a str>,
    /// Per-account base URL override (wins over the provider setting).
    pub base_url: Option<&'a str>,
    pub refresh_token: Option<&'a str>,
    /// Credential expiry in epoch milliseconds.
    pub expires_at: Option<i64>,
}

impl<'a> AccountView<'a> {
    pub fn from_account(
        account: &'a crate::store::Account,
        provider_base: Option<&'a str>,
    ) -> Self {
        Self {
            id: &account.id,
            name: &account.name,
            credential: &account.credential,
            region: account.region.as_deref(),
            base_url: account.base_url.as_deref().or(provider_base),
            refresh_token: account.refresh_token.as_deref(),
            expires_at: account.expires_at,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<serde_json::Value>,
}

impl ChatDelta {
    pub fn is_empty(&self) -> bool {
        self.role.is_none()
            && self.content.as_deref().unwrap_or("").is_empty()
            && self.reasoning_content.as_deref().unwrap_or("").is_empty()
            && self.tool_calls.is_none()
            && self.finish_reason.is_none()
            && self.usage.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteModel {
    /// Public model id exposed by the gateway (display name for Qoder keys).
    pub id: String,
    /// Name the upstream expects in its request body.
    pub upstream_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub vision: bool,
}

impl RemoteModel {
    pub fn simple(id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            name: id.clone(),
            upstream_id: id.clone(),
            id,
            context_length: None,
            reasoning: false,
            vision: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Balance {
    /// Currently usable balance (e.g. current-cycle remaining).
    pub amount: f64,
    /// Upstream unit, e.g. `credits` or `tokens`.
    pub unit: String,
    /// Current plan/monthly quota, when the upstream exposes it.
    #[serde(default)]
    pub monthly_quota: Option<f64>,
    /// Total granted quota including rewards, when exposed.
    #[serde(default)]
    pub total_quota: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckinOutcome {
    /// `success` | `already` | `skipped`
    pub status: String,
    pub message: String,
    #[serde(default)]
    pub reward: Option<f64>,
}

/// A started interactive login (browser authorization + polling).
#[derive(Debug, Clone)]
pub struct LoginStart {
    /// Provider-local flow id. The gateway prefixes it with the provider id.
    pub flow_id: String,
    pub url: String,
    pub message: String,
    pub expires_at: Option<i64>,
}

#[derive(Debug, Clone)]
pub enum LoginPoll {
    Pending { message: String },
    Success(LoginAccount),
    Expired { message: String },
    Failed { message: String },
}

/// Credentials produced by a successful interactive login.
#[derive(Debug, Clone, Default)]
pub struct LoginAccount {
    pub name: String,
    /// Stable upstream identity used for re-login dedupe.
    pub remote_id: Option<String>,
    pub credential: String,
    pub refresh_token: Option<String>,
    pub expires_at: Option<i64>,
    pub region: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("凭证无效: {0}")]
    Auth(String),
    #[error("请求被拒绝: {0}")]
    BadRequest(String),
    #[error("额度已用尽: {0}")]
    Quota(String),
    #[error("上游限流: {message}")]
    RateLimited {
        message: String,
        retry_after_ms: i64,
    },
    #[error("上游不可用: {0}")]
    Unavailable(String),
}

impl ProviderError {
    pub fn cooldown_ms(&self) -> Option<i64> {
        match self {
            ProviderError::Auth(_) => Some(24 * 3600 * 1000),
            ProviderError::Quota(_) => Some(12 * 3600 * 1000),
            ProviderError::RateLimited { retry_after_ms, .. } => Some(if *retry_after_ms > 0 {
                *retry_after_ms
            } else {
                60_000
            }),
            _ => None,
        }
    }

    pub fn http_status(&self) -> u16 {
        match self {
            ProviderError::Auth(_) => 401,
            ProviderError::BadRequest(_) => 400,
            ProviderError::Quota(_) => 429,
            ProviderError::RateLimited { .. } => 429,
            ProviderError::Unavailable(_) => 502,
        }
    }
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn id(&self) -> &'static str;
    fn name(&self) -> &'static str;
    fn default_base_url(&self) -> &'static str;
    /// Models used when the live listing fails (or no account is available).
    fn fallback_models(&self) -> Vec<RemoteModel>;

    /// Stable account identity derivable from a credential without network
    /// access (e.g. WorkBuddy's JWT email). Used to backfill older accounts so
    /// a re-login updates them instead of creating a duplicate.
    fn credential_identity(&self, _credential: &str) -> Option<String> {
        None
    }

    /// Whether this provider implements browser/device authorization.
    fn supports_login(&self) -> bool {
        false
    }

    /// Start an interactive login. `base` is the provider-level Base URL
    /// override (may point at a self-hosted or test upstream).
    async fn start_login(
        &self,
        _client: &reqwest::Client,
        _region: Option<&str>,
        _base: Option<&str>,
    ) -> Result<LoginStart, ProviderError> {
        Err(ProviderError::BadRequest("该提供商未实现浏览器登录".into()))
    }

    /// Poll a started login. Terminal results consume the flow.
    async fn poll_login(
        &self,
        _client: &reqwest::Client,
        _flow_id: &str,
    ) -> Result<LoginPoll, ProviderError> {
        Err(ProviderError::BadRequest("该提供商未实现浏览器登录".into()))
    }

    async fn cancel_login(&self, _flow_id: &str) {}

    async fn models(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
    ) -> Result<Vec<RemoteModel>, ProviderError>;

    /// Always returns a stream; the gateway aggregates for `stream=false`.
    async fn chat(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
        request: &serde_json::Value,
        upstream_model: &str,
    ) -> Result<ChatStream, ProviderError>;

    async fn checkin(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
    ) -> Result<CheckinOutcome, ProviderError>;

    /// Best-effort upstream credit balance. `Ok(None)` means the provider or
    /// account does not expose a balance endpoint.
    async fn balance(
        &self,
        _client: &reqwest::Client,
        _account: &AccountView<'_>,
    ) -> Result<Option<Balance>, ProviderError> {
        Ok(None)
    }

    /// Cheap credential probe used by the "test" button.
    async fn validate(
        &self,
        client: &reqwest::Client,
        account: &AccountView<'_>,
    ) -> Result<String, ProviderError> {
        let models = self.models(client, account).await?;
        Ok(format!("可用，发现 {} 个模型", models.len()))
    }
}

/// Builds an account view using the provider's stored base URL override.
pub fn account_view<'a>(store: &'a Store, account: &'a crate::store::Account) -> AccountView<'a> {
    let base = store
        .data
        .providers
        .get(&account.provider)
        .and_then(|p| p.base_url.as_deref());
    AccountView::from_account(account, base)
}
