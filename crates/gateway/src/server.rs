//! Gateway HTTP server: OpenAI API, admin API and the embedded console.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, patch, post};
use axum::{Json, Router};
use futures::future::join_all;
use futures::StreamExt;
use gateway_core::openai;
use gateway_core::provider::{account_view, AccountView};
use gateway_core::{
    Account, ChatDelta, CheckinRecord, LoginPoll, ModelRegistry, Provider, ProviderError,
    RemoteModel, Store,
};

use crate::stats::{usage_cached_tokens, usage_credits, usage_tokens, RequestLog, StatsStore};
use serde_json::{json, Value};

#[derive(Clone)]
pub struct ModelLimit {
    pub until_ms: i64,
    pub reason: String,
}

#[derive(Clone)]
pub struct SessionPin {
    pub provider: String,
    pub upstream_model: String,
    pub at: i64,
}

pub type ModelLimits = Arc<tokio::sync::Mutex<HashMap<String, ModelLimit>>>;
pub type SessionPins = Arc<tokio::sync::Mutex<HashMap<String, SessionPin>>>;

#[derive(Clone)]
pub struct CreditSnapshot {
    pub at: i64,
    pub total: Option<f64>,
    pub monthly_quota: Option<f64>,
    pub total_quota: Option<f64>,
    pub unit: Option<String>,
    pub units: Vec<String>,
    pub accounts: Vec<Value>,
}

#[derive(Clone)]
pub(crate) struct PendingLogin {
    pub(crate) name: Option<String>,
    pub(crate) started_at: i64,
}

pub struct AppState {
    pub store: Arc<RwLock<Store>>,
    pub providers: Vec<Arc<dyn Provider>>,
    pub registry: Arc<RwLock<ModelRegistry>>,
    pub client: reqwest::Client,
    pub started_at: i64,
    /// Per-provider round-robin cursor.
    pub cursors: tokio::sync::Mutex<HashMap<String, usize>>,
    /// Interactive login flows keyed by `provider:flow_id`.
    pub logins: tokio::sync::Mutex<HashMap<String, PendingLogin>>,
    pub stats: Arc<StatsStore>,
    pub credits: tokio::sync::Mutex<Option<CreditSnapshot>>,
    pub model_limits: ModelLimits,
    pub session_pins: SessionPins,
}

impl AppState {
    pub fn provider(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.providers.iter().find(|p| p.id() == id).cloned()
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(console))
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{id}", get(get_model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/api/admin/overview", get(admin_overview))
        .route("/api/admin/stats", get(admin_stats))
        .route("/api/admin/logs", get(admin_logs))
        .route("/api/admin/accounts", get(admin_accounts))
        .route(
            "/api/admin/accounts/{id}",
            patch(admin_patch_account).delete(admin_delete_account),
        )
        .route("/api/admin/accounts/{id}/test", post(admin_test_account))
        .route(
            "/api/admin/accounts/{id}/checkin",
            post(admin_checkin_account),
        )
        .route("/api/admin/providers", get(admin_providers))
        .route("/api/admin/providers/{id}", patch(admin_patch_provider))
        .route("/api/admin/models", get(admin_models))
        .route("/api/admin/models/refresh", post(admin_refresh_models))
        .route(
            "/api/admin/settings",
            get(admin_settings).patch(admin_patch_settings),
        )
        .route("/api/admin/login/start", post(admin_login_start))
        .route("/api/admin/login/{flow_id}", get(admin_login_poll))
        .route(
            "/api/admin/login/{flow_id}/cancel",
            post(admin_login_cancel),
        )
        .route("/api/admin/checkin", post(admin_checkin_all))
        .layer(axum::middleware::from_fn(no_store_headers))
        .with_state(state)
}

async fn no_store_headers(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-store"),
    );
    response
}

async fn console() -> Response {
    // The console is embedded in the binary. Never let a browser keep an old
    // HTML/JS version after the service is rebuilt and restarted.
    Response::builder()
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-store, max-age=0")
        .header(header::PRAGMA, "no-cache")
        .header(header::EXPIRES, "0")
        .header(header::X_CONTENT_TYPE_OPTIONS, "nosniff")
        .body(axum::body::Body::from(include_str!("console.html")))
        .unwrap()
}

fn authorized(headers: &HeaderMap, state: &AppState) -> bool {
    let expected = state.store.read().unwrap().data.settings.api_key.clone();
    let token = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))
        .unwrap_or("");
    !expected.is_empty() && token == expected
}

fn json_error(status: StatusCode, message: impl Into<String>, kind: &str) -> Response {
    (status, Json(openai::error_body(message, kind, None))).into_response()
}

fn provider_error_response(err: &ProviderError) -> Response {
    let status = StatusCode::from_u16(err.http_status()).unwrap_or(StatusCode::BAD_GATEWAY);
    let kind = match err {
        ProviderError::Auth(_) => "authentication_error",
        ProviderError::BadRequest(_) => "invalid_request_error",
        ProviderError::Quota(_) | ProviderError::RateLimited { .. } => "rate_limit_error",
        ProviderError::Unavailable(_) => "upstream_error",
    };
    json_error(status, err.to_string(), kind)
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

/// Fetch models for every enabled provider. Uses the first working account per
/// provider; falls back to the provider's static list when the live call fails.
pub async fn refresh_registry(state: &AppState) -> ModelRegistry {
    let mut contributions = Vec::new();
    let mut statuses = Vec::new();
    for provider in &state.providers {
        // Snapshot settings and owned account views, then release the store
        // lock before any network call.
        let (enabled, accounts): (bool, Vec<OwnedView>) = {
            let store = state.store.read().unwrap();
            let enabled = store.provider_settings(provider.id()).enabled;
            let accounts = if enabled {
                store
                    .data
                    .accounts
                    .iter()
                    .filter(|a| a.provider == provider.id() && a.available(gateway_core::now_ms()))
                    .map(|a| view_owned(account_view(&store, a)))
                    .collect()
            } else {
                Vec::new()
            };
            (enabled, accounts)
        };
        if !enabled {
            statuses.push((provider.id().to_string(), "disabled".to_string(), 0, None));
            continue;
        }

        let mut fetched: Option<Vec<RemoteModel>> = None;
        let mut last_error: Option<String> = None;
        for account in &accounts {
            match provider.models(&state.client, &account.as_view()).await {
                Ok(models) if !models.is_empty() => {
                    fetched = Some(models);
                    break;
                }
                Ok(_) => {
                    last_error = Some("上游返回空模型列表".into());
                }
                Err(err) => {
                    last_error = Some(err.to_string());
                }
            }
        }
        match fetched {
            Some(models) => {
                let count = models.len();
                tracing::info!(
                    provider = provider.id(),
                    count,
                    "models refreshed (dynamic)"
                );
                // Persist the last known-good catalog; a later minimal
                // `/v3/config` response must not regress to a stale list.
                {
                    let mut store = state.store.write().unwrap();
                    if store.cache_models(provider.id(), models.clone()) {
                        if let Err(err) = store.save() {
                            tracing::warn!(provider = provider.id(), error = %err, "failed to save model cache");
                        }
                    }
                }
                contributions.push((provider.id().to_string(), models, false));
                statuses.push((
                    provider.id().to_string(),
                    "dynamic".to_string(),
                    count,
                    None,
                ));
            }
            None => {
                let cache = state.store.read().unwrap().cached_models(provider.id());
                let (models, source, fallback_flag) = match cache {
                    Some(entry) if !entry.models.is_empty() => (entry.models, "cache", false),
                    _ => (provider.fallback_models(), "fallback", true),
                };
                let count = models.len();
                let error = last_error.or_else(|| {
                    if accounts.is_empty() {
                        Some("没有可用账号".into())
                    } else {
                        None
                    }
                });
                tracing::warn!(
                    provider = provider.id(),
                    count,
                    source,
                    error = ?error,
                    "model discovery failed; using cached/fallback catalog"
                );
                contributions.push((provider.id().to_string(), models, fallback_flag));
                statuses.push((provider.id().to_string(), source.to_string(), count, error));
            }
        }
    }
    let mut registry = ModelRegistry::build(contributions);
    for (provider, source, count, error) in statuses {
        registry.set_source(provider, source, count, error);
    }
    registry
}

async fn list_models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    let registry = state.registry.read().unwrap();
    let data: Vec<Value> = registry
        .entries()
        .iter()
        .map(|e| {
            json!({
                "id": e.exposed_id,
                "object": "model",
                "created": registry.refreshed_at() / 1000,
                "owned_by": if e.variants.len() == 1 { e.variants[0].provider.clone() } else { "merged".to_string() },
                "display_name": e.name,
                "context_length": e.context_length,
                "providers": e.providers(),
                "aliases": e.aliases(),
                "merged": e.variants.len() > 1,
                "capabilities": {
                    "reasoning": e.reasoning,
                    "vision": e.vision,
                }
            })
        })
        .collect();
    Json(json!({"object": "list", "data": data})).into_response()
}

async fn get_model(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    let registry = state.registry.read().unwrap();
    match registry.resolve(&id) {
        Some((e, _)) => Json(json!({
            "id": e.exposed_id,
            "object": "model",
            "created": registry.refreshed_at() / 1000,
            "owned_by": if e.variants.len() == 1 { e.variants[0].provider.clone() } else { "merged".to_string() },
            "display_name": e.name,
            "providers": e.providers(),
            "aliases": e.aliases(),
        }))
        .into_response(),
        None => json_error(
            StatusCode::NOT_FOUND,
            format!("未知模型: {id}"),
            "not_found",
        ),
    }
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

fn candidate_accounts(state: &AppState, provider: &str) -> Vec<Account> {
    let store = state.store.read().unwrap();
    let now = gateway_core::now_ms();
    store
        .data
        .accounts
        .iter()
        .filter(|a| a.provider == provider && a.available(now))
        .cloned()
        .collect()
}

async fn ordered_candidates(state: &AppState, provider: &str) -> Vec<Account> {
    let accounts = candidate_accounts(state, provider);
    if accounts.len() <= 1 {
        return accounts;
    }
    let mut cursors = state.cursors.lock().await;
    let cursor = cursors.entry(provider.to_string()).or_insert(0);
    let start = *cursor % accounts.len();
    *cursor = cursor.wrapping_add(1);
    let mut ordered = accounts[start..].to_vec();
    ordered.extend_from_slice(&accounts[..start]);
    ordered
}

fn mark_failure(state: &AppState, account_id: &str, err: &ProviderError) {
    let mut store = state.store.write().unwrap();
    store.record_failure(account_id, &err.to_string(), err.cooldown_ms());
}

fn mark_success(state: &AppState, account_id: &str) {
    if let Ok(mut store) = state.store.write() {
        store.record_success(account_id);
    }
}

async fn open_stream(
    state: &Arc<AppState>,
    provider: Arc<dyn Provider>,
    accounts: Vec<Account>,
    request: &Value,
    upstream_model: &str,
) -> Result<ChatStreamResult, ProviderError> {
    let mut last_err = ProviderError::Unavailable("没有可用账号".into());
    let mut limit_hint: Option<(i64, String)> = None;
    for account in accounts {
        let view = {
            let store = state.store.read().unwrap();
            view_owned(account_view(&store, &account))
        };
        match provider
            .chat(&state.client, &view.as_view(), request, upstream_model)
            .await
        {
            Ok(mut stream) => {
                // Read the first event before committing the response. Qoder
                // and WorkBuddy can accept the HTTP request and then report
                // quota/auth failures in the first SSE envelope; without this
                // the gateway would emit a 200 stream instead of failing over.
                match stream.next().await {
                    Some(Err(err)) => {
                        tracing::warn!(
                            provider = provider.id(),
                            account = %account.name,
                            error = %err,
                            "stream rejected before first chunk"
                        );
                        if let Some((duration, reason)) = limit_from_error(&err) {
                            // Plan/subscription limits can be account-specific;
                            // try the next account and record only if all fail.
                            limit_hint = Some((duration, reason));
                        } else if matches!(err, ProviderError::BadRequest(_)) {
                            return Err(err);
                        }
                        mark_failure(state, &account.id, &err);
                        last_err = err;
                        continue;
                    }
                    Some(Ok(first)) => {
                        mark_success(state, &account.id);
                        let stream = Box::pin(futures::stream::iter([Ok(first)]).chain(stream));
                        return Ok(ChatStreamResult {
                            stream,
                            account_id: account.id.clone(),
                            account_name: account.name.clone(),
                            provider_id: provider.id().to_string(),
                            upstream_model: upstream_model.to_string(),
                        });
                    }
                    None => {
                        mark_success(state, &account.id);
                        return Ok(ChatStreamResult {
                            stream,
                            account_id: account.id.clone(),
                            account_name: account.name.clone(),
                            provider_id: provider.id().to_string(),
                            upstream_model: upstream_model.to_string(),
                        });
                    }
                }
            }
            Err(err) => {
                tracing::warn!(provider = provider.id(), account = %account.name, error = %err, "chat open failed");
                if let Some((duration, reason)) = limit_from_error(&err) {
                    limit_hint = Some((duration, reason));
                } else if matches!(err, ProviderError::BadRequest(_)) {
                    return Err(err);
                }
                mark_failure(state, &account.id, &err);
                last_err = err;
            }
        }
    }
    if let Some((duration, reason)) = limit_hint {
        // Only mark the model limited after every account for this provider
        // failed; a later account may have succeeded, so per-account 429s must
        // not poison the whole provider for future requests.
        record_limit_hint(
            &state.model_limits,
            provider.id(),
            upstream_model,
            duration,
            &reason,
        )
        .await;
    }
    Err(last_err)
}

struct ChatStreamResult {
    stream: gateway_core::ChatStream,
    account_id: String,
    account_name: String,
    provider_id: String,
    upstream_model: String,
}

const SESSION_TTL_MS: i64 = 30 * 60 * 1000;
const CREDIT_FRESH_MS: i64 = 10 * 60 * 1000;

fn limit_key(provider: &str, upstream_model: &str) -> String {
    format!("{provider}\u{0}{upstream_model}")
}

fn limit_from_error(err: &ProviderError) -> Option<(i64, String)> {
    let reason = err.to_string();
    match err {
        ProviderError::RateLimited { retry_after_ms, .. } => {
            Some(((*retry_after_ms).max(30_000), reason))
        }
        ProviderError::BadRequest(message) => {
            let lower = message.to_ascii_lowercase();
            if lower.contains("rate limit")
                || lower.contains("too many requests")
                || lower.contains("quota")
                || lower.contains("daily")
                || lower.contains("exceeded")
                || lower.contains("pricing")
                || lower.contains("try again")
                || message.contains("额度")
                || message.contains("限流")
                || message.contains("订阅")
            {
                Some((3_600_000, reason))
            } else {
                None
            }
        }
        _ => None,
    }
}

async fn record_limit_hint(
    limits: &ModelLimits,
    provider: &str,
    upstream_model: &str,
    duration_ms: i64,
    reason: &str,
) {
    let key = limit_key(provider, upstream_model);
    let until = gateway_core::now_ms() + duration_ms;
    tracing::warn!(
        provider,
        upstream_model,
        until_ms = until,
        reason,
        "model time-window limit recorded"
    );
    limits.lock().await.insert(
        key,
        ModelLimit {
            until_ms: until,
            reason: reason.chars().take(300).collect(),
        },
    );
}

async fn note_model_limit(
    limits: &ModelLimits,
    provider: &str,
    upstream_model: &str,
    err: &ProviderError,
) {
    if let Some((duration_ms, reason)) = limit_from_error(err) {
        record_limit_hint(limits, provider, upstream_model, duration_ms, &reason).await;
    }
}

fn session_key(headers: &HeaderMap, request: &Value, model: &str) -> String {
    let explicit = headers
        .get("x-session-id")
        .or_else(|| headers.get("x-conversation-id"))
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .or_else(|| {
            ["session_id", "conversation_id"]
                .iter()
                .find_map(|key| request.get(*key).and_then(Value::as_str))
                .map(str::to_string)
        });
    if let Some(id) = explicit.filter(|id| !id.trim().is_empty()) {
        return format!("{model}:explicit:{id}");
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    model.hash(&mut hasher);
    if let Some(user) = request.get("user").and_then(Value::as_str) {
        user.hash(&mut hasher);
    }
    if let Some(messages) = request.get("messages").and_then(Value::as_array) {
        // Hash only the opening system message and the first user message.
        // Later turns append messages, so hashing a fixed prefix keeps the
        // same conversation on the same model variant across requests.
        let mut has_system = false;
        let mut has_user = false;
        for message in messages {
            let role = message.get("role").and_then(Value::as_str).unwrap_or("");
            if role == "system" && !has_system {
                has_system = true;
            } else if role == "user" && !has_user {
                has_user = true;
            } else {
                continue;
            }
            serde_json::to_string(message)
                .unwrap_or_default()
                .hash(&mut hasher);
            if has_user {
                break;
            }
        }
    }
    format!("{model}:h:{:x}", hasher.finish())
}

fn provider_credit_score(snapshot: &Option<CreditSnapshot>, provider: &str) -> f64 {
    let Some(snapshot) = snapshot else {
        return 0.0;
    };
    if gateway_core::now_ms() - snapshot.at > CREDIT_FRESH_MS {
        return 0.0;
    }
    snapshot
        .accounts
        .iter()
        .filter(|row| row.get("provider").and_then(Value::as_str) == Some(provider))
        .filter(|row| row.get("unit").and_then(Value::as_str) == Some("credits"))
        .filter_map(|row| row.get("credits").and_then(Value::as_f64))
        .sum()
}

async fn order_variants(
    state: &AppState,
    variants: &[(String, String)],
    session: &str,
) -> Result<Vec<(String, String)>, (i64, String)> {
    let now = gateway_core::now_ms();
    let pin = {
        let mut pins = state.session_pins.lock().await;
        pins.retain(|_, pin| now - pin.at <= SESSION_TTL_MS);
        pins.get(session).cloned()
    };
    let credits = state.credits.lock().await.clone();
    let limits = {
        let mut limits = state.model_limits.lock().await;
        limits.retain(|_, limit| limit.until_ms > now);
        limits.clone()
    };

    let mut available: Vec<(usize, (String, String), f64, bool)> = Vec::new();
    let mut earliest: Option<(i64, String)> = None;
    for (index, (provider, upstream)) in variants.iter().enumerate() {
        if let Some(limit) = limits.get(&limit_key(provider, upstream)) {
            let entry = (limit.until_ms, limit.reason.clone());
            if earliest
                .as_ref()
                .map(|(t, _)| limit.until_ms < *t)
                .unwrap_or(true)
            {
                earliest = Some(entry);
            }
            continue;
        }
        let pinned = pin
            .as_ref()
            .map(|p| p.provider == *provider && p.upstream_model == *upstream)
            .unwrap_or(false);
        let score = provider_credit_score(&credits, provider);
        available.push((index, (provider.clone(), upstream.clone()), score, pinned));
    }
    if available.is_empty() {
        return Err(earliest.unwrap_or((now + 60_000, "所有提供商当前不可用".into())));
    }
    available.sort_by(|a, b| {
        b.3.cmp(&a.3)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.0.cmp(&b.0))
    });
    Ok(available
        .into_iter()
        .map(|(_, variant, _, _)| variant)
        .collect())
}

async fn pin_session(state: &AppState, session: &str, provider: &str, upstream_model: &str) {
    state.session_pins.lock().await.insert(
        session.to_string(),
        SessionPin {
            provider: provider.to_string(),
            upstream_model: upstream_model.to_string(),
            at: gateway_core::now_ms(),
        },
    );
}

fn base_log(model: &str, stream: bool) -> RequestLog {
    RequestLog {
        id: 0,
        at: 0,
        model: model.to_string(),
        upstream_model: None,
        provider: None,
        account: None,
        stream,
        status: "error".into(),
        http_status: 500,
        duration_ms: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        cached_tokens: 0,
        credits: None,
        error: None,
    }
}

fn finish_chat_log(
    state: &AppState,
    mut log: RequestLog,
    started: std::time::Instant,
    status: &str,
    http_status: u16,
    usage: Option<&Value>,
    error: Option<String>,
) {
    log.status = status.into();
    log.http_status = http_status;
    log.duration_ms = started.elapsed().as_millis() as i64;
    if let Some(usage) = usage {
        let (prompt, completion, total) = usage_tokens(usage);
        log.prompt_tokens = prompt;
        log.completion_tokens = completion;
        log.total_tokens = total;
        log.cached_tokens = usage_cached_tokens(usage);
        log.credits = usage_credits(usage);
    }
    log.error = error;
    state.stats.record(log);
}

#[allow(clippy::too_many_arguments)]
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> Response {
    let started = std::time::Instant::now();
    let wants_stream = openai::wants_stream(&request);
    if !authorized(&headers, &state) {
        finish_chat_log(
            &state,
            base_log("-", wants_stream),
            started,
            "error",
            401,
            None,
            Some("无效的 API Key".into()),
        );
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    let Some(model_id) = request.get("model").and_then(Value::as_str) else {
        finish_chat_log(
            &state,
            base_log("-", wants_stream),
            started,
            "error",
            400,
            None,
            Some("缺少 model 字段".into()),
        );
        return json_error(
            StatusCode::BAD_REQUEST,
            "缺少 model 字段",
            "invalid_request_error",
        );
    };
    if request
        .get("messages")
        .and_then(Value::as_array)
        .map(|m| m.is_empty())
        .unwrap_or(true)
    {
        finish_chat_log(
            &state,
            base_log(model_id, wants_stream),
            started,
            "error",
            400,
            None,
            Some("messages 不能为空".into()),
        );
        return json_error(
            StatusCode::BAD_REQUEST,
            "messages 不能为空",
            "invalid_request_error",
        );
    }
    struct RoutePlan {
        variants: Vec<(String, String)>,
    }
    let plan = {
        let registry = state.registry.read().unwrap();
        registry
            .resolve(model_id)
            .map(|(_entry, variants)| RoutePlan {
                variants: variants
                    .into_iter()
                    .map(|v| (v.provider.clone(), v.model.upstream_id.clone()))
                    .collect(),
            })
    };
    let Some(plan) = plan else {
        let msg = format!("未知模型: {model_id}");
        finish_chat_log(
            &state,
            base_log(model_id, wants_stream),
            started,
            "error",
            404,
            None,
            Some(msg.clone()),
        );
        return json_error(
            StatusCode::NOT_FOUND,
            format!("{msg}。请调用 /v1/models 查看可用模型"),
            "model_not_found",
        );
    };

    let session = session_key(&headers, &request, model_id);
    let ordered = match order_variants(&state, &plan.variants, &session).await {
        Ok(variants) => variants,
        Err((until, reason)) => {
            let when = chrono::DateTime::from_timestamp_millis(until)
                .map(|dt| {
                    dt.with_timezone(&chrono::Local)
                        .format("%Y-%m-%d %H:%M:%S")
                        .to_string()
                })
                .unwrap_or_else(|| until.to_string());
            let msg = format!("模型 {model_id} 处于时段限额：{reason}；最早恢复 {when}");
            let mut log = base_log(model_id, wants_stream);
            log.provider = plan.variants.first().map(|(p, _)| p.clone());
            finish_chat_log(&state, log, started, "error", 429, None, Some(msg.clone()));
            return json_error(StatusCode::TOO_MANY_REQUESTS, msg, "rate_limit_error");
        }
    };

    let mut stream_result: Option<ChatStreamResult> = None;
    let mut last_err: Option<ProviderError> = None;
    let mut unavailable: Vec<String> = Vec::new();
    for (variant_provider, upstream_model) in &ordered {
        let Some(provider) = state.provider(variant_provider) else {
            unavailable.push(format!("{variant_provider}: 未加载"));
            continue;
        };
        let provider_enabled = state
            .store
            .read()
            .unwrap()
            .provider_settings(variant_provider)
            .enabled;
        if !provider_enabled {
            unavailable.push(format!("{variant_provider}: 已停用"));
            continue;
        }
        let accounts = ordered_candidates(&state, variant_provider).await;
        if accounts.is_empty() {
            unavailable.push(format!("{variant_provider}: 无可用账号"));
            continue;
        }
        match open_stream(&state, provider, accounts, &request, upstream_model).await {
            Ok(result) => {
                stream_result = Some(result);
                break;
            }
            Err(err) => {
                tracing::warn!(provider = %variant_provider, model = %upstream_model, error = %err, "model variant failed");
                last_err = Some(err);
            }
        }
    }
    let Some(stream_result) = stream_result else {
        if let Some(err) = last_err {
            let http = err.http_status();
            let mut log = base_log(model_id, wants_stream);
            log.provider = plan.variants.first().map(|(p, _)| p.clone());
            finish_chat_log(
                &state,
                log,
                started,
                "error",
                http,
                None,
                Some(err.to_string()),
            );
            return provider_error_response(&err);
        }
        let msg = format!("模型 {model_id} 暂不可用：{}", unavailable.join("；"));
        let mut log = base_log(model_id, wants_stream);
        log.provider = plan.variants.first().map(|(p, _)| p.clone());
        finish_chat_log(&state, log, started, "error", 503, None, Some(msg.clone()));
        return json_error(StatusCode::SERVICE_UNAVAILABLE, msg, "no_account");
    };

    pin_session(
        &state,
        &session,
        &stream_result.provider_id,
        &stream_result.upstream_model,
    )
    .await;

    let id = openai::new_completion_id();
    let created = chrono::Utc::now().timestamp();
    let model_name = model_id.to_string();
    let include_usage = openai::stream_include_usage(&request);
    let mut log = base_log(model_id, wants_stream);
    log.provider = Some(stream_result.provider_id.clone());
    log.account = Some(stream_result.account_name.clone());
    log.upstream_model = Some(stream_result.upstream_model.clone());

    if wants_stream {
        return sse_response(
            stream_result.stream,
            id,
            created,
            model_name,
            include_usage,
            state.stats.clone(),
            state.model_limits.clone(),
            log,
            started,
        );
    }

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut finish_reason: Option<String> = None;
    let mut usage: Option<Value> = None;
    let mut stream = stream_result.stream;
    while let Some(item) = stream.next().await {
        match item {
            Ok(delta) => {
                if let Some(c) = delta.content {
                    content.push_str(&c);
                }
                if let Some(r) = delta.reasoning_content {
                    reasoning.push_str(&r);
                }
                if let Some(calls) = delta.tool_calls {
                    merge_tool_calls(&mut tool_calls, calls);
                }
                if delta.finish_reason.is_some() {
                    finish_reason = delta.finish_reason;
                }
                if delta.usage.is_some() {
                    usage = delta.usage;
                }
            }
            Err(err) => {
                mark_failure(&state, &stream_result.account_id, &err);
                note_model_limit(
                    &state.model_limits,
                    &stream_result.provider_id,
                    &stream_result.upstream_model,
                    &err,
                )
                .await;
                let http = err.http_status();
                finish_chat_log(
                    &state,
                    log,
                    started,
                    "error",
                    http,
                    None,
                    Some(err.to_string()),
                );
                return provider_error_response(&err);
            }
        }
    }
    let finish = finish_reason.unwrap_or_else(|| {
        if tool_calls.is_empty() {
            "stop".into()
        } else {
            "tool_calls".into()
        }
    });
    let calls = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls.as_slice())
    };
    finish_chat_log(&state, log, started, "success", 200, usage.as_ref(), None);
    Json(openai::completion_response(
        &id,
        created,
        &model_name,
        &content,
        Some(&reasoning),
        calls,
        &finish,
        usage.unwrap_or_else(openai::zero_usage),
    ))
    .into_response()
}

fn merge_tool_calls(buffer: &mut Vec<Value>, incoming: Vec<Value>) {
    for call in incoming {
        let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
        while buffer.len() <= index {
            buffer.push(
                json!({"id": "", "type": "function", "function": {"name": "", "arguments": ""}}),
            );
        }
        let slot = &mut buffer[index];
        if let Some(id) = call.get("id").and_then(Value::as_str) {
            if !id.is_empty() {
                slot["id"] = Value::String(id.into());
            }
        }
        if let Some(t) = call.get("type").and_then(Value::as_str) {
            slot["type"] = Value::String(t.into());
        }
        if let Some(function) = call.get("function") {
            if let Some(name) = function.get("name").and_then(Value::as_str) {
                if !name.is_empty() {
                    slot["function"]["name"] = Value::String(name.into());
                }
            }
            if let Some(args) = function.get("arguments").and_then(Value::as_str) {
                let current = slot["function"]["arguments"].as_str().unwrap_or("");
                slot["function"]["arguments"] = Value::String(format!("{current}{args}"));
            }
        }
    }
    for call in buffer.iter_mut() {
        call.as_object_mut().map(|o| o.remove("index"));
    }
}

#[allow(clippy::too_many_arguments)]
fn sse_response(
    stream: gateway_core::ChatStream,
    id: String,
    created: i64,
    model: String,
    include_usage: bool,
    stats: Arc<StatsStore>,
    limits: ModelLimits,
    mut log: RequestLog,
    started: std::time::Instant,
) -> Response {
    let limit_provider = log.provider.clone().unwrap_or_default();
    let limit_upstream = log.upstream_model.clone().unwrap_or_default();
    let body_stream = async_stream::stream! {
        let mut sent_role = false;
        let mut finish_sent = false;
        let mut usage: Option<Value> = None;
        let mut stream_error: Option<String> = None;

        let mut stream = stream;
        while let Some(item) = stream.next().await {
            match item {
                Ok(mut delta) => {
                    if !sent_role {
                        if delta.role.is_none() {
                            delta.role = Some("assistant".into());
                        }
                        sent_role = true;
                    }
                    if delta.usage.is_some() {
                        usage = delta.usage.take();
                    }
                    if delta.finish_reason.is_some() {
                        let reason = delta.finish_reason.clone();
                        delta.finish_reason = None;
                        if !delta.is_empty() {
                            let chunk = openai::delta_chunk(&id, created, &model, &delta, None);
                            yield Ok::<Bytes, std::convert::Infallible>(Bytes::from(format!("data: {chunk}\n\n")));
                        }
                        let empty = ChatDelta::default();
                        let chunk = openai::delta_chunk(&id, created, &model, &empty, reason.as_deref());
                        yield Ok(Bytes::from(format!("data: {chunk}\n\n")));
                        finish_sent = true;
                        continue;
                    }
                    if delta.is_empty() {
                        continue;
                    }
                    let chunk = openai::delta_chunk(&id, created, &model, &delta, None);
                    yield Ok(Bytes::from(format!("data: {chunk}\n\n")));
                }
                Err(err) => {
                    if !limit_provider.is_empty() && !limit_upstream.is_empty() {
                        note_model_limit(&limits, &limit_provider, &limit_upstream, &err).await;
                    }
                    stream_error = Some(err.to_string());
                    let payload = openai::error_body(err.to_string(), "upstream_error", None);
                    yield Ok(Bytes::from(format!("data: {payload}\n\n")));
                    break;
                }
            }
        }
        if !finish_sent {
            let chunk = openai::delta_chunk(&id, created, &model, &ChatDelta::default(), Some("stop"));
            yield Ok(Bytes::from(format!("data: {chunk}\n\n")));
        }
        if include_usage {
            if let Some(usage) = usage.as_ref() {
                let chunk = openai::usage_chunk(&id, created, &model, usage.clone());
                yield Ok(Bytes::from(format!("data: {chunk}\n\n")));
            }
        }
        log.status = if stream_error.is_some() { "error".into() } else { "success".into() };
        log.http_status = if stream_error.is_some() { 502 } else { 200 };
        log.duration_ms = started.elapsed().as_millis() as i64;
        if let Some(usage) = usage.as_ref() {
            let (prompt, completion, total) = usage_tokens(usage);
            log.prompt_tokens = prompt;
            log.completion_tokens = completion;
            log.total_tokens = total;
            log.cached_tokens = usage_cached_tokens(usage);
            log.credits = usage_credits(usage);
        }
        log.error = stream_error;
        stats.record(log);
        yield Ok(Bytes::from("data: [DONE]\n\n"));
    };
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(axum::body::Body::from_stream(body_stream))
        .unwrap()
}

// ---------------------------------------------------------------------------
// Admin API
// ---------------------------------------------------------------------------

fn account_json(state: &AppState, account: &Account) -> Value {
    let now = gateway_core::now_ms();
    let status = if !account.enabled {
        "disabled"
    } else if account.cooldown_until.map(|t| t > now).unwrap_or(false) {
        "cooldown"
    } else {
        "active"
    };
    let provider_name = state
        .provider(&account.provider)
        .map(|p| p.name().to_string())
        .unwrap_or_else(|| account.provider.clone());
    json!({
        "id": account.id,
        "provider": account.provider,
        "provider_name": provider_name,
        "name": account.name,
        "credential_hint": account.credential_hint(),
        "region": account.region,
        "base_url": account.base_url,
        "enabled": account.enabled,
        "status": status,
        "cooldown_until": account.cooldown_until,
        "disabled_reason": account.disabled_reason,
        "last_error": account.last_error,
        "last_checkin": account.last_checkin,
        "created_at": account.created_at,
        "expires_at": account.expires_at,
        "remote_id": account.remote_id,
        "has_refresh_token": account.refresh_token.is_some(),
    })
}

async fn admin_overview(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    let registry = state.registry.read().unwrap();
    let store = state.store.read().unwrap();
    let accounts = store.data.accounts.len();
    let active = store
        .data
        .accounts
        .iter()
        .filter(|a| a.available(gateway_core::now_ms()))
        .count();
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": (gateway_core::now_ms() - state.started_at) / 1000,
        "accounts": accounts,
        "active_accounts": active,
        "models": registry.entries().len(),
        "providers": state.providers.iter().map(|p| p.id()).collect::<Vec<_>>(),
        "models_used_fallback": registry.used_fallback,
        "models_refreshed_at": registry.refreshed_at(),
    }))
    .into_response()
}

async fn admin_stats(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    let totals = state.stats.totals();
    let log_total = state.stats.total_logs();
    let now = gateway_core::now_ms();
    let cached = state.credits.lock().await.clone();
    let credits = match cached {
        Some(snapshot) if now - snapshot.at < 60_000 => snapshot,
        _ => {
            let snapshot = fetch_credits(&state).await;
            *state.credits.lock().await = Some(snapshot.clone());
            snapshot
        }
    };
    Json(json!({
        "totals": totals,
        "log_total": log_total,
        "credits": {
            "total": credits.total,
            "monthly_quota": credits.monthly_quota,
            "total_quota": credits.total_quota,
            "unit": credits.unit,
            "units": credits.units,
            "accounts": credits.accounts,
            "at": credits.at,
        },
    }))
    .into_response()
}

#[derive(serde::Deserialize)]
struct LogQuery {
    page: Option<usize>,
    page_size: Option<usize>,
}

async fn admin_logs(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<LogQuery>,
) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    let page = query.page.unwrap_or(1).max(1);
    let page_size = query.page_size.unwrap_or(10).clamp(1, 50);
    let (total, logs) = state.stats.page(page, page_size);
    let pages = total.div_ceil(page_size).max(1);
    Json(json!({
        "logs": logs,
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": pages,
    }))
    .into_response()
}

async fn fetch_credits(state: &Arc<AppState>) -> CreditSnapshot {
    let accounts: Vec<(Account, OwnedView, Arc<dyn Provider>)> = {
        let store = state.store.read().unwrap();
        store
            .data
            .accounts
            .iter()
            .filter_map(|account| {
                let settings = store.provider_settings(&account.provider);
                if !account.enabled || !settings.enabled {
                    return None;
                }
                let provider = state.provider(&account.provider)?;
                Some((
                    account.clone(),
                    view_owned(account_view(&store, account)),
                    provider,
                ))
            })
            .collect()
    };
    let tasks = accounts.into_iter().map(|(account, view, provider)| {
        let client = state.client.clone();
        async move {
            let result = provider.balance(&client, &view.as_view()).await;
            let (credits, unit, monthly, total, error) = match result {
                Ok(Some(balance)) => (
                    Some(balance.amount),
                    Some(balance.unit),
                    balance.monthly_quota,
                    balance.total_quota,
                    None,
                ),
                Ok(None) => (None, None, None, None, None),
                Err(err) => (None, None, None, None, Some(err.to_string())),
            };
            json!({
                "provider": account.provider,
                "account_id": account.id,
                "name": account.name,
                "credits": credits,
                "unit": unit,
                "monthly_quota": monthly,
                "total_quota": total,
                "error": error,
            })
        }
    });
    let rows: Vec<Value> = join_all(tasks).await;
    let mut total = 0.0;
    let mut monthly = 0.0;
    let mut quota = 0.0;
    let mut monthly_found = false;
    let mut quota_found = false;
    let mut units = std::collections::BTreeSet::new();
    for row in &rows {
        if let Some(n) = row.get("credits").and_then(Value::as_f64) {
            total += n;
            if let Some(unit) = row.get("unit").and_then(Value::as_str) {
                units.insert(unit.to_string());
            }
        }
        if let Some(n) = row.get("monthly_quota").and_then(Value::as_f64) {
            monthly += n;
            monthly_found = true;
        }
        if let Some(n) = row.get("total_quota").and_then(Value::as_f64) {
            quota += n;
            quota_found = true;
        }
    }
    let units: Vec<String> = units.into_iter().collect();
    let unit = if units.len() == 1 {
        units.first().cloned()
    } else {
        None
    };
    CreditSnapshot {
        at: gateway_core::now_ms(),
        total: if units.len() <= 1 && !units.is_empty() {
            Some(total)
        } else {
            None
        },
        monthly_quota: if units.len() <= 1 && monthly_found {
            Some(monthly)
        } else {
            None
        },
        total_quota: if units.len() <= 1 && quota_found {
            Some(quota)
        } else {
            None
        },
        unit,
        units,
        accounts: rows,
    }
}

async fn admin_accounts(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    let store = state.store.read().unwrap();
    let accounts: Vec<Value> = store
        .data
        .accounts
        .iter()
        .map(|a| account_json(&state, a))
        .collect();
    Json(json!({"accounts": accounts})).into_response()
}

async fn admin_patch_account(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    if body.get("credential").is_some() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "不支持手动设置凭证，请使用授权登录",
            "invalid_request_error",
        );
    }
    {
        let mut store = state.store.write().unwrap();
        let Some(account) = store.data.accounts.iter_mut().find(|a| a.id == id) else {
            return json_error(StatusCode::NOT_FOUND, "账号不存在", "not_found");
        };
        if let Some(name) = body.get("name").and_then(Value::as_str) {
            account.name = name.to_string();
        }
        let old_region = account.region.clone();
        let old_base = account.base_url.clone();
        if let Some(region) = body.get("region") {
            account.region = region
                .as_str()
                .filter(|s| !s.is_empty())
                .map(str::to_string);
        }
        if let Some(base) = body.get("base_url") {
            account.base_url = base.as_str().filter(|s| !s.is_empty()).map(str::to_string);
        }
        if let Some(enabled) = body.get("enabled").and_then(Value::as_bool) {
            account.enabled = enabled;
            if enabled {
                account.cooldown_until = None;
                account.disabled_reason = None;
                account.last_error = None;
            }
        }
        // The cached catalog belongs to the previous upstream/realm.
        if old_region != account.region || old_base != account.base_url {
            let provider = account.provider.clone();
            store.data.model_cache.remove(&provider);
        }
        let _ = store.save();
    }
    spawn_registry_refresh(state.clone());
    Json(json!({"ok": true})).into_response()
}

async fn admin_delete_account(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    let removed = {
        let mut store = state.store.write().unwrap();
        store.remove_account(&id).unwrap_or(false)
    };
    if removed {
        spawn_registry_refresh(state.clone());
        Json(json!({"ok": true})).into_response()
    } else {
        json_error(StatusCode::NOT_FOUND, "账号不存在", "not_found")
    }
}

async fn admin_test_account(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    let (provider, view, account) = {
        let store = state.store.read().unwrap();
        let Some(account) = store.account(&id).cloned() else {
            return json_error(StatusCode::NOT_FOUND, "账号不存在", "not_found");
        };
        let Some(provider) = state.provider(&account.provider) else {
            return json_error(
                StatusCode::BAD_REQUEST,
                "提供商未加载",
                "invalid_request_error",
            );
        };
        let view = view_owned(account_view(&store, &account));
        (provider, view, account)
    };
    match provider.validate(&state.client, &view.as_view()).await {
        Ok(message) => {
            mark_success(&state, &account.id);
            Json(json!({"ok": true, "message": message})).into_response()
        }
        Err(err) => json_error(
            StatusCode::BAD_REQUEST,
            format!("测试失败: {err}"),
            "authentication_error",
        ),
    }
}

/// Owned copy of an [`AccountView`], needed when the store lock is released
/// before an await.
struct OwnedView {
    id: String,
    name: String,
    credential: String,
    region: Option<String>,
    base_url: Option<String>,
    refresh_token: Option<String>,
    expires_at: Option<i64>,
}

impl OwnedView {
    fn as_view(&self) -> AccountView<'_> {
        AccountView {
            id: &self.id,
            name: &self.name,
            credential: &self.credential,
            region: self.region.as_deref(),
            base_url: self.base_url.as_deref(),
            refresh_token: self.refresh_token.as_deref(),
            expires_at: self.expires_at,
        }
    }
}

fn view_owned(view: AccountView<'_>) -> OwnedView {
    OwnedView {
        id: view.id.to_string(),
        name: view.name.to_string(),
        credential: view.credential.to_string(),
        region: view.region.map(str::to_string),
        base_url: view.base_url.map(str::to_string),
        refresh_token: view.refresh_token.map(str::to_string),
        expires_at: view.expires_at,
    }
}

async fn admin_login_start(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    let provider_id = body.get("provider").and_then(Value::as_str).unwrap_or("");
    let Some(provider) = state.provider(provider_id) else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "未知提供商",
            "invalid_request_error",
        );
    };
    if !provider.supports_login() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "该提供商不支持浏览器登录",
            "invalid_request_error",
        );
    }
    let region = body.get("region").and_then(Value::as_str);
    let name = body
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string);
    let base = state
        .store
        .read()
        .unwrap()
        .provider_settings(provider_id)
        .base_url;
    match provider
        .start_login(&state.client, region, base.as_deref())
        .await
    {
        Ok(start) => {
            let flow_id = format!("{}:{}", provider.id(), start.flow_id);
            state.logins.lock().await.insert(
                flow_id.clone(),
                PendingLogin {
                    name,
                    started_at: gateway_core::now_ms(),
                },
            );
            Json(json!({
                "flow_id": flow_id,
                "url": start.url,
                "message": start.message,
                "provider": provider.id(),
                "expires_at": start.expires_at,
            }))
            .into_response()
        }
        Err(err) => provider_error_response(&err),
    }
}

async fn admin_login_poll(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(flow_id): Path<String>,
) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    let Some(pending) = state.logins.lock().await.get(&flow_id).cloned() else {
        return json_error(StatusCode::NOT_FOUND, "登录会话不存在或已结束", "not_found");
    };
    if gateway_core::now_ms() - pending.started_at > 10 * 60 * 1000 {
        state.logins.lock().await.remove(&flow_id);
        return Json(json!({"status": "expired", "message": "登录链接已过期，请重新获取"}))
            .into_response();
    }
    let Some((provider_id, provider_flow)) = flow_id.split_once(':') else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "无效的 flow_id",
            "invalid_request_error",
        );
    };
    let Some(provider) = state.provider(provider_id) else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "未知提供商",
            "invalid_request_error",
        );
    };
    match provider.poll_login(&state.client, provider_flow).await {
        Ok(LoginPoll::Pending { message }) => {
            Json(json!({"status": "pending", "message": message})).into_response()
        }
        Ok(LoginPoll::Expired { message }) => {
            state.logins.lock().await.remove(&flow_id);
            Json(json!({"status": "expired", "message": message})).into_response()
        }
        Ok(LoginPoll::Failed { message }) => {
            state.logins.lock().await.remove(&flow_id);
            Json(json!({"status": "failed", "message": message})).into_response()
        }
        Ok(LoginPoll::Success(login)) => {
            state.logins.lock().await.remove(&flow_id);
            if login.credential.trim().is_empty() {
                return json_error(
                    StatusCode::BAD_GATEWAY,
                    "授权成功但未返回凭证",
                    "upstream_error",
                );
            }
            let (account, account_id) = {
                let mut store = state.store.write().unwrap();
                let display_name = pending
                    .name
                    .clone()
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or_else(|| {
                        if login.name.trim().is_empty() {
                            provider.name().to_string()
                        } else {
                            login.name.clone()
                        }
                    });
                // Dedupe by stable upstream identity, then by credential.
                // NEVER by display name: fallback names collide and would
                // overwrite unrelated accounts.
                let existing = login
                    .remote_id
                    .as_deref()
                    .filter(|id| !id.is_empty())
                    .and_then(|remote| {
                        store.data.accounts.iter().position(|a| {
                            a.provider == provider.id() && a.remote_id.as_deref() == Some(remote)
                        })
                    })
                    .or_else(|| {
                        store.data.accounts.iter().position(|a| {
                            a.provider == provider.id() && a.credential == login.credential
                        })
                    });
                let mut new_account = false;
                let id = match existing {
                    Some(index) => {
                        let a = &mut store.data.accounts[index];
                        a.credential = login.credential.clone();
                        a.refresh_token = login.refresh_token.clone();
                        a.expires_at = login.expires_at;
                        if login.remote_id.is_some() {
                            a.remote_id = login.remote_id.clone();
                        }
                        a.region = login.region.clone().or(a.region.clone());
                        a.base_url = login.base_url.clone();
                        a.enabled = true;
                        a.cooldown_until = None;
                        a.disabled_reason = None;
                        a.last_error = None;
                        a.id.clone()
                    }
                    None => {
                        new_account = true;
                        let mut a = Account::new(provider.id(), &display_name, &login.credential);
                        a.refresh_token = login.refresh_token.clone();
                        a.expires_at = login.expires_at;
                        a.remote_id = login.remote_id.clone();
                        a.region = login.region.clone();
                        a.base_url = login.base_url.clone();
                        let id = a.id.clone();
                        store.data.accounts.push(a);
                        id
                    }
                };
                if new_account {
                    // A new account may have a different plan/model catalog.
                    store.data.model_cache.remove(provider.id());
                }
                if let Err(err) = store.save() {
                    return json_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("保存账号失败: {err}"),
                        "server_error",
                    );
                }
                let account = store
                    .account(&id)
                    .map(|a| account_json(&state, a))
                    .unwrap_or(Value::Null);
                (account, id)
            };
            // Await the refresh so the login response already reflects the
            // new account's models (a background refresh races the UI reload).
            let registry = refresh_registry(&state).await;
            *state.registry.write().unwrap() = registry;
            Json(json!({
                "status": "success",
                "account_id": account_id,
                "account": account,
            }))
            .into_response()
        }
        Err(err) => match err {
            // Transient upstream failures should not abort the user's login
            // window; keep the flow alive and retry on the next poll.
            ProviderError::Unavailable(message) => Json(json!({
                "status": "pending",
                "message": format!("授权轮询暂时失败，正在重试: {message}"),
            }))
            .into_response(),
            other => {
                state.logins.lock().await.remove(&flow_id);
                provider_error_response(&other)
            }
        },
    }
}

async fn admin_login_cancel(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(flow_id): Path<String>,
) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    state.logins.lock().await.remove(&flow_id);
    if let Some((provider_id, provider_flow)) = flow_id.split_once(':') {
        if let Some(provider) = state.provider(provider_id) {
            provider.cancel_login(provider_flow).await;
        }
    }
    Json(json!({"ok": true})).into_response()
}

async fn admin_checkin_account(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    match run_checkin(&state, Some(id), false).await {
        Ok(items) => Json(json!({"ok": true, "results": items})).into_response(),
        Err(e) => json_error(StatusCode::BAD_REQUEST, e, "invalid_request_error"),
    }
}

async fn admin_checkin_all(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    match run_checkin(&state, None, false).await {
        Ok(items) => Json(json!({"ok": true, "results": items})).into_response(),
        Err(e) => json_error(StatusCode::BAD_REQUEST, e, "invalid_request_error"),
    }
}

async fn run_checkin(
    state: &Arc<AppState>,
    account_id: Option<String>,
    only_auto_providers: bool,
) -> Result<Vec<Value>, String> {
    let accounts: Vec<Account> = {
        let store = state.store.read().unwrap();
        store
            .data
            .accounts
            .iter()
            .filter(|a| account_id.as_ref().map(|id| &a.id == id).unwrap_or(true))
            .filter(|a| a.enabled)
            .filter(|a| {
                if !only_auto_providers {
                    return true;
                }
                store
                    .data
                    .providers
                    .get(&a.provider)
                    .map(|p| p.enabled && p.auto_checkin)
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    };
    if let Some(id) = &account_id {
        if accounts.is_empty() {
            return Err(format!("账号 {id} 不存在或已停用"));
        }
    }
    let mut results = Vec::new();
    for account in accounts {
        let Some(provider) = state.provider(&account.provider) else {
            continue;
        };
        let view = {
            let store = state.store.read().unwrap();
            view_owned(account_view(&store, &account))
        };
        let outcome = provider.checkin(&state.client, &view.as_view()).await;
        match outcome {
            Ok(outcome) => {
                let record = CheckinRecord {
                    date: gateway_core::today_local(),
                    status: outcome.status.clone(),
                    message: outcome.message.clone(),
                    reward: outcome.reward,
                    at: gateway_core::now_ms(),
                };
                {
                    let mut store = state.store.write().unwrap();
                    if let Some(a) = store.data.accounts.iter_mut().find(|a| a.id == account.id) {
                        a.last_checkin = Some(record);
                        // A successful/already/skipped check-in proves the
                        // credential works; drop the stale red error shown by
                        // the console (e.g. a previous 10001 misclassification).
                        a.last_error = None;
                        a.cooldown_until = None;
                    }
                    let _ = store.save();
                }
                results.push(json!({
                    "account_id": account.id,
                    "name": account.name,
                    "provider": account.provider,
                    "status": outcome.status,
                    "message": outcome.message,
                    "reward": outcome.reward,
                }));
            }
            Err(err) => {
                mark_failure(state, &account.id, &err);
                results.push(json!({
                    "account_id": account.id,
                    "name": account.name,
                    "provider": account.provider,
                    "status": "error",
                    "message": err.to_string(),
                }));
            }
        }
    }
    Ok(results)
}

async fn admin_providers(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    let store = state.store.read().unwrap();
    let registry = state.registry.read().unwrap();
    let providers: Vec<Value> = state
        .providers
        .iter()
        .map(|p| {
            let settings = store.provider_settings(p.id());
            let account_count = store
                .data
                .accounts
                .iter()
                .filter(|a| a.provider == p.id())
                .count();
            let model_count = registry
                .entries()
                .iter()
                .filter(|e| e.variants.iter().any(|v| v.provider == p.id()))
                .count();
            json!({
                "id": p.id(),
                "name": p.name(),
                "default_base_url": p.default_base_url(),
                "supports_login": p.supports_login(),
                "enabled": settings.enabled,
                "auto_checkin": settings.auto_checkin,
                "base_url": settings.base_url,
                "account_count": account_count,
                "model_count": model_count,
                "models_source": registry.sources.get(p.id()).map(|s| s.source.clone()),
                "model_error": registry.sources.get(p.id()).and_then(|s| s.error.clone()),
            })
        })
        .collect();
    Json(json!({"providers": providers})).into_response()
}

async fn admin_patch_provider(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    if state.provider(&id).is_none() {
        return json_error(StatusCode::NOT_FOUND, "提供商不存在", "not_found");
    }
    {
        let mut store = state.store.write().unwrap();
        let old_base = store
            .data
            .providers
            .get(&id)
            .and_then(|p| p.base_url.clone());
        {
            let settings = store.data.providers.entry(id.clone()).or_default();
            if let Some(enabled) = body.get("enabled").and_then(Value::as_bool) {
                settings.enabled = enabled;
            }
            if let Some(auto) = body.get("auto_checkin").and_then(Value::as_bool) {
                settings.auto_checkin = auto;
            }
            if let Some(base) = body.get("base_url") {
                settings.base_url = base.as_str().filter(|s| !s.is_empty()).map(str::to_string);
            }
        }
        let new_base = store
            .data
            .providers
            .get(&id)
            .and_then(|p| p.base_url.clone());
        if old_base != new_base {
            // A cached catalog from the old Base URL must not leak into the
            // new upstream; refresh will repopulate it.
            store.data.model_cache.remove(&id);
        }
        let _ = store.save();
    }
    spawn_registry_refresh(state.clone());
    Json(json!({"ok": true})).into_response()
}

fn spawn_registry_refresh(state: Arc<AppState>) {
    tokio::spawn(async move {
        let registry = refresh_registry(&state).await;
        *state.registry.write().unwrap() = registry;
    });
}

async fn admin_models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    let registry = state.registry.read().unwrap();
    let models: Vec<Value> = registry
        .entries()
        .iter()
        .map(|e| {
            json!({
                "id": e.exposed_id,
                "provider": e.variants.first().map(|v| v.provider.clone()).unwrap_or_default(),
                "providers": e.providers(),
                "upstream_id": e.variants.first().map(|v| v.model.upstream_id.clone()).unwrap_or_default(),
                "upstream_ids": e.variants.iter().map(|v| v.model.upstream_id.clone()).collect::<Vec<_>>(),
                "name": e.name,
                "context_length": e.context_length,
                "reasoning": e.reasoning,
                "vision": e.vision,
                "merged": e.variants.len() > 1,
                "variants": e.variants.iter().map(|v| json!({
                    "provider": v.provider.clone(),
                    "upstream_id": v.model.upstream_id.clone(),
                    "id": v.model.id.clone(),
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    Json(json!({
        "models": models,
        "refreshed_at": registry.refreshed_at(),
        "used_fallback": registry.used_fallback,
    }))
    .into_response()
}

async fn admin_refresh_models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    let registry = refresh_registry(&state).await;
    let count = registry.entries().len();
    *state.registry.write().unwrap() = registry;
    Json(json!({"ok": true, "models": count})).into_response()
}

fn settings_json(state: &AppState) -> Value {
    let store = state.store.read().unwrap();
    let s = &store.data.settings;
    let chars: Vec<char> = s.api_key.chars().collect();
    let head: String = chars.iter().take(6).collect();
    let tail: String = chars.iter().skip(chars.len().saturating_sub(4)).collect();
    let masked = format!("{head}…{tail}");
    json!({
        "host": s.host,
        "port": s.port,
        "api_key_masked": masked,
        "auto_checkin": s.auto_checkin,
        "checkin_time": s.checkin_time,
        "request_timeout_secs": s.request_timeout_secs,
        "last_auto_checkin": store.data.last_auto_checkin,
    })
}

async fn admin_settings(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    Json(settings_json(&state)).into_response()
}

async fn admin_patch_settings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if !authorized(&headers, &state) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "无效的 API Key",
            "authentication_error",
        );
    }
    {
        let mut store = state.store.write().unwrap();
        let s = &mut store.data.settings;
        if let Some(v) = body.get("auto_checkin").and_then(Value::as_bool) {
            s.auto_checkin = v;
        }
        if let Some(v) = body.get("checkin_time").and_then(Value::as_str) {
            if valid_hhmm(v) {
                s.checkin_time = v.to_string();
            } else {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "checkin_time 必须是 HH:MM",
                    "invalid_request_error",
                );
            }
        }
        if let Some(v) = body.get("request_timeout_secs").and_then(Value::as_u64) {
            s.request_timeout_secs = v.clamp(10, 3600);
        }
        let _ = store.save();
    }
    Json(settings_json(&state)).into_response()
}

fn valid_hhmm(value: &str) -> bool {
    let Some((h, m)) = value.split_once(':') else {
        return false;
    };
    matches!((h.parse::<u32>(), m.parse::<u32>()), (Ok(h), Ok(m)) if h < 24 && m < 60)
}

// ---------------------------------------------------------------------------
// Background scheduler
// ---------------------------------------------------------------------------

pub fn spawn_scheduler(state: Arc<AppState>) {
    // Prime balances shortly after startup so routing can use them.
    {
        let state = state.clone();
        tokio::spawn(async move {
            let snapshot = fetch_credits(&state).await;
            *state.credits.lock().await = Some(snapshot);
        });
    }
    tokio::spawn(async move {
        let mut last_model_refresh = 0i64;
        let mut last_credit_refresh = gateway_core::now_ms();
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            let now = chrono::Local::now();
            let auto = {
                let store = state.store.read().unwrap();
                store.data.settings.auto_checkin
            };
            if auto {
                let (due, done_today) = {
                    let store = state.store.read().unwrap();
                    let time = store.data.settings.checkin_time.clone();
                    let due = now.format("%H:%M").to_string() >= time;
                    let done = store.data.last_auto_checkin.as_deref()
                        == Some(gateway_core::today_local().as_str());
                    (due, done)
                };
                if due && !done_today {
                    tracing::info!("running scheduled check-in");
                    match run_checkin(&state, None, true).await {
                        Ok(results) => {
                            tracing::info!(count = results.len(), "scheduled check-in finished")
                        }
                        Err(e) => tracing::warn!(error = %e, "scheduled check-in failed"),
                    }
                    let mut store = state.store.write().unwrap();
                    store.data.last_auto_checkin = Some(gateway_core::today_local());
                    let _ = store.save();
                }
            }
            if gateway_core::now_ms() - last_model_refresh > 3_600_000 {
                last_model_refresh = gateway_core::now_ms();
                let registry = refresh_registry(&state).await;
                *state.registry.write().unwrap() = registry;
            }
            if gateway_core::now_ms() - last_credit_refresh > 300_000 {
                last_credit_refresh = gateway_core::now_ms();
                let snapshot = fetch_credits(&state).await;
                *state.credits.lock().await = Some(snapshot);
            }
        }
    });
}

#[cfg(test)]
mod routing_tests {
    use super::*;
    use axum::http::HeaderValue;

    fn request(messages: Value) -> Value {
        serde_json::json!({"model": "glm-5.3", "messages": messages})
    }

    #[test]
    fn session_key_is_stable_across_appended_turns() {
        let headers = HeaderMap::new();
        let first = request(serde_json::json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello"}
        ]));
        let later = request(serde_json::json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"}
        ]));
        let other = request(serde_json::json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "different"}
        ]));
        let key = session_key(&headers, &first, "glm-5.3");
        assert_eq!(key, session_key(&headers, &later, "glm-5.3"));
        assert_ne!(key, session_key(&headers, &other, "glm-5.3"));
    }

    #[test]
    fn explicit_session_header_wins() {
        let mut headers = HeaderMap::new();
        headers.insert("x-session-id", HeaderValue::from_static("abc"));
        let key = session_key(&headers, &request(serde_json::json!([])), "glm-5.3");
        assert_eq!(key, "glm-5.3:explicit:abc");
    }

    #[test]
    fn records_model_time_window_limits() {
        let err = ProviderError::RateLimited {
            message: "slow down".into(),
            retry_after_ms: 120_000,
        };
        assert_eq!(limit_from_error(&err).unwrap().0, 120_000);
        let err = ProviderError::BadRequest("daily limit exceeded".into());
        assert!(limit_from_error(&err).is_some());
        let err = ProviderError::BadRequest("bad message".into());
        assert!(limit_from_error(&err).is_none());
    }

    #[test]
    fn credit_score_ignores_other_units() {
        let snapshot = Some(CreditSnapshot {
            at: gateway_core::now_ms(),
            total: Some(150.0),
            monthly_quota: None,
            total_quota: None,
            unit: Some("credits".into()),
            units: vec!["credits".into()],
            accounts: vec![
                serde_json::json!({"provider":"qoder","credits":100.0,"unit":"credits"}),
                serde_json::json!({"provider":"qoder","credits":50.0,"unit":"tokens"}),
                serde_json::json!({"provider":"workbuddy","credits":0.0,"unit":"credits"}),
            ],
        });
        assert_eq!(provider_credit_score(&snapshot, "qoder"), 100.0);
        assert_eq!(provider_credit_score(&snapshot, "workbuddy"), 0.0);
        assert_eq!(provider_credit_score(&None, "qoder"), 0.0);
    }
    fn routing_state(dir: &std::path::Path) -> AppState {
        let store = Store::load(dir, &[]).unwrap();
        AppState {
            store: Arc::new(RwLock::new(store)),
            providers: Vec::new(),
            registry: Arc::new(RwLock::new(ModelRegistry::default())),
            client: reqwest::Client::new(),
            started_at: gateway_core::now_ms(),
            cursors: tokio::sync::Mutex::new(HashMap::new()),
            logins: tokio::sync::Mutex::new(HashMap::new()),
            stats: Arc::new(StatsStore::load(dir)),
            credits: tokio::sync::Mutex::new(None),
            model_limits: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            session_pins: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    #[tokio::test]
    async fn routing_prefers_credits_then_session_pin_then_limits() {
        let dir = std::env::temp_dir().join(format!(
            "apip-routing-test-{}-{}",
            std::process::id(),
            gateway_core::now_ms()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let state = routing_state(&dir);
        let variants = vec![
            ("workbuddy".to_string(), "glm-5.3".to_string()),
            ("qoder".to_string(), "gmodel".to_string()),
        ];
        *state.credits.lock().await = Some(CreditSnapshot {
            at: gateway_core::now_ms(),
            total: Some(100.0),
            monthly_quota: None,
            total_quota: None,
            unit: Some("credits".into()),
            units: vec!["credits".into()],
            accounts: vec![
                serde_json::json!({"provider":"workbuddy","credits":0.0,"unit":"credits"}),
                serde_json::json!({"provider":"qoder","credits":100.0,"unit":"credits"}),
            ],
        });

        let ordered = order_variants(&state, &variants, "session-a")
            .await
            .unwrap();
        assert_eq!(ordered[0].0, "qoder");

        state.session_pins.lock().await.insert(
            "session-a".into(),
            SessionPin {
                provider: "workbuddy".into(),
                upstream_model: "glm-5.3".into(),
                at: gateway_core::now_ms(),
            },
        );
        let ordered = order_variants(&state, &variants, "session-a")
            .await
            .unwrap();
        assert_eq!(ordered[0].0, "workbuddy");

        state.model_limits.lock().await.insert(
            limit_key("workbuddy", "glm-5.3"),
            ModelLimit {
                until_ms: gateway_core::now_ms() + 60_000,
                reason: "daily limit".into(),
            },
        );
        let ordered = order_variants(&state, &variants, "session-a")
            .await
            .unwrap();
        assert_eq!(ordered.len(), 1);
        assert_eq!(ordered[0].0, "qoder");

        state.model_limits.lock().await.insert(
            limit_key("qoder", "gmodel"),
            ModelLimit {
                until_ms: gateway_core::now_ms() + 60_000,
                reason: "quota".into(),
            },
        );
        assert!(order_variants(&state, &variants, "session-a")
            .await
            .is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
    struct FakeProvider;

    #[async_trait::async_trait]
    impl Provider for FakeProvider {
        fn id(&self) -> &'static str {
            "fake"
        }
        fn name(&self) -> &'static str {
            "Fake"
        }
        fn default_base_url(&self) -> &'static str {
            "http://localhost"
        }
        fn fallback_models(&self) -> Vec<RemoteModel> {
            Vec::new()
        }
        async fn models(
            &self,
            _client: &reqwest::Client,
            _account: &gateway_core::AccountView<'_>,
        ) -> Result<Vec<RemoteModel>, ProviderError> {
            Ok(Vec::new())
        }
        async fn chat(
            &self,
            _client: &reqwest::Client,
            account: &gateway_core::AccountView<'_>,
            _request: &Value,
            _upstream_model: &str,
        ) -> Result<gateway_core::ChatStream, ProviderError> {
            if account.credential == "bad" {
                Err(ProviderError::RateLimited {
                    message: "per-account limit".into(),
                    retry_after_ms: 60_000,
                })
            } else if account.credential == "plan" {
                Err(ProviderError::BadRequest(
                    "该模型当前不可用或需要订阅".into(),
                ))
            } else {
                Ok(Box::pin(futures::stream::empty()))
            }
        }
        async fn checkin(
            &self,
            _client: &reqwest::Client,
            _account: &gateway_core::AccountView<'_>,
        ) -> Result<gateway_core::CheckinOutcome, ProviderError> {
            Ok(gateway_core::CheckinOutcome {
                status: "skipped".into(),
                message: "test".into(),
                reward: None,
            })
        }
    }

    #[tokio::test]
    async fn per_account_limit_does_not_poison_provider_when_next_account_succeeds() {
        let dir = std::env::temp_dir().join(format!(
            "apip-limit-test-{}-{}",
            std::process::id(),
            gateway_core::now_ms()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut base = routing_state(&dir);
        let provider: Arc<dyn Provider> = Arc::new(FakeProvider);
        base.providers = vec![provider.clone()];
        let state = Arc::new(base);
        let accounts = vec![
            Account::new("fake", "bad", "bad"),
            Account::new("fake", "good", "good"),
        ];
        let result = open_stream(
            &state,
            provider.clone(),
            accounts,
            &serde_json::json!({}),
            "m",
        )
        .await;
        assert!(result.is_ok());
        assert!(state.model_limits.lock().await.is_empty());

        let plan_then_good = vec![
            Account::new("fake", "plan", "plan"),
            Account::new("fake", "good2", "good"),
        ];
        let result = open_stream(
            &state,
            provider.clone(),
            plan_then_good,
            &serde_json::json!({}),
            "m",
        )
        .await;
        assert!(result.is_ok());
        assert!(state.model_limits.lock().await.is_empty());

        let all_bad = vec![
            Account::new("fake", "bad", "bad"),
            Account::new("fake", "bad2", "bad"),
        ];
        let result = open_stream(
            &state,
            provider.clone(),
            all_bad,
            &serde_json::json!({}),
            "m",
        )
        .await;
        assert!(result.is_err());
        assert!(state
            .model_limits
            .lock()
            .await
            .contains_key(&limit_key("fake", "m")));

        state.model_limits.lock().await.clear();
        let all_plan = vec![
            Account::new("fake", "plan", "plan"),
            Account::new("fake", "plan2", "plan"),
        ];
        let result = open_stream(&state, provider, all_plan, &serde_json::json!({}), "m").await;
        assert!(result.is_err());
        assert!(state
            .model_limits
            .lock()
            .await
            .contains_key(&limit_key("fake", "m")));
        let _ = std::fs::remove_dir_all(dir);
    }
}
