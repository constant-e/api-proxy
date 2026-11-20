//! Persistent configuration and account storage.
//!
//! A single JSON file keeps personal deployments simple. Writes are atomic
//! (temp file + rename) and the file is chmod 600 on Unix because it holds
//! upstream credentials.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::provider::RemoteModel;

pub const DEFAULT_PORT: u16 = 8787;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub host: String,
    pub port: u16,
    /// Bearer key clients use for both `/v1` and the admin API.
    pub api_key: String,
    pub auto_checkin: bool,
    /// Local time `HH:MM`.
    pub checkin_time: String,
    pub request_timeout_secs: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: DEFAULT_PORT,
            api_key: generate_api_key(),
            auto_checkin: true,
            checkin_time: "08:00".into(),
            request_timeout_secs: 180,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderSettings {
    pub enabled: bool,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default = "default_true")]
    pub auto_checkin: bool,
}

fn default_true() -> bool {
    true
}

impl Default for ProviderSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: None,
            auto_checkin: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckinRecord {
    pub date: String,
    pub status: String,
    pub message: String,
    #[serde(default)]
    pub reward: Option<f64>,
    pub at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub id: String,
    pub provider: String,
    pub name: String,
    /// Upstream API key / OAuth token / Qoder PAT. Never returned by the API.
    pub credential: String,
    /// OAuth refresh token, when the login flow supplied one.
    #[serde(default)]
    pub refresh_token: Option<String>,
    /// Credential expiry in epoch milliseconds (0/None = unknown).
    #[serde(default)]
    pub expires_at: Option<i64>,
    /// Stable upstream account identity (user id / email) when known.
    #[serde(default)]
    pub remote_id: Option<String>,
    /// Provider-specific region (`cn` / `global` for Qoder).
    #[serde(default)]
    pub region: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    pub enabled: bool,
    pub created_at: i64,
    #[serde(default)]
    pub cooldown_until: Option<i64>,
    #[serde(default)]
    pub disabled_reason: Option<String>,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub last_checkin: Option<CheckinRecord>,
}

impl Account {
    pub fn new(provider: &str, name: &str, credential: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string()[..8].to_string(),
            provider: provider.to_string(),
            name: name.to_string(),
            credential: credential.trim().to_string(),
            refresh_token: None,
            expires_at: None,
            remote_id: None,
            region: None,
            base_url: None,
            enabled: true,
            created_at: now_ms(),
            cooldown_until: None,
            disabled_reason: None,
            last_error: None,
            last_checkin: None,
        }
    }

    pub fn available(&self, now: i64) -> bool {
        self.enabled && self.cooldown_until.map(|t| t <= now).unwrap_or(true)
    }

    /// Last four characters, for display only.
    pub fn credential_hint(&self) -> String {
        let chars: Vec<char> = self.credential.trim().chars().collect();
        if chars.len() <= 4 {
            "****".into()
        } else {
            format!("…{}", chars[chars.len() - 4..].iter().collect::<String>())
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCacheEntry {
    pub models: Vec<RemoteModel>,
    pub at: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StoreFile {
    pub settings: Settings,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderSettings>,
    #[serde(default)]
    pub accounts: Vec<Account>,
    /// `YYYY-MM-DD` of the last scheduler-driven check-in run.
    #[serde(default)]
    pub last_auto_checkin: Option<String>,
    /// Last successfully discovered model catalog per provider. Survives
    /// restarts and upstream `/v3/config` throttling (which sometimes returns
    /// a minimal payload with no models).
    #[serde(default)]
    pub model_cache: BTreeMap<String, ModelCacheEntry>,
}

pub struct Store {
    pub path: PathBuf,
    pub data: StoreFile,
}

impl Store {
    /// Load `data.json`, creating defaults and provider entries on first run.
    pub fn load(dir: &Path, providers: &[&str]) -> anyhow::Result<Self> {
        fs::create_dir_all(dir)?;
        let path = dir.join("data.json");
        let mut data = if path.exists() {
            let raw = fs::read(&path)?;
            serde_json::from_slice::<StoreFile>(&raw)
                .map_err(|e| anyhow::anyhow!("invalid {}: {e}", path.display()))?
        } else {
            StoreFile::default()
        };
        for p in providers {
            data.providers.entry((*p).into()).or_default();
        }
        let store = Self { path, data };
        store.save()?;
        Ok(store)
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let tmp = self.path.with_extension("json.tmp");
        let raw = serde_json::to_vec_pretty(&self.data)?;
        {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(&raw)?;
            f.sync_all()?;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600));
        }
        fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    pub fn provider_settings(&self, id: &str) -> ProviderSettings {
        self.data.providers.get(id).cloned().unwrap_or_default()
    }

    pub fn cached_models(&self, provider: &str) -> Option<ModelCacheEntry> {
        self.data.model_cache.get(provider).cloned()
    }

    pub fn cache_models(&mut self, provider: &str, models: Vec<RemoteModel>) -> bool {
        if models.is_empty() {
            return false;
        }
        self.data.model_cache.insert(
            provider.to_string(),
            ModelCacheEntry {
                models,
                at: now_ms(),
            },
        );
        true
    }

    pub fn account(&self, id: &str) -> Option<&Account> {
        self.data.accounts.iter().find(|a| a.id == id)
    }

    pub fn accounts_for(&self, provider: &str) -> Vec<Account> {
        self.data
            .accounts
            .iter()
            .filter(|a| a.provider == provider)
            .cloned()
            .collect()
    }

    pub fn remove_account(&mut self, id: &str) -> anyhow::Result<bool> {
        let before = self.data.accounts.len();
        self.data.accounts.retain(|a| a.id != id);
        let removed = self.data.accounts.len() != before;
        if removed {
            self.save()?;
        }
        Ok(removed)
    }

    pub fn record_failure(&mut self, id: &str, error: &str, cooldown_ms: Option<i64>) {
        if let Some(a) = self.data.accounts.iter_mut().find(|a| a.id == id) {
            a.last_error = Some(error.chars().take(300).collect());
            if let Some(ms) = cooldown_ms {
                a.cooldown_until = Some(now_ms() + ms);
                if ms >= 24 * 3600 * 1000 {
                    a.enabled = false;
                    a.disabled_reason = Some("上游拒绝或额度耗尽，已自动停用".into());
                }
            }
            let _ = self.save();
        }
    }

    /// Startup migration: clear stale check-in errors that were misclassified
    /// before tolerant 4xx/code-10001 handling existed. If today's check-in is
    /// already recorded, only clear the error/cooldown; otherwise also write an
    /// `already` record so the console reflects reality without re-checking in.
    pub fn migrate_stale_checkin_errors(&mut self) -> bool {
        let today = today_local();
        let now = now_ms();
        let mut changed = false;
        for account in &mut self.data.accounts {
            if !matches!(account.provider.as_str(), "workbuddy" | "qoder") {
                continue;
            }
            let done_today = account
                .last_checkin
                .as_ref()
                .map(|r| {
                    r.date == today
                        && matches!(r.status.as_str(), "success" | "already" | "skipped")
                })
                .unwrap_or(false);
            let already_error = account
                .last_error
                .as_deref()
                .map(is_already_checkin_error)
                .unwrap_or(false);
            if !done_today && !already_error {
                continue;
            }
            if account.last_error.is_some() {
                account.last_error = None;
                changed = true;
            }
            if account.cooldown_until.is_some() {
                account.cooldown_until = None;
                changed = true;
            }
            if !done_today {
                account.last_checkin = Some(CheckinRecord {
                    date: today.clone(),
                    status: "already".into(),
                    message: "今日已签到（启动迁移）".into(),
                    reward: None,
                    at: now,
                });
                changed = true;
            }
        }
        changed
    }

    pub fn record_success(&mut self, id: &str) {
        if let Some(a) = self.data.accounts.iter_mut().find(|a| a.id == id) {
            a.last_error = None;
            a.cooldown_until = None;
        }
    }
}

pub fn data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("APIPROXY_DATA_DIR") {
        return PathBuf::from(dir);
    }
    #[cfg(target_os = "linux")]
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".local/share/apiproxy");
    }
    #[cfg(target_os = "macos")]
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join("Library/Application Support/apiproxy");
    }
    PathBuf::from(".apiproxy")
}

fn is_already_checkin_error(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    lower.contains("10001")
        || error.contains("已签到")
        || error.contains("今天已签到")
        || lower.contains("already checked in")
        || lower.contains("already signed")
        || lower.contains("already claimed")
}

pub fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

pub fn today_local() -> String {
    chrono::Local::now().format("%Y-%m-%d").to_string()
}

fn generate_api_key() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 24];
    rand::thread_rng().fill_bytes(&mut bytes);
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("sk-{hex}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caches_last_good_catalog() {
        let mut store = Store {
            path: PathBuf::from("/tmp/apiproxy-cache-test.json"),
            data: StoreFile::default(),
        };
        let models = vec![RemoteModel::simple("hy4std")];
        assert!(store.cache_models("workbuddy", models));
        let cached = store.cached_models("workbuddy").unwrap();
        assert_eq!(cached.models.len(), 1);
        assert_eq!(cached.models[0].id, "hy4std");
        assert!(!store.cache_models("workbuddy", Vec::new()));
    }

    #[test]
    fn migrates_stale_already_checkin_error() {
        let mut store = Store {
            path: PathBuf::from("/tmp/apiproxy-migration-test.json"),
            data: StoreFile::default(),
        };
        let mut account = Account::new("workbuddy", "wb", "eyJ.fake.token");
        account.last_error = Some(
            "请求被拒绝: HTTP 400: {\"code\":10001,\"msg\":\"今天已签到，请明天再来\"}".into(),
        );
        account.cooldown_until = Some(now_ms() + 3_600_000);
        store.data.accounts.push(account);

        assert!(store.migrate_stale_checkin_errors());
        let account = &store.data.accounts[0];
        assert!(account.last_error.is_none());
        assert!(account.cooldown_until.is_none());
        let record = account.last_checkin.as_ref().unwrap();
        assert_eq!(record.status, "already");
        assert_eq!(record.date, today_local());

        // Idempotent: nothing changes on the second pass.
        assert!(!store.migrate_stale_checkin_errors());
    }
}
