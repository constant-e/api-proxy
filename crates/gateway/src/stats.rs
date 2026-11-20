//! Request accounting and a bounded, persistent request log.

use std::collections::VecDeque;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MAX_LOGS: usize = 1_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestLog {
    pub id: u64,
    pub at: i64,
    pub model: String,
    #[serde(default)]
    pub upstream_model: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub account: Option<String>,
    pub stream: bool,
    pub status: String,
    pub http_status: u16,
    pub duration_ms: i64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    #[serde(default)]
    pub cached_tokens: u64,
    #[serde(default)]
    pub credits: Option<f64>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Totals {
    pub requests: u64,
    pub errors: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cached_tokens: u64,
}

struct Inner {
    logs: VecDeque<RequestLog>,
    totals: Totals,
    next_id: u64,
}

pub struct StatsStore {
    path: PathBuf,
    inner: Mutex<Inner>,
}

impl StatsStore {
    pub fn load(data_dir: &Path) -> Self {
        let _ = fs::create_dir_all(data_dir);
        let path = data_dir.join("requests.jsonl");
        let mut inner = Inner {
            logs: VecDeque::new(),
            totals: Totals::default(),
            next_id: 1,
        };
        if let Ok(file) = fs::File::open(&path) {
            for line in BufReader::new(file).lines().map_while(Result::ok) {
                let Ok(log) = serde_json::from_str::<RequestLog>(&line) else {
                    continue;
                };
                inner.next_id = inner.next_id.max(log.id + 1);
                add_totals(&mut inner.totals, &log);
                inner.logs.push_back(log);
                while inner.logs.len() > MAX_LOGS {
                    inner.logs.pop_front();
                }
            }
        }
        Self {
            path,
            inner: Mutex::new(inner),
        }
    }

    pub fn record(&self, mut log: RequestLog) -> u64 {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        log.id = inner.next_id;
        inner.next_id += 1;
        if log.at == 0 {
            log.at = gateway_core::now_ms();
        }
        add_totals(&mut inner.totals, &log);
        inner.logs.push_back(log.clone());
        while inner.logs.len() > MAX_LOGS {
            inner.logs.pop_front();
        }
        let id = log.id;
        drop(inner);
        // Do not persist unauthenticated requests; a local port probe could
        // otherwise flood requests.jsonl. They still appear in memory/stats.
        if log.http_status != 401 {
            if let Ok(line) = serde_json::to_string(&log) {
                let mut opts = OpenOptions::new();
                opts.create(true).append(true);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    opts.mode(0o600);
                }
                match opts.open(&self.path) {
                    Ok(mut file) => {
                        if let Err(err) = writeln!(file, "{line}") {
                            tracing::warn!(error = %err, "failed to append request log");
                        }
                    }
                    Err(err) => tracing::warn!(error = %err, "failed to open request log"),
                }
            }
        }
        id
    }

    pub fn totals(&self) -> Totals {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .totals
            .clone()
    }

    pub fn total_logs(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .logs
            .len()
    }

    /// Newest first, one page only.
    pub fn page(&self, page: usize, page_size: usize) -> (usize, Vec<RequestLog>) {
        let page = page.max(1);
        let size = page_size.clamp(1, 50);
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let total = inner.logs.len();
        let logs = inner
            .logs
            .iter()
            .rev()
            .skip((page - 1) * size)
            .take(size)
            .cloned()
            .collect();
        (total, logs)
    }

    /// Newest first, capped at `limit` (tests/diagnostics).
    #[cfg(test)]
    pub fn snapshot(&self, limit: usize) -> (Totals, Vec<RequestLog>) {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let logs = inner
            .logs
            .iter()
            .rev()
            .take(limit.min(MAX_LOGS))
            .cloned()
            .collect();
        (inner.totals.clone(), logs)
    }
}

fn add_totals(totals: &mut Totals, log: &RequestLog) {
    totals.requests += 1;
    if log.status != "success" {
        totals.errors += 1;
    }
    totals.prompt_tokens += log.prompt_tokens;
    totals.completion_tokens += log.completion_tokens;
    totals.total_tokens += log.total_tokens;
    totals.cached_tokens += log.cached_tokens;
}

pub fn usage_cached_tokens(usage: &Value) -> u64 {
    usage
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(Value::as_u64)
        .or_else(|| usage.get("cached_tokens").and_then(Value::as_u64))
        .or_else(|| usage.get("prompt_cache_hit_tokens").and_then(Value::as_u64))
        .unwrap_or(0)
}

pub fn usage_credits(usage: &Value) -> Option<f64> {
    ["credits", "credit", "total_credits", "cost", "cost_credits"]
        .iter()
        .find_map(|key| {
            usage.get(*key).and_then(|v| {
                v.as_f64()
                    .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
            })
        })
}

pub fn usage_tokens(usage: &Value) -> (u64, u64, u64) {
    let pick = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
    let prompt = pick("prompt_tokens");
    let completion = pick("completion_tokens");
    let mut total = pick("total_tokens");
    if total == 0 {
        total = prompt + completion;
    }
    (prompt, completion, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log(model: &str, usage: (u64, u64, u64), status: &str) -> RequestLog {
        RequestLog {
            id: 0,
            at: 0,
            model: model.into(),
            upstream_model: None,
            provider: Some("workbuddy".into()),
            account: Some("acct".into()),
            stream: true,
            status: status.into(),
            http_status: 200,
            duration_ms: 12,
            prompt_tokens: usage.0,
            completion_tokens: usage.1,
            total_tokens: usage.2,
            cached_tokens: 1,
            credits: Some(0.5),
            error: None,
        }
    }

    #[test]
    fn records_totals_and_reloads() {
        let dir =
            std::env::temp_dir().join(format!("apiproxy-stats-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let store = StatsStore::load(&dir);
        store.record(log("Hy4", (10, 5, 15), "success"));
        store.record(log("Hy4", (2, 3, 5), "error"));
        let (totals, logs) = store.snapshot(10);
        assert_eq!(totals.requests, 2);
        assert_eq!(totals.errors, 1);
        assert_eq!(totals.total_tokens, 20);
        assert_eq!(totals.cached_tokens, 2);
        assert_eq!(logs.len(), 2);
        drop(store);

        let reloaded = StatsStore::load(&dir);
        let (totals, logs) = reloaded.snapshot(10);
        assert_eq!(totals.requests, 2);
        assert_eq!(logs[0].model, "Hy4");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn extracts_usage() {
        let usage =
            serde_json::json!({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7});
        assert_eq!(usage_tokens(&usage), (3, 4, 7));
        let usage = serde_json::json!({"prompt_tokens": 3, "completion_tokens": 4});
        assert_eq!(usage_tokens(&usage), (3, 4, 7));
    }
}
