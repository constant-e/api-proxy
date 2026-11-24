//! `apip` CLI: starts the gateway in the background and manages it.

mod server;
mod stats;

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use gateway_core::{ModelRegistry, Provider, Store};
use serde_json::{json, Value};

use server::AppState;

#[derive(Parser)]
#[command(
    name = "apip",
    version,
    about = "将国产 AI 平台封装为 OpenAI Completions 协议的本地网关"
)]
struct Cli {
    /// 数据目录（默认 ~/.local/share/apiproxy）
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,
    /// 覆盖服务地址，例如 http://127.0.0.1:8787
    #[arg(long = "server", global = true)]
    server: Option<String>,
    /// 覆盖 API Key（默认读取 data.json）
    #[arg(long, global = true)]
    api_key: Option<String>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// 后台启动本地网关与网页 Console
    Serve {
        #[arg(long)]
        port: Option<u16>,
        /// 前台运行（供 systemd/docker 使用，会阻塞终端）
        #[arg(long)]
        foreground: bool,
    },
    /// 停止后台网关
    Stop,
    /// 查看后台网关状态
    Status,
    /// 管理上游账号
    Accounts {
        #[command(subcommand)]
        cmd: AccountsCmd,
    },
    /// 查看合并后的模型列表
    Models {
        /// 立即刷新上游模型
        #[arg(long)]
        refresh: bool,
    },
    /// 浏览器授权登录并自动添加账号
    Login {
        #[arg(long)]
        provider: String,
        /// Qoder / WorkBuddy: cn / global
        #[arg(long)]
        region: Option<String>,
        /// 可选：账号名称，默认使用上游返回的昵称
        #[arg(long)]
        name: Option<String>,
        /// 不自动打开浏览器，只打印授权链接
        #[arg(long)]
        no_open: bool,
    },
    /// 查看已加载的提供商
    Providers,
    /// 立即执行签到
    Checkin {
        /// 只签到指定账号 ID
        #[arg(long)]
        account: Option<String>,
    },
    /// 查看可编辑设置
    Settings,
    /// 查看本机 API Key
    Key,
    /// 用网关发一条对话请求
    Chat {
        model: String,
        message: String,
        #[arg(long, default_value = "You are a helpful assistant.")]
        system: String,
    },
}

#[derive(Subcommand)]
enum AccountsCmd {
    List,
    Remove {
        id: String,
    },
    Enable {
        id: String,
    },
    Disable {
        id: String,
    },
    /// 测试凭证并返回账号信息
    Test {
        id: String,
    },
}

fn providers() -> Vec<Arc<dyn Provider>> {
    vec![
        Arc::new(provider_workbuddy::WorkBuddyProvider::new()),
        Arc::new(provider_qoder::QoderProvider::new()),
    ]
}

fn provider_ids() -> Vec<&'static str> {
    providers().iter().map(|p| p.id()).collect()
}

fn data_dir(cli: &Cli) -> PathBuf {
    cli.data_dir.clone().unwrap_or_else(gateway_core::data_dir)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,hyper=warn,reqwest=warn".into()),
        )
        .with_target(false)
        .init();

    let mut cli = Cli::parse();
    let command = cli.command.take();
    match command {
        None => start_background(cli, None).await,
        Some(Command::Serve { port, foreground }) => {
            if foreground {
                serve(cli, port).await
            } else {
                start_background(cli, port).await
            }
        }
        Some(Command::Stop) => stop(cli).await,
        Some(Command::Status) => status(cli).await,
        Some(Command::Accounts { cmd }) => accounts(cli, cmd).await,
        Some(Command::Models { refresh }) => models(cli, refresh).await,
        Some(Command::Login {
            provider,
            region,
            name,
            no_open,
        }) => login(cli, provider, region, name, no_open).await,
        Some(Command::Providers) => list_providers(cli).await,
        Some(Command::Checkin { account }) => checkin(cli, account).await,
        Some(Command::Settings) => settings(cli).await,
        Some(Command::Key) => {
            let store = load_store(&cli)?;
            println!("{}", store.data.settings.api_key);
            Ok(())
        }
        Some(Command::Chat {
            model,
            message,
            system,
        }) => chat(cli, model, message, system).await,
    }
}

fn load_store(cli: &Cli) -> Result<Store> {
    let dir = data_dir(cli);
    Store::load(&dir, &provider_ids())
}

async fn serve(cli: Cli, port_override: Option<u16>) -> Result<()> {
    let dir = data_dir(&cli);
    let mut store = Store::load(&dir, &provider_ids())?;
    let port_changed = port_override
        .map(|p| p != store.data.settings.port)
        .unwrap_or(false);
    if let Some(port) = port_override {
        store.data.settings.port = port;
    }
    let all_providers = providers();
    // Backfill stable identities for accounts created before remote_id
    // existed, so a later browser login updates instead of duplicating.
    let mut changed = false;
    for account in &mut store.data.accounts {
        if account.remote_id.is_none() {
            if let Some(provider) = all_providers.iter().find(|p| p.id() == account.provider) {
                if let Some(identity) = provider.credential_identity(&account.credential) {
                    account.remote_id = Some(identity);
                    changed = true;
                }
            }
        }
    }
    if store.migrate_stale_checkin_errors() {
        changed = true;
    }
    if changed || port_changed {
        store.save()?;
    }
    let settings = store.data.settings.clone();
    let key = settings.api_key.clone();
    let host = settings.host.clone();
    let port = port_override.unwrap_or(settings.port);

    // A total request timeout would truncate long SSE streams. Bound the
    // connect phase and each read instead, which keeps idle connections from
    // hanging while allowing arbitrarily long, data-producing completions.
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .read_timeout(std::time::Duration::from_secs(
            settings.request_timeout_secs.max(10),
        ))
        .build()?;
    let state = Arc::new(AppState {
        store: Arc::new(RwLock::new(store)),
        providers: all_providers,
        registry: Arc::new(RwLock::new(ModelRegistry::default())),
        client,
        started_at: gateway_core::now_ms(),
        cursors: tokio::sync::Mutex::new(Default::default()),
        logins: tokio::sync::Mutex::new(Default::default()),
        stats: Arc::new(crate::stats::StatsStore::load(&dir)),
        credits: tokio::sync::Mutex::new(None),
        model_limits: Arc::new(tokio::sync::Mutex::new(Default::default())),
        session_pins: Arc::new(tokio::sync::Mutex::new(Default::default())),
    });

    let registry = server::refresh_registry(&state).await;
    *state.registry.write().unwrap() = registry;
    server::spawn_scheduler(state.clone());

    let app = server::router(state.clone());
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("无法监听 {addr}"))?;
    let pid_path = dir.join("apip.pid");
    let _ = std::fs::write(&pid_path, std::process::id().to_string());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&pid_path, std::fs::Permissions::from_mode(0o600));
    }
    let shown_host = if host == "0.0.0.0" || host == "::" {
        "127.0.0.1"
    } else {
        &host
    };
    println!("APIProxy 已启动");
    println!("  Console : http://{shown_host}:{port}/");
    println!("  API     : http://{shown_host}:{port}/v1");
    println!("  API Key : {key}");
    println!("  数据目录: {}", dir.display());
    tracing::info!(%addr, "listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    if std::fs::read_to_string(&pid_path)
        .map(|p| p.trim() == std::process::id().to_string())
        .unwrap_or(false)
    {
        let _ = std::fs::remove_file(&pid_path);
    }
    Ok(())
}

fn local_base(settings: &gateway_core::Settings) -> String {
    let host = if settings.host == "0.0.0.0" || settings.host == "::" {
        "127.0.0.1"
    } else {
        &settings.host
    };
    format!("http://{host}:{}", settings.port)
}

async fn health_ok(base: &str) -> bool {
    reqwest::Client::new()
        .get(format!("{base}/health"))
        .timeout(std::time::Duration::from_millis(800))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

fn tail_log(path: &std::path::Path, lines: usize) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .rev()
        .take(lines)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n")
}

async fn start_background(cli: Cli, port_override: Option<u16>) -> Result<()> {
    let requested_dir = data_dir(&cli);
    let mut store = Store::load(&requested_dir, &provider_ids())?;
    // Store::load creates the directory. Canonicalize before spawning so a
    // relative --data-dir is not resolved twice (once by the child's
    // current_dir and again by --data-dir).
    let dir = std::fs::canonicalize(&requested_dir)
        .with_context(|| format!("无法解析数据目录 {}", requested_dir.display()))?;
    if let Some(port) = port_override {
        if store.data.settings.port != port {
            store.data.settings.port = port;
            store.save()?;
        }
    }
    let settings = store.data.settings.clone();
    let base = local_base(&settings);
    let key = settings.api_key.clone();
    if health_ok(&base).await {
        println!("网关已在运行");
        println!("  Console : {base}/");
        println!("  API     : {base}/v1");
        println!("  API Key : {key}");
        return Ok(());
    }

    let log_path = dir.join("apip.log");
    let _ = std::fs::create_dir_all(&dir);
    let log = {
        let mut opts = std::fs::OpenOptions::new();
        opts.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        opts.open(&log_path)
            .with_context(|| format!("无法打开日志 {}", log_path.display()))?
    };
    let stderr = log.try_clone()?;
    let exe = std::env::current_exe()?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--data-dir")
        .arg(&dir)
        .arg("serve")
        .arg("--foreground")
        .current_dir(&dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(log))
        .stderr(std::process::Stdio::from(stderr));
    #[cfg(unix)]
    unsafe {
        use std::os::unix::process::CommandExt;
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = cmd.spawn().context("无法启动后台网关")?;
    let pid = child.id();
    let pid_path = dir.join("apip.pid");
    let _ = std::fs::write(&pid_path, pid.to_string());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&pid_path, std::fs::Permissions::from_mode(0o600));
    }

    for _ in 0..75 {
        if health_ok(&base).await {
            println!("网关已在后台启动");
            println!("  Console : {base}/");
            println!("  API     : {base}/v1");
            println!("  API Key : {key}");
            println!("  PID     : {pid}");
            println!("  日志    : {}", log_path.display());
            return Ok(());
        }
        if let Ok(Some(code)) = child.try_wait() {
            anyhow::bail!(
                "后台网关启动失败（退出码 {code}）:\n{}",
                tail_log(&log_path, 20)
            );
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
    let _ = child.kill();
    anyhow::bail!("后台网关启动超时:\n{}", tail_log(&log_path, 20));
}

async fn status(cli: Cli) -> Result<()> {
    let dir = data_dir(&cli);
    let store = Store::load(&dir, &provider_ids())?;
    let base = local_base(&store.data.settings);
    let pid = std::fs::read_to_string(dir.join("apip.pid"))
        .ok()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty());
    if health_ok(&base).await {
        println!("状态: 运行中");
        println!("  Console : {base}/");
        println!("  API     : {base}/v1");
        if let Some(pid) = pid {
            println!("  PID     : {pid}");
        }
    } else {
        println!("状态: 未运行");
        println!("  地址: {base}");
    }
    Ok(())
}

async fn stop(cli: Cli) -> Result<()> {
    let dir = data_dir(&cli);
    let store = Store::load(&dir, &provider_ids())?;
    let base = local_base(&store.data.settings);
    let pid_path = dir.join("apip.pid");
    let pid = std::fs::read_to_string(&pid_path)
        .ok()
        .and_then(|p| p.trim().parse::<i32>().ok());
    let Some(pid) = pid else {
        if health_ok(&base).await {
            anyhow::bail!(
                "网关在运行，但没有 PID 文件；请手动停止或使用 --foreground 启动的服务管理器"
            );
        }
        println!("网关未运行");
        return Ok(());
    };
    #[cfg(unix)]
    unsafe {
        if libc::kill(pid, libc::SIGTERM) == -1 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::ESRCH) {
                return Err(err).context("无法停止后台网关");
            }
        }
    }
    #[cfg(not(unix))]
    anyhow::bail!("当前平台不支持 apip stop");
    for _ in 0..50 {
        if !health_ok(&base).await {
            let _ = std::fs::remove_file(&pid_path);
            println!("网关已停止");
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
    anyhow::bail!("已发送停止信号，但网关仍未退出；请检查 PID {pid}")
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut terminate =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(signal) => signal,
                Err(_) => {
                    let _ = tokio::signal::ctrl_c().await;
                    tracing::info!("shutting down");
                    return;
                }
            };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
    tracing::info!("shutting down");
}

fn admin_base(cli: &Cli) -> Result<String> {
    if let Some(base) = &cli.server {
        return Ok(base.trim_end_matches('/').to_string());
    }
    let store = load_store(cli)?;
    let host = if store.data.settings.host == "0.0.0.0" || store.data.settings.host == "::" {
        "127.0.0.1".to_string()
    } else {
        store.data.settings.host.clone()
    };
    Ok(format!("http://{host}:{}", store.data.settings.port))
}

fn api_key(cli: &Cli) -> Result<String> {
    if let Some(key) = &cli.api_key {
        return Ok(key.clone());
    }
    if let Ok(key) = std::env::var("APIPROXY_API_KEY") {
        if !key.is_empty() {
            return Ok(key);
        }
    }
    Ok(load_store(cli)?.data.settings.api_key.clone())
}

async fn api(cli: &Cli, method: reqwest::Method, path: &str, body: Option<Value>) -> Result<Value> {
    let base = admin_base(cli)?;
    let key = api_key(cli)?;
    let client = reqwest::Client::new();
    let mut req = client
        .request(method, format!("{base}{path}"))
        .bearer_auth(key);
    if let Some(body) = body {
        req = req.json(&body);
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("无法连接 {base}，请先运行 `apip serve`"))?;
    let status = resp.status();
    let value: Value = resp.json().await.unwrap_or(Value::Null);
    let error_message = value
        .pointer("/error/message")
        .and_then(Value::as_str)
        .or_else(|| value.get("message").and_then(Value::as_str))
        .or_else(|| value.get("error").and_then(Value::as_str));
    let ok_false = value.get("ok").and_then(Value::as_bool) == Some(false);
    // Some upstreams answer unknown routes with HTTP 200 + {"error": ...};
    // treat any explicit error payload as failure even on 2xx.
    let explicit_error = value.get("error").map(|v| !v.is_null()).unwrap_or(false);
    if !status.is_success() || explicit_error || ok_false {
        anyhow::bail!("{}", error_message.unwrap_or("请求失败"));
    }
    Ok(value)
}

async fn accounts(cli: Cli, cmd: AccountsCmd) -> Result<()> {
    match cmd {
        AccountsCmd::List => {
            let value = api(&cli, reqwest::Method::GET, "/api/admin/accounts", None).await?;
            let accounts = value["accounts"].as_array().cloned().unwrap_or_default();
            if accounts.is_empty() {
                println!("暂无账号。用 `apip login --provider <workbuddy|qoder>` 授权登录。");
                return Ok(());
            }
            println!(
                "{:<10} {:<10} {:<16} {:<8} {:<10} 最近签到",
                "ID", "提供商", "名称", "状态", "凭证"
            );
            for a in accounts {
                println!(
                    "{:<10} {:<10} {:<16} {:<8} {:<10} {}",
                    a["id"].as_str().unwrap_or(""),
                    a["provider"].as_str().unwrap_or(""),
                    a["name"].as_str().unwrap_or(""),
                    a["status"].as_str().unwrap_or(""),
                    a["credential_hint"].as_str().unwrap_or(""),
                    a["last_checkin"]["message"].as_str().unwrap_or("-"),
                );
            }
        }
        AccountsCmd::Remove { id } => {
            api(
                &cli,
                reqwest::Method::DELETE,
                &format!("/api/admin/accounts/{id}"),
                None,
            )
            .await?;
            println!("已删除 {id}");
        }
        AccountsCmd::Enable { id } => {
            api(
                &cli,
                reqwest::Method::PATCH,
                &format!("/api/admin/accounts/{id}"),
                Some(json!({"enabled": true})),
            )
            .await?;
            println!("已启用 {id}");
        }
        AccountsCmd::Disable { id } => {
            api(
                &cli,
                reqwest::Method::PATCH,
                &format!("/api/admin/accounts/{id}"),
                Some(json!({"enabled": false})),
            )
            .await?;
            println!("已停用 {id}");
        }
        AccountsCmd::Test { id } => {
            let value = api(
                &cli,
                reqwest::Method::POST,
                &format!("/api/admin/accounts/{id}/test"),
                Some(json!({})),
            )
            .await?;
            println!("{}", value["message"].as_str().unwrap_or("凭证有效"));
        }
    }
    Ok(())
}

async fn models(cli: Cli, refresh: bool) -> Result<()> {
    if refresh {
        let value = api(
            &cli,
            reqwest::Method::POST,
            "/api/admin/models/refresh",
            Some(json!({})),
        )
        .await?;
        println!(
            "已刷新，共 {} 个模型",
            value["models"].as_u64().unwrap_or(0)
        );
    }
    let value = api(&cli, reqwest::Method::GET, "/api/admin/models", None).await?;
    let models = value["models"].as_array().cloned().unwrap_or_default();
    println!("{:<28} {:<24} 上游 ID", "模型 ID", "提供商");
    for m in models {
        let join = |key: &str, fallback: &str| {
            m[key]
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| m[fallback].as_str().unwrap_or("").to_string())
        };
        let providers = join("providers", "provider");
        let upstreams = join("upstream_ids", "upstream_id");
        let marker = if m["merged"].as_bool().unwrap_or(false) {
            " [合并]"
        } else {
            ""
        };
        println!(
            "{:<28} {:<24} {}{}",
            m["id"].as_str().unwrap_or(""),
            providers,
            upstreams,
            marker,
        );
    }
    Ok(())
}

fn open_browser(url: &str) -> bool {
    let mut command = if cfg!(target_os = "macos") {
        let mut c = std::process::Command::new("open");
        c.arg(url);
        c
    } else if cfg!(target_os = "windows") {
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", "start", "", url]);
        c
    } else {
        let mut c = std::process::Command::new("xdg-open");
        c.arg(url);
        c
    };
    command
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .is_ok()
}

async fn login(
    cli: Cli,
    provider: String,
    region: Option<String>,
    name: Option<String>,
    no_open: bool,
) -> Result<()> {
    let value = api(
        &cli,
        reqwest::Method::POST,
        "/api/admin/login/start",
        Some(json!({
            "provider": provider,
            "region": region,
            "name": name,
        })),
    )
    .await?;
    let flow_id = value["flow_id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("服务未返回 flow_id"))?
        .to_string();
    let url = value["url"].as_str().unwrap_or("").to_string();
    if url.is_empty() {
        anyhow::bail!("服务未返回授权链接");
    }
    println!("{}", value["message"].as_str().unwrap_or("请打开授权链接"));
    println!("{url}");
    if !no_open {
        if open_browser(&url) {
            println!("已尝试在浏览器中打开授权页");
        } else {
            println!("无法自动打开浏览器，请手动复制上面的链接");
        }
    }
    println!("等待授权完成（最多 10 分钟）…");
    let mut last_message = String::new();
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let poll = match api(
            &cli,
            reqwest::Method::GET,
            &format!("/api/admin/login/{flow_id}"),
            None,
        )
        .await
        {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let status = poll["status"].as_str().unwrap_or("");
        let message = poll["message"].as_str().unwrap_or("").to_string();
        if status == "pending" {
            if !message.is_empty() && message != last_message {
                println!("{message}");
                last_message = message;
            }
            continue;
        }
        if status == "success" {
            let account = poll["account"].clone();
            println!(
                "登录成功：{}（ID {}，提供商 {}）",
                account["name"].as_str().unwrap_or(""),
                poll["account_id"].as_str().unwrap_or(""),
                account["provider"].as_str().unwrap_or(""),
            );
            return Ok(());
        }
        anyhow::bail!(
            "登录未完成：{}",
            if message.is_empty() {
                status.to_string()
            } else {
                message
            }
        );
    }
}

async fn list_providers(cli: Cli) -> Result<()> {
    let value = api(&cli, reqwest::Method::GET, "/api/admin/providers", None).await?;
    for p in value["providers"].as_array().cloned().unwrap_or_default() {
        println!(
            "{:<10} {:<22} 账号 {:<3} 模型 {:<3} {}",
            p["id"].as_str().unwrap_or(""),
            p["name"].as_str().unwrap_or(""),
            p["account_count"].as_u64().unwrap_or(0),
            p["model_count"].as_u64().unwrap_or(0),
            if p["enabled"].as_bool().unwrap_or(true) {
                "启用"
            } else {
                "停用"
            },
        );
    }
    Ok(())
}

async fn checkin(cli: Cli, account: Option<String>) -> Result<()> {
    let (method, path) = match &account {
        Some(id) => (
            reqwest::Method::POST,
            format!("/api/admin/accounts/{id}/checkin"),
        ),
        None => (reqwest::Method::POST, "/api/admin/checkin".to_string()),
    };
    let value = api(&cli, method, &path, Some(json!({}))).await?;
    for item in value["results"].as_array().cloned().unwrap_or_default() {
        println!(
            "{:<16} {:<8} {}",
            item["name"].as_str().unwrap_or(""),
            item["status"].as_str().unwrap_or(""),
            item["message"].as_str().unwrap_or(""),
        );
    }
    Ok(())
}

async fn settings(cli: Cli) -> Result<()> {
    let value = api(&cli, reqwest::Method::GET, "/api/admin/settings", None).await?;
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

async fn chat(cli: Cli, model: String, message: String, system: String) -> Result<()> {
    let value = api(
        &cli,
        reqwest::Method::POST,
        "/v1/chat/completions",
        Some(json!({
            "model": model,
            "stream": false,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": message}
            ]
        })),
    )
    .await?;
    if let Some(text) = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
    {
        println!("{text}");
    } else {
        println!("{}", serde_json::to_string_pretty(&value)?);
    }
    Ok(())
}
