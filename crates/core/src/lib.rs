//! Shared gateway framework: persistent store, provider trait, model merging,
//! and OpenAI wire helpers.
//!
//! Adding a provider: create a crate that implements [`Provider`], register it
//! in `crates/gateway/src/main.rs`, and it immediately gets account rotation,
//! failover, model merging, console management and scheduled check-in.

pub mod openai;
pub mod provider;
pub mod registry;
pub mod store;

pub use provider::{
    account_view, AccountView, Balance, ChatDelta, ChatStream, CheckinOutcome, LoginAccount,
    LoginPoll, LoginStart, Provider, ProviderError, RemoteModel,
};
pub use registry::{ModelEntry, ModelRegistry, ProviderModelStatus};
pub use store::{
    data_dir, now_ms, today_local, Account, CheckinRecord, ProviderSettings, Settings, Store,
    StoreFile,
};
