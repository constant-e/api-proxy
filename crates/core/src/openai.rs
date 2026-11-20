//! Minimal OpenAI-compatible JSON helpers shared by the server and providers.

use serde_json::{json, Value};

use crate::provider::ChatDelta;

pub fn error_body(message: impl Into<String>, kind: &str, code: Option<&str>) -> Value {
    json!({
        "error": {
            "message": message.into(),
            "type": kind,
            "param": Value::Null,
            "code": code,
        }
    })
}

pub fn stream_include_usage(request: &Value) -> bool {
    request
        .get("stream_options")
        .and_then(|v| v.get("include_usage"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

pub fn wants_stream(request: &Value) -> bool {
    request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

pub fn zero_usage() -> Value {
    json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
}

fn chunk_base(id: &str, created: i64, model: &str) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
    })
}

/// One streaming chunk carrying `delta`. `finish_reason` is emitted by the
/// caller in a separate terminal chunk.
pub fn delta_chunk(
    id: &str,
    created: i64,
    model: &str,
    delta: &ChatDelta,
    finish_reason: Option<&str>,
) -> Value {
    let mut d = serde_json::Map::new();
    if let Some(role) = &delta.role {
        d.insert("role".into(), Value::String(role.clone()));
    }
    if let Some(content) = &delta.content {
        if !content.is_empty() {
            d.insert("content".into(), Value::String(content.clone()));
        }
    }
    if let Some(reasoning) = &delta.reasoning_content {
        if !reasoning.is_empty() {
            d.insert("reasoning_content".into(), Value::String(reasoning.clone()));
        }
    }
    if let Some(tool_calls) = &delta.tool_calls {
        d.insert("tool_calls".into(), Value::Array(tool_calls.clone()));
    }
    let mut chunk = chunk_base(id, created, model);
    chunk["choices"] = json!([{
        "index": 0,
        "delta": Value::Object(d),
        "finish_reason": finish_reason,
    }]);
    chunk
}

pub fn usage_chunk(id: &str, created: i64, model: &str, usage: Value) -> Value {
    let mut chunk = chunk_base(id, created, model);
    chunk["usage"] = usage;
    chunk
}

#[allow(clippy::too_many_arguments)]
pub fn completion_response(
    id: &str,
    created: i64,
    model: &str,
    content: &str,
    reasoning: Option<&str>,
    tool_calls: Option<&[Value]>,
    finish_reason: &str,
    usage: Value,
) -> Value {
    let mut message = serde_json::Map::new();
    message.insert("role".into(), Value::String("assistant".into()));
    if let Some(calls) = tool_calls {
        message.insert("content".into(), Value::Null);
        message.insert("tool_calls".into(), Value::Array(calls.to_vec()));
    } else {
        message.insert("content".into(), Value::String(content.to_string()));
    }
    if let Some(reasoning) = reasoning {
        if !reasoning.is_empty() {
            message.insert(
                "reasoning_content".into(),
                Value::String(reasoning.to_string()),
            );
        }
    }
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": Value::Object(message),
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    })
}

pub fn new_completion_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4().simple())
}
