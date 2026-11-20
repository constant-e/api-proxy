//! Qoder COSY protocol primitives: custom base64, request signing, RSA/AES
//! session sealing, and header construction.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use rsa::pkcs8::DecodePublicKey;
use rsa::{Pkcs1v15Encrypt, RsaPublicKey};
use serde_json::{json, Value};

const CUSTOM_ALPHABET: &[u8] = b"_doRTgHZBKcGVjlvpC,@aFSx#DPuNJme&i*MzLOEn)sUrthbf%Y^w.(kIQyXqWA!";
const STD_ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const SIGNATURE_SECRET: &str = "d2FyLCB3YXIgbmV2ZXIgY2hhbmdlcw==";
const SERVER_PUBKEY: &str = "-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDA8iMH5c02LilrsERw9t6Pv5Nc
4k6Pz1EaDicBMpdpxKduSZu5OANqUq8er4GM95omAGIOPOh+Nx0spthYA2BqGz+l
6HRkPJ7S236FZz73In/KVuLnwI8JJ2CbuJap8kvheCCZpmAWpb/cPx/3Vr/J6I17
XcW+ML9FoCI6AOvOzwIDAQAB
-----END PUBLIC KEY-----";

/// Qoder's base64 variant: standard base64, rotated by `n/3`, then chars
/// remapped through a custom alphabet with `$` for padding.
pub fn encode(plain: &[u8]) -> String {
    let std = STANDARD.encode(plain);
    let bytes = std.as_bytes();
    let n = bytes.len();
    let a = n / 3;
    let mut rotated = Vec::with_capacity(n);
    rotated.extend_from_slice(&bytes[n - a..]);
    rotated.extend_from_slice(&bytes[a..n - a]);
    rotated.extend_from_slice(&bytes[..a]);
    rotated
        .into_iter()
        .map(|c| {
            if c == b'=' {
                '$'
            } else {
                let idx = STD_ALPHABET.iter().position(|x| *x == c).unwrap_or(0);
                CUSTOM_ALPHABET[idx] as char
            }
        })
        .collect()
}

pub fn decode(encoded: &str) -> Result<Vec<u8>, String> {
    let bytes = encoded.as_bytes();
    let n = bytes.len();
    let mut mapped = Vec::with_capacity(n);
    for c in bytes {
        mapped.push(if *c == b'$' {
            b'='
        } else {
            let idx = CUSTOM_ALPHABET
                .iter()
                .position(|x| x == c)
                .ok_or_else(|| format!("非法字符: {}", *c as char))?;
            STD_ALPHABET[idx]
        });
    }
    let a = n / 3;
    let mut std = vec![0u8; n];
    std[..a].copy_from_slice(&mapped[n - a..]);
    std[a..n - a].copy_from_slice(&mapped[a..n - a]);
    std[n - a..].copy_from_slice(&mapped[..a]);
    STANDARD
        .decode(&std)
        .map_err(|e| format!("base64 解码失败: {e}"))
}

pub fn md5_hex(s: &str) -> String {
    format!("{:x}", md5::compute(s.as_bytes()))
}

pub fn sign(date: &str) -> String {
    md5_hex(&format!("cosy&{SIGNATURE_SECRET}&{date}"))
}

/// Request signature: MD5(payload + \n + cosyKey + \n + date + \n + body + \n + path).
pub fn sign_request(payload: &str, cosy_key: &str, date: &str, body: &str, path: &str) -> String {
    md5_hex(&format!("{payload}\n{cosy_key}\n{date}\n{body}\n{path}"))
}

pub fn compose_bearer(payload: &str, sig: &str) -> String {
    format!("Bearer COSY.{payload}.{sig}")
}

fn rsa_encrypt(plain: &[u8]) -> Result<Vec<u8>, String> {
    let key = RsaPublicKey::from_public_key_pem(SERVER_PUBKEY).map_err(|e| e.to_string())?;
    let mut rng = rand::thread_rng();
    key.encrypt(&mut rng, Pkcs1v15Encrypt, plain)
        .map_err(|e| e.to_string())
}

fn aes_encrypt(plain: &[u8], key: &[u8]) -> Result<Vec<u8>, String> {
    use aes::cipher::{block_padding::Pkcs7, BlockEncryptMut, KeyIvInit};
    let enc = cbc::Encryptor::<aes::Aes128>::new_from_slices(key, &key[..16])
        .map_err(|e| e.to_string())?;
    Ok(enc.encrypt_padded_vec_mut::<Pkcs7>(plain))
}

#[derive(Debug, Clone)]
pub struct Identity {
    pub name: String,
    pub uid: String,
    pub user_type: String,
    pub security_oauth_token: String,
    pub refresh_token: String,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub cosy_key: String,
    pub info: String,
    pub identity: Identity,
    pub machine_id: String,
    pub machine_token: String,
    pub machine_type: String,
    pub expire_ms: i64,
    pub created_ms: i64,
}

impl Session {
    pub fn new(
        identity: Identity,
        machine_id: String,
        machine_token: String,
        machine_type: String,
        expire_ms: i64,
    ) -> Result<Self, String> {
        let temp_key = uuid::Uuid::new_v4().to_string()[..16].to_string();
        let encrypted = rsa_encrypt(temp_key.as_bytes())?;
        let cosy_key = STANDARD.encode(encrypted);
        let identity_json = json!({
            "name": identity.name,
            "aid": identity.uid,
            "uid": identity.uid,
            "yx_uid": "",
            "organization_id": "",
            "organization_name": "",
            "user_type": identity.user_type,
            "security_oauth_token": identity.security_oauth_token,
            "refresh_token": identity.refresh_token,
        });
        let sealed = aes_encrypt(identity_json.to_string().as_bytes(), temp_key.as_bytes())?;
        Ok(Self {
            cosy_key,
            info: STANDARD.encode(sealed),
            identity,
            machine_id,
            machine_token,
            machine_type,
            expire_ms,
            created_ms: gateway_core::now_ms(),
        })
    }

    pub fn needs_refresh(&self) -> bool {
        if self.expire_ms > 0 {
            gateway_core::now_ms() > self.expire_ms - 2 * 3600 * 1000
        } else {
            gateway_core::now_ms() - self.created_ms > 30 * 60 * 1000
        }
    }
}

/// Stable machine identity derived from the account id (official clients keep
/// this constant per installation; regenerating per request looks abusive).
pub fn machine_identity(seed: &str) -> (String, String, String) {
    let h = md5_hex(seed);
    let h2 = md5_hex(&format!("{seed}:token"));
    let h3 = md5_hex(&format!("{seed}:type"));
    let machine_id = format!(
        "{}-{}-4{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[13..16],
        &h[16..20],
        &h[20..32]
    );
    let machine_token: String = format!("{h}{h2}").chars().take(50).collect();
    let machine_type: String = h3.chars().take(18).collect();
    (machine_id, machine_token, machine_type)
}

fn date_header() -> String {
    chrono_like_date()
}

/// RFC1123 GMT date without pulling chrono into this crate.
fn chrono_like_date() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    format_http_date(now)
}

pub fn format_http_date(epoch: i64) -> String {
    // Civil date from days since epoch (Howard Hinnant's algorithm).
    let days = epoch.div_euclid(86_400);
    let secs = epoch.rem_euclid(86_400);
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mth = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if mth <= 2 { y + 1 } else { y };
    let weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let weekday = weekdays[((days + 3).rem_euclid(7)) as usize];
    format!(
        "{weekday}, {d:02} {} {year} {h:02}:{m:02}:{s:02} GMT",
        months[(mth - 1) as usize]
    )
}

pub fn signed_headers(
    machine_id: &str,
    machine_token: &str,
    machine_type: &str,
    date: &str,
    signature: &str,
) -> Vec<(&'static str, String)> {
    vec![
        ("cosy-machinetoken", machine_token.into()),
        ("cosy-machinetype", machine_type.into()),
        ("login-version", "v2".into()),
        ("appcode", "cosy".into()),
        ("accept", "application/json".into()),
        ("accept-encoding", "identity".into()),
        ("cosy-version", "0.1.43".into()),
        ("cosy-clienttype", "5".into()),
        ("date", date.into()),
        ("signature", signature.into()),
        ("content-type", "application/json".into()),
        ("cosy-machineid", machine_id.into()),
        ("user-agent", "Go-http-client/2.0".into()),
    ]
}

pub fn bearer_headers(
    session: &Session,
    date: &str,
    bearer: &str,
    accept: &str,
) -> Vec<(&'static str, String)> {
    vec![
        ("cosy-data-policy", "AGREE".into()),
        ("content-type", "application/json".into()),
        ("cosy-machinetype", session.machine_type.clone()),
        ("cosy-clienttype", "5".into()),
        ("cosy-date", date.into()),
        ("cosy-user", session.identity.uid.clone()),
        ("cosy-key", session.cosy_key.clone()),
        ("accept", accept.into()),
        ("authorization", bearer.into()),
        ("accept-encoding", "identity".into()),
        ("cosy-version", "0.1.43".into()),
        ("cosy-machineid", session.machine_id.clone()),
        ("cosy-machinetoken", session.machine_token.clone()),
        ("login-version", "v2".into()),
        ("user-agent", "Go-http-client/2.0".into()),
    ]
}

pub fn build_payload_b64(info: &str) -> String {
    let payload = json!({
        "cosyVersion": "0.1.43",
        "ideVersion": "",
        "info": info,
        "requestId": uuid::Uuid::new_v4().to_string(),
        "version": "v1",
    });
    STANDARD.encode(payload.to_string().as_bytes())
}

pub fn signed_body(session: &Session, body: &str, path_without_algo: &str) -> (String, String) {
    let date = (gateway_core::now_ms() / 1000).to_string();
    let payload = build_payload_b64(&session.info);
    let sig = sign_request(&payload, &session.cosy_key, &date, body, path_without_algo);
    let bearer = compose_bearer(&payload, &sig);
    (date, bearer)
}

pub fn signed_post_headers<'a>(
    machine_id: &'a str,
    machine_token: &'a str,
    machine_type: &'a str,
) -> Vec<(&'static str, String)> {
    let date = date_header();
    let sig = sign(&date);
    signed_headers(machine_id, machine_token, machine_type, &date, &sig)
}

/// Outer envelope used by jobToken / userStatus calls.
pub fn sealed_outer(inner: &Value) -> String {
    let outer = json!({"payload": inner.to_string(), "encodeVersion": "1"});
    encode(outer.to_string().as_bytes())
}

/// Decode a response that may be an outer envelope (`payload` string) or a
/// plain JSON object.
pub fn unseal(value: &Value) -> Value {
    if let Some(payload) = value.get("payload").and_then(Value::as_str) {
        if let Ok(bytes) = decode(payload) {
            if let Ok(v) = serde_json::from_slice(&bytes) {
                return v;
            }
        }
        if let Ok(v) = STANDARD.decode(payload) {
            if let Ok(v) = serde_json::from_slice(&v) {
                return v;
            }
        }
        if let Ok(v) = serde_json::from_str(payload) {
            return v;
        }
    }
    value.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_base64_roundtrip() {
        for s in ["", "a", "hello world", "{\"中文\":1}", &"x".repeat(100)] {
            let enc = encode(s.as_bytes());
            assert_eq!(decode(&enc).unwrap(), s.as_bytes());
        }
    }

    #[test]
    fn http_date_matches_epoch() {
        assert_eq!(format_http_date(0), "Thu, 01 Jan 1970 00:00:00 GMT");
    }

    #[test]
    fn machine_identity_is_stable_and_shaped() {
        let (id, token, ty) = machine_identity("account-1");
        let again = machine_identity("account-1");
        assert_eq!((&id, &token, &ty), (&again.0, &again.1, &again.2));
        assert_eq!(id.len(), 36);
        assert_eq!(token.len(), 50);
        assert_eq!(ty.len(), 18);
    }
}
