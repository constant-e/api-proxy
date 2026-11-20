//! Merged model catalog across providers.
//!
//! Model ids from different platforms often differ only in case or separators
//! (`DeepSeek-V4-Pro` vs `deepseek-v4-pro`, `GLM-5.3` vs `glm-5.3`). Entries are
//! grouped by a normalized key so one id can route to every provider offering
//! that model. All original ids plus `provider/id` remain valid aliases.

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

use crate::provider::RemoteModel;

#[derive(Debug, Clone)]
pub struct ModelVariant {
    pub provider: String,
    pub model: RemoteModel,
}

#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// Single public id for the merged family.
    pub exposed_id: String,
    /// Human-facing name.
    pub name: String,
    /// Normalized identity used for grouping.
    pub canonical_key: String,
    /// Providers in preference order (registration order).
    pub variants: Vec<ModelVariant>,
    pub context_length: Option<u32>,
    pub reasoning: bool,
    pub vision: bool,
}

impl ModelEntry {
    pub fn providers(&self) -> Vec<String> {
        self.variants.iter().map(|v| v.provider.clone()).collect()
    }

    pub fn aliases(&self) -> Vec<String> {
        let mut out = vec![self.exposed_id.clone()];
        for v in &self.variants {
            for id in [&v.model.id, &v.model.name] {
                if !id.is_empty() && !out.contains(id) {
                    out.push(id.clone());
                }
            }
            let prefixed = format!("{}/{}", v.provider, v.model.id);
            if !out.contains(&prefixed) {
                out.push(prefixed);
            }
            let official_prefixed = format!("{}/{}", v.provider, self.exposed_id);
            if !out.contains(&official_prefixed) {
                out.push(official_prefixed);
            }
        }
        out
    }
}

#[derive(Debug, Clone)]
struct Route {
    entry: usize,
    /// Set for `provider/model` aliases: only that variant is tried.
    variant: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderModelStatus {
    /// `dynamic`, `fallback`, or `disabled`.
    pub source: String,
    pub count: usize,
    pub error: Option<String>,
}

#[derive(Debug, Default)]
pub struct ModelRegistry {
    entries: Vec<ModelEntry>,
    by_id: HashMap<String, Route>,
    refreshed_at: i64,
    /// True when at least one provider had to use its static fallback list.
    pub used_fallback: bool,
    /// Per-provider model discovery outcome, surfaced in the admin API.
    pub sources: BTreeMap<String, ProviderModelStatus>,
}

/// Case/separator-insensitive identity for a model id. Keeps dots and digits,
/// because they carry versions (`glm-5.3` must not equal `glm-53`).
pub fn normalize_model_id(id: &str) -> String {
    // Case-insensitive, and separators are dropped entirely:
    // "DeepSeek-V4.1-Flash" == "deepseek v4.1 flash" == "deepseek_v4.1_flash".
    // Dots are preserved because they carry versions (`glm-5.3` != `glm-53`).
    id.trim()
        .chars()
        .flat_map(char::to_lowercase)
        .filter(|ch| !matches!(ch, ' ' | '-' | '_' | '/' | '\\' | '\t'))
        .collect()
}

/// Normalized alias -> official public model id. Aliases here are deliberately
/// explicit (not guessed): two upstream ids are merged only when they denote
/// the same official model. Downstream clients should use the official id.
const OFFICIAL_MODELS: &[(&str, &str)] = &[
    ("DeepSeek-V4.1-Flash", "DeepSeek-V4.1-Flash"),
    ("DeepSeek-Flash", "DeepSeek-V4.1-Flash"),
    ("DeepSeek-V4-Pro", "DeepSeek-V4-Pro"),
    ("DeepSeek-V4-Flash", "DeepSeek-V4-Flash"),
    ("DeepSeek-V3.2", "DeepSeek-V3.2"),
    ("DeepSeek-V3", "DeepSeek-V3"),
    ("DeepSeek-R1", "DeepSeek-R1"),
    ("GLM-5.3", "GLM-5.3"),
    ("GLM-5.3-Flash", "GLM-5.3-Flash"),
    ("GLM-5.2", "GLM-5.2"),
    ("GLM-5.1", "GLM-5.1"),
    ("GLM-5v-Turbo", "GLM-5v-Turbo"),
    ("Kimi-K3", "Kimi-K3"),
    ("Kimi-K3.1-Pro", "Kimi-K3.1-Pro"),
    ("Kimi-K2.8-Preview", "Kimi-K2.8-Preview"),
    ("Kimi-K2.7", "Kimi-K2.7"),
    ("Kimi-K2.6", "Kimi-K2.6"),
    ("MiniMax-M3", "MiniMax-M3"),
    ("MiniMax-M3.1-Flash", "MiniMax-M3.1-Flash"),
    ("MiniMax-M2.7", "MiniMax-M2.7"),
    ("Qwen3.8-Max", "Qwen3.8-Max"),
    ("Qwen3.8-Flash", "Qwen3.8-Flash"),
    ("Qwen3.7-Max", "Qwen3.7-Max"),
    ("Qwen3.7-Plus", "Qwen3.7-Plus"),
    ("Qwen3.7-Flash", "Qwen3.7-Flash"),
    ("Qwen4-Max", "Qwen4-Max"),
    ("Qwen4-Max-Preview", "Qwen4-Max-Preview"),
    ("Step-5-Preview", "Step-5-Preview"),
    ("Hy4", "Hy4"),
    ("hy4std", "Hy4"),
    ("Space-Bunny", "Space-Bunny"),
    ("space-bunny", "Space-Bunny"),
    ("Xiaomi MiMo-V2.6-Pro", "Xiaomi MiMo-V2.6-Pro"),
    ("mimo-v2.6-pro", "Xiaomi MiMo-V2.6-Pro"),
    ("kimi-k3-1", "Kimi-K3"),
];

fn official_model(normalized: &str) -> Option<&'static str> {
    OFFICIAL_MODELS
        .iter()
        .find(|(alias, _)| normalize_model_id(alias) == normalized)
        .map(|(_, official)| *official)
}

fn identity_key(normalized: &str) -> String {
    official_model(normalized)
        .map(normalize_model_id)
        .unwrap_or_else(|| normalized.to_string())
}

fn pick_public_id(variants: &[ModelVariant]) -> String {
    // Prefer a display-style id (contains uppercase) over an all-lowercase
    // upstream id, then fall back to provider order. "GLM-5.3" beats "glm-5.3".
    variants
        .iter()
        .find(|v| v.model.id.chars().any(char::is_uppercase))
        .map(|v| v.model.id.clone())
        .unwrap_or_else(|| variants[0].model.id.clone())
}

fn pick_name(variants: &[ModelVariant], exposed_id: &str) -> String {
    for v in variants {
        let name = v.model.name.trim();
        if !name.is_empty() && name != v.model.id {
            return name.to_string();
        }
    }
    exposed_id.to_string()
}

impl ModelRegistry {
    pub fn build(contributions: Vec<(String, Vec<RemoteModel>, bool)>) -> Self {
        // provider -> canonical key -> first model (dedupes within a provider)
        let mut groups: Vec<ModelEntry> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        let mut used_fallback = false;

        for (provider, models, fallback) in contributions {
            used_fallback |= fallback;
            let mut seen_in_provider = std::collections::HashSet::new();
            for model in models {
                let normalized = normalize_model_id(&model.id);
                if normalized.is_empty() {
                    continue;
                }
                let key = identity_key(&normalized);
                let variant_key = format!("{provider}\u{0}{key}");
                if !seen_in_provider.insert(variant_key) {
                    continue;
                }
                match index.get(&key) {
                    Some(i) => groups[*i].variants.push(ModelVariant {
                        provider: provider.clone(),
                        model,
                    }),
                    None => {
                        index.insert(key.clone(), groups.len());
                        groups.push(ModelEntry {
                            exposed_id: String::new(),
                            name: String::new(),
                            canonical_key: key,
                            variants: vec![ModelVariant {
                                provider: provider.clone(),
                                model,
                            }],
                            context_length: None,
                            reasoning: false,
                            vision: false,
                        });
                    }
                }
            }
        }

        for entry in &mut groups {
            let official = entry
                .variants
                .iter()
                .find_map(|v| official_model(&normalize_model_id(&v.model.id)).map(str::to_string));
            entry.exposed_id = official.unwrap_or_else(|| pick_public_id(&entry.variants));
            entry.name = pick_name(&entry.variants, &entry.exposed_id);
            entry.context_length = entry
                .variants
                .iter()
                .filter_map(|v| v.model.context_length)
                .max();
            entry.reasoning = entry.variants.iter().any(|v| v.model.reasoning);
            entry.vision = entry.variants.iter().any(|v| v.model.vision);
        }
        groups.sort_by(|a, b| {
            a.exposed_id
                .to_lowercase()
                .cmp(&b.exposed_id.to_lowercase())
        });

        let mut by_id = HashMap::new();
        for (i, entry) in groups.iter().enumerate() {
            by_id.insert(
                entry.exposed_id.clone(),
                Route {
                    entry: i,
                    variant: None,
                },
            );
            by_id.insert(
                entry.canonical_key.clone(),
                Route {
                    entry: i,
                    variant: None,
                },
            );
            // All official aliases that resolve to this canonical identity.
            for (alias, official) in OFFICIAL_MODELS {
                if normalize_model_id(official) == entry.canonical_key {
                    for key in [(*alias).to_string(), normalize_model_id(alias)] {
                        by_id.entry(key).or_insert(Route {
                            entry: i,
                            variant: None,
                        });
                    }
                }
            }
            for (vi, v) in entry.variants.iter().enumerate() {
                for alias in [&v.model.id, &v.model.name] {
                    if !alias.is_empty() {
                        by_id.entry(alias.clone()).or_insert(Route {
                            entry: i,
                            variant: None,
                        });
                    }
                    let normalized = normalize_model_id(alias);
                    if !normalized.is_empty() {
                        by_id.entry(normalized).or_insert(Route {
                            entry: i,
                            variant: None,
                        });
                    }
                }
                by_id.insert(
                    format!(
                        "{}/{}",
                        v.provider.to_lowercase(),
                        normalize_model_id(&v.model.id)
                    ),
                    Route {
                        entry: i,
                        variant: Some(vi),
                    },
                );
                by_id.insert(
                    format!(
                        "{}/{}",
                        v.provider.to_lowercase(),
                        normalize_model_id(&entry.exposed_id)
                    ),
                    Route {
                        entry: i,
                        variant: Some(vi),
                    },
                );
            }
        }

        Self {
            entries: groups,
            by_id,
            refreshed_at: crate::store::now_ms(),
            used_fallback,
            sources: BTreeMap::new(),
        }
    }

    /// Resolve a client model id. Returns the merged entry and the variants to
    /// try, in preference order; `provider/id` pins one variant.
    pub fn resolve(&self, id: &str) -> Option<(&ModelEntry, Vec<&ModelVariant>)> {
        let route = self
            .by_id
            .get(id)
            .or_else(|| {
                let (provider, rest) = id.split_once('/')?;
                let key = format!("{}/{}", provider.to_lowercase(), normalize_model_id(rest));
                self.by_id.get(&key)
            })
            .or_else(|| self.by_id.get(&normalize_model_id(id)))?;
        let entry = self.entries.get(route.entry)?;
        let variants = match route.variant {
            Some(i) => entry.variants.get(i).into_iter().collect(),
            None => entry.variants.iter().collect(),
        };
        Some((entry, variants))
    }

    pub fn set_source(
        &mut self,
        provider: impl Into<String>,
        source: impl Into<String>,
        count: usize,
        error: Option<String>,
    ) {
        self.sources.insert(
            provider.into(),
            ProviderModelStatus {
                source: source.into(),
                count,
                error,
            },
        );
    }

    pub fn entries(&self) -> &[ModelEntry] {
        &self.entries
    }

    pub fn refreshed_at(&self) -> i64 {
        self.refreshed_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(id: &str) -> RemoteModel {
        RemoteModel::simple(id)
    }

    #[test]
    fn unique_models_keep_bare_ids() {
        let registry = ModelRegistry::build(vec![
            ("a".into(), vec![model("x")], false),
            ("b".into(), vec![model("y")], false),
        ]);
        assert!(registry.resolve("x").is_some());
        assert!(registry.resolve("y").is_some());
        assert!(registry.resolve("a/x").is_some());
        assert_eq!(registry.entries().len(), 2);
    }

    #[test]
    fn case_and_separator_differences_merge() {
        let registry = ModelRegistry::build(vec![
            (
                "workbuddy".into(),
                vec![model("deepseek-v4-pro"), model("glm-5.3")],
                false,
            ),
            (
                "qoder".into(),
                vec![model("DeepSeek-V4-Pro"), model("GLM-5.3")],
                false,
            ),
        ]);
        assert_eq!(registry.entries().len(), 2);
        let (entry, variants) = registry.resolve("DeepSeek-V4-Pro").unwrap();
        assert_eq!(variants.len(), 2);
        assert_eq!(entry.exposed_id, "DeepSeek-V4-Pro");
        assert_eq!(
            registry.resolve("deepseek-v4-pro").unwrap().0.exposed_id,
            entry.exposed_id
        );
        assert_eq!(
            registry
                .resolve("workbuddy/deepseek-v4-pro")
                .unwrap()
                .1
                .len(),
            1
        );
        assert_eq!(
            registry
                .resolve("WORKBUDDY/DeepSeek_V4_Pro")
                .unwrap()
                .1
                .len(),
            1
        );
    }

    #[test]
    fn separators_around_versions_are_ignored() {
        assert_eq!(
            normalize_model_id("Qwen-3.8-Max"),
            normalize_model_id("qwen3.8 max")
        );
        assert_eq!(
            normalize_model_id("DeepSeek V4 Pro"),
            normalize_model_id("deepseek-v4-pro")
        );
        assert_eq!(
            normalize_model_id("GLM-5.3-Flash"),
            normalize_model_id("glm5.3flash")
        );
    }

    #[test]
    fn official_alias_merges_deepseek_flash() {
        let registry = ModelRegistry::build(vec![
            ("qoder".into(), vec![model("DeepSeek-Flash")], false),
            (
                "workbuddy".into(),
                vec![model("deepseek-v4.1-flash")],
                false,
            ),
        ]);
        assert_eq!(registry.entries().len(), 1);
        let (entry, variants) = registry.resolve("deepseek-flash").unwrap();
        assert_eq!(variants.len(), 2);
        assert_eq!(entry.exposed_id, "DeepSeek-V4.1-Flash");
        assert_eq!(
            registry
                .resolve("DeepSeek-V4.1-Flash")
                .unwrap()
                .0
                .canonical_key,
            entry.canonical_key
        );
        assert_eq!(
            registry
                .resolve("workbuddy/DeepSeek-V4.1-Flash")
                .unwrap()
                .1
                .len(),
            1
        );
    }

    #[test]
    fn current_workbuddy_names_map_officially() {
        let registry = ModelRegistry::build(vec![(
            "workbuddy".into(),
            vec![
                model("hy4std"),
                model("space-bunny"),
                model("mimo-v2.6-pro"),
                model("kimi-k3-1"),
            ],
            false,
        )]);
        assert_eq!(registry.resolve("hy4std").unwrap().0.exposed_id, "Hy4");
        assert_eq!(
            registry.resolve("space-bunny").unwrap().0.exposed_id,
            "Space-Bunny"
        );
        assert_eq!(
            registry.resolve("mimo-v2.6-pro").unwrap().0.exposed_id,
            "Xiaomi MiMo-V2.6-Pro"
        );
        assert_eq!(
            registry.resolve("kimi-k3-1").unwrap().0.exposed_id,
            "Kimi-K3"
        );
    }

    #[test]
    fn hy4_free_and_paid_stay_distinct() {
        let registry = ModelRegistry::build(vec![(
            "workbuddy".into(),
            vec![model("hy4flash"), model("hy4flash-free")],
            false,
        )]);
        assert_eq!(registry.entries().len(), 2);
        assert_ne!(
            registry.resolve("hy4flash").unwrap().0.exposed_id,
            registry.resolve("hy4flash-free").unwrap().0.exposed_id
        );
    }

    #[test]
    fn versions_are_not_collapsed() {
        let registry = ModelRegistry::build(vec![(
            "a".into(),
            vec![model("glm-5.3"), model("glm-53")],
            false,
        )]);
        assert_eq!(registry.entries().len(), 2);
    }
}
