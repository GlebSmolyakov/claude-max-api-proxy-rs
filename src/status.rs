//! What the proxy learns while it runs: uptime, the CLI version, the
//! subscription limits the CLI reports on every turn, and which real model
//! ids stand behind the aliases.

use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::sync::RwLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::models::ALIASES;
use crate::types::claude_cli::{ModelUsage, RateLimitInfo};

pub struct RuntimeStatus {
    started: Instant,
    cli_version: String,
    inner: RwLock<Inner>,
}

#[derive(Default)]
struct Inner {
    rate_limits: Option<RateLimits>,
    /// Alias → the model id it resolved to on the last turn that used it.
    aliases: BTreeMap<String, String>,
    /// Model id → what `modelUsage` said about it.
    models: BTreeMap<String, ModelLimits>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RateLimits {
    pub status: Option<String>,
    /// Window name (`five_hour`, `seven_day`, …) → how much is used.
    pub windows: BTreeMap<String, RateLimitWindow>,
    /// Unix seconds when the CLI last reported these numbers.
    pub reported_at: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RateLimitWindow {
    /// Share of the window already used, from 0.0 to 1.0.
    pub utilization: Option<f64>,
    pub resets_at: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq)]
pub struct ModelLimits {
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
}

impl RuntimeStatus {
    pub fn new(cli_version: String) -> Self {
        Self {
            started: Instant::now(),
            cli_version,
            inner: RwLock::default(),
        }
    }

    pub fn uptime_secs(&self) -> u64 {
        self.started.elapsed().as_secs()
    }

    pub fn cli_version(&self) -> &str {
        &self.cli_version
    }

    pub fn record_rate_limit(&self, info: &RateLimitInfo) {
        let windows = info
            .windows
            .iter()
            .map(|(name, w)| {
                let window = RateLimitWindow {
                    utilization: w.utilization,
                    resets_at: w.resets_at,
                };
                (name.clone(), window)
            })
            .collect();
        let limits = RateLimits {
            status: info.status.clone(),
            windows,
            reported_at: unix_now(),
        };
        self.inner.write().unwrap().rate_limits = Some(limits);
    }

    /// Remember what an alias resolved to. Full ids are not aliases and are skipped.
    pub fn record_model(&self, requested: &str, resolved: &str) {
        if ALIASES.contains(&requested) {
            self.inner
                .write()
                .unwrap()
                .aliases
                .insert(requested.to_string(), resolved.to_string());
        }
    }

    pub fn record_model_usage(&self, usage: &HashMap<String, ModelUsage>) {
        let mut inner = self.inner.write().unwrap();
        for (id, facts) in usage {
            inner.models.insert(
                id.clone(),
                ModelLimits {
                    context_window: facts.context_window,
                    max_output_tokens: facts.max_output_tokens,
                },
            );
        }
    }

    pub fn rate_limits(&self) -> Option<RateLimits> {
        self.inner.read().unwrap().rate_limits.clone()
    }

    pub fn aliases(&self) -> BTreeMap<String, String> {
        self.inner.read().unwrap().aliases.clone()
    }

    pub fn models(&self) -> BTreeMap<String, ModelLimits> {
        self.inner.read().unwrap().models.clone()
    }
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::claude_cli::RateLimitWindow as CliWindow;

    #[test]
    fn uptime_starts_near_zero() {
        assert!(RuntimeStatus::new("x".into()).uptime_secs() < 2);
    }

    #[test]
    fn records_rate_limits() {
        let status = RuntimeStatus::new("x".into());
        assert_eq!(status.rate_limits(), None);
        let info = RateLimitInfo {
            status: Some("allowed".into()),
            windows: HashMap::from([(
                "five_hour".to_string(),
                CliWindow { utilization: Some(0.25), resets_at: Some(10) },
            )]),
        };
        status.record_rate_limit(&info);
        let limits = status.rate_limits().unwrap();
        assert_eq!(limits.status.as_deref(), Some("allowed"));
        assert_eq!(limits.windows["five_hour"].utilization, Some(0.25));
    }

    #[test]
    fn records_aliases_but_not_full_ids() {
        let status = RuntimeStatus::new("x".into());
        status.record_model("haiku", "claude-haiku-4-5-20251001");
        status.record_model("claude-sonnet-5", "claude-sonnet-5");
        assert_eq!(
            status.aliases(),
            BTreeMap::from([("haiku".to_string(), "claude-haiku-4-5-20251001".to_string())])
        );
    }

    #[test]
    fn records_model_limits() {
        let status = RuntimeStatus::new("x".into());
        status.record_model_usage(&HashMap::from([(
            "claude-haiku-4-5-20251001".to_string(),
            ModelUsage { context_window: Some(200_000), max_output_tokens: Some(32_000) },
        )]));
        assert_eq!(status.models()["claude-haiku-4-5-20251001"].context_window, Some(200_000));
    }
}
