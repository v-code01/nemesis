//! Scenario YAML schema and loader for nemesis-sim fault injection.
//!
//! A scenario file describes a timeline of synthetic fault events (ECC errors,
//! bandwidth degradation, thermal throttling) that the sim replay engine will
//! inject into the metric stream.  This module owns only deserialization; the
//! replay loop is a future extension.

use serde::Deserialize;
use std::path::Path;

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct Scenario {
    pub scenario:    String,
    pub description: String,
    pub timeline:    Vec<TimelineEvent>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct TimelineEvent {
    pub t:      String,
    pub gpu:    Option<String>,
    pub action: String,
    pub rate:   Option<f32>,
}

impl TimelineEvent {
    /// Parse the `t` field (e.g. `"30s"`) into whole seconds.
    ///
    /// Returns `None` if the field is missing the `'s'` suffix or is not a valid integer,
    /// so callers can distinguish a parse failure from a legitimate zero-second timestamp.
    #[allow(dead_code)]
    pub fn t_seconds(&self) -> Option<u64> {
        self.t.trim_end_matches('s').parse().ok()
    }
}

/// Load and deserialise a scenario YAML file.
#[allow(dead_code)]
pub fn load_scenario(path: &Path) -> anyhow::Result<Scenario> {
    let content = std::fs::read_to_string(path)?;
    Ok(serde_yaml::from_str(&content)?)
}
