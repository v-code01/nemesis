//! Scenario YAML schema and loader for nemesis-sim fault injection.
//!
//! A scenario file describes a timeline of synthetic fault events (ECC errors,
//! bandwidth degradation, thermal throttling) that the sim replay engine will
//! inject into the metric stream.  This module owns only deserialization; the
//! replay loop is a future extension.

#![allow(dead_code)]

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Scenario {
    pub scenario:    String,
    pub description: String,
    pub timeline:    Vec<TimelineEvent>,
}

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
    /// Strips a trailing `'s'` suffix before parsing; returns 0 on parse failure
    /// rather than panicking, so malformed YAML produces a no-op rather than a crash.
    pub fn t_seconds(&self) -> u64 {
        self.t.trim_end_matches('s').parse().unwrap_or(0)
    }
}

/// Load and deserialise a scenario YAML file.
pub fn load_scenario(path: &Path) -> anyhow::Result<Scenario> {
    let content = std::fs::read_to_string(path)?;
    Ok(serde_yaml::from_str(&content)?)
}
