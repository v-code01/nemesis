// TelemetryStore: thread-safe per-GPU MetricRing map + broadcast event bus.
//
// Design decisions:
//   - `parking_lot::RwLock` is chosen over `std::sync::RwLock` for:
//       * No poisoning API surface (panics instead, which is fine here).
//       * Shorter critical sections: upgradable reads, fair scheduling.
//   - The broadcast channel is sized at 1 024 events.  Slow subscribers
//     that fall behind receive a `Lagged` error from `BroadcastStream` which
//     the service layer converts to a tracing warning and drops the stale
//     events rather than back-pressuring the publisher.
//   - `TelemetryStore` is `Clone` because it wraps an `Arc`; cloning is O(1)
//     and all clones share the same rings and event bus.

use crate::ring::MetricRing;
use nemesis_proto::telemetry::v1::{HardwareEvent, MetricSample};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;

/// Broadcast bus capacity.  1 024 covers ~10 s of events at 100 Hz before a
/// slow subscriber begins to lag; tune upward if event volume increases.
const BUS_CAPACITY: usize = 1_024;

/// Shared, cheaply cloneable telemetry store.
///
/// Thread safety: reads are concurrent; writes lock only the per-GPU
/// `MetricRing` map.  The broadcast sender is lock-free (tokio internal).
#[derive(Clone)]
pub struct TelemetryStore {
    /// Per-GPU circular buffers.  Keyed by `gpu_id` string.
    rings: Arc<RwLock<HashMap<String, MetricRing>>>,
    /// Sender half of the hardware-event broadcast bus.
    /// Retaining this keeps the channel open even when there are no receivers.
    bus_tx: broadcast::Sender<HardwareEvent>,
}

impl TelemetryStore {
    /// Create an empty store with a fresh broadcast channel.
    pub fn new() -> Self {
        let (bus_tx, _) = broadcast::channel(BUS_CAPACITY);
        Self {
            rings: Arc::new(RwLock::new(HashMap::new())),
            bus_tx,
        }
    }

    /// Ingest one `MetricSample`.
    ///
    /// Allocates a new `MetricRing` for `sample.gpu_id` on first contact.
    /// The write lock is held only for the duration of the hash-map lookup
    /// and the O(1) ring push.
    pub fn ingest(&self, sample: MetricSample) {
        let mut rings = self.rings.write();
        rings.entry(sample.gpu_id.clone()).or_default().push(sample);
    }

    /// Return up to `n` most-recent samples for `gpu_id`, oldest first.
    ///
    /// Returns an empty `Vec` if `gpu_id` is unknown.
    pub fn window(&self, gpu_id: &str, n: usize) -> Vec<MetricSample> {
        self.rings
            .read()
            .get(gpu_id)
            .map(|r| r.window(n))
            .unwrap_or_default()
    }

    /// Return the IDs of all GPUs that have ever sent a sample.
    pub fn gpu_ids(&self) -> Vec<String> {
        self.rings.read().keys().cloned().collect()
    }

    /// Broadcast a `HardwareEvent` to all active subscribers.
    ///
    /// Errors (no receivers) are intentionally ignored — the event bus is
    /// best-effort; if nothing is subscribed the event is simply dropped.
    pub fn publish(&self, event: HardwareEvent) {
        let _ = self.bus_tx.send(event);
    }

    /// Subscribe to the hardware-event broadcast bus.
    ///
    /// Each call returns an independent `Receiver`.  Receivers that cannot keep
    /// up will see `broadcast::error::RecvError::Lagged` rather than blocking
    /// the publisher (bounded-capacity semantics).
    pub fn subscribe(&self) -> broadcast::Receiver<HardwareEvent> {
        self.bus_tx.subscribe()
    }
}

impl Default for TelemetryStore {
    fn default() -> Self {
        Self::new()
    }
}
