// MetricRing: fixed-capacity circular buffer for MetricSample telemetry.
//
// Invariants:
//   - `buf` has exactly WINDOW slots; indices are always taken mod WINDOW.
//   - `head` points to the next write slot (one past the newest entry).
//   - `len` tracks how many valid samples are present (saturates at WINDOW).
//   - `window(n)` always returns samples in oldest-first order.
//
// Capacity: 36_000 samples = 60 minutes at 10 Hz (100 ms cadence) per GPU.
// Each MetricSample is ~72 bytes (9 × f32/u64 fields); total ~2.5 MB per GPU.

use nemesis_proto::telemetry::v1::MetricSample;

/// 60 minutes × 60 seconds × 10 samples/second = 36 000 samples.
const WINDOW: usize = 36_000;

/// Lock-free single-producer ring buffer for MetricSample.
///
/// This type is deliberately NOT Send/Sync on its own — callers wrap it in
/// `parking_lot::RwLock<HashMap<String, MetricRing>>` inside `TelemetryStore`.
pub struct MetricRing {
    buf: Vec<Option<MetricSample>>,
    /// Index of the *next* write slot (advances mod WINDOW on every push).
    head: usize,
    /// Number of valid entries; saturates at WINDOW once the ring is full.
    len: usize,
}

impl MetricRing {
    /// Construct an empty ring.  Allocates the full 36 000-slot backing vector
    /// immediately to avoid reallocations on the ingestion hot path.
    pub fn new() -> Self {
        Self {
            // Vec::with_capacity alone would require unsafe or push-based
            // initialisation; vec![None; N] gives us a fully initialised slice.
            buf: vec![None; WINDOW],
            head: 0,
            len: 0,
        }
    }

    /// Append `sample` to the ring.  O(1), no allocation after construction.
    ///
    /// When the ring is full the oldest sample is silently overwritten — this
    /// is the intended semantics for a sliding-window telemetry store.
    #[inline]
    pub fn push(&mut self, sample: MetricSample) {
        self.buf[self.head] = Some(sample);
        self.head = (self.head + 1) % WINDOW;
        // Saturate at WINDOW; once full every push overwrites the oldest slot.
        if self.len < WINDOW {
            self.len += 1;
        }
    }

    /// Return up to `n` of the most recent samples, in **oldest-first** order.
    ///
    /// If `n >= self.len` all stored samples are returned.
    /// Clones each sample — callers own the returned `Vec`.
    ///
    /// Complexity: O(min(n, len)) time and space.
    pub fn window(&self, n: usize) -> Vec<MetricSample> {
        // How many samples we actually return.
        let take = n.min(self.len);
        if take == 0 {
            return Vec::new();
        }

        // Index of the oldest valid sample.
        //
        // Layout after k pushes (k <= WINDOW):
        //   slot (head + WINDOW - len) % WINDOW  ← oldest
        //   ...
        //   slot (head - 1 + WINDOW)  % WINDOW  ← newest
        //
        // When we only want `take` < `len` samples we skip the first
        // `len - take` entries (the oldest ones we don't need).
        let oldest = (self.head + WINDOW - self.len) % WINDOW;
        let skip = self.len - take;

        (0..take)
            .filter_map(|i| {
                // SAFETY: every index in [oldest + skip + i] % WINDOW is within
                // the initialised `self.len` region of the ring, so the slot
                // is Some.  filter_map handles the None case defensively.
                self.buf[(oldest + skip + i) % WINDOW].clone()
            })
            .collect()
    }
}

impl Default for MetricRing {
    fn default() -> Self {
        Self::new()
    }
}
