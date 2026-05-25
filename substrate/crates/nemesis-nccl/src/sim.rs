//! Simulated NCCL backend for testing and staging environments.
//!
//! `NcclSim` uses a seeded LCG PRNG to produce deterministic, plausible
//! latencies without needing the `rand` crate.  The LCG parameters are from
//! Knuth (MMIX): multiplier 6364136223846793005, addend 1442695040888963407.
//!
//! Latency model:
//!   - Mean: 4 000 000 000 ns  (4 s — realistic ncclCommSplit on 8-GPU host)
//!   - Sigma: 500 000 000 ns   (0.5 s jitter)
//!   - Floor: 100 000 000 ns   (100 ms — never unrealistically fast)
//!   - For tests: actual `tokio::time::sleep` is capped at 10 ms so test
//!     suites complete in < 1 s, while `duration_ns` still reflects realistic
//!     simulated values.
//!
//! Thread safety: `SimState` is protected by a `parking_lot::Mutex` so the
//! struct is `Send + Sync` and therefore satisfies `NcclBackend`.

use crate::backend::{ExpandMetrics, NcclBackend, ShrinkMetrics};
use nemesis_proto::healer::v1::{ExpandRequest, ShrinkRequest};
use parking_lot::Mutex;
use std::sync::Arc;

/// Mean simulated shrink latency in nanoseconds (4 seconds).
const SHRINK_MEAN_NS: u64 = 4_000_000_000;
/// Sigma of simulated shrink latency in nanoseconds (0.5 seconds).
const SHRINK_SIGMA_NS: u64 = 500_000_000;
/// Minimum simulated latency floor in nanoseconds (100 ms).
const SHRINK_FLOOR_NS: u64 = 100_000_000;
/// Maximum real sleep injected during tests (10 ms — avoids suite timeouts).
const MAX_REAL_SLEEP_NS: u64 = 10_000_000;

struct SimState {
    /// Current number of active ranks after shrinks/expands.
    active_ranks: u32,
    /// LCG state — advances on every duration sample.
    seed: u64,
}

/// Simulated NCCL communicator backend.
///
/// Create one instance per test or per job registration in the service layer.
/// The `seed` parameter makes outcomes fully deterministic for property-based
/// and regression tests.
pub struct NcclSim {
    state: Arc<Mutex<SimState>>,
}

impl NcclSim {
    /// Create a new simulator for a communicator with `world_size` initial ranks.
    ///
    /// `seed` controls the LCG so that callers can reproduce latency sequences.
    pub fn new(world_size: u32, seed: u64) -> Self {
        Self {
            state: Arc::new(Mutex::new(SimState {
                active_ranks: world_size,
                seed,
            })),
        }
    }

    /// Advance the LCG and return a realistic shrink duration in nanoseconds.
    ///
    /// Uses a signed jitter in [-SIGMA, +SIGMA) derived from the high 31 bits
    /// of the LCG output, then clamps to SHRINK_FLOOR_NS.
    fn seeded_duration_ns(state: &mut SimState) -> u64 {
        // Knuth MMIX LCG — full period over u64.
        state.seed = state
            .seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Extract signed jitter from upper 31 bits to avoid modulo bias over the full u64.
        let jitter = ((state.seed >> 33) as i64) % (SHRINK_SIGMA_NS as i64);
        let raw = SHRINK_MEAN_NS as i64 + jitter;
        raw.max(SHRINK_FLOOR_NS as i64) as u64
    }
}

#[async_trait::async_trait]
impl NcclBackend for NcclSim {
    /// Simulate a shrink: sample latency, sleep ≤ 10 ms, decrement active ranks.
    async fn shrink(&self, req: &ShrinkRequest) -> anyhow::Result<ShrinkMetrics> {
        // Sample duration while holding the lock for a minimal critical section.
        let duration_ns = {
            let mut state = self.state.lock();
            Self::seeded_duration_ns(&mut state)
        };

        // Inject a tiny real sleep so async machinery executes; cap at 10 ms so
        // test suites are not slowed to real NCCL timescales.
        tokio::time::sleep(tokio::time::Duration::from_nanos(
            duration_ns.min(MAX_REAL_SLEEP_NS),
        ))
        .await;

        let active_rank_count = {
            let mut state = self.state.lock();
            state.active_ranks = state
                .active_ranks
                .saturating_sub(req.exclude_ranks.len() as u32);
            state.active_ranks
        };

        Ok(ShrinkMetrics {
            success: true,
            duration_ns,
            active_rank_count,
            error: String::new(),
        })
    }

    /// Simulate an expand: sample latency, sleep ≤ 10 ms, increment active ranks.
    async fn expand(&self, req: &ExpandRequest) -> anyhow::Result<ExpandMetrics> {
        let duration_ns = {
            let mut state = self.state.lock();
            Self::seeded_duration_ns(&mut state)
        };

        tokio::time::sleep(tokio::time::Duration::from_nanos(
            duration_ns.min(MAX_REAL_SLEEP_NS),
        ))
        .await;

        let active_rank_count = {
            let mut state = self.state.lock();
            // Saturating add prevents wrapping if ranks somehow exceeds world_size.
            state.active_ranks = state
                .active_ranks
                .saturating_add(req.new_gpu_ids.len() as u32);
            state.active_ranks
        };

        Ok(ExpandMetrics {
            success: true,
            duration_ns,
            active_rank_count,
            error: String::new(),
        })
    }
}
