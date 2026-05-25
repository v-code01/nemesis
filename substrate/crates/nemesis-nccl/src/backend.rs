//! NcclBackend trait and shared metric types.
//!
//! `NcclBackend` abstracts over the simulated (`NcclSim`) and real (`NcclReal`,
//! feature-gated on `cuda`) NCCL communicator implementations.  Both methods
//! accept protobuf request types directly so the service layer stays thin.
//!
//! Invariants:
//!   - `shrink`: `active_rank_count` in the returned metrics must equal the
//!     pre-call rank count minus `exclude_ranks.len()`.
//!   - `expand`: `active_rank_count` must equal pre-call count plus `new_gpu_ids.len()`.
//!   - `duration_ns` must be > 0 on success.
//!   - `error` is empty on success; non-empty on failure (even if `success == false`
//!     is also set).

use nemesis_proto::healer::v1::{ExpandRequest, ShrinkRequest};

/// Outcome of a communicator shrink operation.
#[derive(Debug, Clone)]
pub struct ShrinkMetrics {
    /// Whether the shrink succeeded.
    pub success: bool,
    /// Wall-clock duration of the entire operation in nanoseconds.
    pub duration_ns: u64,
    /// Number of ranks that remain active after shrinking.
    pub active_rank_count: u32,
    /// Human-readable error message; empty on success.
    pub error: String,
}

/// Outcome of a communicator expand operation.
#[derive(Debug, Clone)]
pub struct ExpandMetrics {
    /// Whether the expand succeeded.
    pub success: bool,
    /// Wall-clock duration of the entire operation in nanoseconds.
    pub duration_ns: u64,
    /// Number of ranks active after expanding.
    pub active_rank_count: u32,
    /// Human-readable error message; empty on success.
    pub error: String,
}

/// Backend trait that both the simulator and the real NCCL integration implement.
///
/// All methods are async to allow non-blocking simulation of I/O latency and
/// to support real FFI calls that may block a thread pool thread.
#[async_trait::async_trait]
pub trait NcclBackend: Send + Sync {
    /// Shrink the communicator by excluding the ranks in `req.exclude_ranks`.
    async fn shrink(&self, req: &ShrinkRequest) -> anyhow::Result<ShrinkMetrics>;

    /// Expand the communicator by admitting the GPUs in `req.new_gpu_ids`.
    async fn expand(&self, req: &ExpandRequest) -> anyhow::Result<ExpandMetrics>;
}
