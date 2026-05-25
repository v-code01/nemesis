//! Real NCCL hardware backend — compiled only when `feature = "cuda"`.
//!
//! This module provides a stub for the full hardware integration (Task 10).
//! The real implementation will:
//!   1. Notify the training process hook via a Unix domain socket to enter a
//!      safe point (AllReduce quiescence).
//!   2. Wait for the safe-point ACK with a configurable timeout.
//!   3. Call `ncclCommSplit` via C FFI (cudarc crate) to carve a sub-communicator.
//!   4. Call `ncclCommFinalize` + `ncclCommDestroy` on the excluded ranks'
//!      communicator handles.
//!   5. Notify the hook to resume training on the reduced rank set.
//!
//! Expand reverses the process using `ncclCommMerge` (NCCL 2.27+).
//!
//! Feature gate: `cuda` is intentionally NOT in `[features]` of the default
//! Cargo.toml so the crate builds in CI without a CUDA toolkit.

#[cfg(feature = "cuda")]
pub mod real {
    use crate::backend::{ExpandMetrics, NcclBackend, ShrinkMetrics};
    use nemesis_proto::healer::v1::{ExpandRequest, ShrinkRequest};

    /// Real NCCL communicator backend.
    ///
    /// `world_size` must match the number of ranks the process group was
    /// initialized with.  The `comm_handle` field (not yet present) will hold
    /// an opaque pointer to the `ncclComm_t` obtained from the training hook.
    pub struct NcclReal {
        pub world_size: u32,
    }

    #[async_trait::async_trait]
    impl NcclBackend for NcclReal {
        /// Shrink the NCCL communicator by excluding `req.exclude_ranks`.
        ///
        /// # Safety
        /// The full implementation will call C FFI via cudarc.  All FFI calls
        /// must be wrapped in `unsafe` blocks with documented preconditions.
        async fn shrink(&self, req: &ShrinkRequest) -> anyhow::Result<ShrinkMetrics> {
            let t0 = std::time::Instant::now();
            // TODO(Task 10): Unix socket notify → safe-point ACK → ncclCommSplit
            //   → ncclCommFinalize/Destroy on excluded ranks → hook resume.
            let duration_ns = t0.elapsed().as_nanos() as u64;
            Ok(ShrinkMetrics {
                success: true,
                duration_ns,
                active_rank_count: self.world_size - req.exclude_ranks.len() as u32,
                error: String::new(),
            })
        }

        /// Expand the NCCL communicator by merging `req.new_gpu_ids`.
        ///
        /// Requires NCCL 2.27+ for `ncclCommMerge`.
        async fn expand(&self, _req: &ExpandRequest) -> anyhow::Result<ExpandMetrics> {
            unimplemented!("expand via ncclCommMerge — Task 10")
        }
    }
}
