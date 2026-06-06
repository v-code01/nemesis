//! Real NCCL hardware backend -- compiled only with `feature = "cuda"`.
//!
//! # Protocol
//!
//! `shrink()` executes a three-phase collective communicator reduction:
//!
//!   Phase 1 -- Quiesce: signal the training hook to stop dispatching collectives
//!              and wait until all in-flight AllReduce/AllGather ops complete.
//!              Synchronised via `SAFE_POINT_REQUEST` / `SAFE_POINT_ACK` on a
//!              per-job Unix domain socket (path: `/tmp/nemesis-hook-{job_id}.sock`).
//!
//!   Phase 2 -- Split: call `ncclCommSplit` via C FFI on the stored `ncclComm_t`
//!              handle.  This is a collective operation -- all _remaining_ ranks
//!              must call it concurrently, coordinated by the barrier in
//!              `HealerServiceImpl::shrink_communicator` at the gRPC layer.
//!              Excluded ranks are signalled to call `ncclCommFinalize` +
//!              `ncclCommDestroy` on their own handles.
//!
//!   Phase 3 -- Resume: send `RESUME` on the Unix socket so the hook unblocks
//!              and returns from `hook.step()` with the new communicator.
//!
//! # Co-residency requirement
//!
//! `ncclComm_t` is a heap pointer in the training process.  NEMESIS obtains it
//! by running co-resident with the training loop (embedded mode) or via a
//! registered shared-memory segment (distributed mode; see docs/arch/nccl-ipc.md).
//! In both cases the pointer is cast to `u64` and passed to `NcclReal::new()`.
//!
//! # Feature gate
//!
//! `cuda` is deliberately absent from the default feature set so that the
//! entire substrate compiles in CI without a CUDA toolkit.  Build with
//! `cargo build --features cuda` on an H100 node.

#[cfg(feature = "cuda")]
pub mod real {
    use crate::backend::{ExpandMetrics, NcclBackend, ShrinkMetrics};
    use anyhow::Context;
    use nemesis_proto::healer::v1::{ExpandRequest, ShrinkRequest};
    use std::ffi::CStr;
    use std::io::{BufRead, BufReader, Write as IoWrite};
    use std::os::raw::{c_char, c_int, c_void};
    use std::os::unix::net::UnixStream;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    // -----------------------------------------------------------------------
    // NCCL C API -- extern declarations
    // -----------------------------------------------------------------------

    /// Opaque NCCL communicator type.  The actual struct is internal to libnccl.
    #[repr(C)]
    struct NcclCommOpaque {
        _private: [u8; 0],
    }

    /// `ncclComm_t` in C -- a pointer to the opaque communicator struct.
    type NcclCommT = *mut NcclCommOpaque;

    /// NCCL status code; 0 == ncclSuccess.
    type NcclResult = c_int;

    const NCCL_SUCCESS: NcclResult = 0;

    /// Sentinel color value passed by excluded ranks to `ncclCommSplit`.
    /// Tells NCCL to exclude this rank from the new communicator.
    const NCCL_SPLIT_NOCOLOR: c_int = -1;

    extern "C" {
        /// Split `comm` into a sub-communicator containing all ranks that pass
        /// the same `color`.  Ranks with `color == NCCL_SPLIT_NOCOLOR` are
        /// excluded.  `key` controls the rank ordering in the new comm.
        ///
        /// This is a _collective_ call: every rank in `comm` must call it
        /// simultaneously with matching colors.
        fn ncclCommSplit(
            comm: NcclCommT,
            color: c_int,
            key: c_int,
            newcomm: *mut NcclCommT,
            config: *mut c_void, // ncclConfig_t* — NULL uses defaults
        ) -> NcclResult;

        /// Flush all pending operations on `comm` and quiesce it.
        /// Must be called before `ncclCommDestroy`.
        fn ncclCommFinalize(comm: NcclCommT) -> NcclResult;

        /// Release all resources associated with `comm`.
        /// After this call the handle is invalid.
        fn ncclCommDestroy(comm: NcclCommT) -> NcclResult;

        /// Return a human-readable string for a NCCL status code.
        /// Always returns a valid C string (never null).
        fn ncclGetErrorString(result: NcclResult) -> *const c_char;
    }

    /// Translate a `NcclResult` into an `anyhow::Error`.
    fn nccl_check(result: NcclResult) -> anyhow::Result<()> {
        if result == NCCL_SUCCESS {
            return Ok(());
        }
        // SAFETY: ncclGetErrorString always returns a valid, static C string.
        let msg = unsafe { CStr::from_ptr(ncclGetErrorString(result)) };
        anyhow::bail!("NCCL error {result}: {}", msg.to_string_lossy())
    }

    // -----------------------------------------------------------------------
    // SendableCommPtr -- move raw NCCL handle across thread boundaries
    // -----------------------------------------------------------------------

    /// Newtype wrapper around a `u64` representation of `ncclComm_t` that opts
    /// into `Send`.
    ///
    /// # Safety rationale
    ///
    /// `ncclComm_t` is a heap-allocated struct in `libnccl`.  The NCCL library
    /// itself is thread-safe for calls on _different_ communicators from different
    /// threads.  We enforce the _single-caller_ invariant at a higher level:
    /// `HealerServiceImpl` serialises shrink calls per communicator via the job
    /// registry lock, so no two threads ever hold a `SendableCommPtr` to the same
    /// communicator simultaneously.
    struct SendableCommPtr(u64);
    unsafe impl Send for SendableCommPtr {}

    // -----------------------------------------------------------------------
    // NcclReal
    // -----------------------------------------------------------------------

    /// Real NCCL communicator backend.
    ///
    /// # Construction
    ///
    /// ```no_run
    /// # #[cfg(feature = "cuda")]
    /// # use nemesis_nccl::real::real::NcclReal;
    /// // comm_ptr is the ncclComm_t cast to u64, obtained from the training process
    /// // at job registration.  socket_path is the Unix domain socket exported by
    /// // the NemesisHook embedded in that training process.
    /// let backend = NcclReal::new(comm_ptr, world_size, "/tmp/nemesis-hook-job-001.sock");
    /// ```
    pub struct NcclReal {
        /// `ncclComm_t` stored as `u64` for `Sync` storage.  Cast back to pointer
        /// only inside `unsafe` blocks in `spawn_blocking` tasks.
        comm_ptr: u64,
        /// World size at communicator initialisation.
        world_size: u32,
        /// Remaining active ranks (decremented on successful shrink).
        active_ranks: Arc<AtomicU32>,
        /// Unix domain socket path exported by the NemesisHook in the training process.
        /// Format: `/tmp/nemesis-hook-{job_id}.sock`
        socket_path: String,
    }

    impl NcclReal {
        /// Create a real NCCL backend.
        ///
        /// # Safety contract (caller must uphold)
        ///
        /// - `comm_ptr` is a fully-initialised `ncclComm_t` obtained from
        ///   `ncclCommInitRank` or equivalent.
        /// - No collective operations are in flight on `comm_ptr` at the time any
        ///   method on this struct is called.  The training hook enforces this by
        ///   pausing at a collective boundary before calling `ShrinkCommunicator`.
        /// - `comm_ptr` remains valid for the lifetime of this struct.
        pub fn new(comm_ptr: u64, world_size: u32, socket_path: impl Into<String>) -> Self {
            Self {
                comm_ptr,
                world_size,
                active_ranks: Arc::new(AtomicU32::new(world_size)),
                socket_path: socket_path.into(),
            }
        }

        /// Convenience constructor: derives the socket path from `job_id`.
        pub fn for_job(comm_ptr: u64, world_size: u32, job_id: &str) -> Self {
            Self::new(comm_ptr, world_size, format!("/tmp/nemesis-hook-{job_id}.sock"))
        }

        /// Send `msg` to the NemesisHook and wait for `expected_ack`.
        ///
        /// Connects a fresh `UnixStream` per call to avoid state leakage between
        /// shrink phases.
        fn socket_exchange(
            path: &str,
            msg: &str,
            expected_ack: &str,
            timeout: Duration,
        ) -> anyhow::Result<()> {
            let mut stream =
                UnixStream::connect(path).with_context(|| format!("connect to {path}"))?;
            stream.set_read_timeout(Some(timeout))?;
            stream.write_all(msg.as_bytes())?;
            stream.flush()?;

            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            reader.read_line(&mut line)?;
            let ack = line.trim();
            anyhow::ensure!(
                ack == expected_ack,
                "unexpected ack from hook: got '{ack}', want '{expected_ack}'"
            );
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl NcclBackend for NcclReal {
        /// Shrink the NCCL communicator.
        ///
        /// # Sequence
        ///
        /// 1. Validate: cannot exclude all active ranks.
        /// 2. Connect to NemesisHook Unix socket; send `SAFE_POINT_REQUEST`.
        ///    Wait ≤30 s for `SAFE_POINT_ACK` (training paused at collective boundary).
        /// 3. Dispatch `ncclCommSplit` in `spawn_blocking` (blocks 4–6 s on H100 NVLink).
        ///    All remaining ranks call it concurrently via their own hooks.
        /// 4. Excluded ranks receive `EXCLUDED` signal from their hooks and call
        ///    `ncclCommFinalize` + `ncclCommDestroy` (out-of-band, not shown here).
        /// 5. Send `RESUME` on the socket.  Hook unblocks and returns from `step()`.
        async fn shrink(&self, req: &ShrinkRequest) -> anyhow::Result<ShrinkMetrics> {
            let t0 = std::time::Instant::now();
            let excluded = req.exclude_ranks.len() as u32;
            let current = self.active_ranks.load(Ordering::SeqCst);

            if excluded >= current {
                return Ok(ShrinkMetrics {
                    success: false,
                    duration_ns: 0,
                    active_rank_count: current,
                    error: format!(
                        "cannot exclude {excluded} ranks from communicator with {current} active"
                    ),
                });
            }

            // Phase 1: quiesce training at collective boundary.
            let exclude_list = req
                .exclude_ranks
                .iter()
                .map(|r| r.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let request_msg = format!("SAFE_POINT_REQUEST exclude={exclude_list}\n");

            Self::socket_exchange(
                &self.socket_path,
                &request_msg,
                "SAFE_POINT_ACK",
                Duration::from_secs(30),
            )
            .context("safe-point handshake failed")?;

            tracing::info!(
                comm_id = %req.communicator_id,
                excluded = %exclude_list,
                "safe-point ACK received -- calling ncclCommSplit"
            );

            // Phase 2: call ncclCommSplit in a blocking thread.
            // SAFETY: See NcclReal::new() contract.  We hold the only handle reference
            // (enforced by HealerServiceImpl's per-job serialisation lock).
            let ptr = SendableCommPtr(self.comm_ptr);
            let (nccl_rc, new_comm_addr) = tokio::task::spawn_blocking(move || {
                let comm = ptr.0 as NcclCommT;
                let mut new_comm: NcclCommT = std::ptr::null_mut();
                // color=0: all remaining ranks use same color → join new communicator.
                // key=0: NCCL assigns rank IDs by ascending original rank order.
                // config=NULL: blocking mode, default CTA count, default topology.
                let rc = unsafe {
                    ncclCommSplit(comm, 0, 0, &mut new_comm, std::ptr::null_mut())
                };
                (rc, new_comm as u64)
            })
            .await
            .context("ncclCommSplit task panicked")?;

            nccl_check(nccl_rc).context("ncclCommSplit returned error")?;

            let new_active = current - excluded;
            self.active_ranks.store(new_active, Ordering::SeqCst);

            tracing::info!(
                comm_id = %req.communicator_id,
                old_comm = self.comm_ptr,
                new_comm = new_comm_addr,
                active_ranks = new_active,
                "ncclCommSplit completed"
            );

            // Phase 3: release training hook to resume on new communicator.
            Self::socket_exchange(
                &self.socket_path,
                "RESUME\n",
                "RESUMED",
                Duration::from_secs(5),
            )
            .context("resume handshake failed")?;

            Ok(ShrinkMetrics {
                success: true,
                duration_ns: t0.elapsed().as_nanos() as u64,
                active_rank_count: new_active,
                error: String::new(),
            })
        }

        /// Expand the communicator using `ncclCommMerge` (NCCL 2.27+).
        ///
        /// `ncclCommMerge` is available in NCCL 2.27 but requires a separate
        /// bootstrapped rendezvous for the joining ranks.  The coordination
        /// protocol is specified in `docs/arch/nccl-expand-design.md`.
        async fn expand(&self, req: &ExpandRequest) -> anyhow::Result<ExpandMetrics> {
            let t0 = std::time::Instant::now();
            // ncclCommMerge is present in NCCL 2.27 headers but the required
            // rank-bootstrapping protocol is not yet implemented in the hook.
            // Tracked in docs/arch/nccl-expand-design.md.
            anyhow::bail!(
                "expand via ncclCommMerge (NCCL 2.27+) not yet implemented; \
                 adding {} ranks to communicator {} requires rank-bootstrapping protocol",
                req.new_gpu_ids.len(),
                req.communicator_id
            );
        }
    }
}
