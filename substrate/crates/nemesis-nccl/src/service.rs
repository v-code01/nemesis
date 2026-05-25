//! gRPC service implementation for HealerService.
//!
//! `HealerServiceImpl` wires an `Arc<dyn NcclBackend>` into the generated tonic
//! server trait.  Job-to-communicator mapping and per-job world sizes are tracked
//! in `RwLock`-protected `HashMap`s so concurrent RPCs from different jobs do not
//! contend on a single mutex.
//!
//! Concurrency model:
//!   - `register_job`: acquires write locks on `job_comms` and `world_sizes`;
//!     serialises registration (acceptable — jobs register once at startup).
//!   - `shrink_communicator` / `expand_communicator`: calls through to the
//!     backend without holding any locks; the backend itself is `Send + Sync`.
//!   - `execute_playbook` / `list_playbooks`: stateless stubs; Phase 2 will
//!     dispatch into a persisted playbook library.

use crate::backend::NcclBackend;
use crate::sim::NcclSim;
use nemesis_proto::healer::v1::{
    healer_service_server::HealerService, ExpandRequest, ExpandResult, PlaybookLibrary,
    PlaybookRequest, PlaybookResult, RegisterJobRequest, RegisterJobResponse, ShrinkRequest,
    ShrinkResult,
};
use nemesis_proto::telemetry::v1::Void;
use parking_lot::RwLock;
use std::{collections::HashMap, sync::Arc};
use tonic::{Request, Response, Status};
use uuid::Uuid;

/// Concrete implementation of the `HealerService` gRPC service.
///
/// Use `HealerServiceImpl::new_sim` to construct an instance backed by the
/// simulated NCCL backend for development and testing.
pub struct HealerServiceImpl {
    /// Backend handling actual communicator operations.
    backend: Arc<dyn NcclBackend>,
    /// Maps job_id → communicator_id (UUID string).
    job_comms: Arc<RwLock<HashMap<String, String>>>,
    /// Maps job_id → world_size at registration time.
    world_sizes: Arc<RwLock<HashMap<String, u32>>>,
}

impl HealerServiceImpl {
    /// Construct a `HealerServiceImpl` backed by `NcclSim`.
    ///
    /// `default_world_size` seeds the simulator's rank count.  `seed` controls
    /// the LCG so that test scenarios produce deterministic latency sequences.
    pub fn new_sim(default_world_size: u32, seed: u64) -> Self {
        Self {
            backend: Arc::new(NcclSim::new(default_world_size, seed)),
            job_comms: Arc::new(RwLock::new(HashMap::new())),
            world_sizes: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[tonic::async_trait]
impl HealerService for HealerServiceImpl {
    /// Register a training job and return a new communicator UUID.
    ///
    /// Subsequent shrink/expand RPCs reference this communicator_id.
    async fn register_job(
        &self,
        req: Request<RegisterJobRequest>,
    ) -> Result<Response<RegisterJobResponse>, Status> {
        let r = req.into_inner();
        let comm_id = Uuid::new_v4().to_string();
        self.job_comms
            .write()
            .insert(r.job_id.clone(), comm_id.clone());
        self.world_sizes.write().insert(r.job_id, r.world_size);
        tracing::info!(comm_id = %comm_id, "job registered");
        Ok(Response::new(RegisterJobResponse {
            communicator_id: comm_id,
        }))
    }

    /// Shrink the NCCL communicator by removing `exclude_ranks`.
    async fn shrink_communicator(
        &self,
        req: Request<ShrinkRequest>,
    ) -> Result<Response<ShrinkResult>, Status> {
        let r = req.into_inner();
        tracing::info!(
            comm_id = %r.communicator_id,
            excluded = r.exclude_ranks.len(),
            "shrink requested"
        );
        let metrics = self
            .backend
            .shrink(&r)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(ShrinkResult {
            success: metrics.success,
            duration_ns: metrics.duration_ns,
            active_rank_count: metrics.active_rank_count,
            error: metrics.error,
        }))
    }

    /// Expand the NCCL communicator by admitting `new_gpu_ids`.
    async fn expand_communicator(
        &self,
        req: Request<ExpandRequest>,
    ) -> Result<Response<ExpandResult>, Status> {
        let r = req.into_inner();
        tracing::info!(
            comm_id = %r.communicator_id,
            added = r.new_gpu_ids.len(),
            "expand requested"
        );
        let metrics = self
            .backend
            .expand(&r)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(ExpandResult {
            success: metrics.success,
            duration_ns: metrics.duration_ns,
            active_rank_count: metrics.active_rank_count,
            error: metrics.error,
        }))
    }

    /// Execute a named playbook; returns a stub result.
    ///
    /// Phase 2 will dispatch into a persisted YAML playbook library and execute
    /// structured action sequences (drain → shrink → checkpoint → resume).
    async fn execute_playbook(
        &self,
        req: Request<PlaybookRequest>,
    ) -> Result<Response<PlaybookResult>, Status> {
        let r = req.into_inner();
        tracing::info!(playbook = %r.name, "playbook execution recorded");
        Ok(Response::new(PlaybookResult {
            success: true,
            duration_ns: 0,
            actions_taken: vec![format!("recorded: {}", r.name)],
            error: String::new(),
        }))
    }

    /// List available playbooks; returns an empty library (Phase 2 stub).
    async fn list_playbooks(
        &self,
        _req: Request<Void>,
    ) -> Result<Response<PlaybookLibrary>, Status> {
        Ok(Response::new(PlaybookLibrary { playbooks: vec![] }))
    }
}
