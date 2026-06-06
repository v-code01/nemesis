//! gRPC service implementation for the SchedulerService.
//!
//! `SchedulerServiceImpl` wires the topology DSL pipeline (parse → type_check → solve)
//! into the generated tonic server trait.
//!
//! # Reservation lifecycle
//!
//! ```text
//! schedule(job) → solver assigns gpu_ids → reserved += gpu_ids
//!                                          job_reservations[job_id] = gpu_ids
//! release(job)  → reserved -= job_reservations[job_id]
//!                 job_reservations.remove(job_id)
//! ```
//!
//! # Concurrency model
//!
//!   - `ClusterGraph` mutations happen from the telemetry service; both services
//!     share the same `Arc<RwLock<ClusterGraph>>`.
//!   - `validate` and `get_topology` acquire only a read lock on the graph.
//!   - `schedule` holds the `reserved` write lock through the entire solve + insert,
//!     serialising all placement decisions.  Correct for the current single-region
//!     control plane; a multi-region extension would need per-rack locking.
//!   - `release` acquires the `reserved` write lock, removes the job's GPUs, and
//!     drops the job_reservations entry atomically with respect to other RPCs.

use crate::{checker::type_check, parser::parse, solver::TopologySolver};
use nemesis_graph::ClusterGraph;
use nemesis_proto::telemetry::v1::{ClusterTopology, Void};
use nemesis_proto::topology::v1::{
    scheduler_service_server::SchedulerService, JobSpec, PlacementResult as ProtoPlacementResult,
    ReleaseRequest, ValidationResult,
};
use parking_lot::RwLock;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tonic::{Request, Response, Status};

// ---------------------------------------------------------------------------
// Service impl
// ---------------------------------------------------------------------------

/// Concrete implementation of the `SchedulerService` gRPC service.
pub struct SchedulerServiceImpl {
    /// Placement engine; holds a clone of the `Arc` to the same graph.
    solver: TopologySolver,
    /// Shared cluster graph -- used directly for `get_topology`.
    graph: Arc<RwLock<ClusterGraph>>,
    /// Set of gpu_ids currently reserved by active jobs.
    reserved: Arc<RwLock<HashSet<String>>>,
    /// Maps job_id → Vec<gpu_id> for the GPUs reserved by that job.
    /// Used by `release` to reclaim exactly the GPUs allocated to a job.
    job_reservations: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl SchedulerServiceImpl {
    /// Construct a new service instance backed by `graph`.
    pub fn new(graph: Arc<RwLock<ClusterGraph>>) -> Self {
        Self {
            solver: TopologySolver::new(graph.clone()),
            graph,
            reserved: Arc::new(RwLock::new(HashSet::new())),
            job_reservations: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

// ---------------------------------------------------------------------------
// Trait impl
// ---------------------------------------------------------------------------

#[tonic::async_trait]
impl SchedulerService for SchedulerServiceImpl {
    /// Parse and type-check the DSL without attempting placement.
    async fn validate(
        &self,
        request: Request<JobSpec>,
    ) -> Result<Response<ValidationResult>, Status> {
        let spec_str = &request.get_ref().topology_dsl;

        let result = match parse(spec_str) {
            Err(e) => ValidationResult {
                valid: false,
                errors: vec![e],
            },
            Ok(spec) => {
                let errors = type_check(&spec);
                ValidationResult {
                    valid: errors.is_empty(),
                    errors,
                }
            }
        };

        Ok(Response::new(result))
    }

    async fn schedule(
        &self,
        request: Request<JobSpec>,
    ) -> Result<Response<ProtoPlacementResult>, Status> {
        let job = request.get_ref();
        let spec = parse(&job.topology_dsl).map_err(Status::invalid_argument)?;
        let errors = type_check(&spec);
        if !errors.is_empty() {
            return Ok(Response::new(ProtoPlacementResult {
                placed: false,
                gpu_ids: vec![],
                rejection_reason: errors.join("; "),
                subgraph: None,
            }));
        }

        // Hold both write locks for the entire solve + insert to prevent
        // double-allocation across concurrent RPCs.
        let mut reserved = self.reserved.write();
        let mut job_res = self.job_reservations.write();

        let result = self.solver.solve(&spec);
        if result.placed {
            // Post-solve conflict check: another RPC may have allocated these
            // GPUs between when the graph was read and now.
            if result.gpu_ids.iter().any(|id| reserved.contains(id)) {
                return Ok(Response::new(ProtoPlacementResult {
                    placed: false,
                    gpu_ids: vec![],
                    rejection_reason: "GPUs already reserved by a concurrent job".to_string(),
                    subgraph: None,
                }));
            }
            for id in &result.gpu_ids {
                reserved.insert(id.clone());
            }
            // Record which GPUs belong to this job for release.
            job_res.insert(job.job_id.clone(), result.gpu_ids.clone());
        }

        Ok(Response::new(ProtoPlacementResult {
            placed: result.placed,
            gpu_ids: result.gpu_ids,
            rejection_reason: result.rejection_reason,
            subgraph: None,
        }))
    }

    /// Release GPU reservations held by `job_id`.
    ///
    /// All GPUs previously allocated to this job are returned to the free pool
    /// so the scheduler can place new jobs on them.  Idempotent: releasing an
    /// already-released or never-scheduled job_id is a no-op.
    async fn release(
        &self,
        request: Request<ReleaseRequest>,
    ) -> Result<Response<Void>, Status> {
        let job_id = &request.get_ref().job_id;

        let mut reserved = self.reserved.write();
        let mut job_res = self.job_reservations.write();

        if let Some(gpu_ids) = job_res.remove(job_id.as_str()) {
            let count = gpu_ids.len();
            for id in &gpu_ids {
                reserved.remove(id);
            }
            tracing::info!(job_id = %job_id, released = count, "GPU reservations released");
        } else {
            tracing::debug!(job_id = %job_id, "release: no reservations found (idempotent)");
        }

        Ok(Response::new(Void {}))
    }

    /// Return a proto-serialised snapshot of the full cluster topology.
    async fn get_topology(
        &self,
        _request: Request<Void>,
    ) -> Result<Response<ClusterTopology>, Status> {
        Ok(Response::new(self.graph.read().to_proto()))
    }
}
