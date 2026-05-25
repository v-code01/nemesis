//! gRPC service implementation for the SchedulerService.
//!
//! `SchedulerServiceImpl` wires the topology DSL pipeline (parse → type_check → solve)
//! into the generated tonic server trait.  A `parking_lot::RwLock<HashSet<String>>` tracks
//! the set of currently-reserved gpu_ids so that a future `release` RPC can reclaim them.
//!
//! Concurrency model:
//!   - `ClusterGraph` mutations (mark_unhealthy etc.) happen from the telemetry service;
//!     both services share the same `Arc<RwLock<ClusterGraph>>`.
//!   - `validate` and `get_topology` acquire only a read lock.
//!   - `schedule` acquires the `reserved` write lock before calling the solver and holds
//!     it through the insert, serialising all placement decisions for Phase 1 correctness.
//!   - `release` is a Phase 1 stub; no GPU reservations are removed.

use crate::{checker::type_check, parser::parse, solver::TopologySolver};
use nemesis_graph::ClusterGraph;
use nemesis_proto::topology::v1::{
    scheduler_service_server::SchedulerService, JobSpec, PlacementResult as ProtoPlacementResult,
    ReleaseRequest, ValidationResult,
};
use nemesis_proto::telemetry::v1::{ClusterTopology, Void};
use parking_lot::RwLock;
use std::{collections::HashSet, sync::Arc};
use tonic::{Request, Response, Status};

// ---------------------------------------------------------------------------
// Service impl
// ---------------------------------------------------------------------------

/// Concrete implementation of the `SchedulerService` gRPC service.
///
/// Wraps:
///   - A shared `ClusterGraph` (read for solve + topology export).
///   - A `TopologySolver` over that same graph.
///   - A reservation set tracking which gpu_ids are currently allocated.
pub struct SchedulerServiceImpl {
    /// Placement engine; holds a clone of the `Arc` to the same graph.
    solver: TopologySolver,
    /// Shared cluster graph — used directly for `get_topology`.
    graph: Arc<RwLock<ClusterGraph>>,
    /// Set of gpu_ids currently reserved by active jobs.
    /// Separate lock from the graph to allow independent mutation.
    reserved: Arc<RwLock<HashSet<String>>>,
}

impl SchedulerServiceImpl {
    /// Construct a new service instance backed by `graph`.
    ///
    /// Both the solver and the service share the same `Arc<RwLock<ClusterGraph>>`
    /// so topology mutations are visible to all future solve calls without cloning.
    pub fn new(graph: Arc<RwLock<ClusterGraph>>) -> Self {
        Self {
            solver: TopologySolver::new(graph.clone()),
            graph,
            reserved: Arc::new(RwLock::new(HashSet::new())),
        }
    }
}

// ---------------------------------------------------------------------------
// Trait impl
// ---------------------------------------------------------------------------

#[tonic::async_trait]
impl SchedulerService for SchedulerServiceImpl {
    /// Parse and type-check the DSL without attempting placement.
    ///
    /// Useful for client-side pre-flight validation before submitting a job.
    async fn validate(
        &self,
        request: Request<JobSpec>,
    ) -> Result<Response<ValidationResult>, Status> {
        let spec_str = &request.get_ref().topology_dsl;

        let result = match parse(spec_str) {
            Err(e) => ValidationResult { valid: false, errors: vec![e] },
            Ok(spec) => {
                let errors = type_check(&spec);
                ValidationResult { valid: errors.is_empty(), errors }
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
        // Hold the write lock for the entire solve + insert to prevent double-allocation
        // across concurrent RPCs. Serialised scheduling is correct for Phase 1.
        let mut reserved = self.reserved.write();
        let result = self.solver.solve(&spec);
        if result.placed {
            // Post-solve conflict check: another RPC may have allocated these GPUs
            // between when the graph was read and now.
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
        }
        Ok(Response::new(ProtoPlacementResult {
            placed: result.placed,
            gpu_ids: result.gpu_ids,
            rejection_reason: result.rejection_reason,
            subgraph: None,
        }))
    }

    // Phase 1 stub: release does not yet remove GPU reservations.
    // Full implementation requires a job_id → gpu_ids map (Phase 2).
    async fn release(
        &self,
        _request: Request<ReleaseRequest>,
    ) -> Result<Response<Void>, Status> {
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
