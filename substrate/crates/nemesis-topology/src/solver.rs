//! Topology solver: maps a `TopologySpec` onto a live `ClusterGraph`.
//!
//! Design:
//!   - `TopologySolver` holds a shared read-lock handle to `ClusterGraph`.
//!   - `solve` dispatches on the spec variant:
//!       * Atom   → delegates to `find_nvlink_clique` (TP) or `find_ib_path` (PP)
//!                  or a greedy healthy-GPU pick (DP).
//!       * Conjunction → solves each arm independently; GPUs are unioned (may
//!                       overlap in a real system — callers dedup as needed).
//!       * Disjunction → tries each alternative in order, returns the first success.
//!
//! Invariants maintained:
//!   - The graph is never mutated inside the solver (read-lock only).
//!   - `PlacementResult::placed == true` iff `gpu_ids` is non-empty and
//!     represents a valid assignment.
//!   - `PlacementResult::placed == false` iff `rejection_reason` is non-empty.

use crate::parser::{Constraint, ParallelDim, TopologySpec};
use nemesis_graph::ClusterGraph;
use parking_lot::RwLock;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The outcome of a single `TopologySolver::solve` call.
pub struct PlacementResult {
    /// True when the solver found a valid GPU assignment.
    pub placed:           bool,
    /// The gpu_ids assigned to this job; empty when `placed == false`.
    pub gpu_ids:          Vec<String>,
    /// Human-readable reason for rejection; empty when `placed == true`.
    pub rejection_reason: String,
}

impl PlacementResult {
    /// Construct a successful placement.
    #[inline]
    fn ok(gpu_ids: Vec<String>) -> Self {
        Self { placed: true, gpu_ids, rejection_reason: String::new() }
    }

    /// Construct a failed placement with a reason.
    #[inline]
    fn rejected(reason: impl Into<String>) -> Self {
        Self { placed: false, gpu_ids: Vec::new(), rejection_reason: reason.into() }
    }
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Stateless placement engine that operates on a shared `ClusterGraph`.
///
/// `TopologySolver` holds a reference-counted read-write lock over the graph.
/// All `solve` calls acquire only a read lock and never modify the graph.
pub struct TopologySolver {
    graph: Arc<RwLock<ClusterGraph>>,
}

impl TopologySolver {
    /// Construct a solver backed by the given shared graph.
    pub fn new(graph: Arc<RwLock<ClusterGraph>>) -> Self {
        Self { graph }
    }

    /// Attempt to place `spec` on the current cluster state.
    ///
    /// Returns a `PlacementResult` whose `placed` field indicates success.
    /// Does not mutate the graph or the reservation set; callers are responsible
    /// for marking GPUs reserved after a successful placement.
    pub fn solve(&self, spec: &TopologySpec) -> PlacementResult {
        match spec {
            TopologySpec::Atom(dim, constraints) => self.solve_atom(dim, constraints),
            TopologySpec::Conjunction(l, r) => self.solve_conjunction(l, r),
            TopologySpec::Disjunction(alts) => self.solve_disjunction(alts),
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Resolve a single `Atom` node against the graph.
    ///
    /// - TP → NVLink clique search; requires min_bw from any `NvlMin` constraint.
    /// - PP → IB path search; respects `IbMax` hop limit.
    /// - DP → greedy first-N healthy GPUs (no topology constraint beyond count).
    fn solve_atom(&self, dim: &ParallelDim, constraints: &[Constraint]) -> PlacementResult {
        let graph = self.graph.read();
        match dim {
            ParallelDim::Tp(n) => {
                // Extract the minimum NVLink bandwidth from constraints.
                // Absence of the constraint is interpreted as 0 (any bandwidth).
                let min_bw = constraints
                    .iter()
                    .find_map(|c| {
                        if let Constraint::NvlMin(bw) = c { Some(*bw) } else { None }
                    })
                    .unwrap_or(0.0_f32);

                match graph.find_nvlink_clique(*n as usize, min_bw) {
                    Some(ids) => PlacementResult::ok(ids),
                    None => PlacementResult::rejected(format!(
                        "no NVLink clique of size {n} with bandwidth >= {min_bw} GB/s"
                    )),
                }
            }

            ParallelDim::Pp(n) => {
                // Extract the maximum IB hop count from constraints.
                // Absence means we accept any hop count (u32::MAX).
                let max_hops = constraints
                    .iter()
                    .find_map(|c| {
                        if let Constraint::IbMax(h) = c { Some(*h) } else { None }
                    })
                    .unwrap_or(u32::MAX);

                match graph.find_ib_path(*n as usize, max_hops) {
                    Some(ids) => PlacementResult::ok(ids),
                    None => PlacementResult::rejected(format!(
                        "no IB path of length {n} with hop count <= {max_hops}"
                    )),
                }
            }

            ParallelDim::Dp(n) => {
                // DP has no topological constraint beyond needing N healthy GPUs.
                let ids: Vec<String> = graph
                    .healthy_gpu_ids()
                    .into_iter()
                    .take(*n as usize)
                    .collect();

                if ids.len() == *n as usize {
                    PlacementResult::ok(ids)
                } else {
                    PlacementResult::rejected(format!(
                        "insufficient healthy GPUs for DP{n}: need {n}, have {}",
                        ids.len()
                    ))
                }
            }
        }
    }

    /// Solve a `Conjunction`: both arms must succeed independently.
    ///
    /// GPU sets are concatenated. In a real scheduler the allocator would
    /// need to ensure disjointness; that invariant is enforced at reservation
    /// time in the service layer, not here.
    fn solve_conjunction(&self, l: &TopologySpec, r: &TopologySpec) -> PlacementResult {
        let left = self.solve(l);
        if !left.placed {
            return left; // short-circuit: no point trying right
        }
        let right = self.solve(r);
        if !right.placed {
            return right;
        }
        let mut combined = left.gpu_ids;
        combined.extend(right.gpu_ids);
        PlacementResult::ok(combined)
    }

    /// Solve a `Disjunction`: try each alternative in declaration order; return
    /// the first successful placement.
    ///
    /// If all alternatives fail, return a rejection that names the strategy.
    fn solve_disjunction(&self, alts: &[TopologySpec]) -> PlacementResult {
        for alt in alts {
            let result = self.solve(alt);
            if result.placed {
                return result;
            }
        }
        PlacementResult::rejected("no alternative could be placed on the current cluster")
    }
}
