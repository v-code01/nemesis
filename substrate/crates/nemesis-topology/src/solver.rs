//! Topology solver: maps a `TopologySpec` onto a live `ClusterGraph`.
//!
//! Design:
//!   - `TopologySolver` holds a shared read-lock handle to `ClusterGraph`.
//!   - `solve` dispatches on the spec variant:
//!       * Atom — delegates to `find_nvlink_clique` (TP) or `find_ib_path` (PP)
//!         or a greedy healthy-GPU pick (DP).
//!       * Conjunction — solves each arm; rejects if arms overlap (disjoint GPU sets required).
//!       * Disjunction — tries each alternative in order, returns the first success.
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
    pub fn ok(gpu_ids: Vec<String>) -> Self {
        Self { placed: true, gpu_ids, rejection_reason: String::new() }
    }

    /// Construct a failed placement with a reason.
    #[inline]
    pub fn rejected(reason: impl Into<String>) -> Self {
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
                // Sort for deterministic placement across calls.
                let mut all_healthy = graph.healthy_gpu_ids();
                all_healthy.sort();
                let ids: Vec<String> = all_healthy.into_iter().take(*n as usize).collect();

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

    /// Solve a `Conjunction`: both arms must succeed independently and with disjoint GPUs.
    fn solve_conjunction(&self, l: &TopologySpec, r: &TopologySpec) -> PlacementResult {
        let left = self.solve(l);
        if !left.placed {
            return left;
        }
        let right = self.solve(r);
        if !right.placed {
            return right;
        }
        // Enforce disjointness: conjunction arms must not share GPUs.
        let left_set: std::collections::HashSet<&str> =
            left.gpu_ids.iter().map(String::as_str).collect();
        if right.gpu_ids.iter().any(|id| left_set.contains(id.as_str())) {
            return PlacementResult::rejected(
                "conjunction arms overlap: insufficient distinct GPUs for both dimensions",
            );
        }
        let mut combined = left.gpu_ids;
        combined.extend(right.gpu_ids);
        PlacementResult::ok(combined)
    }

    /// Solve a `Disjunction`: try each alternative in declaration order; return
    /// the first successful placement.
    ///
    /// If all alternatives fail, return a rejection listing each alternative's reason.
    fn solve_disjunction(&self, alts: &[TopologySpec]) -> PlacementResult {
        let mut reasons = Vec::with_capacity(alts.len());
        for alt in alts {
            let result = self.solve(alt);
            if result.placed {
                return result;
            }
            reasons.push(result.rejection_reason);
        }
        PlacementResult::rejected(format!(
            "no alternative could be placed on the current cluster; tried: [{}]",
            reasons.join("; ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nemesis_graph::{ClusterGraph, LinkKind};
    use parking_lot::RwLock;
    use std::sync::Arc;

    fn eight_gpu_nvlink_graph() -> Arc<RwLock<ClusterGraph>> {
        let mut g = ClusterGraph::new();
        for i in 0..8usize {
            g.add_gpu(&format!("gpu-{i}"), "node-0", 0);
        }
        for i in 0..8usize {
            for j in (i + 1)..8 {
                g.add_link(
                    &format!("gpu-{i}"),
                    &format!("gpu-{j}"),
                    LinkKind::NvLink,
                    600.0,
                    0,
                );
            }
        }
        Arc::new(RwLock::new(g))
    }

    fn four_gpu_ib_graph() -> Arc<RwLock<ClusterGraph>> {
        let mut g = ClusterGraph::new();
        for i in 0..4usize {
            g.add_gpu(&format!("gpu-{i}"), &format!("node-{i}"), 0);
        }
        for i in 0..3usize {
            g.add_link(
                &format!("gpu-{i}"),
                &format!("gpu-{}", i + 1),
                LinkKind::InfiniBand,
                200.0,
                1,
            );
        }
        Arc::new(RwLock::new(g))
    }

    #[test]
    fn solve_tp8_nvl_succeeds() {
        let solver = TopologySolver::new(eight_gpu_nvlink_graph());
        let spec = crate::parser::parse("TP8_NVL12").unwrap();
        let r = solver.solve(&spec);
        assert!(r.placed, "rejection: {}", r.rejection_reason);
        assert_eq!(r.gpu_ids.len(), 8);
    }

    #[test]
    fn solve_tp8_nvl_fails_on_small_graph() {
        // Only 2 GPUs — can't form an 8-GPU clique
        let mut g = ClusterGraph::new();
        g.add_gpu("gpu-0", "node-0", 0);
        g.add_gpu("gpu-1", "node-0", 0);
        g.add_link("gpu-0", "gpu-1", LinkKind::NvLink, 600.0, 0);
        let solver = TopologySolver::new(Arc::new(RwLock::new(g)));
        let spec = crate::parser::parse("TP8_NVL12").unwrap();
        let r = solver.solve(&spec);
        assert!(!r.placed);
        assert!(!r.rejection_reason.is_empty());
    }

    #[test]
    fn solve_pp4_ib_succeeds() {
        let solver = TopologySolver::new(four_gpu_ib_graph());
        let spec = crate::parser::parse("PP4_IB2").unwrap();
        let r = solver.solve(&spec);
        assert!(r.placed, "rejection: {}", r.rejection_reason);
        assert_eq!(r.gpu_ids.len(), 4);
    }

    #[test]
    fn solve_conjunction_tp4_pp2() {
        // 4-GPU NVLink clique + 2-GPU IB path on same graph
        let mut g = ClusterGraph::new();
        for i in 0..4usize {
            g.add_gpu(&format!("gpu-{i}"), "node-0", 0);
        }
        for i in 0..4usize {
            for j in (i + 1)..4 {
                g.add_link(&format!("gpu-{i}"), &format!("gpu-{j}"), LinkKind::NvLink, 600.0, 0);
            }
        }
        g.add_gpu("gpu-4", "node-1", 0);
        g.add_gpu("gpu-5", "node-2", 0);
        g.add_link("gpu-4", "gpu-5", LinkKind::InfiniBand, 200.0, 1);
        let solver = TopologySolver::new(Arc::new(RwLock::new(g)));
        let spec = crate::parser::parse("TP4+PP2").unwrap();
        let r = solver.solve(&spec);
        assert!(r.placed, "rejection: {}", r.rejection_reason);
        assert_eq!(r.gpu_ids.len(), 6);
        // Conjunction arms must be disjoint
        let unique: std::collections::HashSet<&str> = r.gpu_ids.iter().map(String::as_str).collect();
        assert_eq!(unique.len(), 6, "conjunction arms returned overlapping gpu_ids");
    }

    #[test]
    fn solve_disjunction_takes_first() {
        let solver = TopologySolver::new(eight_gpu_nvlink_graph());
        // Both alternatives can be placed; first (TP8_NVL12) should win
        let spec = crate::parser::parse("TP8_NVL12|TP8_NVL50").unwrap();
        let r = solver.solve(&spec);
        assert!(r.placed, "rejection: {}", r.rejection_reason);
        assert_eq!(r.gpu_ids.len(), 8);
    }

    #[test]
    fn solve_dp_any() {
        let solver = TopologySolver::new(eight_gpu_nvlink_graph());
        let spec = crate::parser::parse("DP4").unwrap();
        let r = solver.solve(&spec);
        assert!(r.placed, "rejection: {}", r.rejection_reason);
        assert_eq!(r.gpu_ids.len(), 4);
    }
}
