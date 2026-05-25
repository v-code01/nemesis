use nemesis_graph::{ClusterGraph, LinkKind};
use nemesis_topology::{parser::parse, solver::TopologySolver};
use parking_lot::RwLock;
use std::sync::Arc;

/// Eight GPUs on a single node fully meshed with NVLink at 600 GB/s each.
fn eight_gpu_nvlink_cluster() -> Arc<RwLock<ClusterGraph>> {
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

/// Four GPUs on four distinct nodes connected in a chain via InfiniBand at 200 GB/s, 1 IB hop each.
fn four_gpu_ib_chain() -> Arc<RwLock<ClusterGraph>> {
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

// ---------------------------------------------------------------------------
// TP placement tests
// ---------------------------------------------------------------------------

#[test]
fn tp8_nvl12_placed_on_nvlink_cluster() {
    let graph = eight_gpu_nvlink_cluster();
    let solver = TopologySolver::new(graph);
    let spec = parse("TP8_NVL12").unwrap();
    let result = solver.solve(&spec);
    assert!(result.placed, "expected placement; rejection: {}", result.rejection_reason);
    assert_eq!(result.gpu_ids.len(), 8);
}

#[test]
fn tp8_high_bandwidth_rejected_on_low_bandwidth_cluster() {
    // All links are 100 GB/s; TP8_NVL600 requires 600 GB/s.
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
                100.0,
                0,
            );
        }
    }
    let graph = Arc::new(RwLock::new(g));
    let solver = TopologySolver::new(graph);
    let spec = parse("TP8_NVL600").unwrap();
    let result = solver.solve(&spec);
    assert!(!result.placed, "expected rejection");
    assert!(
        !result.rejection_reason.is_empty(),
        "rejection_reason must not be empty"
    );
}

// ---------------------------------------------------------------------------
// PP placement tests
// ---------------------------------------------------------------------------

#[test]
fn pp4_ib2_placed_on_ib_chain() {
    let graph = four_gpu_ib_chain();
    let solver = TopologySolver::new(graph);
    let spec = parse("PP4_IB2").unwrap();
    let result = solver.solve(&spec);
    assert!(result.placed, "expected placement; rejection: {}", result.rejection_reason);
    assert_eq!(result.gpu_ids.len(), 4);
}

#[test]
fn pp4_ib0_rejected_on_multi_hop_chain() {
    // Chain has IB hop count 1 per link; IB0 (max 0 hops) cannot be satisfied.
    let graph = four_gpu_ib_chain();
    let solver = TopologySolver::new(graph);
    let spec = parse("PP4_IB0").unwrap();
    let result = solver.solve(&spec);
    assert!(!result.placed, "expected rejection for IB0 on a 1-hop chain");
    assert!(!result.rejection_reason.is_empty());
}

// ---------------------------------------------------------------------------
// Disjunction fallback test
// ---------------------------------------------------------------------------

#[test]
fn disjunction_falls_back_to_second_alternative() {
    // NVLink at 100 GB/s — first alt (NVL600) fails, second (NVL50) succeeds.
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
                100.0,
                0,
            );
        }
    }
    let graph = Arc::new(RwLock::new(g));
    let solver = TopologySolver::new(graph);
    let spec = parse("TP8_NVL600|TP8_NVL50").unwrap();
    let result = solver.solve(&spec);
    assert!(result.placed, "second alternative should have been placed");
    assert_eq!(result.gpu_ids.len(), 8);
}

#[test]
fn disjunction_all_fail_returns_rejection() {
    // All links at 10 GB/s; both NVL600 and NVL50 fail.
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
                10.0,
                0,
            );
        }
    }
    let graph = Arc::new(RwLock::new(g));
    let solver = TopologySolver::new(graph);
    let spec = parse("TP8_NVL600|TP8_NVL50").unwrap();
    let result = solver.solve(&spec);
    assert!(!result.placed, "no alternative should succeed with 10 GB/s links");
    assert!(!result.rejection_reason.is_empty());
}
