use nemesis_graph::{ClusterGraph, LinkKind};

#[test]
fn add_gpu_node_is_queryable() {
    let mut g = ClusterGraph::new();
    g.add_gpu("gpu-0", "node-0", 0);
    assert!(g.contains_gpu("gpu-0"));
    assert_eq!(g.gpu_count(), 1);
}

#[test]
fn nvlink_edge_has_correct_bandwidth() {
    let mut g = ClusterGraph::new();
    g.add_gpu("gpu-0", "node-0", 0);
    g.add_gpu("gpu-1", "node-0", 0);
    g.add_link("gpu-0", "gpu-1", LinkKind::NvLink, 600.0, 0);
    let bw = g.link_bandwidth("gpu-0", "gpu-1").unwrap();
    assert!((bw - 600.0).abs() < f32::EPSILON);
}

#[test]
fn mark_unhealthy_excludes_from_healthy_set() {
    let mut g = ClusterGraph::new();
    g.add_gpu("gpu-0", "node-0", 0);
    g.add_gpu("gpu-1", "node-0", 0);
    g.mark_unhealthy("gpu-0");
    let healthy = g.healthy_gpu_ids();
    assert!(!healthy.contains(&"gpu-0".to_string()));
    assert!(healthy.contains(&"gpu-1".to_string()));
}

#[test]
fn nvlink_clique_found_in_8gpu_mesh() {
    let mut g = ClusterGraph::new();
    for i in 0..8 {
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
    let clique = g.find_nvlink_clique(8, 200.0);
    assert!(clique.is_some());
    assert_eq!(clique.unwrap().len(), 8);
}

#[test]
fn nvlink_clique_not_found_when_bandwidth_too_low() {
    let mut g = ClusterGraph::new();
    for i in 0..8 {
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
    let clique = g.find_nvlink_clique(8, 600.0);
    assert!(clique.is_none());
}

#[test]
fn ib_path_found_in_linear_chain() {
    let mut g = ClusterGraph::new();
    for i in 0..4 {
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
    let path = g.find_ib_path(4, 2);
    assert!(path.is_some());
    assert_eq!(path.unwrap().len(), 4);
}
