// ClusterGraph: undirected petgraph-backed cluster topology model.
//
// Invariants:
//   - id_index is the sole source of truth for gpu_id -> NodeIndex mapping.
//   - Every NodeIndex stored in id_index is valid for `graph` (never removed).
//   - Edges are undirected; edges_connecting(a, b) == edges_connecting(b, a).
//   - `healthy` flag is the only mechanism to exclude nodes from scheduling.

use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// Discriminates the physical interconnect technology of a topology edge.
#[derive(Debug, Clone, PartialEq)]
pub enum LinkKind {
    NvLink,
    PCIe,
    InfiniBand,
}

/// Per-GPU vertex payload stored in the graph.
#[derive(Debug, Clone)]
pub struct GpuNode {
    pub gpu_id:      String,
    pub node_id:     String,
    pub numa_domain: u32,
    /// False once mark_unhealthy() is called; excluded from all solver outputs.
    pub healthy:     bool,
}

/// Per-edge payload stored in the graph.
#[derive(Debug, Clone)]
pub struct Link {
    pub kind:           LinkKind,
    /// Bidirectional peak bandwidth in GB/s.
    pub bandwidth_gbps: f32,
    /// InfiniBand switch-hop count between the two endpoints (0 for NVLink/PCIe).
    pub ib_hop_count:   u32,
}

/// Undirected topology graph over GPU nodes.
///
/// petgraph's UnGraph is used because NVLink and PCIe fabrics are
/// electrically symmetric — directionality adds no useful information.
pub struct ClusterGraph {
    graph:    UnGraph<GpuNode, Link>,
    /// Maps gpu_id -> NodeIndex for O(1) lookups. Never shrinks.
    id_index: HashMap<String, NodeIndex>,
}

impl ClusterGraph {
    pub fn new() -> Self {
        Self {
            graph:    UnGraph::new_undirected(),
            id_index: HashMap::new(),
        }
    }

    /// Add a GPU vertex. Panics if gpu_id is already present (enforces uniqueness).
    pub fn add_gpu(&mut self, gpu_id: &str, node_id: &str, numa_domain: u32) {
        assert!(
            !self.id_index.contains_key(gpu_id),
            "duplicate gpu_id: {gpu_id}"
        );
        let idx = self.graph.add_node(GpuNode {
            gpu_id:      gpu_id.to_string(),
            node_id:     node_id.to_string(),
            numa_domain,
            healthy:     true,
        });
        self.id_index.insert(gpu_id.to_string(), idx);
    }

    /// Add a directed-less link between two already-added GPUs.
    ///
    /// Multiple edges between the same pair are allowed (e.g. bonded IB links).
    pub fn add_link(
        &mut self,
        src: &str,
        dst: &str,
        kind: LinkKind,
        bandwidth_gbps: f32,
        ib_hop_count: u32,
    ) {
        let s = self.id_index[src];
        let d = self.id_index[dst];
        self.graph.add_edge(s, d, Link { kind, bandwidth_gbps, ib_hop_count });
    }

    /// Mark a GPU as unhealthy. Subsequent solver calls exclude it.
    pub fn mark_unhealthy(&mut self, gpu_id: &str) {
        if let Some(&idx) = self.id_index.get(gpu_id) {
            self.graph[idx].healthy = false;
        }
    }

    /// O(1) membership test.
    #[inline]
    pub fn contains_gpu(&self, gpu_id: &str) -> bool {
        self.id_index.contains_key(gpu_id)
    }

    /// Total GPU count (healthy + unhealthy).
    #[inline]
    pub fn gpu_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns gpu_ids of all healthy GPUs. Order is graph-internal (stable for
    /// a given sequence of mutations but not guaranteed across builds).
    pub fn healthy_gpu_ids(&self) -> Vec<String> {
        self.graph
            .node_indices()
            .filter(|&i| self.graph[i].healthy)
            .map(|i| self.graph[i].gpu_id.clone())
            .collect()
    }

    /// Bandwidth of the first edge between src and dst, or None if no edge exists.
    pub fn link_bandwidth(&self, src: &str, dst: &str) -> Option<f32> {
        let s = *self.id_index.get(src)?;
        let d = *self.id_index.get(dst)?;
        self.graph
            .edges_connecting(s, d)
            .next()
            .map(|e| e.weight().bandwidth_gbps)
    }

    /// Find a clique of `size` healthy nodes where every pairwise NVLink edge
    /// has bandwidth >= min_bandwidth_gbps.
    ///
    /// NVLink meshes are at most 8 nodes, so exhaustive Bron–Kerbosch enumeration
    /// is tractable (C(8,8) = 1 candidate in the best case). Returns the first
    /// clique found; does not guarantee the maximum clique.
    pub fn find_nvlink_clique(
        &self,
        size: usize,
        min_bandwidth_gbps: f32,
    ) -> Option<Vec<String>> {
        let candidates: Vec<NodeIndex> = self
            .graph
            .node_indices()
            .filter(|&i| self.graph[i].healthy)
            .collect();

        // Early exit: not enough candidates.
        if candidates.len() < size {
            return None;
        }

        self.clique_search(&candidates, size, min_bandwidth_gbps, &mut Vec::with_capacity(size))
    }

    /// Recursive backtracking clique search over a candidate slice.
    ///
    /// `candidates` is the suffix of nodes still available for selection.
    /// `current` is the in-progress clique being built.
    fn clique_search(
        &self,
        candidates: &[NodeIndex],
        target: usize,
        min_bw: f32,
        current: &mut Vec<NodeIndex>,
    ) -> Option<Vec<String>> {
        if current.len() == target {
            return Some(
                current
                    .iter()
                    .map(|&i| self.graph[i].gpu_id.clone())
                    .collect(),
            );
        }
        // Pruning: not enough remaining candidates to complete the clique.
        if current.len() + candidates.len() < target {
            return None;
        }
        for (pos, &node) in candidates.iter().enumerate() {
            // node must have a qualifying NVLink edge to every node already in clique.
            if current.iter().all(|&c| self.has_nvlink_edge(c, node, min_bw)) {
                current.push(node);
                if let Some(result) =
                    self.clique_search(&candidates[pos + 1..], target, min_bw, current)
                {
                    return Some(result);
                }
                current.pop();
            }
        }
        None
    }

    /// Returns true iff there is at least one NVLink edge between a and b
    /// with bandwidth >= min_bw.
    #[inline]
    fn has_nvlink_edge(&self, a: NodeIndex, b: NodeIndex, min_bw: f32) -> bool {
        self.graph.edges_connecting(a, b).any(|e| {
            e.weight().kind == LinkKind::NvLink && e.weight().bandwidth_gbps >= min_bw
        })
    }

    /// Find a simple path of exactly `size` healthy nodes connected by
    /// InfiniBand edges where each hop has ib_hop_count <= max_hops.
    ///
    /// Uses DFS from each healthy node. Returns None if no such path exists.
    pub fn find_ib_path(&self, size: usize, max_hops: u32) -> Option<Vec<String>> {
        let candidates: Vec<NodeIndex> = self
            .graph
            .node_indices()
            .filter(|&i| self.graph[i].healthy)
            .collect();

        if candidates.len() < size {
            return None;
        }

        for &start in &candidates {
            let mut path = Vec::with_capacity(size);
            path.push(start);
            if let Some(full_path) = self.dfs_ib_path(start, size, max_hops, &mut path) {
                return Some(
                    full_path
                        .iter()
                        .map(|&i| self.graph[i].gpu_id.clone())
                        .collect(),
                );
            }
        }
        None
    }

    /// DFS expanding `path` by one IB-connected healthy node at a time.
    fn dfs_ib_path(
        &self,
        current: NodeIndex,
        target: usize,
        max_hops: u32,
        path: &mut Vec<NodeIndex>,
    ) -> Option<Vec<NodeIndex>> {
        if path.len() == target {
            return Some(path.clone());
        }
        for edge in self.graph.edges(current) {
            // UnGraph edges report both directions; normalise to neighbour.
            let next = if edge.source() == current {
                edge.target()
            } else {
                edge.source()
            };
            // Skip: already visited, unhealthy, or wrong link kind / hop count.
            if path.contains(&next) {
                continue;
            }
            if !self.graph[next].healthy {
                continue;
            }
            if edge.weight().kind == LinkKind::InfiniBand
                && edge.weight().ib_hop_count <= max_hops
            {
                path.push(next);
                if let Some(result) = self.dfs_ib_path(next, target, max_hops, path) {
                    return Some(result);
                }
                path.pop();
            }
        }
        None
    }

    /// Serialize the topology to the generated proto ClusterTopology message.
    ///
    /// Proto module path (from nemesis-proto/src/lib.rs):
    ///   nemesis_proto::nemesis::telemetry::v1  (canonical)
    ///   nemesis_proto::telemetry::v1            (flat re-export alias)
    ///
    /// Link::kind is stored as i32 matching the proto enum:
    ///   Nvlink=0, Pcie=1, Infiniband=2
    pub fn to_proto(&self) -> nemesis_proto::telemetry::v1::ClusterTopology {
        use nemesis_proto::telemetry::v1::{ClusterTopology, GpuNode as ProtoGpu, Link as ProtoLink};

        let nodes: Vec<ProtoGpu> = self
            .graph
            .node_indices()
            .map(|i| {
                let n = &self.graph[i];
                ProtoGpu {
                    gpu_id:      n.gpu_id.clone(),
                    node_id:     n.node_id.clone(),
                    numa_domain: n.numa_domain,
                    healthy:     n.healthy,
                }
            })
            .collect();

        let edges: Vec<ProtoLink> = self
            .graph
            .edge_indices()
            .map(|e| {
                let (a, b) = self.graph.edge_endpoints(e).unwrap();
                let w = &self.graph[e];
                ProtoLink {
                    src:            self.graph[a].gpu_id.clone(),
                    dst:            self.graph[b].gpu_id.clone(),
                    // Raw i32 values matching link::Kind enum order.
                    kind:           match w.kind {
                        LinkKind::NvLink     => 0, // Kind::Nvlink
                        LinkKind::PCIe       => 1, // Kind::Pcie
                        LinkKind::InfiniBand => 2, // Kind::Infiniband
                    },
                    bandwidth_gbps: w.bandwidth_gbps,
                    ib_hop_count:   w.ib_hop_count,
                }
            })
            .collect();

        ClusterTopology { nodes, edges }
    }
}

impl Default for ClusterGraph {
    fn default() -> Self {
        Self::new()
    }
}
