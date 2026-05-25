use nemesis_nccl::{backend::NcclBackend, sim::NcclSim};
use nemesis_proto::healer::v1::{ExpandRequest, ShrinkRequest};

#[tokio::test]
async fn sim_shrink_succeeds_and_returns_duration() {
    let sim = NcclSim::new(8, 42);
    let req = ShrinkRequest {
        communicator_id: "comm-0".to_string(),
        job_id: "job-0".to_string(),
        exclude_ranks: vec![7],
    };
    let result = sim.shrink(&req).await.unwrap();
    assert!(result.success);
    assert!(result.duration_ns > 0);
    assert_eq!(result.active_rank_count, 7);
}

#[tokio::test]
async fn sim_expand_succeeds_after_shrink() {
    let sim = NcclSim::new(8, 42);
    let shrink_req = ShrinkRequest {
        communicator_id: "comm-0".to_string(),
        job_id: "job-0".to_string(),
        exclude_ranks: vec![7],
    };
    sim.shrink(&shrink_req).await.unwrap();
    let expand_req = ExpandRequest {
        communicator_id: "comm-0".to_string(),
        job_id: "job-0".to_string(),
        new_gpu_ids: vec!["gpu-8".to_string()],
    };
    let result = sim.expand(&expand_req).await.unwrap();
    assert!(result.success);
    assert_eq!(result.active_rank_count, 8);
}

#[tokio::test]
async fn sim_shrink_duration_under_30_seconds() {
    let sim = NcclSim::new(64, 42);
    let req = ShrinkRequest {
        communicator_id: "comm-0".to_string(),
        job_id: "job-0".to_string(),
        exclude_ranks: vec![63],
    };
    let result = sim.shrink(&req).await.unwrap();
    assert!(
        result.duration_ns < 30_000_000_000,
        "duration {}ns exceeds 30s",
        result.duration_ns
    );
}

#[tokio::test]
async fn sim_shrink_exhausting_all_ranks_returns_error() {
    let sim = NcclSim::new(4, 42);
    let req = ShrinkRequest {
        communicator_id: "comm-0".to_string(),
        job_id: "job-0".to_string(),
        exclude_ranks: vec![0, 1, 2, 3], // all 4 ranks
    };
    let result = sim.shrink(&req).await.unwrap();
    assert!(!result.success, "shrinking all ranks must fail");
    assert!(!result.error.is_empty(), "error message must be present");
    assert_eq!(
        result.active_rank_count, 4,
        "rank count must be unchanged on failure"
    );
}
