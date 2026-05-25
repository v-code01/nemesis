use nemesis_proto::telemetry::v1::telemetry_service_client::TelemetryServiceClient;
use nemesis_proto::telemetry::v1::Void;
use std::time::{Duration, Instant};
use tonic::transport::Channel;

struct ChildGuard(tokio::process::Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.start_kill();
    }
}

#[tokio::test]
async fn substrate_serves_get_cluster_state() {
    let addr = "http://[::1]:50061";

    // Pre-build the binary so cargo run doesn't re-compile during the test
    std::process::Command::new("cargo")
        .args(["build", "--bin", "nemesis-substrate"])
        .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
        .status()
        .expect("cargo build failed");

    let child = tokio::process::Command::new("cargo")
        .args(["run", "--bin", "nemesis-substrate", "--", "--port", "50061"])
        .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
        .spawn()
        .expect("failed to spawn substrate");
    let _guard = ChildGuard(child);

    // Retry connect loop — waits up to 60s for the server to bind
    let deadline = Instant::now() + Duration::from_secs(60);
    let channel = loop {
        match Channel::from_static(addr).connect().await {
            Ok(c) => break c,
            Err(_) if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(300)).await;
            }
            Err(e) => panic!("substrate did not start within 60s: {e}"),
        }
    };

    let mut client = TelemetryServiceClient::new(channel);
    let response = client
        .get_cluster_state(Void {})
        .await
        .expect("get_cluster_state RPC failed");
    assert!(
        response.get_ref().taken_ns > 0,
        "taken_ns must be a non-zero Unix timestamp"
    );
    // _guard drops here, killing the child process
}
