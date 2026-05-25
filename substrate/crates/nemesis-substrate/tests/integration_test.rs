use nemesis_proto::telemetry::v1::telemetry_service_client::TelemetryServiceClient;
use nemesis_proto::telemetry::v1::Void;
use tonic::transport::Channel;

#[tokio::test]
async fn substrate_serves_get_cluster_state() {
    let addr = "http://[::1]:50061";
    let mut child = tokio::process::Command::new("cargo")
        .args(["run", "--bin", "nemesis-substrate", "--", "--port", "50061"])
        .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
        .spawn()
        .expect("failed to spawn substrate");

    // Wait for the binary to start and bind the port
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

    let channel = Channel::from_static(addr).connect().await.unwrap();
    let mut client = TelemetryServiceClient::new(channel);
    let response = client.get_cluster_state(Void {}).await.unwrap();
    // taken_ns is SystemTime::now() in nanos — always > 0
    assert!(response.get_ref().taken_ns > 0);

    child.kill().await.ok();
}
