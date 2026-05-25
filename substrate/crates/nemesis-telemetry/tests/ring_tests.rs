// Integration tests for MetricRing.
//
// Proto module path (confirmed from nemesis-proto/src/lib.rs):
//   nemesis_proto::telemetry::v1::MetricSample
//
// These tests verify:
//   1. Basic push + window retrieval.
//   2. window(n) capped at available samples.
//   3. Oldest-first ordering is preserved.

use nemesis_telemetry::ring::MetricRing;

fn make_sample(gpu_id: &str, ecc_corr: f32) -> nemesis_proto::telemetry::v1::MetricSample {
    nemesis_proto::telemetry::v1::MetricSample {
        gpu_id:                       gpu_id.to_string(),
        timestamp_ns:                 0,
        ecc_correctable_rate:         ecc_corr,
        ecc_uncorrectable_rate:       0.0,
        temperature_celsius:          40.0,
        nvlink_bandwidth_gbps:        600.0,
        ib_bandwidth_gbps:            200.0,
        sm_utilization:               0.8,
        memory_bandwidth_utilization: 0.7,
    }
}

#[test]
fn ring_stores_and_retrieves_sample() {
    let mut ring = MetricRing::new();
    ring.push(make_sample("gpu-0", 0.1));
    let window = ring.window(1);
    assert_eq!(window.len(), 1);
    assert!(
        (window[0].ecc_correctable_rate - 0.1).abs() < f32::EPSILON,
        "expected 0.1, got {}",
        window[0].ecc_correctable_rate,
    );
}

#[test]
fn ring_returns_at_most_available_samples() {
    let mut ring = MetricRing::new();
    ring.push(make_sample("gpu-0", 1.0));
    ring.push(make_sample("gpu-0", 2.0));
    let window = ring.window(100);
    assert_eq!(window.len(), 2);
}

#[test]
fn ring_preserves_insertion_order() {
    let mut ring = MetricRing::new();
    ring.push(make_sample("gpu-0", 1.0));
    ring.push(make_sample("gpu-0", 2.0));
    ring.push(make_sample("gpu-0", 3.0));
    let window = ring.window(3);
    // Oldest first.
    assert!(
        (window[0].ecc_correctable_rate - 1.0).abs() < f32::EPSILON,
        "first element should be oldest (1.0), got {}",
        window[0].ecc_correctable_rate,
    );
    assert!(
        (window[2].ecc_correctable_rate - 3.0).abs() < f32::EPSILON,
        "last element should be newest (3.0), got {}",
        window[2].ecc_correctable_rate,
    );
}
