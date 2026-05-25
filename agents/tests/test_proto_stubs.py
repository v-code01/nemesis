def test_telemetry_stubs_importable():
    from nemesis.grpc import telemetry_pb2, telemetry_pb2_grpc
    assert hasattr(telemetry_pb2, "HardwareEvent")
    assert hasattr(telemetry_pb2_grpc, "TelemetryServiceStub")

def test_topology_stubs_importable():
    from nemesis.grpc import topology_pb2, topology_pb2_grpc
    assert hasattr(topology_pb2, "JobSpec")
    assert hasattr(topology_pb2_grpc, "SchedulerServiceStub")

def test_healer_stubs_importable():
    from nemesis.grpc import healer_pb2, healer_pb2_grpc
    assert hasattr(healer_pb2, "ShrinkRequest")
    assert hasattr(healer_pb2_grpc, "HealerServiceStub")

def test_hardware_event_kind_values():
    from nemesis.grpc import telemetry_pb2
    assert telemetry_pb2.HardwareEvent.HARDWARE_FAILURE_PREDICTED == 0
    assert telemetry_pb2.HardwareEvent.NVLINK_DEGRADED == 3
