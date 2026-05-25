import sys
from pathlib import Path

# grpcio-tools generates bare `import telemetry_pb2` in topology_pb2 and healer_pb2.
# Adding this directory to sys.path makes all stubs findable by each other
# regardless of how the package is imported.
_grpc_dir = str(Path(__file__).parent)
if _grpc_dir not in sys.path:
    sys.path.insert(0, _grpc_dir)
