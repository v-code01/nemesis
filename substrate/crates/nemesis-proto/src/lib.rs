// The generated tonic/prost code uses super:: chains to resolve cross-package
// references. The depth of those chains is determined by the proto package
// hierarchy: nemesis.telemetry.v1, nemesis.topology.v1, nemesis.healer.v1.
//
// Generated cross-refs use up to 3 supers (from inside a service mod):
//   super::super::super::telemetry::v1  (topology/healer -> telemetry)
//
// We mirror the proto package hierarchy so that every super chain resolves:
//   crate::nemesis::topology::v1::scheduler_service_client (include level + 1)
//     super       -> crate::nemesis::topology::v1  (include level)
//     super::super -> crate::nemesis::topology
//     super::super::super -> crate::nemesis
//     super::super::super::telemetry::v1 -> crate::nemesis::telemetry::v1  ✓
//
// Re-export flat aliases at the crate root for ergonomic downstream use.

pub mod nemesis {
    pub mod telemetry {
        pub mod v1 {
            tonic::include_proto!("nemesis.telemetry.v1");
        }
    }

    pub mod topology {
        pub mod v1 {
            tonic::include_proto!("nemesis.topology.v1");
        }
    }

    pub mod healer {
        pub mod v1 {
            tonic::include_proto!("nemesis.healer.v1");
        }
    }
}

// Flat re-exports so callers can use nemesis_proto::telemetry, ::topology, ::healer
pub use nemesis::telemetry as telemetry;
pub use nemesis::topology as topology;
pub use nemesis::healer as healer;
