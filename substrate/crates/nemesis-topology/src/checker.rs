//! Type-checker for topology specifications.
//!
//! Enforces four rules:
//!
//! 1. **TP degree must be a power of 2** — required for ring/tree all-reduce correctness.
//! 2. **TP degree ≤ 8 when an NVL constraint is present** — NVSwitch domains top out at 8 GPUs
//!    per switch in current NVLink 3.0/4.0 hardware.
//! 3. Conjunction GPU count is correct by construction (product of degrees) — no runtime check
//!    needed beyond recursive descent into each arm.
//! 4. **Disjunction alternatives must all have the same `gpu_count`** — the scheduler cannot
//!    reserve a variable number of GPUs for a single logical placement.

use crate::parser::{Constraint, ParallelDim, TopologySpec};

/// Run all type rules against `spec` and collect human-readable error messages.
///
/// Returns an empty `Vec` when the spec is well-typed.
pub fn type_check(spec: &TopologySpec) -> Vec<String> {
    let mut errors = Vec::new();
    check(spec, &mut errors);
    errors
}

fn check(spec: &TopologySpec, errors: &mut Vec<String>) {
    match spec {
        TopologySpec::Atom(dim, constraints) => check_atom(dim, constraints, errors),
        TopologySpec::Conjunction(l, r) => {
            check(l, errors);
            check(r, errors);
        }
        TopologySpec::Disjunction(alts) => {
            // Recursively type-check each alternative.
            for alt in alts {
                check(alt, errors);
            }
            // Rule 4: all alternatives must agree on gpu_count so the allocator
            // knows exactly how many devices to reserve.
            let counts: Vec<u32> = alts.iter().map(|a| a.gpu_count()).collect();
            if counts.windows(2).any(|w| w[0] != w[1]) {
                errors.push(format!(
                    "disjunction alternatives have mismatched gpu_count: {counts:?}"
                ));
            }
        }
    }
}

fn check_atom(dim: &ParallelDim, constraints: &[Constraint], errors: &mut Vec<String>) {
    if let ParallelDim::Tp(n) = dim {
        // Rule 1: TP degree must be a power of 2.
        // Invariant: n == 0 is also illegal (zero-rank TP is nonsensical).
        if *n == 0 || (*n & (*n - 1)) != 0 {
            errors.push(format!(
                "TP degree {n} must be a power of 2 (e.g. 1, 2, 4, 8, 16 …)"
            ));
        }

        // Rule 2: NVLink mesh maximum is 8 GPUs per NVSwitch domain.
        // TP > 8 with an NVL constraint cannot be satisfied by any current hardware.
        let has_nvl = constraints
            .iter()
            .any(|c| matches!(c, Constraint::NvlMin(_)));
        if has_nvl && *n > 8 {
            errors.push(format!(
                "TP{n} with NVLink constraint exceeds NVLink mesh maximum of 8 GPUs per NVSwitch domain"
            ));
        }
    }
}
