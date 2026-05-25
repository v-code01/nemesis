use nemesis_topology::parser::{parse, Constraint, ParallelDim, TopologySpec};

#[test]
fn parse_tp8_nvl12() {
    let spec = parse("TP8_NVL12").unwrap();
    match spec {
        TopologySpec::Atom(dim, constraints) => {
            assert_eq!(dim, ParallelDim::Tp(8));
            assert!(constraints.contains(&Constraint::NvlMin(12.0)));
        }
        _ => panic!("expected Atom"),
    }
}

#[test]
fn parse_conjunction() {
    let spec = parse("TP8_NVL12+PP4_IB2").unwrap();
    assert!(matches!(spec, TopologySpec::Conjunction(_, _)));
}

#[test]
fn parse_disjunction() {
    let spec = parse("TP8_NVL12|TP8_NVL6").unwrap();
    assert!(matches!(spec, TopologySpec::Disjunction(_)));
}

#[test]
fn parse_error_on_invalid_token() {
    assert!(parse("INVALID8").is_err());
}

#[test]
fn gpu_count_tp8() {
    let spec = parse("TP8_NVL12").unwrap();
    assert_eq!(spec.gpu_count(), 8);
}

#[test]
fn gpu_count_conjunction() {
    let spec = parse("TP8_NVL12+PP4_IB2").unwrap();
    assert_eq!(spec.gpu_count(), 32);
}
