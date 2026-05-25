use nemesis_topology::{checker::type_check, parser::parse};

#[test]
fn tp8_nvl12_is_valid() {
    let spec = parse("TP8_NVL12").unwrap();
    let errors = type_check(&spec);
    assert!(errors.is_empty(), "errors: {errors:?}");
}

#[test]
fn tp3_is_invalid_not_power_of_two() {
    let spec = parse("TP3_NVL12").unwrap();
    let errors = type_check(&spec);
    assert!(
        errors.iter().any(|e| e.contains("power of 2")),
        "expected power-of-2 error, got: {errors:?}"
    );
}

#[test]
fn tp16_is_invalid_exceeds_nvlink_max() {
    let spec = parse("TP16_NVL12").unwrap();
    let errors = type_check(&spec);
    assert!(
        errors.iter().any(|e| e.contains("NVLink mesh maximum")),
        "expected NVLink max error, got: {errors:?}"
    );
}

#[test]
fn disjunction_mismatched_gpu_count_is_invalid() {
    let spec = parse("TP8_NVL12|TP4_NVL12").unwrap();
    let errors = type_check(&spec);
    assert!(
        errors.iter().any(|e| e.contains("gpu_count")),
        "expected gpu_count mismatch error, got: {errors:?}"
    );
}

#[test]
fn disjunction_same_gpu_count_is_valid() {
    let spec = parse("TP8_NVL12|TP8_NVL6").unwrap();
    let errors = type_check(&spec);
    assert!(errors.is_empty(), "errors: {errors:?}");
}
