fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CARGO_MANIFEST_DIR is the crate root: substrate/crates/nemesis-proto/
    // proto/ lives two levels up at the repo root; canonicalize so protoc
    // receives absolute, symlink-resolved paths with no ".." components.
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    // substrate/crates/nemesis-proto/ is 3 levels deep from repo root,
    // so proto/ is at ../../../proto relative to CARGO_MANIFEST_DIR.
    let proto_dir = manifest_dir.join("../../../proto").canonicalize()?;

    // extern_path maps proto package names to Rust module paths so that
    // cross-package references (topology -> telemetry, healer -> telemetry)
    // resolve correctly against our lib.rs module layout instead of using
    // the generated super::super::... chains which break when modules are
    // included at the crate root.
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[
                proto_dir.join("telemetry.proto"),
                proto_dir.join("topology.proto"),
                proto_dir.join("healer.proto"),
            ],
            &[&proto_dir],
        )?;
    Ok(())
}
