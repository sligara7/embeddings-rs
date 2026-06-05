//! Contract drift gate for the embeddings-rs OpenAPI interface.
//!
//! `contract/openapi.json` is the frozen, code-derived interface contract for
//! this service (the "black box" boundary — see the interface-first effort).
//! This test fails if the spec the binary emits today differs from the
//! committed contract, so a change to a route or a request/response struct
//! can't silently drift the interface out from under consumers.
//!
//! On an intentional interface change: regenerate with `make dump-openapi`
//! (or `cargo run --release -- --dump-openapi > contract/openapi.json`),
//! review the diff, and commit it alongside the code change.

use std::process::Command;

#[test]
fn openapi_contract_is_fresh() {
    // CARGO_BIN_EXE_<bin> is set by cargo for integration tests and points at
    // the freshly-built binary — so we exercise the real `--dump-openapi`
    // path, not a re-derivation. `--dump-openapi` exits before loading the
    // ONNX model, so this needs no weights and is CI-safe.
    let bin = env!("CARGO_BIN_EXE_embeddings-rs");
    let output = Command::new(bin)
        .arg("--dump-openapi")
        .output()
        .expect("run `embeddings-rs --dump-openapi`");

    assert!(
        output.status.success(),
        "--dump-openapi exited non-zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let live = String::from_utf8(output.stdout).expect("spec is utf-8");
    let committed = include_str!("../contract/openapi.json");

    if live != committed {
        panic!(
            "OpenAPI contract drift: contract/openapi.json is stale.\n\
             The spec the code produces no longer matches the committed contract.\n\
             If this change to the interface is intentional, regenerate and commit it:\n\
             \n    make dump-openapi\n\n\
             (or: cargo run --release -- --dump-openapi > contract/openapi.json)"
        );
    }
}
