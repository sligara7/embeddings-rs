# OpenAPI contract tooling for embeddings-rs.
#
# Mirrors the Python services' `dump-openapi` / `check-openapi` targets.
# `contract/openapi.json` is the frozen, code-derived interface contract; the
# integration test `tests/openapi_contract.rs` gates it against drift.

.PHONY: dump-openapi check-openapi

# Regenerate the committed contract from the code. Run this after an
# INTENTIONAL interface change (new route, changed request/response struct),
# review the diff, and commit it. `--dump-openapi` exits before loading the
# ONNX model, so no weights are needed.
dump-openapi:
	cargo run --release --quiet -- --dump-openapi > contract/openapi.json

# Fail if the committed contract no longer matches what the code emits.
check-openapi:
	cargo test --release --test openapi_contract
