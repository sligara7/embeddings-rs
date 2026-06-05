# embeddings-rs

Rust/[`ort`](https://ort.pyke.io) (ONNX Runtime) parity sidecar for
`nomic-ai/nomic-embed-text-v1.5` — a drop-in replacement for the original
Python `embeddings` service. Pairs with
[`dynograph-foundation`](https://github.com/sligara7/dynograph-foundation),
which embeds and stores vectors through this contract.

Same HTTP contract, same vectors:

| Endpoint | Request | Response |
|----------|---------|----------|
| `POST /embed` | `{text, task_type}` | `{vector, dimensions}` |
| `POST /embed-batch` | `{texts, task_type}` | `{vectors, count, dimensions}` |
| `GET /health` | — | `{status, model, dimensions}` |

## Contract (OpenAPI)

The HTTP contract is **code-derived**, not hand-maintained: request/response
schemas come from `#[derive(ToSchema)]` on the real structs and paths from
`#[utoipa::path]` on the handlers (via `utoipa` + `utoipa-axum`), so the spec
can't silently drift from the routing.

- **Frozen artifact**: `contract/openapi.json` is the committed interface
  contract (the "black box" boundary). It is gated against drift by
  `tests/openapi_contract.rs`, which runs `--dump-openapi` and fails if the
  spec the code emits no longer matches the committed file.
- **Offline export** (no model needed — CI-friendly):
  `embeddings-rs --dump-openapi > contract/openapi.json`, or `make dump-openapi`.
- **Check for drift**: `make check-openapi` (runs the contract test).
- **Served at runtime**: `GET /openapi.json` (matches the FastAPI services).

After an *intentional* interface change, regenerate (`make dump-openapi`),
review the diff, and commit it with the code. `contract/openapi.json` is the
canonical, versioned interface contract for this service; downstream consumers
should track it from this repo.

## Parity

It must reproduce the Python sidecar's vectors closely enough to be
cosine-compatible with the corpus already stored in DynoGraph's RocksDB, so
**no re-embedding is required** to switch. Three rules make or break that, all
enforced in `src/main.rs`:

1. **Prefix** — `"{task_type}: {text}"`, byte-for-byte.
2. **Masked mean pooling** — average hidden states over real tokens only.
3. **No normalization** — the Python config returns *raw* mean-pooled vectors
   (`|v| ≈ 22`). Do **not** L2-normalize, or dot-product / distance / centroid
   consumers drift.

Verified by the `parity_embeddings.py` script (from the storyflow project this
was extracted from — reference `:8401` vs candidate `:8402`): cosine
`1.0000000`, magnitude ratio `1.0`, zero samples below `0.99999` over a
1–800-word length sweep across both task types.

## Build & run (local, native)

```bash
# Weights aren't vendored (gitignored). Fetch once into models/:
mkdir -p models
base=https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main
curl -sSL "$base/onnx/model.onnx" -o models/model.onnx
curl -sSL "$base/tokenizer.json"  -o models/tokenizer.json

# Host lacks pkg-config? Point openssl-sys at the system install:
OPENSSL_NO_PKG_CONFIG=1 OPENSSL_DIR=/usr \
  OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu OPENSSL_INCLUDE_DIR=/usr/include \
  cargo build --release

MODEL_DIR=models EMBEDDING_PORT=8402 ./target/release/embeddings-rs
```

The Docker build needs neither step — it installs `pkg-config`/`libssl-dev` and
downloads the weights itself.

## Docker / cutover

Profile-gated so the default stack is untouched:

```bash
docker compose --profile rust-embeddings up -d --build embeddings_rs
# A/B against the Python sidecar on :8401 (parity_embeddings.py lives in
# the storyflow project):
REFERENCE_URL=http://localhost:8401 CANDIDATE_URL=http://localhost:8402 \
  python path/to/parity_embeddings.py
```

To make it the real backend, repoint `dynograph` in `docker-compose.yml`:
`EMBEDDING_URL=http://embeddings_rs:8401` and swap its `depends_on` from
`embeddings` to `embeddings_rs`.

## Concurrency

ort's `run` needs `&mut self`, so a session can't be shared for concurrent
inference. Instead the service keeps a **pool of `EMBEDDING_POOL_SIZE` sessions**
("lanes", default 2), each handed out per in-flight request via a semaphore.
Requests beyond the lane count queue (no busy-wait). Measured ~2.5× throughput
at 3 lanes vs. serialized on a 24-request burst.

Tradeoff: **each lane is another ~550 MB copy of the weights in RAM**, so scale
`EMBEDDING_POOL_SIZE` and `mem_limit` together. Intra-op threads are split
across lanes (`cores / pool_size`) so N lanes don't oversubscribe the CPU. Set
`EMBEDDING_POOL_SIZE=1` to get back the lean single-session profile.
