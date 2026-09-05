//! Rust/ort parity sidecar for nomic-embed-text-v1.5.
//!
//! A drop-in for the Python `embeddings` service. It MUST reproduce that
//! service's vectors exactly enough to be cosine-compatible with the corpus
//! already stored in DynoGraph's RocksDB. The three things that make or break
//! parity (learned from scripts/parity_embeddings.py against the live sidecar):
//!
//! 1. Prefix: format!("{task_type}: {text}") — the "{}: {}" form, byte-for-byte.
//! 2. Pooling: MASKED mean over the sequence (ignore padding positions).
//! 3. Normalization: NONE. The Python ST config returns raw mean-pooled vectors
//!    (|v| ~= 22). Do NOT L2-normalize here.
//!
//! Contract (identical to server.py):
//!   POST /embed        {text, task_type}  -> {vector, dimensions}
//!   POST /embed-batch  {texts, task_type} -> {vectors, count, dimensions}
//!   GET  /health       -> {status, model, dimensions}

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{extract::State, Json};
use ort::ep;
use ort::session::Session;
use ort::value::Value;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use utoipa::{OpenApi, ToSchema};
use utoipa_axum::{router::OpenApiRouter, routes};

mod pool;
use pool::LanePool;

const MODEL_NAME: &str = "nomic-ai/nomic-embed-text-v1.5";
const MAX_LEN: usize = 8192;

/// A pool of N independent ONNX sessions ("lanes"). ort's `run` needs `&mut
/// self`, so a session can't be shared for concurrent inference — instead we
/// keep N of them and hand one out per in-flight request. N lanes cost N x the
/// model weights in RAM (see EMBEDDING_POOL_SIZE / mem_limit).
///
/// The lane accounting lives in [`LanePool`], where holding the lane IS holding
/// the right to use it — see that module for the outage this replaced.
struct AppState {
    pool: LanePool<Session>,
    tokenizer: Tokenizer,
    dim: usize,
    /// Max tokens (rows × padded seq-len) per ONNX `run()`. Bounds forward-pass
    /// activation memory — which scales with batch×seq (+ O(seq²) attention), NOT
    /// with the number of texts. See [`embed_prefixed`].
    max_batch_tokens: usize,
}

/// Why a request could not be served. Both arms are LOUD: logged here and
/// returned to the caller as a real status code, never swallowed into an empty
/// success. A 200 with no vectors would be the silent failure this service was
/// built out of.
enum EmbedError {
    /// Every lane is permanently gone, so waiting could never succeed.
    PoolExhausted,
    /// The inference task panicked (or was aborted). The lane itself is safe —
    /// its guard returned it while unwinding.
    Inference,
}

impl IntoResponse for EmbedError {
    fn into_response(self) -> Response {
        match self {
            EmbedError::PoolExhausted => (
                StatusCode::SERVICE_UNAVAILABLE,
                "embedding pool exhausted: no usable inference lanes remain",
            ),
            EmbedError::Inference => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "embedding inference failed",
            ),
        }
        .into_response()
    }
}

impl AppState {
    /// Acquire a lane, run inference off the async runtime, return the lane.
    ///
    /// The lane guard is **moved into the blocking closure**, which is what makes
    /// the pool leak-proof: a blocking task always runs to completion, so the
    /// guard's `Drop` returns the lane on success, on panic, and on caller
    /// cancellation alike. Nothing in this function returns the lane by hand,
    /// because anything done by hand is skipped by an early exit.
    async fn embed(self: &Arc<Self>, prefixed: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut lane = self.pool.acquire().await.ok_or_else(|| {
            tracing::error!(
                capacity = self.pool.capacity(),
                lost = self.pool.lost(),
                "refusing request: every inference lane has been lost"
            );
            EmbedError::PoolExhausted
        })?;

        let state = self.clone();
        tokio::task::spawn_blocking(move || {
            embed_prefixed(
                lane.get_mut(),
                &state.tokenizer,
                &prefixed,
                state.max_batch_tokens,
            )
        })
        .await
        .map_err(|err| {
            // The task panicked. Surface it — do NOT unwrap it into a panic of
            // our own, and do NOT return an empty vector that would read as
            // success to the caller.
            tracing::error!(error = %err, "inference task failed; lane was returned by its guard");
            EmbedError::Inference
        })
    }
}

#[derive(Deserialize, ToSchema)]
struct EmbedRequest {
    /// Text to embed.
    #[schema(example = "The dragon coiled around the obsidian spire.")]
    text: String,
    /// nomic task prefix; prepended as `"{task_type}: {text}"` before encoding.
    #[serde(default = "default_task")]
    #[schema(example = "search_document", default = "search_document")]
    task_type: String,
}

#[derive(Deserialize, ToSchema)]
struct EmbedBatchRequest {
    /// Texts to embed. Processed in token-bounded sub-batches internally, so a
    /// large request (or one with a very long text) is handled without a memory
    /// spike; vectors are returned in input order.
    texts: Vec<String>,
    #[serde(default = "default_task")]
    #[schema(example = "search_document", default = "search_document")]
    task_type: String,
}

fn default_task() -> String {
    "search_document".to_string()
}

#[derive(Serialize, ToSchema)]
struct EmbedResponse {
    /// Raw (un-normalized) masked-mean-pooled embedding; `|v|` ~= 22.
    vector: Vec<f32>,
    /// Always 768 for nomic-embed-text-v1.5.
    dimensions: usize,
}

#[derive(Serialize, ToSchema)]
struct EmbedBatchResponse {
    vectors: Vec<Vec<f32>>,
    count: usize,
    dimensions: usize,
}

#[derive(Serialize, ToSchema)]
struct HealthResponse {
    status: String,
    model: String,
    dimensions: usize,
    /// Inference lanes the pool was built with (`EMBEDDING_POOL_SIZE`).
    lanes_total: usize,
    /// Lanes that can still serve a request. Equals `lanes_total` in a healthy
    /// process; a lower number means capacity was permanently lost and is worth
    /// alerting on. When this reaches 0 the endpoint returns **503**, because a
    /// liveness probe that passes while every request fails is worse than none.
    lanes_available: usize,
}

/// Embed a batch of already-prefixed strings on the given session. Returns raw
/// (un-normalized) masked-mean-pooled vectors, one per input.
///
/// A forward pass allocates activations proportional to `batch × seq` (plus
/// O(seq²) attention per layer) where `seq` is the longest text IN THE BATCH —
/// so one 8192-token fragment dragged into a batch of ten pads all ten to 8192
/// and blows peak RAM (the cgroup-v2 OOM that crashed bulk re-embeds). To make
/// peak memory a function of an internal cap rather than of request size, we
/// tokenize once (UNpadded — so we see true lengths), then process the inputs in
/// order-preserving sub-batches each bounded by `max_batch_tokens` (rows × the
/// sub-batch's own max seq). Large requests are transparently handled as several
/// sequential `run()`s. A single text always runs even if it alone exceeds the
/// cap (its sequence can't be split without changing the embedding).
///
/// Each sub-batch is zero-padded to ITS OWN longest member inside
/// [`run_encodings`] — identical tensors to the old global-BatchLongest padding,
/// so embeddings are byte-for-byte unchanged.
fn embed_prefixed(
    session: &mut Session,
    tokenizer: &Tokenizer,
    prefixed: &[String],
    max_batch_tokens: usize,
) -> Vec<Vec<f32>> {
    if prefixed.is_empty() {
        return Vec::new();
    }
    // Unpadded tokenization (padding is disabled on the tokenizer) → true lengths
    // drive sub-batch planning; truncation to MAX_LEN still applies.
    let encodings = tokenizer
        .encode_batch(prefixed.to_vec(), true)
        .expect("tokenization failed");

    let lengths: Vec<usize> = encodings.iter().map(|e| e.get_ids().len()).collect();
    let mut result: Vec<Vec<f32>> = Vec::with_capacity(encodings.len());
    for (start, end) in plan_subbatches(&lengths, max_batch_tokens) {
        result.extend(run_encodings(session, &encodings[start..end]));
    }
    result
}

/// Partition `lengths` (token counts, in input order) into contiguous
/// `[start, end)` sub-batch ranges such that, for each range, `rows × (max length
/// in the range) ≤ max_batch_tokens` — the quantity that bounds a forward pass's
/// activation memory. Order-preserving (so results reassemble trivially). A single
/// item is always emitted as its own range even if it alone exceeds the budget (a
/// text's sequence can't be split without changing its embedding).
fn plan_subbatches(lengths: &[usize], max_batch_tokens: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < lengths.len() {
        let mut end = start;
        let mut max_seq = 0usize;
        while end < lengths.len() {
            let new_max = max_seq.max(lengths[end]);
            let rows = end - start + 1;
            // Always take the first item (rows == 1); otherwise stop before
            // exceeding the budget.
            if rows > 1 && rows * new_max > max_batch_tokens {
                break;
            }
            max_seq = new_max;
            end += 1;
        }
        ranges.push((start, end));
        start = end;
    }
    ranges
}

/// Run one sub-batch of (unpadded) encodings through the model and return raw
/// masked-mean-pooled vectors, one per encoding. Zero-pads to the sub-batch's own
/// longest member. Split out of [`embed_prefixed`] so the token-budget planner can
/// call it per sub-batch.
fn run_encodings(session: &mut Session, encodings: &[tokenizers::Encoding]) -> Vec<Vec<f32>> {
    let batch = encodings.len();
    if batch == 0 {
        return Vec::new();
    }
    let seq = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(0);

    // Flat row-major [batch, seq] tensors fed to ort as (shape, Vec).
    let mut ids = vec![0i64; batch * seq];
    let mut mask = vec![0i64; batch * seq];
    let types = vec![0i64; batch * seq]; // single segment -> all zeros
    for (b, enc) in encodings.iter().enumerate() {
        for (s, (&id, &m)) in enc
            .get_ids()
            .iter()
            .zip(enc.get_attention_mask().iter())
            .enumerate()
        {
            ids[b * seq + s] = id as i64;
            mask[b * seq + s] = m as i64;
        }
    }

    let shape = vec![batch, seq];
    // SessionOutputs borrow the session, so extract owned data before they drop.
    // last_hidden_state: [batch, seq, dim].
    let (seq_out, dim, data): (usize, usize, Vec<f32>) = {
        let outputs = session
            .run(ort::inputs![
                "input_ids" => Value::from_array((shape.clone(), ids)).unwrap(),
                "attention_mask" => Value::from_array((shape.clone(), mask.clone())).unwrap(),
                "token_type_ids" => Value::from_array((shape.clone(), types)).unwrap(),
            ])
            .expect("onnx inference failed");
        let (out_shape, out_data) = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .expect("extract last_hidden_state");
        (
            out_shape[1] as usize,
            out_shape[2] as usize,
            out_data.to_vec(),
        )
    };

    // MASKED mean pooling — sum hidden over real tokens, divide by token count.
    let mut result = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut pooled = vec![0.0f32; dim];
        let mut count = 0.0f32;
        for s in 0..seq_out {
            if mask[b * seq + s] == 0 {
                continue;
            }
            count += 1.0;
            let base = (b * seq_out + s) * dim;
            for d in 0..dim {
                pooled[d] += data[base + d];
            }
        }
        if count > 0.0 {
            for v in pooled.iter_mut() {
                *v /= count;
            }
        }
        // NO normalization — parity with the Python sidecar.
        result.push(pooled);
    }
    result
}

/// Embed a single text into a 768-dim vector.
#[utoipa::path(
    post, path = "/embed", tag = "embeddings",
    request_body = EmbedRequest,
    responses(
        (status = 200, description = "Embedding vector", body = EmbedResponse),
        (status = 500, description = "Inference failed"),
        (status = 503, description = "No usable inference lanes remain")
    )
)]
async fn handle_embed(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, EmbedError> {
    let prefixed = vec![format!("{}: {}", req.task_type, req.text)];
    let vectors = state.embed(prefixed).await?;
    let vector = vectors.into_iter().next().unwrap_or_default();
    let dimensions = vector.len();
    Ok(Json(EmbedResponse { vector, dimensions }))
}

/// Embed multiple texts in one call.
#[utoipa::path(
    post, path = "/embed-batch", tag = "embeddings",
    request_body = EmbedBatchRequest,
    responses(
        (status = 200, description = "Embedding vectors", body = EmbedBatchResponse),
        (status = 500, description = "Inference failed"),
        (status = 503, description = "No usable inference lanes remain")
    )
)]
async fn handle_embed_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedBatchRequest>,
) -> Result<Json<EmbedBatchResponse>, EmbedError> {
    let prefixed: Vec<String> = req
        .texts
        .iter()
        .map(|t| format!("{}: {}", req.task_type, t))
        .collect();
    let vectors = state.embed(prefixed).await?;
    let count = vectors.len();
    let dimensions = vectors.first().map(|v| v.len()).unwrap_or(state.dim);
    Ok(Json(EmbedBatchResponse {
        vectors,
        count,
        dimensions,
    }))
}

/// Liveness + model info + real inference-lane capacity.
///
/// Reports the pool's ACTUAL state and fails with 503 once no lane can serve a
/// request. The previous version returned a hardcoded `"healthy"` and touched
/// neither the pool nor inference — which is why `docker ps` reported healthy
/// for the seven days this service was completely dead.
#[utoipa::path(get, path = "/health", tag = "embeddings",
    responses(
        (status = 200, description = "Healthy", body = HealthResponse),
        (status = 503, description = "No usable inference lanes remain")
    ))]
async fn handle_health(
    State(state): State<Arc<AppState>>,
) -> Result<Json<HealthResponse>, EmbedError> {
    let lanes_available = state.pool.available_capacity();
    if lanes_available == 0 {
        tracing::error!("health: no usable inference lanes remain");
        return Err(EmbedError::PoolExhausted);
    }
    Ok(Json(HealthResponse {
        status: "healthy".to_string(),
        model: MODEL_NAME.to_string(),
        dimensions: state.dim,
        lanes_total: state.pool.capacity(),
        lanes_available,
    }))
}

/// OpenAPI document. Paths are merged in from the `#[utoipa::path]` handlers via
/// `routes!` in `api_router`, and schemas from the `ToSchema` derives — so the
/// contract is code-derived, not hand-kept.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "embeddings-rs",
        description = "Embedding sidecar for nomic-embed-text-v1.5 (768-dim). Drop-in for the Python `embeddings` service.",
        version = "0.1.1"
    ),
    tags((name = "embeddings", description = "Text embedding endpoints"))
)]
struct ApiDoc;

/// Build the route table once. Returns the (stateless) router plus the OpenAPI
/// doc with all paths merged in. `--dump-openapi` uses only the doc half (no
/// model load); the server attaches state to the router half.
fn api_router() -> (axum::Router<Arc<AppState>>, utoipa::openapi::OpenApi) {
    OpenApiRouter::with_openapi(ApiDoc::openapi())
        .routes(routes!(handle_embed))
        .routes(routes!(handle_embed_batch))
        .routes(routes!(handle_health))
        .split_for_parts()
}

#[tokio::main]
async fn main() {
    // Offline contract export: emit the spec and exit WITHOUT loading the model,
    // so CI can regenerate contract/openapi.json with no weights present.
    // e.g. `embeddings-rs --dump-openapi > contract/openapi.json`
    if std::env::args().any(|a| a == "--dump-openapi") {
        let (_, api) = api_router();
        println!("{}", api.to_pretty_json().expect("serialize openapi"));
        return;
    }

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "models".to_string());
    let port: u16 = std::env::var("EMBEDDING_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8402);

    // Tokenizer: configure truncation to the model's 8192 ceiling and pad to
    // the longest sequence in each batch (matches sentence-transformers).
    let mut tokenizer =
        Tokenizer::from_file(format!("{model_dir}/tokenizer.json")).expect("load tokenizer.json");
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: MAX_LEN,
            ..Default::default()
        }))
        .expect("set truncation");
    // Padding is done per-sub-batch in `run_encodings` (zero-fill to the sub-batch
    // max), NOT globally — global BatchLongest would pad every text to the single
    // longest in the request, defeating the token-budget chunking. Disable it so
    // `encode_batch` returns true lengths for the planner.
    tokenizer.with_padding(None);

    // Pool of N independent sessions ("lanes"). Each session is another full
    // copy of the model weights in RAM (~550 MB), so scale with mem_limit.
    let pool_size: usize = std::env::var("EMBEDDING_POOL_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(2);
    // Split intra-op threads across lanes so N concurrent inferences don't
    // oversubscribe the CPU (N lanes x C/N threads ~= C cores total).
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let intra = std::cmp::max(1, cores / pool_size);

    // Tokens-per-run cap (rows × padded seq). Floored at MAX_LEN so a single
    // full-length text can always run. Default 16384 = two full-length texts'
    // worth of activations — keeps peak well inside a 3 GB lane budget while
    // still batching many short texts together.
    let max_batch_tokens: usize = std::env::var("EMBEDDING_MAX_BATCH_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(|n: usize| n.max(MAX_LEN))
        .unwrap_or(16_384);

    // CPU arena allocator. ON by default, which is ORT's own default and the
    // behaviour every deployed version has had — this flag changes nothing unless
    // it is set, deliberately, because the trade is real in both directions.
    //
    // WHY IT IS REACHABLE AT ALL (measured on production 2026-09-04): with the
    // arena ON, ORT's BFCArena extends on a strict POWER-OF-TWO ladder and never
    // returns the high-water. Observed grabs were exactly 64/128/256/512 MiB,
    // 1 GiB, 2 GiB, with the largest pair asking 1.90 GiB and TAKING 2.00 GiB —
    // so the rung after 2 GiB is 4 GiB, which cannot be reached inside a 4 GiB
    // container that already holds the weights. The sidecar sat at 3.8 GiB of its
    // 4 GiB cap AT IDLE, CPU 0.00%, a day after a bulk run, because only a process
    // restart releases the arena. That is what stalls a bulk re-embed at ~600
    // nodes and why it "recovers" by dying.
    //
    // ⚠️ AND `arena_extend_strategy = kSameAsRequested` IS NOT AVAILABLE HERE.
    // In ort 2.0.0-rc.12 that option exists only on the CUDA/ROCm/MIGraphX/CANN
    // providers (src/ep/{cuda,rocm,migraphx,cann}.rs). The CPU provider —
    // src/ep/cpu.rs, which is what MLAS runs on the droplet — exposes exactly one
    // arena knob: with_arena_allocator(bool), i.e. EnableCpuMemArena /
    // DisableCpuMemArena. So getting off the ladder on CPU means turning the arena
    // OFF, not re-strategising it.
    //
    // THE TRADE, stated rather than assumed: OFF means every tensor goes to the
    // system allocator at its requested size and is freed when it drops — no
    // ladder, no permanent high-water — at the cost of malloc/free per allocation
    // instead of arena reuse. For a SEQUENTIAL BULK re-embed that is very likely
    // the right side; for low-latency serving it may not be. Nobody has measured
    // the serving cost, so the default stays ON.
    let cpu_arena: bool = std::env::var("EMBEDDING_CPU_ARENA")
        .ok()
        .map(|v| !matches!(v.trim(), "0" | "false" | "no" | "off"))
        .unwrap_or(true);

    let mut sessions = Vec::with_capacity(pool_size);
    for _ in 0..pool_size {
        let session = Session::builder()
            .expect("session builder")
            .with_execution_providers([ep::CPU::default().with_arena_allocator(cpu_arena).build()])
            .expect("register CPU execution provider")
            .with_intra_threads(intra)
            .expect("set intra threads")
            .commit_from_file(format!("{model_dir}/model.onnx"))
            .expect("load model.onnx");
        sessions.push(session);
    }
    tracing::info!(
        "loaded {pool_size} session lane(s), {intra} intra-op thread(s) each ({cores} cores); \
         max_batch_tokens={max_batch_tokens}, cpu_arena={cpu_arena}"
    );
    if !cpu_arena {
        // Say it loudly and separately: this is a non-default allocator mode, and
        // a reader diagnosing throughput must not have to infer it from a field at
        // the end of another line.
        tracing::warn!(
            "CPU ARENA DISABLED (EMBEDDING_CPU_ARENA): allocations are sized as \
             requested and released on drop, so RSS should no longer hold a \
             power-of-two high-water — expect lower peak memory and some \
             per-request allocation overhead"
        );
    }

    let state = Arc::new(AppState {
        pool: LanePool::new(sessions),
        tokenizer,
        dim: 768,
        max_batch_tokens,
    });

    let (router, api) = api_router();
    // Serve the spec at /openapi.json too (matches the FastAPI services, so the
    // contract tooling's online mode works the same way against this service).
    let spec = api.to_pretty_json().expect("serialize openapi");
    let app = router
        .route(
            "/openapi.json",
            axum::routing::get(move || {
                let spec = spec.clone();
                async move {
                    (
                        [(axum::http::header::CONTENT_TYPE, "application/json")],
                        spec,
                    )
                }
            }),
        )
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    tracing::info!("embeddings-rs ({MODEL_NAME}) listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    axum::serve(listener, app).await.expect("serve");
}

#[cfg(test)]
mod tests {
    use super::plan_subbatches;

    /// Every emitted sub-batch must respect the token budget (except a lone
    /// over-budget item, which must still appear as its own range), and the
    /// ranges must tile the input contiguously in order.
    fn assert_valid(lengths: &[usize], budget: usize) -> Vec<(usize, usize)> {
        let ranges = plan_subbatches(lengths, budget);
        // Contiguous cover in order.
        let mut cursor = 0;
        for &(s, e) in &ranges {
            assert_eq!(s, cursor, "ranges must be contiguous: {ranges:?}");
            assert!(e > s, "ranges must be non-empty: {ranges:?}");
            let rows = e - s;
            let max_seq = lengths[s..e].iter().copied().max().unwrap_or(0);
            if rows > 1 {
                assert!(
                    rows * max_seq <= budget,
                    "multi-item range {s}..{e} (rows={rows}, max_seq={max_seq}) exceeds budget {budget}"
                );
            }
            cursor = e;
        }
        assert_eq!(
            cursor,
            lengths.len(),
            "ranges must cover all items: {ranges:?}"
        );
        ranges
    }

    #[test]
    fn empty_input() {
        assert!(plan_subbatches(&[], 16384).is_empty());
    }

    #[test]
    fn all_short_pack_into_one() {
        assert_eq!(assert_valid(&[10, 10, 10], 100), vec![(0, 3)]);
    }

    #[test]
    fn splits_when_budget_exceeded() {
        // 2*50=100 (ok), 3*50=150 (>100) → split into pairs.
        assert_eq!(assert_valid(&[50, 50, 50, 50], 100), vec![(0, 2), (2, 4)]);
    }

    #[test]
    fn single_over_budget_item_runs_alone() {
        assert_eq!(assert_valid(&[8192], 100), vec![(0, 1)]);
        // Over-budget item flushes on its own, neighbors batch separately.
        assert_eq!(
            assert_valid(&[10, 9000, 10], 100),
            vec![(0, 1), (1, 2), (2, 3)]
        );
    }

    #[test]
    fn long_then_short_respects_running_max() {
        // start at 8000: rows1=8000; rows2 → 2*8000=16000 ≤16384 ok;
        // rows3 → 3*8000=24000 >16384 break.
        assert_eq!(assert_valid(&[8000, 10, 10], 16384), vec![(0, 2), (2, 3)]);
    }

    #[test]
    fn budget_floored_at_one_full_text() {
        // A realistic mix never violates the bound at the default budget.
        let lengths = [8192, 12, 40, 8000, 7, 7, 7, 6000, 5, 5, 5, 5];
        assert_valid(&lengths, 16384);
    }
}
