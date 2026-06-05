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

use std::sync::{Arc, Mutex};

use axum::{extract::State, Json};
use ort::session::Session;
use ort::value::Value;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tokio::sync::Semaphore;
use utoipa::{OpenApi, ToSchema};
use utoipa_axum::{router::OpenApiRouter, routes};

const MODEL_NAME: &str = "nomic-ai/nomic-embed-text-v1.5";
const MAX_LEN: usize = 8192;

/// A pool of N independent ONNX sessions ("lanes"). ort's `run` needs `&mut
/// self`, so a session can't be shared for concurrent inference — instead we
/// keep N of them and hand one out per in-flight request. `sem` has exactly N
/// permits, so a permit-holder is always guaranteed a session to pop. N lanes
/// cost N x the model weights in RAM (see EMBEDDING_POOL_SIZE / mem_limit).
struct AppState {
    sessions: Mutex<Vec<Session>>,
    sem: Semaphore,
    tokenizer: Tokenizer,
    dim: usize,
}

impl AppState {
    /// Acquire a lane, run inference off the async runtime, return the lane.
    async fn embed(self: &Arc<Self>, prefixed: Vec<String>) -> Vec<Vec<f32>> {
        // Wait until a lane is free (no busy-wait; the permit gates concurrency).
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        let mut session = self
            .sessions
            .lock()
            .unwrap()
            .pop()
            .expect("permit guarantees an available session");

        let state = self.clone();
        let (result, session) = tokio::task::spawn_blocking(move || {
            let v = embed_prefixed(&mut session, &state.tokenizer, &prefixed);
            (v, session)
        })
        .await
        .expect("inference task panicked");

        // Return the lane BEFORE the permit drops, so the next waiter finds one.
        // (If the blocking task had panicked the session would be lost and the
        // pool would shrink by one; restart: unless-stopped heals that.)
        self.sessions.lock().unwrap().push(session);
        result
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
    /// Texts to embed; padded to the longest in the batch.
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
}

/// Embed a batch of already-prefixed strings on the given session. Returns raw
/// (un-normalized) masked-mean-pooled vectors, one per input.
fn embed_prefixed(
    session: &mut Session,
    tokenizer: &Tokenizer,
    prefixed: &[String],
) -> Vec<Vec<f32>> {
    // Tokenize (padding to longest + truncation are configured on the tokenizer).
    let encodings = tokenizer
        .encode_batch(prefixed.to_vec(), true)
        .expect("tokenization failed");

    let batch = encodings.len();
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
    responses((status = 200, description = "Embedding vector", body = EmbedResponse))
)]
async fn handle_embed(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> Json<EmbedResponse> {
    let prefixed = vec![format!("{}: {}", req.task_type, req.text)];
    let vectors = state.embed(prefixed).await;
    let vector = vectors.into_iter().next().unwrap_or_default();
    let dimensions = vector.len();
    Json(EmbedResponse { vector, dimensions })
}

/// Embed multiple texts in one call.
#[utoipa::path(
    post, path = "/embed-batch", tag = "embeddings",
    request_body = EmbedBatchRequest,
    responses((status = 200, description = "Embedding vectors", body = EmbedBatchResponse))
)]
async fn handle_embed_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedBatchRequest>,
) -> Json<EmbedBatchResponse> {
    let prefixed: Vec<String> = req
        .texts
        .iter()
        .map(|t| format!("{}: {}", req.task_type, t))
        .collect();
    let vectors = state.embed(prefixed).await;
    let count = vectors.len();
    let dimensions = vectors.first().map(|v| v.len()).unwrap_or(state.dim);
    Json(EmbedBatchResponse {
        vectors,
        count,
        dimensions,
    })
}

/// Liveness + model info.
#[utoipa::path(get, path = "/health", tag = "embeddings",
    responses((status = 200, description = "Healthy", body = HealthResponse)))]
async fn handle_health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        model: MODEL_NAME.to_string(),
        dimensions: state.dim,
    })
}

/// OpenAPI document. Paths are merged in from the `#[utoipa::path]` handlers via
/// `routes!` in `api_router`, and schemas from the `ToSchema` derives — so the
/// contract is code-derived, not hand-kept.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "embeddings-rs",
        description = "Embedding sidecar for nomic-embed-text-v1.5 (768-dim). Drop-in for the Python `embeddings` service.",
        version = "0.1.0"
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
    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    }));

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

    let mut sessions = Vec::with_capacity(pool_size);
    for _ in 0..pool_size {
        let session = Session::builder()
            .expect("session builder")
            .with_intra_threads(intra)
            .expect("set intra threads")
            .commit_from_file(format!("{model_dir}/model.onnx"))
            .expect("load model.onnx");
        sessions.push(session);
    }
    tracing::info!(
        "loaded {pool_size} session lane(s), {intra} intra-op thread(s) each ({cores} cores)"
    );

    let state = Arc::new(AppState {
        sem: Semaphore::new(pool_size),
        sessions: Mutex::new(sessions),
        tokenizer,
        dim: 768,
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
