# Rust/ort parity sidecar for nomic-embed-text-v1.5 (drop-in for the Python
# `embeddings` service). Multi-stage: a Rust builder that compiles the binary,
# downloads the fp32 ONNX weights, and stashes the onnxruntime shared lib that
# ort's `download-binaries` feature fetched; then a slim runtime.
#
# Parity with the Python sidecar is verified by dev_storyflow/scripts/parity_embeddings.py
# (cosine 1.0 / magnitude 1.0 over a length+task_type sweep).

# ----------------------------------------------------------------------------
# trixie (glibc 2.41), NOT bookworm (2.36): ort's download-binaries statically
# links a prebuilt onnxruntime compiled against glibc >=2.38 (it references the
# C23 __isoc23_* symbols), so an older base fails to link.
FROM rust:1-trixie AS builder

ARG SERVICE_TREE_SHA=unknown
LABEL service_tree_sha="$SERVICE_TREE_SHA"

# pkg-config + libssl-dev: ort's binary downloader (ureq -> native-tls) links openssl.
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release

# Pre-download the fp32 ONNX weights + tokenizer at build time (matches the
# Python image, which pre-pulls the model so the container starts fast).
#
# `-f` fails the build on HTTP errors and `--retry*` rides out transient ones,
# but Hugging Face can also answer 200 OK with an HTML interstitial (bot/JS
# challenge or rate-limit page) in place of the file. That is exactly what
# silently baked a ~3 KB web page in place of the model in the first v0.1.0
# image. So after downloading we VALIDATE the bytes and hard-fail the build
# rather than ship an unusable image: model.onnx must be ~550 MB (an HTML page
# is a few KB) and tokenizer.json must be JSON (starts with `{`, not `<`).
RUN mkdir -p /build/models \
    && base="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main" \
    && curl -fSL --retry 5 --retry-delay 5 --retry-all-errors \
        -A "embeddings-rs-docker-build" \
        "$base/onnx/model.onnx" -o /build/models/model.onnx \
    && curl -fSL --retry 5 --retry-delay 5 --retry-all-errors \
        -A "embeddings-rs-docker-build" \
        "$base/tokenizer.json" -o /build/models/tokenizer.json \
    && onnx_sz=$(wc -c < /build/models/model.onnx) \
    && if [ "$onnx_sz" -lt 100000000 ]; then \
        echo "ERROR: model.onnx is $onnx_sz bytes (expected ~547MB) — HF likely served HTML:"; \
        head -c 200 /build/models/model.onnx; echo; exit 1; \
    fi \
    && if [ "$(head -c1 /build/models/tokenizer.json)" != "{" ]; then \
        echo "ERROR: tokenizer.json is not JSON — HF likely served HTML:"; \
        head -c 200 /build/models/tokenizer.json; echo; exit 1; \
    fi \
    && echo "weights OK: model.onnx=${onnx_sz}B tokenizer.json=$(wc -c < /build/models/tokenizer.json)B"

# ort downloaded a matching libonnxruntime.so during the build — stash it so the
# runtime stage can load it (we link dynamically, not statically).
RUN mkdir -p /build/ortlib \
    && find /build/target/release -name 'libonnxruntime.so*' -exec cp -a {} /build/ortlib/ \; \
    && ls -la /build/ortlib/

# ----------------------------------------------------------------------------
# trixie runtime too: onnxruntime is statically linked into the binary but still
# resolves C23 glibc symbols at runtime, so the runtime glibc must be >=2.38.
FROM debian:trixie-slim AS runtime

# libgomp1: onnxruntime's CPU EP needs OpenMP. libssl3/ca-certs for completeness.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libssl3 ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/target/release/embeddings-rs /app/embeddings-rs
COPY --from=builder /build/models /app/models
COPY --from=builder /build/ortlib/ /usr/local/lib/
RUN ldconfig

ENV MODEL_DIR=/app/models
ENV EMBEDDING_PORT=8401
EXPOSE 8401

CMD ["/app/embeddings-rs"]
