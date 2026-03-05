# syntax=docker/dockerfile:1.7

# Multi-stage Dockerfile for the IronClaw agent (cloud deployment).
#
# Build (local):
#   docker build --platform linux/amd64 -t ironclaw:latest .
#
# Build (CI with sccache — automatically uses GHA cache):
#   docker build --platform linux/amd64 \
#     --secret id=actions_cache_url,env=ACTIONS_CACHE_URL \
#     --secret id=actions_runtime_token,env=ACTIONS_RUNTIME_TOKEN \
#     -t ironclaw:latest .
#
# Run:
#   docker run --env-file .env -p 3000:3000 ironclaw:latest

# Stage 1: Build
FROM rust:1.92-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev cmake gcc g++ curl \
    && rm -rf /var/lib/apt/lists/* \
    && rustup target add wasm32-wasip2 \
    && cargo install wasm-tools

# Install sccache (prebuilt binary — ~3s vs ~2min from source).
# In CI, sccache uses GitHub Actions cache as backend so individual
# compilation units survive across builds. For local builds without
# GHA secrets, sccache is installed but not activated.
ARG SCCACHE_VERSION=0.9.1
RUN curl -sSfL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
    | tar xz --strip-components=1 -C /usr/local/bin "sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl/sccache" \
    && chmod +x /usr/local/bin/sccache

WORKDIR /app

# Copy dependency manifests first (layer cached when Cargo.lock unchanged)
COPY Cargo.toml Cargo.lock ./
COPY channels-src/ channels-src/

# Pre-build dependencies only (dummy src for cache layer).
# This layer is invalidated only by Cargo.toml/Cargo.lock changes.
RUN mkdir -p src && echo "fn main() {}" > src/main.rs \
    && mkdir -p tests && touch tests/empty.rs \
    && mkdir -p migrations registry wit \
    && echo "fn main() {}" > build.rs

RUN --mount=type=secret,id=actions_cache_url \
    --mount=type=secret,id=actions_runtime_token \
    --mount=type=cache,id=ironclaw-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=ironclaw-cargo-git,target=/usr/local/cargo/git \
    if [ -f /run/secrets/actions_cache_url ]; then \
      export ACTIONS_CACHE_URL="$(cat /run/secrets/actions_cache_url)" ; \
      export ACTIONS_RUNTIME_TOKEN="$(cat /run/secrets/actions_runtime_token)" ; \
      export SCCACHE_GHA_ENABLED=true ; \
      export RUSTC_WRAPPER=sccache ; \
    fi && \
    cargo build --release --bin ironclaw 2>&1 || true && \
    if [ "${RUSTC_WRAPPER:-}" = "sccache" ]; then sccache --show-stats; fi

# Copy real source (only this layer changes on source edits)
COPY build.rs build.rs
COPY src/ src/
COPY tests/ tests/
COPY migrations/ migrations/
COPY registry/ registry/
COPY wit/ wit/

RUN --mount=type=secret,id=actions_cache_url \
    --mount=type=secret,id=actions_runtime_token \
    --mount=type=cache,id=ironclaw-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=ironclaw-cargo-git,target=/usr/local/cargo/git \
    if [ -f /run/secrets/actions_cache_url ]; then \
      export ACTIONS_CACHE_URL="$(cat /run/secrets/actions_cache_url)" ; \
      export ACTIONS_RUNTIME_TOKEN="$(cat /run/secrets/actions_runtime_token)" ; \
      export SCCACHE_GHA_ENABLED=true ; \
      export RUSTC_WRAPPER=sccache ; \
    fi && \
    cargo build --release --bin ironclaw && \
    if [ "${RUSTC_WRAPPER:-}" = "sccache" ]; then sccache --show-stats; fi && \
    install -D /app/target/release/ironclaw /app-out/ironclaw

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app-out/ironclaw /usr/local/bin/ironclaw
COPY --from=builder /app/migrations /app/migrations

# Non-root user
RUN useradd -m -u 1000 -s /bin/bash ironclaw
USER ironclaw

EXPOSE 3000

ENV RUST_LOG=ironclaw=info

ENTRYPOINT ["ironclaw"]
