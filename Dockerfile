# syntax=docker/dockerfile:1.7

# Multi-stage Dockerfile for the IronClaw agent (cloud deployment).
#
# Build (local):
#   docker build --platform linux/amd64 -t ironclaw:latest .
#
# Build (CI with registry cache):
#   docker build --platform linux/amd64 \
#     --cache-from type=registry,ref=registry.wynndrahorad.com/ironclaw:buildcache \
#     --cache-to type=registry,ref=registry.wynndrahorad.com/ironclaw:buildcache,mode=max \
#     -t ironclaw:latest .
#
# Run:
#   docker run --env-file .env -p 3000:3000 ironclaw:latest

# Stage 1: Build
FROM rust:1.92-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev cmake gcc g++ curl \
    && rm -rf /var/lib/apt/lists/* \
    && rustup target add wasm32-wasip2

# Install wasm-tools from prebuilt binary (~3s vs ~2min from source).
ARG WASM_TOOLS_VERSION=1.245.1
RUN curl -sSfL "https://github.com/bytecodealliance/wasm-tools/releases/download/v${WASM_TOOLS_VERSION}/wasm-tools-${WASM_TOOLS_VERSION}-x86_64-linux.tar.gz" \
    | tar xz --strip-components=1 -C /usr/local/bin "wasm-tools-${WASM_TOOLS_VERSION}-x86_64-linux/wasm-tools" \
    && chmod +x /usr/local/bin/wasm-tools

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

RUN --mount=type=cache,id=ironclaw-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=ironclaw-cargo-git,target=/usr/local/cargo/git \
    cargo build --release --bin ironclaw 2>&1 || true

# Copy real source (only this layer changes on source edits)
COPY build.rs build.rs
COPY src/ src/
COPY tests/ tests/
COPY migrations/ migrations/
COPY registry/ registry/
COPY wit/ wit/

RUN --mount=type=cache,id=ironclaw-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=ironclaw-cargo-git,target=/usr/local/cargo/git \
    cargo build --release --bin ironclaw \
    && install -D /app/target/release/ironclaw /app-out/ironclaw

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
