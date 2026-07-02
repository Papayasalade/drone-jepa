#!/bin/sh
# Build the demo WASM bundle with source paths remapped: Rust embeds absolute
# panic-location paths (home dir, toolchain) into the binary otherwise — an
# identity leak in any published bundle. Always build the deployable pkg here.
set -e
cd "$(git rev-parse --show-toplevel)/web-demo/racer"
RUSTFLAGS="--remap-path-prefix=$HOME/.rustup=/rustup --remap-path-prefix=$HOME/.cargo=/cargo --remap-path-prefix=$HOME=/src" \
  wasm-pack build --target web --out-dir ../web/pkg -- --features wasm
