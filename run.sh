#!/usr/bin/env bash
# Build and run the Vulkan tutorial app.
# Usage: ./run.sh [--clean]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build"
APP_DIR="$BUILD_DIR/GaussianSplatting"

if [[ "${1:-}" == "--clean" ]]; then
  echo ">> Removing build directory"
  rm -rf "$BUILD_DIR"
fi

echo ">> Configuring"
cmake -S "$ROOT" -B "$BUILD_DIR"

echo ">> Building"
cmake --build "$BUILD_DIR"

echo ">> Running (working dir: $APP_DIR)"
cd "$APP_DIR"
exec ./GaussianSplatting
