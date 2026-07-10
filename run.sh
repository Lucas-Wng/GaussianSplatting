#!/usr/bin/env bash
# Build and run the Gaussian Splatting viewer.
# Usage: ./run.sh [--clean] [scene.ply]
#   --clean      wipe build/ first
#   scene.ply    which capture to load (default: cactus.ply). Accepts a bare filename
#                (resolved against the run dir's models/), or any relative/absolute path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/build"
APP_DIR="$BUILD_DIR/GaussianSplatting"

CLEAN=0
PLY_ARG=""
for arg in "$@"; do
  if [[ "$arg" == "--clean" ]]; then
    CLEAN=1
  else
    PLY_ARG="$arg"
  fi
done

# Resolve the .ply argument (before we cd into the run dir):
#   - an existing file (relative to here, or absolute) -> pass its absolute path
#   - otherwise treat it as a name under the run dir's models/ folder
RUN_ARGS=()
if [[ -n "$PLY_ARG" ]]; then
  if [[ -f "$PLY_ARG" ]]; then
    RUN_ARGS=("$(cd "$(dirname "$PLY_ARG")" && pwd)/$(basename "$PLY_ARG")")
  else
    RUN_ARGS=("models/$(basename "$PLY_ARG")")
  fi
fi

if [[ "$CLEAN" == "1" ]]; then
  echo ">> Removing build directory"
  rm -rf "$BUILD_DIR"
fi

echo ">> Configuring"
cmake -S "$ROOT" -B "$BUILD_DIR"

echo ">> Building"
cmake --build "$BUILD_DIR"

echo ">> Running (working dir: $APP_DIR)${RUN_ARGS[0]+, scene: ${RUN_ARGS[0]}}"
cd "$APP_DIR"
exec ./GaussianSplatting ${RUN_ARGS[0]+"${RUN_ARGS[@]}"}
