#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${1:-./compressed_data}"
DST_DIR="${2:-./data}"
JOBS="${3:-${JOBS:-$(nproc 2>/dev/null || echo 1)}}"

# Validate concurrency value
if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  JOBS=1
fi

# Normalize paths
SRC_DIR="${SRC_DIR%/}"
DST_DIR="${DST_DIR%/}"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source directory does not exist: $SRC_DIR" >&2
  exit 1
fi

decompress_one() {
  set -euo pipefail
  local src="$1"

  # Compute relative path
  local rel="${src#$SRC_DIR/}"
  local dst_dir="$DST_DIR/$(dirname "$rel")"
  mkdir -p "$dst_dir"

  local base
  base="$(basename "$rel")"

  # Handle .gz files only
  if [[ "$base" != *.gz ]]; then
    echo "Skip non-.gz file: $src"
    return 0
  fi

  local out_name="${base%.gz}"
  local dst="$dst_dir/$out_name"

  # If destination exists and is newer, skip decompression
  if [[ -f "$dst" && "$dst" -nt "$src" ]]; then
    echo "Up to date: $dst"
    return 0
  fi

  echo "Decompress: $src -> $dst"
  # Write to a temp file first, then atomically replace on success
  if ! gzip -cd -- "$src" > "$dst.tmp"; then
    echo "Decompression failed: $src" >&2
    rm -f "$dst.tmp"
    return 1
  fi
  mv -f "$dst.tmp" "$dst"
}

export -f decompress_one
export SRC_DIR DST_DIR

# Parallel processing: -P sets concurrency; -0/-print0 handles filenames with spaces or special characters
find "$SRC_DIR" -type f -print0 | xargs -0 -n 1 -P "$JOBS" -I {} bash -c 'decompress_one "$@"' _ "{}"

echo "All done."