#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${1:-./data}"
DST_DIR="${2:-./compressed_data}"
JOBS="${3:-${JOBS:-$(nproc 2>/dev/null || echo 1)}}"
# Validate concurrency value
if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  JOBS=1
fi

# Normalize paths: strip trailing slash
SRC_DIR="${SRC_DIR%/}"
DST_DIR="${DST_DIR%/}"

compress_one() {
  set -euo pipefail
  local src="$1"

  # Compute relative path
  local rel="${src#$SRC_DIR/}"
  local dst_dir="$DST_DIR/$(dirname "$rel")"
  mkdir -p "$dst_dir"

  local base
  base="$(basename "$rel")"

  # Skip common compressed/archive formats
  shopt -s nocasematch
  if [[ "$base" =~ \.(gz|zip|xz|bz2|zst|7z|rar|tar|tgz|tar\.gz)$ ]]; then
    echo "Skip compressed file: $src"
    shopt -u nocasematch
    return 0
  fi
  shopt -u nocasematch

  local dst="$dst_dir/$base.gz"

  # If target exists and is newer, skip recompression
  if [[ -f "$dst" && "$dst" -nt "$src" ]]; then
    echo "Up to date: $dst"
    return 0
  fi

  echo "Compress: $src -> $dst"
  # Write to a temp file first, then atomically replace to avoid partial states
  if ! gzip -c -- "$src" > "$dst.tmp"; then
    echo "Compression failed: $src" >&2
    rm -f "$dst.tmp"
    return 1
  fi
  mv -f "$dst.tmp" "$dst"
}

export -f compress_one
export SRC_DIR DST_DIR

# Parallel processing: -P sets concurrency; -0/-print0 handles filenames with spaces or special characters
find "$SRC_DIR" -type f -print0 | xargs -0 -n 1 -P "$JOBS" -I {} bash -c 'compress_one "$@"' _ "{}"

echo "All done."