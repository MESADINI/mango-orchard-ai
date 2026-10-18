#!/usr/bin/env bash
# Downloads the deployed model weights from the GitHub release and verifies each SHA-256 against
# ml/model_registry.json. Files that are already present and correct are skipped.
#   bash scripts/download_weights.sh [owner/name] [tag]
# Until the release is published, its assets are visible only to collaborators: the script then falls back to the
# GitHub CLI (`gh release download`, after `gh auth login`). Needs curl and python3 (or python).
set -euo pipefail
REPO="${1:-MESADINI/mango-orchard-ai}"
TAG="${2:-v1.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=""
for c in python3 python; do   # skip stubs such as the Windows Store alias, which exist but do not run
  if command -v "$c" >/dev/null && "$c" -c "import sys" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "python3 (or python) is required" >&2; exit 1; }

py() { "$PY" "$@" | tr -d '\r'; }   # Windows Python prints CRLF
sha() { py -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; }

get_asset() {  # name dest
  mkdir -p "$(dirname "$2")"
  if ! curl -fsSL -o "$2.download" "https://github.com/$REPO/releases/download/$TAG/$1"; then
    command -v gh >/dev/null || { echo "Could not download $1 from release $TAG of $REPO (not published yet?). Install the GitHub CLI and run 'gh auth login', or ask the owner to publish the release." >&2; exit 1; }
    echo "  public download failed; trying the GitHub CLI (release may still be a draft)"
    gh release download "$TAG" -R "$REPO" -p "$1" -D "$(dirname "$2")" --clobber
    mv -f "$(dirname "$2")/$1" "$2.download"
  fi
  mv -f "$2.download" "$2"
}

failed=0
n=0
while IFS=$'\t' read -r name path expected; do
  n=$((n+1))
  dest="$ROOT/$path"; file="$(basename "$path")"; meta="${dest%.*}.json"
  if [ -f "$dest" ] && [ "$(sha "$dest")" = "$expected" ]; then
    echo "OK (present)  $file"
  else
    echo "Downloading   $file ..."
    get_asset "$file" "$dest"
    if [ "$(sha "$dest")" = "$expected" ]; then echo "OK (verified) $file  sha256 ${expected:0:12}"
    else rm -f "$dest"; echo "SHA-256 MISMATCH for $file; file removed." >&2; failed=1; fi
  fi
  # the pest detector's metadata (classes, host_verified flags) must describe exactly this checkpoint
  if [ "$name" = "pest_detector" ]; then
    [ -f "$meta" ] || get_asset "$(basename "$meta")" "$meta"
    if [ "$(py -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$meta")" != "$expected" ]; then
      rm -f "$meta"; echo "pest_detector.json does not match the registry; removed." >&2; failed=1
    fi
  fi
done < <(py -c 'import json,sys; [print(m["name"], m["path"], m["sha256"], sep="\t") for m in json.load(open(sys.argv[1]))["models"]]' "$ROOT/ml/model_registry.json")
[ "$n" -gt 0 ] || { echo "no models read from ml/model_registry.json" >&2; exit 1; }
[ "$failed" = 0 ] || exit 1
echo "All model weights match ml/model_registry.json."
