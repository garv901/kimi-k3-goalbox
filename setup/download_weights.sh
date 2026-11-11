#!/bin/bash
# Download the Kimi K3 checkpoint (about 1.56 TB, 96 shards) at the pinned revision.
#
#   setup/download_weights.sh DEST_DIR        (HF_TOKEN optional; it raises the rate limit)
#
# The repository moonshotai/Kimi-K3 is public. Shards mix text and vision tensors, so all
# 96 shards are needed; the converter reads only the text model.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DEST=${1:?usage: download_weights.sh DEST_DIR}
REV=f831ab66814297da540d832a5235f8e904f29d06
mkdir -p "$DEST"
"$REPO/.venv/bin/python" -s - "$DEST" "$REV" <<'EOF'
import sys, time
from huggingface_hub import snapshot_download
dest, rev = sys.argv[1], sys.argv[2]
for attempt in range(20):
    try:
        path = snapshot_download("moonshotai/Kimi-K3", revision=rev, local_dir=dest, max_workers=16,
                                 allow_patterns=["model-*.safetensors", "*.json", "*.py", "tiktoken.model"])
        print("downloaded to", path); break
    except Exception as exc:  # rate limits and dropped connections: resume
        print(f"retry {attempt}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        time.sleep(min(600, 30 * (attempt + 1)))
else:
    sys.exit("download failed")
EOF
"$REPO/.venv/bin/python" -s - "$DEST" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
files = set(json.loads((d / "model.safetensors.index.json").read_text())["weight_map"].values())
missing = [f for f in sorted(files) if not (d / f).is_file()]
print(f"{len(files) - len(missing)}/{len(files)} shards present")
sys.exit(1 if missing else 0)
EOF
