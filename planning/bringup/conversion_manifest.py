"""CPU-only: write a reproducible manifest for one convert() output (planning/bringup/README.md).

Records the source directory, pinned revision, sha256 of config.json and the safetensors index, the HF shards the
converted layers read (from the index; sha256 of each when --hash-shards), the converter's sha256 and the repository
commit, the command line, every output file with size and sha256, DONE.json, and the trial log/time files if present.

Usage: python -s planning/bringup/conversion_manifest.py --hf HF_DIR --out OUT_DIR --tp 4 --layers 4 \
           --repo <repo> [--hash-shards] [--log trial.log --time trial.time.txt] > manifest.json"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

P = 'language_model.model.'


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--hf', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--tp', type=int, required=True); ap.add_argument('--layers', type=int, required=True)
    ap.add_argument('--repo', required=True); ap.add_argument('--hash-shards', action='store_true')
    ap.add_argument('--log'); ap.add_argument('--time')
    args = ap.parse_args()
    hf, out, repo = Path(args.hf), Path(args.out), Path(args.repo)
    index = json.loads((hf/'model.safetensors.index.json').read_text())['weight_map']
    prefixes = [P+'embed_tokens.', 'language_model.lm_head.', P+'norm.', P+'output_attn_res_']
    prefixes += [f'{P}layers.{layer}.' for layer in range(args.layers)]
    shards = sorted({file for name, file in index.items() if any(name.startswith(p) for p in prefixes)})
    manifest = {
        'written_utc': datetime.now(timezone.utc).isoformat(), 'host': platform.node(),
        'source': {'dir': str(hf), 'pinned_revision': 'f831ab66814297da540d832a5235f8e904f29d06',
                   'config_sha256': sha256(hf/'config.json'), 'index_sha256': sha256(hf/'model.safetensors.index.json'),
                   'shards_read': [{'file': f, 'bytes': (hf/f).stat().st_size,
                                    **({'sha256': sha256(hf/f)} if args.hash_shards else {})} for f in shards]},
        'converter': {'file': 'megabench/tasks/kimi_checkpoint.py', 'sha256': sha256(repo/'megabench/tasks/kimi_checkpoint.py'),
                      'repo_commit': subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True,
                                                    text=True).stdout.strip(),
                      'command': f'python -s -m megabench.tasks.kimi_checkpoint --src {hf} --out {out} --tp {args.tp} --layers {args.layers}'},
        'tp': args.tp, 'layers': args.layers, 'outputs': []}
    total = 0
    for rank in range(args.tp):
        directory = out/f'tp{args.tp}'/f'rank{rank}'
        for path in sorted(directory.glob('*.safetensors')):
            size = path.stat().st_size; total += size
            manifest['outputs'].append({'file': str(path.relative_to(out)), 'bytes': size, 'sha256': sha256(path)})
    manifest['output_bytes_total'] = total
    done = out/f'tp{args.tp}'/'DONE.json'
    manifest['done_json'] = json.loads(done.read_text()) if done.exists() else None
    for key, value in (('log', args.log), ('time', args.time)):
        if value and Path(value).exists():
            manifest[key] = Path(value).read_text()[-6000:]
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
