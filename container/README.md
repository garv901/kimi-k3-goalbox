# Container: build and run

The image holds only tools (CUDA 13.0 devel + torch 2.14.1+cu130 base, git/tmux/curl/gh, pytest,
Nsight, Claude Code 2.1.326). The harness, benchmark and seed are fetched at run time into
`/srv/ss` by `scripts/bootstrap_node.sh`.

> Not built or tested by us: the package was assembled without Docker or GPUs. Our runs used the
> same base image digest with these tools installed by hand on a Kubernetes pod (root shell).

## Build
```sh
git clone -b scaffold-search-dsv41 https://github.com/garv901/kimi-k3-goalbox ss && cd ss
docker build -t scaffold-search:dsv41 -f container/Dockerfile container/
```

## Run (Docker)
```sh
docker run -d --name ss --gpus all --ipc=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /data/ss:/srv/ss \
  -e SS_SNAPSHOT_REPO=git@github.com:<you>/<repo>.git \
  scaffold-search:dsv41
docker exec -it ss bash          # everything below runs as root inside the container
```

## Run (Kubernetes)
```yaml
apiVersion: v1
kind: Pod
metadata: {name: scaffold-search}
spec:
  hostIPC: true                       # NCCL shared memory between the 8 ranks (or a large /dev/shm emptyDir)
  containers:
  - name: ss
    image: <registry>/scaffold-search:dsv41
    command: ["sleep", "infinity"]
    resources: {limits: {nvidia.com/gpu: 8}}
    env: [{name: SS_SNAPSHOT_REPO, value: "git@github.com:<you>/<repo>.git"}]
    volumeMounts: [{name: ss, mountPath: /srv/ss}, {name: shm, mountPath: /dev/shm}]
  volumes:
  - {name: ss, persistentVolumeClaim: {claimName: scaffold-search}}   # or hostPath / emptyDir
  - {name: shm, emptyDir: {medium: Memory, sizeLimit: 64Gi}}
```

## Requirements and capabilities
- **GPU access:** all 8 GPUs of the node in the container (`--gpus all` / `nvidia.com/gpu: 8`),
  NVLink between them (the kernel is TP8 with one cooperative launch per rank).
- **Root inside the container**, with the default capability set. Privileged mode is not needed.
  The harness, as root: creates Unix users `ss-<run_id>` and groups `ss-agents`, `ss-gpu`
  (`useradd/groupadd`), chowns and chmods `/dev/nvidia*` to `root:ss-gpu 0660` (CAP_CHOWN,
  CAP_FOWNER), switches uid/gid for run processes (CAP_SETUID/SETGID), and kills run processes
  (CAP_KILL). It does not need CAP_NET_ADMIN or CAP_SYS_ADMIN (the web block is a proxy, see the
  main README). Do not drop these default capabilities and do not run as a non-root user.
- **Hardware counters for run users** (A1's `./profile_candidate_all_ranks`, the stock
  `./profile_candidate`): ncu runs as the unprivileged run user. The host driver may need `NVreg_RestrictProfilingToAdminUsers=0`, otherwise those tools fail
  with ERR_NVGPUCTRPERM (scores are unaffected, the arm's tool is).
- **Network (egress):** GitHub (clone benchmark/seed, push snapshots), PyPI only at image build,
  `claude.ai` / `*.anthropic.com` / `*.claude.com` (Claude Code). Agents are restricted to the
  Anthropic/Claude hosts by the harness's proxy; root is not.
- **Disk:** ~200 GB under `/srv/ss` is plenty (each run keeps its workspace, snapshots and rescore
  trees, a few GB at most; A3 captures are ~0.4 GB each).
- **Volumes:** mount a persistent volume at `/srv/ss` if you want results to survive a container
  restart. Do **not** mount over `/root`: the pinned Claude Code binary lives in
  `/root/.local/share/claude/versions/`. Credentials live in `/root/.claude/` (log in again after
  a restart, or also mount a volume at `/root/.claude`).
- **Processes:** the gpu_lease daemon and the proxy run in tmux sessions `ss-lease` / `ss-proxy`
  inside the container; `scripts/bootstrap_node.sh` restarts them after a container restart.

## Environment variables
| var | used by | meaning |
|---|---|---|
| `SS_SNAPSHOT_REPO` | bootstrap, snapshot.py, rescore.py | git URL for run branches `run/<id>-<arm>` and branch `results`; unset = no pushes |
| `SS_BENCHMARK_URL` | fetch_benchmark.sh | clone URL of `kamahori/vibe-megakernel` (default https) |
| `SS_ALLOW_MISMATCH=1` | bootstrap | continue despite a PROTOCOL.md manifest mismatch (runs invalid) |
| `SS_SKIP_GPU_CHECK=1` | bootstrap | skip the 8x B300 check |
| `GPU_LEASE_*` | gpu_lease.py | lease policy knobs; leave unset (defaults = pre-registered values) |
