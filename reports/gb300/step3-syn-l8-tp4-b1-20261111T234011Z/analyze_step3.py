"""Analyze a distributed evaluator record: per-rank verdicts, launch accounting split, collective sequence agreement."""
import json, sys, collections, re, glob, os
path = sys.argv[1]
r = json.loads(open(path).read().strip().splitlines()[-1])
print("status:", r.get("status"), "| reason:", r.get("reason"))
print("session:", r.get("session"), "| submission_sha256:", r.get("submission_sha256"))
print("hardware:", r.get("hardware")); print("topology:", r.get("topology"))
print("build_ms:", round(r.get("build_ms", 0), 1), "first_call_ms:", round(r.get("first_call_ms_including_jit", 0), 1))
print("aggregate correctness:", r.get("correctness", {}).get("status"), "| aggregate launch_audit:", {k: v for k, v in r.get("launch_audit", {}).items() if k != "per_rank"})
seqs = {}
def classify(name):
    n = name.lower()
    if n.startswith("nccl:"): return "nccl_profiler_record"
    if "ncclDevKernel" in name or "ncclKernel" in name: return "nccl_device_kernel"
    if "memcpy" in n: return "memcpy"
    if "memset" in n: return "memset"
    return "compute_kernel"
for rk in r.get("ranks", []):
    print(f"--- rank {rk['rank']} device {rk['device']} status {rk['status']} reason {rk.get('reason')}")
    hw = rk.get("hardware", {}); print("    gpu:", hw.get("gpu_name"), "peak_allocated_GB:", round(hw.get("peak_allocated_bytes", 0)/1e9, 2))
    c = rk.get("correctness", {}); print("    correctness:", c.get("status"), "trials:", len(c.get("trials", [])))
    worst = 0.0
    for i, t in enumerate(c.get("trials", [])):
        o = t["outputs"] or {}
        ratios = {k: round(v["worst_ratio"], 3) for k, v in o.items() if isinstance(v, dict) and "worst_ratio" in v}
        worst = max([worst] + list(ratios.values()))
        print(f"      trial {i} seed {t['seed']}: ratios {ratios}; next_token {o.get('next_token')}; expert_ids {o.get('expert_ids')}; cache_length {o.get('cache_length')}")
    print("    worst band ratio over trials and outputs:", round(worst, 3))
    la = rk.get("launch_audit", {}); names = la.get("gpu_kernel_names", [])
    cls = collections.Counter(classify(n) for n in names)
    print("    launch_audit:", la.get("status"), "| total profiler device events:", la.get("gpu_kernel_count"), "| budget:", la.get("max_gpu_launches"))
    print("    accounting:", dict(cls), "| actual device kernel launches (compute + nccl device):", cls["compute_kernel"] + cls["nccl_device_kernel"])
    print("    nccl device kernels by name:", collections.Counter(n.split("(")[0] for n in names if classify(n) == "nccl_device_kernel").most_common(6))
    seqs[rk["rank"]] = [n.split("(")[0] for n in names if classify(n) == "nccl_device_kernel"]
    for k in ("candidate_timing", "reference_timing"):
        t = rk.get(k)
        if t: print(f"    {k}: cuda-event p50 {t['cuda_event_p50_ms']:.3f} ms p95 {t['cuda_event_p95_ms']:.3f}; reps {[round(x,1) for x in t['cuda_event_ms']]}")
    if rk.get("traceback"): print("    traceback:", rk["traceback"][-800:])
ranks = sorted(seqs)
if ranks:
    same = all(seqs[k] == seqs[ranks[0]] for k in ranks)
    print(f"--- audit-step NCCL device-kernel sequence identical across ranks {ranks}: {same} (length {len(seqs[ranks[0]])})")
print("--- aggregate (slowest rank per rep)")
for k in ("candidate_timing", "reference_timing"):
    t = r.get(k)
    if t: print(f"  {k}: cuda-event p50 {t['cuda_event_p50_ms']:.3f} ms p95 {t['cuda_event_p95_ms']:.3f} host p50 {t['host_p50_ms']:.3f}")
print("  graph_baseline_unavailable:", r.get("graph_baseline_unavailable"))
for k in ("speedup_vs_eager_host", "speedup_vs_eager_cuda_event", "speedup_vs_best_baseline_cuda_event"): print(" ", k, r.get(k))
# NCCL COLL trace comparison (NCCL_DEBUG_SUBSYS=COLL): one file per rank process
logs = sorted(glob.glob(os.path.join(os.path.dirname(path), "nccl-*.log.txt")))
if logs:
    print("--- NCCL collective trace per process")
    pat = re.compile(r"\[(\d+)\] NCCL INFO (\w+): opCount (\w+) sendbuff \S+ recvbuff \S+ count (\d+) datatype (\d+) op (\d+) root (\d+) comm (\S+)")
    traces = {}
    for f in logs:
        txt = open(f, errors="replace").read()
        warn = txt.count("NCCL WARN"); err = sum(1 for l in txt.splitlines() if "error" in l.lower() and "NCCL WARN" not in l)
        ops = [(m.group(2), m.group(4), m.group(5), m.group(6)) for m in pat.finditer(txt)]
        dev = sorted({m.group(1) for m in pat.finditer(txt)})
        kinds = collections.Counter(o[0] for o in ops)
        traces[f] = ops
        print(f"  {os.path.basename(f)}: cudaDev {dev} collectives {len(ops)} {dict(kinds)} | WARN {warn} | other 'error' lines {err} | Init COMPLETE lines {txt.count('Init COMPLETE')}")
    base = traces[logs[0]]
    print("  collective sequence (kind, count, datatype, op) identical across all processes:", all(t == base for t in traces.values()), "| length", len(base))
    if base:
        print("  first 6:", base[:6]); print("  last 3:", base[-3:])
