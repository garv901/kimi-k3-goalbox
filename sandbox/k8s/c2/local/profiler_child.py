#!/usr/bin/env python3
"""Traced target for the CPU-only profiler experiments (no GPU needed).

E1 clock mapping: emits NVTX marks and records, for each, the three Linux clocks read just before the mark,
so the trace's timestamps can be related to clocks another process (the grader) can read.
E2 injection escape: forks a child with an emptied environment that emits its own marks.
Writes a JSON record of the clocks to the path in argv[1]; argv[2] == "child" marks the escaped child.
"""
import ctypes, ctypes.util, json, os, subprocess, sys, time

def nvtx_mark():
    for name in ("libnvToolsExt.so.1", "libnvToolsExt.so"):
        path = ctypes.util.find_library(name.replace("lib", "", 1).split(".so")[0]) or name
        try:
            lib = ctypes.CDLL(path)
            lib.nvtxMarkA.argtypes = [ctypes.c_char_p]
            return lambda s: lib.nvtxMarkA(s.encode())
        except OSError:
            continue
    import torch   # falls back to torch's NVTX binding (no GPU needed for marks)
    return lambda s: torch.cuda.nvtx.mark(s)

def clocks():
    return {"monotonic_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
            "monotonic_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC),
            "realtime_ns": time.clock_gettime_ns(time.CLOCK_REALTIME),
            "boottime_ns": time.clock_gettime_ns(time.CLOCK_BOOTTIME)}

mark = nvtx_mark()
role = sys.argv[2] if len(sys.argv) > 2 else "parent"
record = {"role": role, "pid": os.getpid(), "uid": os.getuid(), "ld_preload": os.environ.get("LD_PRELOAD", ""), "marks": []}
for i in range(3):
    c = clocks()
    mark(f"exp:{role}:mark{i}")
    record["marks"].append({"label": f"exp:{role}:mark{i}", **c})
    time.sleep(0.2)
if role == "parent":
    # E2: a child started with an emptied environment (no injection variables)
    out = sys.argv[1] + ".child"
    subprocess.run(["env", "-i", "PATH=/usr/bin:/bin", sys.executable, __file__, out, "child"], check=False)
    record["child_record"] = json.load(open(out)) if os.path.exists(out) else None
json.dump(record, open(sys.argv[1], "w"), indent=1)
print(json.dumps({"role": role, "pid": record["pid"], "uid": record["uid"], "ld_preload_set": bool(record["ld_preload"])}))
