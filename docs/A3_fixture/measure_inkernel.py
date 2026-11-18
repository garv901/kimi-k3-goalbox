"""A3 fixture: in-kernel window of the routed w1/w3 phase per (layer, rank) from -DA3_STAMP stamps.
torchrun --nproc_per_node 8 measure_inkernel.py OUT.json [launches]"""
import ctypes, json, os, sys
import torch, torch.distributed as dist
HERE = os.getcwd(); sys.path.insert(0, HERE); sys.path.insert(1, os.path.join(HERE, 'reference'))
from megabench.cases import CASE
from megabench.tasks import dsv41
import ss_task

rank = int(os.environ['RANK']); torch.cuda.set_device(rank)
dist.init_process_group('nccl', device_id=torch.device(f'cuda:{rank}'))
g = dist.new_group(list(range(8)))
so = ss_task.build_ext('-DA3_STAMP') if rank == 0 else None
dist.barrier(group=g)
so = ss_task.build_ext('-DA3_STAMP')
lib = ctypes.CDLL(so)
case = CASE.to_dict()
case['execution'] = dict(rank=rank, world_size=8, tp_rank=rank, ep_rank=0, device=f'cuda:{rank}', backend='nccl', tp_group=g, ep_group=None)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
with torch.inference_mode():
    V = dsv41.make_inputs(CASE, 1234, f'cuda:{rank}', rank=rank)
    dsv41.set_trial_scenario(CASE, V, 0)
    import submission
    run = submission.build(case)
    res = []
    for it in range(n):
        dist.barrier(group=g); torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record(); run(V); en.record(); torch.cuda.synchronize()
        buf = (ctypes.c_uint64 * (2 * 64 * 160 * 2))()
        assert lib.a3_get_stamps(rank, buf) == 0
        nb = torch.cuda.get_device_properties(rank).multi_processor_count
        t = torch.tensor(list(buf), dtype=torch.float64).view(2, 64, 160, 2)[:, :, :nb]
        win = ((t[0, :40, :, 1].max(1).values - t[0, :40, :, 0].min(1).values) / 1000).tolist()
        busy = ((t[0, :40, :, 1] - t[0, :40, :, 0]).max(1).values / 1000).tolist()
        res.append(dict(launch_ms=st.elapsed_time(en), window_us=win, busy_max_us=busy))
out = [None] * 8
dist.all_gather_object(out, res, group=g)
if rank == 0:
    json.dump(out, open(sys.argv[1], 'w'))
    for L in (1, 8):
        print('layer', L, 'window us per rank (median over launches):',
              [round(sorted(r[i]['window_us'][L] for i in range(1, n))[(n - 1) // 2], 1) for r in out])
    print('launch ms rank0:', [round(x['launch_ms'], 2) for x in out[0]])
dist.destroy_process_group()
