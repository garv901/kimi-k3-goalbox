"""A3 verification fixture: instrument the seed for ./capture and ./task_ab (MoE routed w1/w3 phase).
NOT shipped to agents (it would be a hint); kept in docs/A3_fixture/ as the worked verification.
usage: python3 patch_seed.py WS_DIR"""
import sys, os
ws = sys.argv[1]
p = os.path.join(ws, 'mk_tail.cu')
s = open(p).read()
lines = s.split('\n')
# locate N phase
i0 = next(i for i, l in enumerate(lines) if '// ================= N: routing + routed w1/w3 + shared w2' in l)
assert lines[i0 + 1].strip() == 'const int K6 = b.topk;' and lines[i0 + 2].strip() == '{'
i_sh = next(i for i in range(i0, len(lines)) if '// shared expert w2 slice' in lines[i])
assert lines[i_sh - 1].strip() == 'tmark(c, __LINE__);'
body = lines[i0 + 1:i_sh]          # K6 ... tmark (inside the open scope)
func = ['// ---- A3 fixture: routing + routed w1/w3 as a function (task_ab engine)',
        '__device__ __forceinline__ void moe_w13(Ctx& c, Shared& sh, float* dsm, const BW& b, int l, bool draft) {',
        '  const KP& p = *c.pp;',
        '  const int nr = draft ? RDT : RT;',
        '#ifdef A3_STAMP',
        '  if (threadIdx.x == 0) a3_stamp[draft][l][blockIdx.x][0] = gtime();',
        '#endif'] + body[:-1] + [
        '#ifdef A3_STAMP',
        '    __syncthreads();',
        '    if (threadIdx.x == 0) a3_stamp[draft][l][blockIdx.x][1] = gtime();',
        '#endif',
        body[-1], '  }', '}', '']
# shared w2 needs s0, slen, G
call = ['  // ================= N: routing + routed w1/w3 + shared w2',
        '  SS_CAP("moe_in", draft ? 100 + l : l, SS_REG("scratch", c.scr, S_TOTAL, SS_U8),',
        '         SS_REG("comm_self", c.comm[c.rank], 4096 + 2 * HALF_FLOATS * 4 + 4096, SS_U8));',
        '  moe_w13(c, sh, dsm, b, l, draft);',
        '  SS_CAP("moe_w13_out", draft ? 100 + l : l, SS_REG("hr", SC(bf16, S_HR), (size_t)RT * 6 * INTER, SS_BF16),',
        '         SS_REG("ucnt", SC(unsigned, S_UCNT), 128, SS_U32),',
        '         SS_REG("expert_ids", draft ? OUT(int64_t, draft_expert_ids) + (size_t)l * RDT * 3',
        '                                    : OUT(int64_t, expert_ids) + (size_t)l * RT * 6, draft ? RDT * 3 : RT * 6, SS_I64));',
        '  {',
        '    const int sc0 = (36 * c.rank) / WORLD, sc1 = (36 * (c.rank + 1)) / WORLD;',
        '    const int s0 = sc0 * 64, slen = (sc1 - sc0) * 64;',
        '    const int G = gridDim.x;']
new = lines[:i0] + call + lines[i_sh:]
# insert function before run_block
j = next(i for i, l in enumerate(new) if l.startswith('__device__ void run_block('))
new = new[:j] + func + new[j:]
s = '\n'.join(new)
# header include + stamps at top of tail
s = ('#include "ss_task.cuh"\n#ifdef A3_STAMP\n__device__ unsigned long long a3_stamp[2][64][160][2];\n'
     'SS_EXPORT int a3_get_stamps(int dev, void* host) { cudaSetDevice(dev); cudaDeviceSynchronize();'
     ' return (int)cudaMemcpyFromSymbol(host, a3_stamp, sizeof(a3_stamp)); }\n#endif\n') + s
# params capture at kernel start
k = s.index('mk_kernel(const __grid_constant__ KP p) {')
k = s.index('\n', k) + 1
s = s[:k] + '  SS_CAP_PARAMS(&p, sizeof(KP));\n' + s[k:]
# task kernel before host section
h = s.index('// ----------------------------------------------------------------- host')
task = r'''#ifdef SS_TASK
__global__ void __launch_bounds__(NT, 1) task_moe_w13(const __grid_constant__ KP p) {
  extern __shared__ float dsm[];
  __shared__ Shared sh;
  Ctx c;
  c.pp = &p;
  c.scr = (uint8_t*)p.v[P_scratch];
  c.bar = (unsigned*)(c.scr + S_BAR);
  c.epoch = *(volatile unsigned long long*)(c.scr + 64);
  c.rank = (int)p.v[P_rank];
  for (int i = 0; i < WORLD; ++i) c.comm[i] = (uint8_t*)p.v[P_comm0 + i];
  c.lane = threadIdx.x & 31;
  c.warp = threadIdx.x >> 5;
  c.gw = blockIdx.x * WPB + c.warp;
  c.nw = gridDim.x * WPB;
  c.nmark = 8192;   // no tmark writes
  c.seq = 0;
  const int l = ss_task_layer();
  const bool draft = ss_task_aux() != 0;
  BW b = make_bw(p, draft, l, 0);
#ifdef A3_SLOW   // deliberately slowed (still correct) variant: run the phase twice, resetting its counters
  moe_w13(c, sh, dsm, b, l, draft);
  ss_grid_barrier();
  if (blockIdx.x == 0 && threadIdx.x < 128) SC(unsigned, S_UCNT)[threadIdx.x] = 0u;
  ss_grid_barrier();
#endif
#ifdef A3_DELAY  // deliberately slowed variant: every block idles A3_DELAY ns before the phase
  if (threadIdx.x == 0) { const unsigned long long t0 = gtime(); while (gtime() - t0 < (unsigned long long)(A3_DELAY)) {} }
  __syncthreads();
#endif
  moe_w13(c, sh, dsm, b, l, draft);
}
SS_TASK_REGISTER(moe_w13, task_moe_w13, KP, NT, 160 * 1024)
#endif

'''
s = s[:h] + task + s[h:]
open(p, 'w').write(s)
# ss_task.py hooks
open(os.path.join(ws, 'ss_task.py'), 'w').write('''"""A3 hooks: build the kernel with extra flags (keeps the seed default -DNO_PF)."""
import hashlib, os
HERE = os.path.dirname(os.path.abspath(__file__))


def build_ext(flags: str) -> str:
    hdr = hashlib.sha1(open(os.path.join(HERE, 'ss_task.cuh'), 'rb').read()).hexdigest()[:8]
    os.environ['MK_FLAGS'] = f'-DNO_PF {flags} -DSS_HDR_{hdr}'   # header hash: rebuild when ss_task.cuh changes
    import mk_host
    return mk_host.load_ext().__file__
''')
print('patched', p)
