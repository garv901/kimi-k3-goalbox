"""A3 hooks of the toy candidate (self-test). A real candidate's ss_task.py usually only needs
build_ext(); case/make_inputs/build_runner default to the workspace's benchmark case."""
import ctypes
import hashlib
import os
import subprocess
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def build_ext(flags: str) -> str:
    """Build the kernel with extra nvcc flags; return the path of the shared library."""
    src = open(os.path.join(HERE, 'toy.cu')).read() + open(os.path.join(HERE, 'ss_task.cuh')).read()
    so = os.path.join(HERE, 'build', 'toy_' + hashlib.sha1((flags + src).encode()).hexdigest()[:12] + '.so')
    if not os.path.exists(so):
        os.makedirs(os.path.dirname(so), exist_ok=True)
        tmp = f'{so}.tmp{os.getpid()}'
        subprocess.run(['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-gencode=arch=compute_103a,code=sm_103a',
                        '-shared', '-Xcompiler', '-fPIC', f'-I{HERE}', *flags.split(), os.path.join(HERE, 'toy.cu'),
                        '-o', tmp], check=True)
        os.replace(tmp, so)
    return so


def case():
    return types.SimpleNamespace(id='a3-toy', atol=0.003, rtol=0.003, bf16_rtol=0.008, noise_factor=2.0)


def make_inputs(case, seed, trial, device, rank):
    g = torch.Generator(device=device).manual_seed(seed + 7 * trial)
    return {'x': (torch.randn(8, 1024, generator=g, device=device)).to(torch.bfloat16),
            'W': (torch.randn(4, 4096, 1024, generator=g, device=device) / 32).to(torch.bfloat16)}


def build_runner(case, rank, world, group, device):
    assert _LAST[0], 'build_ext() must run first (./capture does)'
    return _Runner(device)


class _Runner:
    def __init__(self, device):
        self.device = device
        self.y = torch.zeros(8, 4096, dtype=torch.float32, device=device)
        self.bar = torch.zeros(2, dtype=torch.int32, device=device)
        self.nsm = torch.cuda.get_device_properties(device).multi_processor_count

    def __call__(self, inputs):
        lib = ctypes.CDLL(_LAST[0])
        x = inputs['x'].clone()
        p = (ctypes.c_uint64 * 4)(x.data_ptr(), inputs['W'].data_ptr(), self.y.data_ptr(), self.bar.data_ptr())
        r = lib.toy_launch(p, self.nsm, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        assert r == 0, r
        return {'x_out': x}


_LAST = [None]
_build = build_ext


def build_ext(flags: str) -> str:  # noqa: F811  (remember the last build for the runner)
    _LAST[0] = _build(flags)
    return _LAST[0]
