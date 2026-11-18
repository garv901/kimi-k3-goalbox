# Container for the scaffold-search runs on one 4x GB300 tray (protocol dsv41-tp4-rack10-v1, linux/arm64).
# Base = NVIDIA CUDA 13.0.1 devel on Ubuntu 24.04, pinned to its linux/arm64 manifest digest
# (nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04, `docker manifest inspect` on 2026-11-18). Ubuntu 24.04's
# system Python is 3.12 (/usr/bin/python3.12); torch 2.14.1+cu130 comes from the PyTorch cu130 index
# (aarch64 wheel), Nsight Compute/Systems from the CUDA apt repo already configured in the base image.
#   docker build -t scaffold-search:dsv41-tp4 -f container/Dockerfile container/
# Every RUN step checks what the harness needs and fails loudly. Built on an arm64 host (or with
# `docker buildx build --platform linux/arm64`).
FROM nvcr.io/nvidia/cuda@sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    DISABLE_AUTOUPDATER=1 \
    CUDA_HOME=/usr/local/cuda \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN [ "$(uname -m)" = aarch64 ] || { echo "this image is for linux/arm64 (got $(uname -m))"; exit 1; }

# OS tools: git/tmux/curl/rsync/pciutils, build tools for nvcc host code, `patch` (trusted.py),
# passwd/procps (useradd, pkill), xz (vendored seed), gnupg (gh apt key); Python 3.12 + headers
# (nvcc builds the torch extension against them) + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
      git tmux curl ca-certificates rsync pciutils build-essential patch xz-utils procps psmisc \
      passwd gnupg less vim-tiny jq \
      python3.12 python3.12-dev python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && test -x /usr/bin/python3.12 && /usr/bin/python3.12 -c "import sys, tomllib; assert sys.version_info[:2] == (3, 12), sys.version"

# GitHub CLI (snapshot pushes + private clones via `gh auth setup-git`); the apt repo ships arm64
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh && rm -rf /var/lib/apt/lists/* \
    && gh --version | head -1

# PyTorch 2.14.1+cu130 (aarch64 wheel) into the system site-packages (/usr/local/lib/python3.12/dist-packages,
# the path the seed's microbenchmark build script and mk_host.load_ext() expect), pytest + numpy for the tests
RUN /usr/bin/python3.12 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1 \
    && /usr/bin/python3.12 -m pip install --no-cache-dir pytest numpy \
    && /usr/bin/python3.12 -c "import torch, pytest, numpy; assert torch.__version__.startswith('2.14.1'), torch.__version__; \
assert torch.version.cuda.startswith('13.0'), torch.version.cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
    && test -d /usr/local/lib/python3.12/dist-packages/torch/include

# Nsight Compute / Systems from the CUDA apt repo (sbsa) configured in the base image
RUN set -e; \
    if ! ls /opt/nvidia/nsight-compute/*/ncu >/dev/null 2>&1 && ! command -v ncu >/dev/null; then \
      apt-get update && apt-get install -y --no-install-recommends cuda-nsight-compute-13-0 cuda-nsight-systems-13-0 \
      && rm -rf /var/lib/apt/lists/*; \
    fi; \
    (ls -d /opt/nvidia/nsight-compute/* 2>/dev/null; command -v ncu || ls /usr/local/cuda/bin/ncu; command -v nvcc; nvcc --version | tail -1)

# Claude Code, pinned to 2.1.326 (native installer, arm64 build -> /root/.local/share/claude/versions/2.1.326,
# which isolation.install_claude() copies to /srv/ss/bin for the run users). Alternative if the
# installer is unavailable: install Node 20+, then `npm i -g @anthropic-ai/claude-code@2.1.326`
# (the harness then links the npm binary instead).
RUN curl -fsSL https://claude.ai/install.sh | bash -s 2.1.326 \
    && test -x /root/.local/share/claude/versions/2.1.326 \
    && /root/.local/bin/claude --version | grep -q '^2\.1\.326'

WORKDIR /srv/ss
CMD ["sleep", "infinity"]
