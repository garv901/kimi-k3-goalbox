# Container for the scaffold-search runs on one 8x B300 node.
# Base = the pinned image the harness was built and tested on (torch 2.14.1+cu130, CUDA 13.0 devel, python3.12 at
# /usr/bin/python3.12, Nsight Compute under /opt/nvidia/nsight-compute/2025.3.1, no curl/git/tmux).
#   docker build -t scaffold-search:dsv41 -f container/Dockerfile container/
# NOT built or tested on our side (no Docker/GPU where this package was made); every RUN step
# checks what the harness needs and fails loudly.
FROM pytorch/pytorch@sha256:4e76657a91117d4373f0a4ba514fb55cbac4c846539aae8f36ed6800254c9a46

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    DISABLE_AUTOUPDATER=1 \
    CUDA_HOME=/usr/local/cuda

# OS tools: git/tmux/curl/rsync/pciutils, build tools for nvcc host code, `patch` (trusted.py),
# passwd/procps (useradd, pkill), xz (vendored seed), gnupg (gh apt key)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git tmux curl ca-certificates rsync pciutils build-essential patch xz-utils procps psmisc \
      passwd gnupg less vim-tiny jq \
    && rm -rf /var/lib/apt/lists/*

# GitHub CLI (snapshot pushes + private clones via `gh auth setup-git`)
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh && rm -rf /var/lib/apt/lists/* \
    && gh --version | head -1

# Python: the harness and every workspace tool call /usr/bin/python3.12 (stdlib + torch); tests need pytest
RUN set -e; \
    if [ ! -x /usr/bin/python3.12 ]; then p=$(command -v python3.12); ln -s "$p" /usr/bin/python3.12; fi; \
    /usr/bin/python3.12 -c "import torch, tomllib; print('torch', torch.__version__, 'cuda', torch.version.cuda)"; \
    /usr/bin/python3.12 -m pip install --no-cache-dir pytest numpy \
      || /usr/bin/python3.12 -m pip install --no-cache-dir --break-system-packages pytest numpy; \
    /usr/bin/python3.12 -c "import pytest, numpy"

# Nsight Compute / Systems: expected in the base image; install from the CUDA apt repo if missing
RUN set -e; \
    if ! ls /opt/nvidia/nsight-compute/*/ncu >/dev/null 2>&1 && ! command -v ncu >/dev/null; then \
      apt-get update && apt-get install -y --no-install-recommends cuda-nsight-compute-13-0 cuda-nsight-systems-13-0 \
      && rm -rf /var/lib/apt/lists/*; \
    fi; \
    (ls -d /opt/nvidia/nsight-compute/* 2>/dev/null; command -v ncu; command -v nvcc; nvcc --version | tail -1) || true

# Claude Code, pinned to 2.1.326 (native installer -> /root/.local/share/claude/versions/2.1.326,
# which isolation.install_claude() copies to /srv/ss/bin for the run users). Alternative if the
# installer is unavailable: install Node 20+, then `npm i -g @anthropic-ai/claude-code@2.1.326`
# (the harness then links the npm binary instead).
RUN curl -fsSL https://claude.ai/install.sh | bash -s 2.1.326 \
    && test -x /root/.local/share/claude/versions/2.1.326 \
    && /root/.local/bin/claude --version | grep -q '^2\.1\.326'

WORKDIR /srv/ss
CMD ["sleep", "infinity"]
