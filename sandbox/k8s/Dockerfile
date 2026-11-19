# Goalbox image for the Kimi K3 TP8 run on GB300 (aarch64 only). One image for the agent pod,
# the proxy pod, the controller pod and the two GPU worker pods; the pod spec decides the role.
#
# Build on a Grace host (no cluster privileges needed):  sandbox/k8s/build-image.sh
# Contents: CUDA 13.4 toolkit (nvcc, ptxas, cuobjdump), Nsight Systems and Nsight Compute,
# a clean Python venv at /opt/goalbox/venv (torch cu130, numpy, safetensors, huggingface_hub,
# ninja), Claude Code pinned to CLAUDE_CODE_VERSION, and two unprivileged users:
#   agent   (uid 1000)  runs Claude Code, GPU jobs and the evaluator ranks
#   capture (uid 2000)  runs the trajectory sidecar; the agent cannot read its files
# The repository itself is not baked in: the controller copies megabench/ and sandbox/ onto the
# read-only benchmark volume, so an evaluator change does not need a new image.
FROM nvidia/cuda:13.4.1-devel-ubuntu24.04

ARG CLAUDE_CODE_VERSION=2.1.322
ARG TORCH_VERSION=2.14.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG NSYS_PKG=nsight-systems-2026.3.2
ARG NCU_PKG=nsight-compute-2026.3.0
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential ninja-build \
        git rsync jq curl ca-certificates procps sqlite3 tini less file \
        ${NSYS_PKG} ${NCU_PKG} \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "$(ls -d /opt/nvidia/nsight-systems/*/bin/nsys | head -1)" /usr/local/bin/nsys \
    && ln -sf "$(ls -d /opt/nvidia/nsight-compute/*/ncu | head -1)" /usr/local/bin/ncu \
    && nsys --version && ncu --version | tail -1 && nvcc --version | tail -1

# The Ubuntu base ships a user 'ubuntu' with uid 1000; the agent takes that uid instead.
RUN (userdel -r ubuntu 2>/dev/null || true) && (groupdel ubuntu 2>/dev/null || true) \
    && groupadd -g 1000 agent && useradd -m -u 1000 -g 1000 -s /bin/bash agent \
    && groupadd -g 2000 capture && useradd -m -u 2000 -g 2000 -s /bin/bash capture

# Clean read-only Python environment (the agent layers its own venv on top of it).
RUN python3 -m venv /opt/goalbox/venv \
    && /opt/goalbox/venv/bin/pip install -q --upgrade pip \
    && /opt/goalbox/venv/bin/pip install -q "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}" \
    && /opt/goalbox/venv/bin/pip install -q numpy safetensors "huggingface_hub[hf_xet]" ninja \
    && /opt/goalbox/venv/bin/python -c 'import torch, safetensors; print("torch", torch.__version__, "cuda", torch.version.cuda)' \
    && find /opt/goalbox -name __pycache__ -prune -exec rm -rf {} + \
    && chmod -R a-w /opt/goalbox

# Claude Code, pinned. The native installer puts the binary under $HOME/.local; keep it in /opt.
RUN curl -fsSL https://claude.ai/install.sh -o /tmp/install-claude.sh \
    && HOME=/opt/claude bash /tmp/install-claude.sh "${CLAUDE_CODE_VERSION}" \
    && rm /tmp/install-claude.sh \
    && ln -sf "$(readlink -f /opt/claude/.local/bin/claude)" /usr/local/bin/claude \
    && chmod -R a+rX /opt/claude \
    && claude --version | grep -F "${CLAUDE_CODE_VERSION}"

ENV PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    CUDA_HOME=/usr/local/cuda \
    DISABLE_AUTOUPDATER=1
LABEL org.goalbox.claude-code="${CLAUDE_CODE_VERSION}" org.goalbox.torch="${TORCH_VERSION}"
# The benchmark harness, sandbox scripts, bring-up tools and goal text, read-only, pinned by the image
# digest (no ConfigMaps to drift). Candidate code never lives here: it is on the work volume.
COPY --chown=0:0 megabench /goalbox/bench/megabench
COPY --chown=0:0 planning/bringup /goalbox/bench/planning/bringup
COPY --chown=0:0 sandbox/lib /goalbox/sandbox/lib
COPY --chown=0:0 sandbox/k8s/agent-entry.sh sandbox/k8s/controller.sh /goalbox/sandbox/k8s/
COPY --chown=0:0 sandbox/agent-tools /goalbox/sandbox/agent-tools
COPY --chown=0:0 sandbox/sandbox-bin /goalbox/bin
COPY --chown=0:0 sandbox/blocked-packages.txt /goalbox/sandbox/blocked-packages.txt
COPY --chown=0:0 goals /goalbox/goal
RUN chmod -R a-w,a+rX /goalbox/bench /goalbox/sandbox /goalbox/bin /goalbox/goal \
    && find /goalbox/bench -name __pycache__ -prune -exec rm -rf {} + \
    && /opt/goalbox/venv/bin/python -m compileall -q /goalbox/bench >/dev/null || true

USER 1000:1000
WORKDIR /home/agent
