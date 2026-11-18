# Overlay S: CUDA skills from mlsys-contest-syfi-fully-agent

The 9 skills in `.claude/skills/` are copied unchanged from
https://github.com/kamahori/mlsys-contest-syfi-fully-agent at commit
a4addd96dbfe3d9bf484bf65fbb99aad80751e95, path `full-agent-pipeline/.claude/skills/`:
cuda-toolkit, cuda-debugging, cutlass-triton, warp-primitives, parallel-patterns,
gpu-memory-analysis, nsight-profiler, gpu-benchmarking, cublas-cudnn (the set that repo's driver
gives its CUDA phase). The repository has no LICENSE file; it is owned by the experiment's
requester (kamahori), who asked for these skills to be used. The off-topic skills (Vulkan, OpenCL,
HIP, NVENC, TensorRT, stencil, unified memory, NCCL, CUDA graphs) and the ncu-mcp server are not
included. Some SKILL.md files mention MCP servers; none are configured in the runs.

## sha256
```
c6e8232cbee559a7897e505915e5b7531f959c1d76dff0c7872348ffc69b2fbc  ./cublas-cudnn/SKILL.md
a84f44bddb7d7c5c08a56a67d9a44b2e7ce98da5fca6d0c65ead8adce58784a0  ./cuda-debugging/SKILL.md
058c0f0261e13c684496c0673b123776d0a3a3fc12909ae3da9f226c8e784335  ./cuda-toolkit/SKILL.md
06c91bebe6e4738d20e1cc88843c96a97287b3dcb98e3e1f3e4b659a3f0169b4  ./cutlass-triton/SKILL.md
be2fe8bf9ecbbefac89e874e91829e5949fc9623f5aa28c8ba0a5f72336e5360  ./gpu-benchmarking/SKILL.md
63bcd630f362e762811341951c2cf2c066dbac0d8d4fed9bc58ac91588e3322a  ./gpu-memory-analysis/SKILL.md
86bebc6de97eb8aa9d0789011c5bad289278099a75b710a549080f228114aa04  ./nsight-profiler/SKILL.md
519adc294626fcbeb2c2a214be6b3e47e70ad0664cffaa5a1a6facc79e2f7566  ./parallel-patterns/SKILL.md
3b3821ecc97a2b9e346e6b39271dad977515379346a4d75b46185c8e2d38130f  ./warp-primitives/SKILL.md
```
