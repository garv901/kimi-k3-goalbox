#!/bin/sh
# passes needed per metric set on this GPU (tiny torch kernel, kernel replay, 1 GPU)
N=/opt/nvidia/nsight-compute/2025.3.1/ncu
P="import torch; x=torch.ones(1<<20,device='cuda'); torch.cuda.synchronize(); y=(x*2).sum(); torch.cuda.synchronize()"
for set in \
 "dram__bytes_read.sum,dram__bytes_write.sum" \
 "dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct,gpu__time_duration.sum" \
 "smsp__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_lsu.sum" \
 "sm__throughput.avg.pct_of_peak_sustained_elapsed" \
 "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tc.sum" \
 "dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct,gpu__time_duration.sum,smsp__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_lsu.sum" \
 "group:smsp__pcsamp_warp_stall_reasons" ; do
  echo "== $set"; $N --metrics "$set" -k regex:reduce /usr/bin/python3.12 -c "$P" 2>&1 | grep -i "pass\|error" | head -3
done
