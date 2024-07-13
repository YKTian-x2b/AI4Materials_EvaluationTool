 #!/bin/bash

#================ nsys
nsys profile --stats=true -f true -o res/TorchRes/torch_gcn_profile_nsys \
python GCN_torch/predict.py GCN_torch/model_best.pth.tar root_dir > res/TorchRes/torch_gcn_profile_nsys.txt

nsys profile --stats=true -f true -o res/PaddleRes/paddle_gcn_profile_nsys \
python GCN_paddle/main.py root_dir > res/PaddleRes/paddle_gcn_profile_nsys.txt




#================ ncu show
sudo su

/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "convLayer" \
--metrics gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
launch__grid_size,\
launch__block_size,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
launch__shared_mem_per_block_dynamic,\
sm__warps_active.avg.pct_of_peak_sustained_active \
--csv python GCN_torch/predict.py GCN_torch/model_best.pth.tar root_dir \
> res/TorchRes/ncu_metrics_show_torch.txt


/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "convLayer" \
--metrics gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
launch__grid_size,\
launch__block_size,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
launch__shared_mem_per_block_dynamic,\
sm__warps_active.avg.pct_of_peak_sustained_active \
--csv python GCN_paddle/main.py root_dir \
> res/PaddleRes/ncu_metrics_show_paddle.txt


#================ ncu profile
/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "convLayer" \
--metrics gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
launch__grid_size,\
launch__block_size,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
launch__shared_mem_per_block_dynamic,\
sm__warps_active.avg.pct_of_peak_sustained_active \
python GCN_torch/predict.py GCN_torch/model_best.pth.tar root_dir \
> res/TorchRes/ncu_metrics_prof_torch.txt

/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "convLayer" \
--metrics gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
launch__grid_size,\
launch__block_size,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
launch__shared_mem_per_block_dynamic,\
sm__warps_active.avg.pct_of_peak_sustained_active \
python GCN_paddle/main.py root_dir \
> res/PaddleRes/ncu_metrics_prof_paddle.txt