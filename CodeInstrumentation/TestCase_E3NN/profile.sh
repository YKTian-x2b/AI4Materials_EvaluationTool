 #!/bin/bash

#================ nsys
#nsys profile --stats=true -f true -o res/TorchRes/torch_gcn_profile_nsys \
#python GCN_torch/predict.py GCN_torch/model_best.pth.tar root_dir > res/TorchRes/torch_gcn_profile_nsys.txt
#
#nsys profile --stats=true -f true -o res/MindsporeRes/mindspore_gcn_profile_nsys \
#python e3nn_mindspore/main.py root_dir > res/MindsporeRes/mindspore_gcn_profile_nsys.txt




#================ ncu show
sudo su

#/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "convLayer" \
#--metrics gpu__time_duration.sum,\
#sm__throughput.avg.pct_of_peak_sustained_elapsed,\
#gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
#gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
#launch__grid_size,\
#launch__block_size,\
#launch__registers_per_thread,\
#launch__shared_mem_per_block_static,\
#launch__shared_mem_per_block_dynamic,\
#sm__warps_active.avg.pct_of_peak_sustained_active \
#--csv python GCN_torch/predict.py GCN_torch/model_best.pth.tar root_dir \
#> res/TorchRes/ncu_metrics_show_torch.txt
#
#
#/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "convLayer" \
#--metrics gpu__time_duration.sum,\
#sm__throughput.avg.pct_of_peak_sustained_elapsed,\
#gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
#gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
#launch__grid_size,\
#launch__block_size,\
#launch__registers_per_thread,\
#launch__shared_mem_per_block_static,\
#launch__shared_mem_per_block_dynamic,\
#sm__warps_active.avg.pct_of_peak_sustained_active \
#--csv python e3nn_mindspore/main.py root_dir \
#> res/MindsporeRes/ncu_metrics_show_mindspore.txt


#================ ncu profile
/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "TensorProduct" \
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
python e3nn_torch/inference.py \
> res/TorchRes/ncu_metrics_prof_torch_TensorProduct.txt

/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "TensorProduct" \
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
python e3nn_mindspore/inference.py \
> res/MindsporeRes/ncu_metrics_prof_mindspore_TensorProduct.txt



/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "FullyConnectedNet" \
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
python e3nn_torch/inference.py \
> res/TorchRes/ncu_metrics_prof_torch_FullyConnectedNet.txt

/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "FullyConnectedNet" \
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
python e3nn_mindspore/inference.py \
> res/MindsporeRes/ncu_metrics_prof_mindspore_FullyConnectedNet.txt


/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "Gate_nvtx" \
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
python e3nn_torch/inference.py \
> res/TorchRes/ncu_metrics_prof_torch_Gate.txt

/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "Gate_nvtx" \
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
python e3nn_mindspore/inference.py \
> res/MindsporeRes/ncu_metrics_prof_mindspore_Gate.txt



/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "Scatter" \
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
python e3nn_torch/inference.py \
> res/TorchRes/ncu_metrics_prof_torch_Scatter.txt

/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "Scatter" \
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
python e3nn_mindspore/inference.py \
> res/MindsporeRes/ncu_metrics_prof_mindspore_Scatter.txt