#================ ncu
sudo /usr/local/NVIDIA-Nsight-Compute/ncu --devices 0 --set basic --list-sections > ../all_sections.txt
sudo /usr/local/NVIDIA-Nsight-Compute/ncu --devices 0 --query-metrics-mode all --query-metrics > ../all_metrics.txt
sudo /usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --devices 0 --set basic --list-sections > ../ncu_2020_sections.txt
sudo /usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --devices 0 --query-metrics-mode all --query-metrics  > ../ncu_2020_metrics.txt


sudo su

# without csv for profile
/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "linear_nvtx" \
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
python demo3.py > metrics_713_paddle.txt

# with --csv for show
/usr/local/cuda-11.1/nsight-compute-2020.2.0/ncu --nvtx --nvtx-include "linear_nvtx" \
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
--csv python demo3.py > metrics_713_paddle_.csv