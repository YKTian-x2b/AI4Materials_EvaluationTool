set -e

CUDA_VISIBLE_DEVICES=4,5,6,7 mpirun --allow-run-as-root -n 4 --output-filename parallel_log_output --merge-stderr-to-stdout python train.py --mode GRAPH \
 --device_target GPU --parallel_mode DATA_PARALLEL


# python train.py --mode PYNATIVE --device_target GPU

# CUDA_VISIBLE_DEVICES=4 python train.py --mode GRAPH --device_target GPU