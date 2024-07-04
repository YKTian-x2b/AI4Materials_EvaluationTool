 #!/bin/bash

nsys profile --stats=true -f true -o res/PaddleRes/paddle_gcn_profile_nsys \
python GCN_paddle/main.py root_dir > res/PaddleRes/paddle_gcn_profile_nsys.txt

nsys profile --stats=true -f true -o res/TorchRes/torch_gcn_profile_nsys \
python GCN_torch/predict.py GCN_torch/model_best.pth.tar root_dir > res/TorchRes/torch_gcn_profile_nsys.txt
