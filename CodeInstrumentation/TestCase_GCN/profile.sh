 #!/bin/bash

nsys profile --stats=true -f true -o res/gcn_profile_nsys \
python main.py root_dir > res/gcn_profile_nsys.txt



python main.py root_dir


#
#PWD=$(pwd)
#
#cd ../../Framework/CGCNN
#
#nsys profile --stats=true -f true -o $PWD/gcn_profile_nsys python CGCNN_paddle/main.py data/root_dir_1000  > $PWD/gcn_profile_nsys.txt