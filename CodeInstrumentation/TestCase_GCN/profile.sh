 #!/bin/bash

nsys profile --stats=true -f true -o res/gcn_profile_nsys \
python main.py root_dir > res/gcn_profile_nsys.txt



# python main.py root_dir
