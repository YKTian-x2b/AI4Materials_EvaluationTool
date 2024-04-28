

conda create -n CGCNN_paddle python=3.8
# paddle也有依赖 如cuda和cudnn
python3.8 -m pip install paddlepaddle-gpu==2.6.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
#
python3.8 -m pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

conda install scikit-learn
# 这会给你一个合适的pymatgen版本
pip install mp_api      
