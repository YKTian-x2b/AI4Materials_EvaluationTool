

conda create -n CGCNN_paddle python=3.8
#
python3.8 -m pip install paddlepaddle-gpu==2.6.1.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
#
python3.8 -m pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

conda install scikit-learn
pip install pymatgen==2020.11.11
