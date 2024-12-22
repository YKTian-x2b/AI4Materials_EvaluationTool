#!/bin/bash

#source /usr/local/Anaconda/bin/activate SyMat
#echo "conda env: "
#echo $CONDA_DEFAULT_ENV
#cd /opt/AI4Sci/AI4S_EvaluationToolset/AI4Material_EvaluationTool
#python frameMain.py --modelName CGCNN --frameName torch
#echo "bash end"



source /home/yujixuan/anaconda3/bin/activate SyMat
echo "conda env: "
echo $CONDA_DEFAULT_ENV
cd /home/yujixuan/AI4Sci/AI4Materials_EvaluationTool
python frameMain.py --modelName CGCNN --frameName torch
echo "bash end"