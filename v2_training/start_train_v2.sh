#!/bin/bash
# 善知识模型训练 v2 启动脚本

source ~/miniforge3/etc/profile.d/conda.sh
conda activate buddhist-llm-py310

cd ~/code/buddhist-72b-distill

echo "=== 善知识模型训练 v2 ===" | tee training_v2.log
echo "开始时间: $(date)" | tee -a training_v2.log
echo "数据: 11353 条" | tee -a training_v2.log
echo "" | tee -a training_v2.log

python train_buddhist_v2.py 2>&1 | tee -a training_v2.log

echo "" | tee -a training_v2.log
echo "结束时间: $(date)" | tee -a training_v2.log
