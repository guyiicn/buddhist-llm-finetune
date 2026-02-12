#!/bin/bash
# 善知识模型训练 v3 启动脚本

source ~/miniforge3/etc/profile.d/conda.sh
conda activate buddhist-llm-py310

cd ~/code/buddhist-72b-distill

echo "=== 善知识模型训练 v3 ===" | tee training_v3.log
echo "开始时间: $(date)" | tee -a training_v3.log
echo "数据: 12285 条 (v2 + 幻觉/边界/身份/唯识专项)" | tee -a training_v3.log
echo "" | tee -a training_v3.log

python train_buddhist_v3.py 2>&1 | tee -a training_v3.log

echo "" | tee -a training_v3.log
echo "结束时间: $(date)" | tee -a training_v3.log
