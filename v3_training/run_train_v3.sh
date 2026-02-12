#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate buddhist-llm-py310
cd ~/code/buddhist-72b-distill
exec python train_buddhist_v3.py
