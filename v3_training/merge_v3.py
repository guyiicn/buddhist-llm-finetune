#!/usr/bin/env python3
"""Merge v2 and v3 training data for 善知识 v3 model"""
import json
import os
import random
from collections import Counter

DATA_DIR = os.path.expanduser("~/code/buddhist-72b-distill/data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # Load v2 training data
    v2_train = load_json(os.path.join(TRAIN_DIR, "buddhist_full_train.json"))
    v2_val = load_json(os.path.join(TRAIN_DIR, "buddhist_full_val.json"))
    print(f"V2 train: {len(v2_train)}, val: {len(v2_val)}")
    
    # Load v3 supplement data
    v3_data = load_json(os.path.join(DATA_DIR, "v3_training_data.json"))
    print(f"V3 supplement: {len(v3_data)}")
    
    # Show v3 distribution
    v3_counts = Counter(d.get("category", "unknown") for d in v3_data)
    print(f"V3 categories: {dict(v3_counts)}")
    
    # Merge v3 into training data
    # v3 data should be added to training set (not validation)
    merged_train = v2_train + v3_data
    random.shuffle(merged_train)
    
    print(f"\nMerged train: {len(merged_train)}")
    
    # Save merged data
    output_train = os.path.join(TRAIN_DIR, "buddhist_v3_train.json")
    output_val = os.path.join(TRAIN_DIR, "buddhist_v3_val.json")
    
    save_json(merged_train, output_train)
    save_json(v2_val, output_val)  # Keep same validation set
    
    print(f"Saved: {output_train}")
    print(f"Saved: {output_val}")
    
    # Update dataset_info.json
    dataset_info = {
        "buddhist_v3_train": {
            "file_name": "buddhist_v3_train.json"
        },
        "buddhist_v3_val": {
            "file_name": "buddhist_v3_val.json"
        }
    }
    save_json(dataset_info, os.path.join(TRAIN_DIR, "dataset_info.json"))
    print("Updated dataset_info.json")

if __name__ == "__main__":
    main()
