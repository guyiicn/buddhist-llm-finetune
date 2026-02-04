#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集合并脚本
合并种子数据、生成的 QA 数据，输出最终训练集
"""

import json
from pathlib import Path
from typing import List, Dict
import random


def load_config() -> Dict:
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        import yaml

        return yaml.safe_load(f)


def load_seed_data() -> List[Dict]:
    """加载种子数据（人工编写的高质量 QA）"""
    seeds_dir = Path(__file__).parent.parent / "seeds"
    seed_file = seeds_dir / "seed_qa.json"

    if seed_file.exists():
        with open(seed_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_generated_qa() -> List[Dict]:
    """加载生成的 QA 数据"""
    qa_dir = Path(__file__).parent.parent / "qa_pairs"
    qa_file = qa_dir / "buddhist_qa.json"

    if qa_file.exists():
        with open(qa_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def merge_datasets(
    seeds: List[Dict], generated: List[Dict], config: Dict
) -> List[Dict]:
    """合并数据集"""

    # 确保格式一致
    formatted_seeds = []
    for item in seeds:
        formatted_seeds.append(
            {
                "instruction": item["instruction"],
                "input": item.get("input", ""),
                "output": item["output"],
                "source": item.get("source", "seed_data"),
            }
        )

    formatted_generated = []
    for item in generated:
        formatted_generated.append(
            {
                "instruction": item["instruction"],
                "input": item.get("input", ""),
                "output": item["output"],
                "source": item.get("source", "generated"),
            }
        )

    # 合并
    merged = formatted_seeds + formatted_generated

    # 去重（基于 instruction）
    seen_instructions = set()
    unique_data = []

    for item in merged:
        if item["instruction"] not in seen_instructions:
            seen_instructions.add(item["instruction"])
            unique_data.append(item)

    return unique_data


def split_train_val(data: List[Dict], train_ratio: float = 0.9, seed: int = 42):
    """分割训练集和验证集"""

    random.seed(seed)
    random.shuffle(data)

    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]

    return train_data, val_data


def save_datasets(train_data: List[Dict], val_data: List[Dict], config: Dict):
    """保存数据集"""

    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    format = config["output"]["format"]

    if format == "alpaca":
        # Alpaca 格式
        train_file = output_dir / "buddhist_train_alpaca.json"
        val_file = output_dir / "buddhist_val_alpaca.json"

        with open(train_file, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(val_file, "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

    else:
        # ShareGPT 格式
        train_file = output_dir / "buddhist_train_sharegpt.json"
        val_file = output_dir / "buddhist_val_sharegpt.json"

        # 转换为 ShareGPT 格式
        def convert_to_sharegpt(item):
            return {
                "conversations": [
                    {"from": "human", "value": item["instruction"]},
                    {"from": "gpt", "value": item["output"]},
                ]
            }

        train_sharegpt = [convert_to_sharegpt(item) for item in train_data]
        val_sharegpt = [convert_to_sharegpt(item) for item in val_data]

        with open(train_file, "w", encoding="utf-8") as f:
            json.dump(train_sharegpt, f, ensure_ascii=False, indent=2)

        with open(val_file, "w", encoding="utf-8") as f:
            json.dump(val_sharegpt, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 训练集: {len(train_data)} 条 → {train_file}")
    print(f"✓ 验证集: {len(val_data)} 条 → {val_file}")

    # 保存数据集统计
    stats = {
        "total": len(train_data) + len(val_data),
        "train": len(train_data),
        "validation": len(val_data),
        "seed_count": len(
            [d for d in train_data + val_data if d.get("source") == "seed_data"]
        ),
        "generated_count": len(
            [d for d in train_data + val_data if d.get("source") == "generated"]
        ),
        "format": format,
        "train_ratio": config["output"]["train_ratio"],
    }

    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 数据集统计:")
    print(f"   - 总计: {stats['total']} 条")
    print(f"   - 种子数据: {stats['seed_count']} 条")
    print(f"   - 生成数据: {stats['generated_count']} 条")
    print(f"   - 训练/验证: {stats['train_ratio']}/{1 - stats['train_ratio']}")


def main():
    print("=" * 60)
    print("佛经数据集合并脚本")
    print("=" * 60)

    # 加载配置
    config = load_config()

    # 加载种子数据
    print("\n加载种子数据...")
    seeds = load_seed_data()
    print(f"✓ 种子数据: {len(seeds)} 条")

    # 加载生成的 QA
    print("\n加载生成的 QA...")
    generated = load_generated_qa()
    print(f"✓ 生成 QA: {len(generated)} 条")

    # 合并
    print("\n合并数据集...")
    merged = merge_datasets(seeds, generated, config)
    print(f"✓ 合并后: {len(merged)} 条")

    # 分割
    train_ratio = config["output"]["train_ratio"]
    random_seed = config["output"]["seed"]
    print(f"\n分割数据集 (train_ratio={train_ratio})...")
    train_data, val_data = split_train_val(merged, train_ratio, random_seed)

    # 保存
    print("\n保存数据集...")
    save_datasets(train_data, val_data, config)

    print("\n" + "=" * 60)
    print("数据集准备完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 检查 output/ 目录下的数据文件")
    print("2. 使用 LLaMA-Factory 开始微调")
    print("   python -m llamafactory.cli train examples/train_lora/buddhist.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
