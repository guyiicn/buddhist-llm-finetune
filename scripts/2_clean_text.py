#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
佛经数据清洗脚本
清洗原始文本，去除噪音
"""

import json
import re
from pathlib import Path
from typing import List, Dict


def load_config() -> Dict:
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        import yaml

        return yaml.safe_load(f)


def load_raw_data() -> List[Dict]:
    """加载原始数据"""
    raw_dir = Path(__file__).parent.parent / "raw"
    meta_file = raw_dir / "metadata.json"

    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # 如果没有元数据，从文本文件读取
    data = []
    for txt_file in raw_dir.glob("*.txt"):
        if txt_file.name == "metadata.json":
            continue

        content = txt_file.read_text(encoding="utf-8")

        # 从内容中提取元数据
        lines = content.split("\n")
        metadata = {}
        text_start = 0

        for i, line in enumerate(lines):
            if line.startswith("# "):
                if "ID:" in line:
                    metadata["id"] = line.split("ID:")[1].strip()
                elif "分类:" in line:
                    metadata["category"] = line.split("分类:")[1].strip()
                text_start = i + 1

        data.append(
            {
                "file": txt_file.stem,
                "content": "\n".join(lines[text_start:]),
                **metadata,
            }
        )

    return data


def clean_text(text: str, config: Dict) -> str:
    """清洗文本"""

    # 1. 移除特殊符号和注释
    text = re.sub(r"\[\\d+\]", "", text)  # [1][2]
    text = re.sub(r"【.*?】", "", text)  # 【注】
    text = re.sub(r"〈.*?〉", "", text)  # 〈注〉

    # 2. 标准化空格和换行
    text = re.sub(r"　", " ", text)  # 全角空格
    text = re.sub(r"\n{3,}", "\n\n", text)  # 多余空行

    # 3. 移除空白字符
    text = text.strip()

    return text


def split_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
    """将文本分成块"""

    # 先按段落分割
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += "\n\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def save_cleaned_data(data: List[Dict], config: Dict):
    """保存清洗后的数据"""
    cleaned_dir = Path(__file__).parent.parent / "cleaned"
    cleaned_dir.mkdir(exist_ok=True)

    chunk_size = config["qa_generation"]["chunk_size"]

    all_chunks = []

    for i, item in enumerate(data):
        if (i + 1) % 10 == 0:
            print(f"  处理: {i + 1}/{len(data)}")
        cleaned_content = clean_text(item["content"], config)

        # 分块
        chunks = split_into_chunks(cleaned_content, chunk_size)

        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "source_id": item.get("id", item.get("file", "")),
                    "source_name": item.get("name", item.get("file", "")),
                    "chunk_index": i,
                    "content": chunk,
                    "category": item.get("category", "未分类"),
                }
            )

    # 保存分块数据
    chunks_file = cleaned_dir / "chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 清洗完成")
    print(f"✓ 生成分块: {len(all_chunks)}")
    print(f"✓ 保存位置: {chunks_file}")


def main():
    print("=" * 60)
    print("佛经数据清洗脚本")
    print("=" * 60)

    # 加载配置
    config = load_config()

    # 加载原始数据
    print("\n加载原始数据...")
    raw_data = load_raw_data()
    print(f"✓ 加载了 {len(raw_data)} 个文件")

    # 清洗并分块
    save_cleaned_data(raw_data, config)

    print("=" * 60)


if __name__ == "__main__":
    main()
