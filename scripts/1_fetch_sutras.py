#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
佛经数据获取脚本
从本地 CBETA 文本库提取佛经原文

CBETA 目录结构：
  /home/nvidia/cbeta/cbeta-text/T/{经号}/
    {经号}_001.txt, {经号}_002.txt, ...  (多卷经典)
    {经号}.yaml                          (元数据)

文本格式：
  - 前 N 行以 # 开头为文件头注释
  - 之后为经文正文（繁体中文）
"""

import json
import re
import yaml
from pathlib import Path
from typing import List, Dict

# ============================================================
# 配置
# ============================================================

CBETA_BASE = Path("/home/nvidia/cbeta/cbeta-text/T")

# 对超大经典设置最大提取字符数，避免数据量过大
# 小经全量提取，大经/大论取有代表性的前 N 字符
MAX_CHARS = {
    "T0099": 60000,  # 杂阿含经 50卷 1.9MB → 取前6万字(约前10卷核心内容)
    "T1579": 50000,  # 瑜伽师地论 100卷 3.1MB → 取前5万字
    "T1509": 50000,  # 大智度论 100卷 3.7MB → 取前5万字
    "T0262": 80000,  # 法华经 7卷 304K → 基本全取
    "T0945": 80000,  # 楞严经 10卷 284K → 基本全取
}

DEFAULT_MAX_CHARS = 200000  # 默认上限(足以容纳大部分经典全文)


def load_config() -> Dict:
    """加载项目配置"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def strip_header(text: str) -> str:
    """去除 CBETA 文件头部的注释行"""
    lines = text.split("\n")
    content_lines = []
    header_done = False
    for line in lines:
        if not header_done:
            if line.startswith("#"):
                continue
            # 跳过紧跟注释后的空行
            if line.strip() == "" and not content_lines:
                continue
            header_done = True
        content_lines.append(line)
    return "\n".join(content_lines)


def clean_cbeta_text(text: str) -> str:
    """清洗 CBETA 文本

    保留经文正文，去除：
    - CBETA 行首页码标记 (如 T08n0235_p0749a01 等)
    - 校勘记引用 [1], [2] 等
    - 过多空行
    """
    # 去除 CBETA 页码行标 (行首的 Txxnxxxx_pxxxxxx 格式)
    text = re.sub(r"^T\d+n\d+_p\d+[a-z]\d+.*$", "", text, flags=re.MULTILINE)

    # 去除校勘记编号 [1] [2] 等
    text = re.sub(r"\[\d+\]", "", text)

    # 去除 CBETA XML 残留标记
    text = re.sub(r"<[^>]+>", "", text)

    # 规范化空行（3个以上连续空行→2个）
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 去除行首行尾多余空格
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 去除首尾空白
    return text.strip()


def extract_sutra(sutra_id: str) -> str:
    """从 CBETA 本地文件提取指定经典的全文

    支持单卷和多卷经典。
    多卷按文件名排序后拼接。
    """
    sutra_dir = CBETA_BASE / sutra_id

    if not sutra_dir.exists():
        print(f"  ⚠ 目录不存在: {sutra_dir}")
        return ""

    # 获取所有 txt 文件（排除 yaml）
    txt_files = sorted(sutra_dir.glob(f"{sutra_id}_*.txt"))

    if not txt_files:
        print(f"  ⚠ 没有找到文本文件: {sutra_dir}")
        return ""

    max_chars = MAX_CHARS.get(sutra_id, DEFAULT_MAX_CHARS)
    collected = []
    total_len = 0

    for txt_file in txt_files:
        raw = txt_file.read_text(encoding="utf-8")
        content = strip_header(raw)
        content = clean_cbeta_text(content)

        if not content:
            continue

        # 检查是否超过上限
        if total_len + len(content) > max_chars:
            remaining = max_chars - total_len
            if remaining > 500:  # 至少保留500字才值得加
                # 在段落边界截断
                truncated = content[:remaining]
                last_para = truncated.rfind("\n\n")
                if last_para > remaining * 0.7:
                    truncated = truncated[:last_para]
                collected.append(truncated)
                total_len += len(truncated)
            break

        collected.append(content)
        total_len += len(content)

    full_text = "\n\n".join(collected)
    return full_text


def save_raw_text(sutra_data: List[Dict]):
    """保存原始文本到 raw 目录"""
    raw_dir = Path(__file__).parent.parent / "raw"
    raw_dir.mkdir(exist_ok=True)

    for sutra in sutra_data:
        filename = f"{sutra['id']}_{sutra['short_name']}.txt"
        filepath = raw_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(sutra["content"])
            f.write("\n")

        char_count = len(sutra["content"])
        print(f"  ✓ {filename} ({char_count:,} 字)")


def main():
    print("=" * 60)
    print("佛经数据提取脚本 (从本地 CBETA 文本库)")
    print("=" * 60)
    print(f"\nCBETA 路径: {CBETA_BASE}")
    print(f"CBETA 存在: {CBETA_BASE.exists()}")

    if not CBETA_BASE.exists():
        print("❌ 错误: CBETA 文本库路径不存在!")
        print("请确认 /home/nvidia/cbeta/cbeta-text/T/ 目录存在")
        return

    # 加载配置
    config = load_config()

    # 合并经和论
    all_texts = []
    total_chars = 0

    # 获取经
    print("\n--- 提取经典 ---")
    for sutra in config["sutras"]:
        print(f"\n提取: {sutra['name']} ({sutra['id']})")

        content = extract_sutra(sutra["id"])
        if not content:
            print(f"  ⚠ 跳过（无内容）: {sutra['id']}")
            continue

        total_chars += len(content)
        print(f"  提取: {len(content):,} 字")

        if sutra["id"] in MAX_CHARS:
            print(f"  (上限: {MAX_CHARS[sutra['id']]:,} 字)")

        all_texts.append(
            {
                "id": sutra["id"],
                "name": sutra["name"],
                "short_name": sutra["short_name"],
                "category": sutra.get("category", "未分类"),
                "content": content,
            }
        )

    # 获取论
    print("\n--- 提取论典 ---")
    if "treatises" in config:
        for treatise in config["treatises"]:
            print(f"\n提取: {treatise['name']} ({treatise['id']})")

            content = extract_sutra(treatise["id"])
            if not content:
                print(f"  ⚠ 跳过（无内容）: {treatise['id']}")
                continue

            total_chars += len(content)
            print(f"  提取: {len(content):,} 字")

            if treatise["id"] in MAX_CHARS:
                print(f"  (上限: {MAX_CHARS[treatise['id']]:,} 字)")

            all_texts.append(
                {
                    "id": treatise["id"],
                    "name": treatise["name"],
                    "short_name": treatise["short_name"],
                    "author": treatise.get("author", "未知"),
                    "category": treatise.get("category", "论部"),
                    "content": content,
                }
            )

    # 保存
    print("\n" + "=" * 60)
    print("保存文件...")
    save_raw_text(all_texts)

    # 保存元数据
    meta_file = Path(__file__).parent.parent / "raw" / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 元数据: metadata.json")

    # 统计
    print(f"\n{'=' * 60}")
    print(f"提取完成!")
    print(f"  经典数量: {len(all_texts)} 部")
    print(f"  总文本量: {total_chars:,} 字")
    print(f"  预计分块: ~{total_chars // 500} 个 (chunk_size=500)")
    print(f"  预计 QA:  ~{(total_chars // 500) * 3} 条 (3 QA/chunk)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
