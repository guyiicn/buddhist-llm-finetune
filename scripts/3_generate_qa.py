#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
佛经 QA 生成脚本 (Qwen API) - 并发版

功能：
- 从 cleaned/chunks.json 读取文本块
- 使用 Qwen API 并发生成 QA 对 (默认 5 并发)
- 支持增量处理（跳过已处理的块）
- 支持 --max-qa 限制总 QA 数量
- 支持 --fresh 清除旧数据重新开始
- 支持 --workers 控制并发数
- 按经典均匀采样，确保覆盖所有 18 部经典

用法：
  export QWEN_API_KEY="sk-xxx"
  python scripts/3_generate_qa.py                          # 处理所有块 (5并发)
  python scripts/3_generate_qa.py --max-qa 500             # 生成约500条
  python scripts/3_generate_qa.py --workers 8              # 8并发加速
  python scripts/3_generate_qa.py --fresh                  # 清除旧数据重新生成
"""

import json
import argparse
import random
import re
import time
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

import requests

# ============================================================
# API 配置
# ============================================================

QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

QA_DIR = Path(__file__).parent.parent / "qa_pairs"
QA_FILE = QA_DIR / "buddhist_qa.json"

# 线程安全锁
_lock = threading.Lock()


def load_cleaned_chunks() -> List[Dict]:
    chunks_file = Path(__file__).parent.parent / "cleaned" / "chunks.json"
    if not chunks_file.exists():
        return []
    with open(chunks_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_qa() -> List[Dict]:
    if QA_FILE.exists():
        with open(QA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def get_processed_chunk_keys(qa_list: List[Dict]) -> set:
    """返回已处理的 (source_id, chunk_index) 集合"""
    keys = set()
    for qa in qa_list:
        source = qa.get("source_id", qa.get("source", ""))
        chunk_idx = qa.get("chunk_index", -1)
        if source and chunk_idx >= 0:
            keys.add((source, chunk_idx))
    return keys


def select_chunks_proportional(
    chunks: List[Dict], max_chunks: int, seed: int = 42
) -> List[Dict]:
    """按经典均匀采样"""
    random.seed(seed)

    by_source = defaultdict(list)
    for chunk in chunks:
        key = chunk.get("source_id", chunk.get("source_name", "unknown"))
        by_source[key].append(chunk)

    n_sources = len(by_source)
    if max_chunks >= len(chunks):
        return chunks

    min_per_source = min(2, max_chunks // n_sources) if n_sources > 0 else 0
    reserved = min_per_source * n_sources
    remaining = max(0, max_chunks - reserved)

    total_chunks = len(chunks)
    selected = []

    for source_id, source_chunks in by_source.items():
        proportion = len(source_chunks) / total_chunks
        extra = round(remaining * proportion)
        n_select = min(min_per_source + extra, len(source_chunks))
        n_select = max(1, n_select)

        if n_select >= len(source_chunks):
            selected.extend(source_chunks)
        else:
            indices = []
            step = len(source_chunks) / n_select
            for i in range(n_select):
                idx = int(i * step)
                indices.append(idx)
            indices = sorted(set(indices))[:n_select]
            for idx in indices:
                selected.append(source_chunks[idx])

    if len(selected) < max_chunks:
        remaining_chunks = [c for c in chunks if c not in selected]
        random.shuffle(remaining_chunks)
        selected.extend(remaining_chunks[: max_chunks - len(selected)])

    if len(selected) > max_chunks:
        random.shuffle(selected)
        selected = selected[:max_chunks]

    return selected


def generate_qa_with_qwen(chunk: Dict, attempt: int = 0) -> List[Dict]:
    """调用 Qwen API 生成 QA 对 (线程安全)"""
    prompt = f"""你是一位精通佛学的学者。请根据以下佛经段落，生成 3 个高质量的问答对。

要求：
1. 问题要自然、多样（可以是理解、翻译、应用、辨析等不同角度）
2. 答案要准确、详细，通常 200-600 字
3. 答案要引用或参考原文内容
4. 用现代中文表达，通俗易懂

来源：{chunk["source_name"]}
内容：
{chunk["content"]}

请以 JSON 格式输出，不要有任何额外文字：
[
  {{"instruction": "问题1", "input": "", "output": "答案1"}},
  {{"instruction": "问题2", "input": "", "output": "答案2"}},
  {{"instruction": "问题3", "input": "", "output": "答案3"}}
]
"""

    try:
        response = requests.post(
            f"{QWEN_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {QWEN_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": QWEN_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一位精通佛学的学者。请严格按照 JSON 格式输出。",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 3000,
            },
            timeout=90,
        )

        response.raise_for_status()
        result = response.json()
        generated_text = result["choices"][0]["message"]["content"]

        json_match = re.search(r"\[.*\]", generated_text, re.DOTALL)
        if json_match:
            qa_list = json.loads(json_match.group())
            valid = []
            for qa in qa_list:
                if isinstance(qa, dict) and qa.get("instruction") and qa.get("output"):
                    valid.append(qa)
            return valid
        return []

    except Exception as e:
        if attempt < 3:
            wait = 2 ** (attempt + 1)
            time.sleep(wait)
            return generate_qa_with_qwen(chunk, attempt + 1)
        return []


def process_chunk(chunk: Dict, index: int, total: int) -> Dict:
    """处理单个 chunk，返回结果 (线程安全)"""
    source = chunk.get("source_name", "?")
    idx = chunk.get("chunk_index", "?")

    qa_pairs = generate_qa_with_qwen(chunk)

    results = []
    if qa_pairs:
        for qa in qa_pairs:
            results.append(
                {
                    "instruction": qa["instruction"],
                    "input": qa.get("input", ""),
                    "output": qa["output"],
                    "source": chunk.get("source_name", ""),
                    "source_id": chunk.get("source_id", ""),
                    "chunk_index": chunk.get("chunk_index", -1),
                    "category": chunk.get("category", ""),
                }
            )

    return {
        "chunk_key": (chunk.get("source_id", ""), chunk.get("chunk_index", -1)),
        "source": source,
        "chunk_index": idx,
        "qa_results": results,
        "success": len(results) > 0,
    }


def save_qa_pairs(qa_pairs: List[Dict]):
    QA_DIR.mkdir(exist_ok=True)
    valid_qa = [qa for qa in qa_pairs if qa.get("instruction") and qa.get("output")]
    with open(QA_FILE, "w", encoding="utf-8") as f:
        json.dump(valid_qa, f, ensure_ascii=False, indent=2)


def print_stats(qa_list: List[Dict]):
    by_source = defaultdict(int)
    for qa in qa_list:
        by_source[qa.get("source", "unknown")] += 1

    print(f"\n--- QA 统计 ---")
    print(f"总计: {len(qa_list)} 条")
    print(f"来源分布:")
    for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count} 条")


def main():
    parser = argparse.ArgumentParser(description="佛经 QA 生成脚本 (并发版)")
    parser.add_argument("--max-qa", type=int, default=0, help="最大 QA 数量 (0=不限制)")
    parser.add_argument("--fresh", action="store_true", help="清除旧数据重新生成")
    parser.add_argument("--dry-run", action="store_true", help="只显示计划，不调用 API")
    parser.add_argument("--workers", type=int, default=5, help="并发线程数 (默认 5)")
    parser.add_argument(
        "--save-every", type=int, default=10, help="每处理 N 个块保存一次 (默认 10)"
    )
    args = parser.parse_args()

    if not QWEN_API_KEY:
        print("❌ 错误: 请设置 QWEN_API_KEY 环境变量")
        return

    print("=" * 60)
    print("佛经 QA 生成脚本 (Qwen API) - 并发版")
    print(f"  模型: {QWEN_MODEL}")
    print(f"  并发: {args.workers} 线程")
    if args.max_qa:
        print(f"  目标: {args.max_qa} 条 QA")
    if args.fresh:
        print(f"  模式: 全新生成")
    print("=" * 60)

    chunks = load_cleaned_chunks()
    if not chunks:
        print("❌ 错误: 没有分块数据，请先运行 2_clean_text.py")
        return

    print(f"\n总分块数: {len(chunks)}")

    if args.fresh:
        all_qa = []
        processed_keys = set()
        print("已清除旧 QA 数据")
    else:
        all_qa = load_existing_qa()
        processed_keys = get_processed_chunk_keys(all_qa)
        print(f"已有 QA: {len(all_qa)} 条 (来自 {len(processed_keys)} 个块)")

    # 确定待处理块
    if args.max_qa and args.max_qa <= len(all_qa):
        print(f"\n已有 {len(all_qa)} 条 >= 目标 {args.max_qa}，无需生成")
        print_stats(all_qa)
        return

    if args.max_qa:
        target_chunks = max(1, (args.max_qa - len(all_qa) + 2) // 3)
        unprocessed = [
            c
            for c in chunks
            if (c.get("source_id", ""), c.get("chunk_index", -1)) not in processed_keys
        ]
        if not unprocessed:
            print("所有分块已处理完成!")
            print_stats(all_qa)
            return
        pending_chunks = select_chunks_proportional(unprocessed, target_chunks)
    else:
        pending_chunks = [
            c
            for c in chunks
            if (c.get("source_id", ""), c.get("chunk_index", -1)) not in processed_keys
        ]

    print(f"待处理块: {len(pending_chunks)}")
    print(f"预计生成: ~{len(pending_chunks) * 3} 条 QA")
    est_minutes = len(pending_chunks) * 7 / args.workers / 60
    print(f"预计耗时: ~{est_minutes:.0f} 分钟 ({args.workers} 并发)")

    # 显示来源分布
    by_source = defaultdict(int)
    for c in pending_chunks:
        by_source[c.get("source_name", "unknown")] += 1
    print(f"\n待处理分布:")
    for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count} 块")

    if args.dry_run:
        return

    if not pending_chunks:
        print("\n没有待处理的块!")
        print_stats(all_qa)
        return

    # 并发生成
    print(f"\n开始并发生成 QA ({args.workers} 线程)...")
    completed = 0
    errors = 0
    generated_count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for i, chunk in enumerate(pending_chunks):
            future = executor.submit(process_chunk, chunk, i, len(pending_chunks))
            futures[future] = i

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if result["success"]:
                with _lock:
                    all_qa.extend(result["qa_results"])
                    generated_count += len(result["qa_results"])
            else:
                errors += 1

            # 定期保存
            if completed % args.save_every == 0:
                with _lock:
                    save_qa_pairs(all_qa)

                elapsed = time.time() - start_time
                rate = completed / elapsed * 60
                remaining = (
                    (len(pending_chunks) - completed) / (rate / 60) if rate > 0 else 0
                )
                print(
                    f"  [{completed}/{len(pending_chunks)}] "
                    f"QA: {len(all_qa)} | "
                    f"速率: {rate:.0f} 块/分 | "
                    f"剩余: ~{remaining:.0f}s | "
                    f"错误: {errors}"
                )

            # 检查目标
            if args.max_qa and len(all_qa) >= args.max_qa:
                print(f"\n✓ 已达到目标: {len(all_qa)} 条")
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break

    # 最终保存
    save_qa_pairs(all_qa)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"生成完成!")
    print(f"  本次生成: {generated_count} 条 QA")
    print(f"  总计: {len(all_qa)} 条 QA")
    print(f"  错误: {errors} 个块")
    print(f"  耗时: {elapsed / 60:.1f} 分钟")
    print(f"  速率: {completed / elapsed * 60:.0f} 块/分")

    print_stats(all_qa)
    print("=" * 60)


if __name__ == "__main__":
    main()
