#!/usr/bin/env python3
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import time
import requests
import re

import os

QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

if not QWEN_API_KEY:
    raise ValueError("Please set QWEN_API_KEY environment variable")

QA_DIR = Path(__file__).parent.parent / "qa_pairs"
QA_FILE = QA_DIR / "buddhist_qa.json"


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


def get_processed_sources(qa_list: List[Dict]) -> set:
    return {qa.get("source", "") for qa in qa_list}


def generate_qa_with_qwen(chunk: Dict, attempt: int = 0) -> List[Dict]:
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
            timeout=60,
        )

        response.raise_for_status()
        result = response.json()
        generated_text = result["choices"][0]["message"]["content"]

        json_match = re.search(r"\[.*\]", generated_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return []

    except Exception as e:
        print(f"  ⚠  API error: {e}")
        if attempt < 3:
            time.sleep(2)
            return generate_qa_with_qwen(chunk, attempt + 1)
        return []


def save_qa_pairs(qa_pairs: List[Dict]):
    QA_DIR.mkdir(exist_ok=True)
    valid_qa = [qa for qa in qa_pairs if qa.get("instruction") and qa.get("output")]
    with open(QA_FILE, "w", encoding="utf-8") as f:
        json.dump(valid_qa, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 60)
    print("佛经 QA 生成脚本 (Qwen API) - 增量模式")
    print("=" * 60)

    chunks = load_cleaned_chunks()
    if not chunks:
        print("错误: 没有可用的分块数据")
        return

    all_qa = load_existing_qa()
    processed = get_processed_sources(all_qa)

    pending_chunks = [c for c in chunks if c["source_name"] not in processed]

    print(
        f"总分块: {len(chunks)}, 已处理: {len(chunks) - len(pending_chunks)}, 待处理: {len(pending_chunks)}"
    )

    if not pending_chunks:
        print("所有分块已处理完成!")
        return

    for chunk in tqdm(pending_chunks, desc="生成 QA"):
        qa_pairs = generate_qa_with_qwen(chunk)

        for qa in qa_pairs:
            all_qa.append(
                {
                    "instruction": qa["instruction"],
                    "input": qa.get("input", ""),
                    "output": qa["output"],
                    "source": chunk["source_name"],
                    "category": chunk["category"],
                }
            )

        save_qa_pairs(all_qa)
        time.sleep(0.5)

    print(f"\n✓ 完成! 共 {len(all_qa)} 条 QA")
    print("=" * 60)


if __name__ == "__main__":
    main()
