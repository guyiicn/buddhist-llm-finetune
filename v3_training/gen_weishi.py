#!/usr/bin/env python3
"""Generate weishi (唯识) data using RAG + qwen-plus API - with incremental saving"""

import json
import os
import time
import random
import requests
from openai import OpenAI

OUTPUT_FILE = os.path.expanduser("~/v3_training_data.json")
QWEN_API_KEY = "sk-8cdb59cb607348ff9ef65478d24d635b"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
RAG_URL = "http://192.168.50.12:8000/search"

client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

WEISHI_QA = [
    # 八识基础
    ("什么是八识？", "八识 识"),
    ("八识分别是什么？它们各有什么作用？", "八识 眼识 耳识 意识 末那识 阿赖耶识"),
    ("前五识是什么？有什么特点？", "前五识 眼识 耳识 鼻识 舌识 身识"),
    ("第六意识有什么特点？与前五识有何不同？", "第六识 意识 分别"),
    ("第七末那识的作用是什么？", "末那识 第七识 我执"),
    ("第八阿赖耶识是什么？为什么说它是根本识？", "阿赖耶识 第八识 种子 含藏"),
    # 阿赖耶识详解
    ("阿赖耶识有哪些别名？", "阿赖耶识 藏识 异熟识 一切种子识"),
    ("阿赖耶识的三相是什么？", "阿赖耶识 自相 因相 果相"),
    ("阿赖耶识与轮回有什么关系？", "阿赖耶识 轮回 业力 种子"),
    ("阿赖耶识是我吗？与外道神我有何区别？", "阿赖耶识 无我 非我 执著"),
    ("'恒转如瀑流'如何理解？", "阿赖耶识 恒转 瀑流 相续"),
    # 末那识详解
    ("末那识为什么叫'染污意'？", "末那识 染污意 我执 四烦恼"),
    ("末那识的四烦恼是什么？", "末那识 我痴 我见 我慢 我爱"),
    ("末那识'恒审思量'是什么意思？", "末那识 恒审思量 执著"),
    ("末那识与第六意识有什么区别？", "末那识 意识 区别"),
    # 种子与熏习
    ("什么是种子？种子有什么特性？", "种子 六义 刹那灭 果俱有"),
    ("种子六义是什么？", "种子 六义 刹那灭 果俱有 恒随转 性决定 待众缘 引自果"),
    ("本有种子和新熏种子有什么区别？", "本有种子 新熏种子"),
    ("什么是熏习？熏习如何形成种子？", "熏习 种子 现行"),
    ("能熏与所熏的关系是什么？", "能熏 所熏 四义"),
    # 三性三无性
    ("什么是三性？请详细解释。", "三性 遍计所执性 依他起性 圆成实性"),
    ("遍计所执性是什么意思？", "遍计所执性 妄想 分别"),
    ("依他起性是什么意思？", "依他起性 缘起 因缘"),
    ("圆成实性是什么意思？", "圆成实性 真如 空性"),
    ("三性之间有什么关系？", "三性 遍计 依他 圆成 关系"),
    ("什么是三无性？与三性有何对应？", "三无性 相无性 生无性 胜义无性"),
    # 转识成智
    ("什么是转识成智？", "转识成智 八识 四智"),
    ("八识如何转为四智？", "转识成智 大圆镜智 平等性智 妙观察智 成所作智"),
    ("大圆镜智是什么？由哪个识转成？", "大圆镜智 阿赖耶识 第八识"),
    ("平等性智是什么？由哪个识转成？", "平等性智 末那识 第七识"),
    ("妙观察智是什么？由哪个识转成？", "妙观察智 意识 第六识"),
    ("成所作智是什么？由哪个识转成？", "成所作智 前五识"),
    ("如何修行才能转识成智？", "转识成智 修行 方法"),
    # 五位百法
    ("什么是五位百法？", "五位百法 心法 心所法 色法 心不相应行法 无为法"),
    ("心法八种是什么？", "心法 八识"),
    ("心所法五十一种如何分类？", "心所法 遍行 别境 善 烦恼 随烦恼 不定"),
    ("什么是遍行心所？有哪几种？", "遍行心所 作意 触 受 想 思"),
    ("什么是别境心所？", "别境心所 欲 胜解 念 定 慧"),
    ("什么是善心所？有哪些？", "善心所 信 惭 愧 无贪 无嗔 无痴"),
    ("什么是烦恼心所？", "烦恼心所 贪 嗔 痴 慢 疑 恶见"),
    ("色法十一种是什么？", "色法 五根 五境 法处所摄色"),
    ("无为法六种是什么？", "无为法 虚空 择灭 非择灭 不动 想受灭 真如"),
    # 唯识核心概念
    ("唯识无境是什么意思？", "唯识无境 万法唯识"),
    ("万法唯识如何理解？外境存在吗？", "万法唯识 外境 识变"),
    ("唯识与唯心有什么区别？", "唯识 唯心 区别"),
    ("什么是识变？", "识变 因能变 果能变"),
    ("见分和相分是什么？", "见分 相分 四分"),
    ("唯识四分说是什么？", "四分 见分 相分 自证分 证自证分"),
    ("什么是所缘缘？", "所缘缘 亲所缘缘 疏所缘缘"),
    ("亲所缘缘和疏所缘缘有什么区别？", "亲所缘缘 疏所缘缘"),
    # 唯识经论
    ("《成唯识论》的主要内容是什么？", "成唯识论 玄奘 唯识"),
    ("《唯识三十颂》讲什么？是谁造的？", "唯识三十颂 世亲"),
    ("《唯识二十论》的核心观点是什么？", "唯识二十论 世亲"),
    ("《瑜伽师地论》与唯识有什么关系？", "瑜伽师地论 弥勒 无著"),
    ("《摄大乘论》的主要思想是什么？", "摄大乘论 无著"),
    ("《解深密经》中的唯识思想有哪些？", "解深密经 三性 阿赖耶识"),
    # 唯识祖师
    ("玄奘大师对唯识学有什么贡献？", "玄奘 唯识 成唯识论 法相宗"),
    ("窥基大师的唯识思想有什么特点？", "窥基 唯识 成唯识论述记"),
    ("护法论师与安慧论师有什么分歧？", "护法 安慧 有相唯识 无相唯识"),
    ("无著菩萨对唯识学有什么贡献？", "无著 瑜伽行派 摄大乘论"),
    ("世亲菩萨的唯识著作有哪些？", "世亲 唯识三十颂 唯识二十论"),
    # 修行相关
    ("唯识学如何指导修行？", "唯识 修行 转依"),
    ("什么是转依？", "转依 转识成智"),
    ("唯识观如何修？", "唯识观 五重唯识观"),
    ("什么是五重唯识观？", "五重唯识观 遣虚存实 舍滥留纯"),
    # 其他重要概念
    ("什么是二取？", "二取 能取 所取"),
    ("什么是法执和我执？", "法执 我执 二执"),
    ("唯识学如何解释梦境？", "唯识 梦境 独影境"),
    ("唯识学如何解释轮回？", "唯识 轮回 阿赖耶识 业力"),
    ("阿陀那识是什么？与阿赖耶识有何关系？", "阿陀那识 阿赖耶识 执持"),
    ("什么是异熟识？", "异熟识 阿赖耶识 果报"),
]


def load_data():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_data(data):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def call_rag(query, top_k=3):
    try:
        response = requests.post(
            RAG_URL, json={"query": query, "top_k": top_k}, timeout=30
        )
        if response.status_code == 200:
            results = response.json()
            if results:
                return "\n\n".join(
                    [r.get("text", "") for r in results if r.get("text")]
                )
    except Exception as e:
        print(f"    RAG error: {e}")
    return None


def call_qwen(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"    API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    return None


def generate_weishi_entry(question, rag_query):
    """Generate a single weishi entry"""
    # Query RAG
    rag_context = call_rag(rag_query, top_k=3)

    if rag_context and len(rag_context) > 100:
        prompt = f"""你是一位精通唯识学的佛教法师，法号"善知识"。请根据以下经论原文，回答问题。

经论原文：
{rag_context[:3000]}

问题：{question}

要求：
1. 以"阿弥陀佛"开头
2. 自称"贫僧"
3. 引用经论时注明出处（如《成唯识论》《瑜伽师地论》等）
4. 解释要通俗易懂但不失准确
5. 控制在200-400字
6. 不要编造不存在的经文或概念"""
    else:
        prompt = f"""你是一位精通唯识学的佛教法师，法号"善知识"。请回答以下唯识学问题。

问题：{question}

要求：
1. 以"阿弥陀佛"开头
2. 自称"贫僧"
3. 基于《成唯识论》《唯识三十颂》《瑜伽师地论》《摄大乘论》等唯识经论回答
4. 解释要通俗易懂但不失准确
5. 控制在200-400字
6. 不要编造不存在的内容"""

    response = call_qwen(prompt)
    return response


def main():
    existing = load_data()
    from collections import Counter

    counts = Counter(d["category"] for d in existing)
    current = counts.get("weishi", 0)
    target = 200
    need = max(0, target - current)

    print(f"Current weishi: {current}, target: {target}, need: {need}")

    if need <= 0:
        print("Weishi data complete!")
        return

    # Shuffle questions and select needed amount
    qa_pairs = list(WEISHI_QA)
    random.shuffle(qa_pairs)

    # Add variations
    prefixes = ["", "请问", "师父，", "请教", "想了解", "能否解释"]

    generated = 0
    batch_size = 10

    for i, (question, rag_query) in enumerate(qa_pairs):
        if generated >= need:
            break

        q_with_prefix = random.choice(prefixes) + question
        print(f"[{generated + 1}/{need}] {question[:30]}...")

        response = generate_weishi_entry(question, rag_query)

        if response and len(response) > 50:
            entry = {
                "instruction": q_with_prefix,
                "input": "",
                "output": response.strip(),
                "category": "weishi",
            }
            existing.append(entry)
            generated += 1
            print(f"    -> OK ({len(response)} chars)")

            # Save every batch_size entries
            if generated % batch_size == 0:
                save_data(existing)
                print(f"    [Saved {len(existing)} entries]")
        else:
            print(f"    -> Failed")

        time.sleep(1)  # Rate limiting

    # Final save
    save_data(existing)

    final_counts = Counter(d["category"] for d in existing)
    print(f"\nDone! Generated {generated} weishi entries")
    print(f"Final distribution: {dict(final_counts)}")
    print(f"Total: {len(existing)}")


if __name__ == "__main__":
    main()
