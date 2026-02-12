#!/usr/bin/env python3
"""Generate additional weishi data - batch 2"""

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

# Additional weishi questions - batch 2
WEISHI_QA_2 = [
    # 深入八识
    ("眼识如何认知色尘？", "眼识 色尘 眼根"),
    ("耳识如何了别声音？", "耳识 声尘 耳根"),
    ("身识感知触尘的过程是怎样的？", "身识 触尘 身根"),
    ("意识的独头意识和五俱意识有什么区别？", "意识 独头意识 五俱意识"),
    ("散位独头意识是什么？", "独头意识 散位 梦中"),
    ("定中独头意识有什么特点？", "定中意识 禅定 意识"),
    ("梦中意识属于什么识？", "梦中意识 独头意识 独影境"),
    ("狂乱意识是什么情况？", "狂乱意识 独头意识"),
    # 识与根境的关系
    ("根、境、识三者是什么关系？", "根 境 识 三缘"),
    ("眼根与眼识有什么区别？", "眼根 眼识 浮尘根 胜义根"),
    ("什么是浮尘根和胜义根？", "浮尘根 胜义根"),
    ("五根各缘什么境？", "五根 五境 色声香味触"),
    # 心所法详解
    ("作意心所的作用是什么？", "作意 心所 警觉"),
    ("触心所是什么意思？", "触 心所 根境识"),
    ("受心所分为几种？", "受 心所 苦受 乐受 舍受"),
    ("想心所的功能是什么？", "想 心所 取像"),
    ("思心所与业有什么关系？", "思 心所 造作 业"),
    ("欲心所是什么？", "欲 心所 希望"),
    ("胜解心所的作用？", "胜解 心所 印持"),
    ("念心所是什么？与正念有何关系？", "念 心所 明记 正念"),
    ("定心所与禅定的关系？", "定 心所 专注 三摩地"),
    ("慧心所是什么？与般若的关系？", "慧 心所 抉择 般若"),
    # 善心所详解
    ("信心所包含哪些内容？", "信 心所 三宝"),
    ("惭与愧有什么区别？", "惭 愧 心所"),
    ("无贪无嗔无痴是什么意思？", "无贪 无嗔 无痴 三善根"),
    ("精进心所的作用？", "精进 心所 勤"),
    ("轻安心所是什么感受？", "轻安 心所 调畅"),
    ("不放逸心所是什么？", "不放逸 心所 防护"),
    ("行舍心所的作用？", "行舍 心所 平等"),
    ("不害心所是什么意思？", "不害 心所 慈悲"),
    # 烦恼心所详解
    ("贪烦恼的特征是什么？", "贪 烦恼 染著"),
    ("嗔烦恼有什么过患？", "嗔 烦恼 忿恨"),
    ("痴烦恼是什么？", "痴 烦恼 无明 愚暗"),
    ("慢烦恼有几种？", "慢 烦恼 七慢"),
    ("疑烦恼障碍什么？", "疑 烦恼 犹豫"),
    ("恶见包含哪五种？", "恶见 五见 身见 边见 邪见 见取见 戒禁取见"),
    ("什么是身见？", "身见 萨迦耶见 我见"),
    ("什么是边见？", "边见 断见 常见"),
    ("什么是邪见？", "邪见 拨无因果"),
    ("什么是见取见？", "见取见 执见为胜"),
    ("什么是戒禁取见？", "戒禁取见 非因计因"),
    # 随烦恼心所
    ("忿与恨有什么区别？", "忿 恨 随烦恼"),
    ("覆与诳有什么不同？", "覆 诳 随烦恼"),
    ("谄与憍有什么特点？", "谄 憍 随烦恼"),
    ("嫉与悭是什么烦恼？", "嫉 悭 随烦恼"),
    ("无惭无愧的过患是什么？", "无惭 无愧 随烦恼"),
    ("掉举与昏沉如何对治？", "掉举 昏沉 随烦恼"),
    ("不信与懈怠有什么危害？", "不信 懈怠 随烦恼"),
    ("放逸与失念的区别？", "放逸 失念 随烦恼"),
    ("散乱与不正知如何理解？", "散乱 不正知 随烦恼"),
    # 不定心所
    ("悔心所是善是恶？", "悔 不定心所"),
    ("眠心所的性质如何？", "眠 不定心所 睡眠"),
    ("寻与伺有什么区别？", "寻 伺 不定心所 粗细"),
    # 色法详解
    ("法处所摄色是什么？", "法处所摄色 极略色 极迥色"),
    ("什么是极略色和极迥色？", "极略色 极迥色"),
    ("定果色是什么？", "定果色 定所引色"),
    ("遍计所执色是什么？", "遍计所执色"),
    ("无表色是什么？", "无表色 受戒"),
    # 心不相应行法
    ("什么是得与非得？", "得 非得 心不相应行"),
    ("什么是同分？", "同分 众同分 心不相应行"),
    ("什么是命根？", "命根 寿命 心不相应行"),
    ("什么是异生性？", "异生性 凡夫 心不相应行"),
    ("四相是什么？", "四相 生住异灭 心不相应行"),
    ("什么是名身句身文身？", "名身 句身 文身 心不相应行"),
    # 唯识修行进阶
    ("资粮位的修行内容？", "资粮位 五位 唯识"),
    ("加行位四寻思是什么？", "加行位 四寻思 暖顶忍世第一"),
    ("见道位证悟什么？", "见道位 真见道 相见道"),
    ("修道位如何修行？", "修道位 十地 唯识"),
    ("究竟位是什么境界？", "究竟位 佛果 转依"),
    # 唯识与其他宗派
    ("唯识与中观的区别？", "唯识 中观 空有"),
    ("唯识与如来藏的关系？", "唯识 如来藏 阿赖耶识"),
    ("法相宗与慈恩宗是一个宗派吗？", "法相宗 慈恩宗 唯识宗"),
    ("唯识学在中国的传承？", "唯识 法相宗 玄奘 窥基"),
    ("唯识学在日本的发展？", "唯识 日本 法相宗"),
    # 重要概念补充
    ("什么是共相与自相？", "共相 自相 唯识"),
    ("什么是现量与比量？", "现量 比量 量论"),
    ("什么是圣教量？", "圣教量 至教量"),
    ("唯识学如何解释因果？", "唯识 因果 种子 现行"),
    ("什么是等流果和异熟果？", "等流果 异熟果 果报"),
    ("什么是增上果？", "增上果 增上缘"),
    ("什么是士用果？", "士用果 作业"),
    ("什么是离系果？", "离系果 涅槃 解脱"),
    # 阿赖耶识进阶
    ("阿赖耶识的五遍行是什么？", "阿赖耶识 遍行 心所"),
    ("为什么说阿赖耶识是无覆无记？", "阿赖耶识 无覆无记"),
    ("阿赖耶识如何执持根身器界？", "阿赖耶识 执持 根身 器界"),
    ("阿赖耶识与业力的关系？", "阿赖耶识 业力 种子"),
    ("阿赖耶识在死亡时的变化？", "阿赖耶识 死亡 中阴"),
    ("阿赖耶识在投胎时的作用？", "阿赖耶识 投胎 结生"),
    # 末那识进阶
    ("末那识为什么恒常执我？", "末那识 我执 恒审思量"),
    ("末那识与阿赖耶识的关系？", "末那识 阿赖耶识 见分"),
    ("末那识转为平等性智的过程？", "末那识 平等性智 转识成智"),
    ("末那识在禅定中的状态？", "末那识 禅定 灭尽定"),
    # 意识进阶
    ("意识的九缘是什么？", "意识 九缘 生起"),
    ("意识在五位无心时如何？", "意识 五位无心 无想定 灭尽定"),
    ("意识与业的造作关系？", "意识 业 造作 思心所"),
    ("意识转为妙观察智的条件？", "意识 妙观察智 转识成智"),
    # 唯识名相
    ("什么是有漏与无漏？", "有漏 无漏 烦恼"),
    ("什么是有为与无为？", "有为 无为 法"),
    ("什么是能变与所变？", "能变 所变 识变"),
    ("什么是能缘与所缘？", "能缘 所缘 见分 相分"),
    ("什么是能熏与所熏？", "能熏 所熏 熏习"),
    ("什么是能依与所依？", "能依 所依 识"),
    # 修行实践
    ("如何观察五蕴皆空？", "五蕴 空 唯识观"),
    ("如何对治我执？", "我执 对治 末那识"),
    ("如何对治法执？", "法执 对治 遍计所执"),
    ("唯识学的止观修法？", "唯识 止观 禅定"),
    ("如何在日常生活中运用唯识？", "唯识 日常 修行"),
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
    rag_context = call_rag(rag_query, top_k=3)

    if rag_context and len(rag_context) > 100:
        prompt = f"""你是一位精通唯识学的佛教法师，法号"善知识"。请根据以下经论原文，回答问题。

经论原文：
{rag_context[:3000]}

问题：{question}

要求：
1. 以"阿弥陀佛"开头
2. 自称"贫僧"
3. 引用经论时注明出处
4. 解释要通俗易懂但不失准确
5. 控制在200-400字
6. 不要编造不存在的经文"""
    else:
        prompt = f"""你是一位精通唯识学的佛教法师，法号"善知识"。请回答以下唯识学问题。

问题：{question}

要求：
1. 以"阿弥陀佛"开头
2. 自称"贫僧"
3. 基于《成唯识论》《唯识三十颂》《瑜伽师地论》等唯识经论回答
4. 解释要通俗易懂但不失准确
5. 控制在200-400字"""

    return call_qwen(prompt)


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

    qa_pairs = list(WEISHI_QA_2)
    random.shuffle(qa_pairs)

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

            if generated % batch_size == 0:
                save_data(existing)
                print(f"    [Saved {len(existing)} entries]")
        else:
            print(f"    -> Failed")

        time.sleep(1)

    save_data(existing)

    final_counts = Counter(d["category"] for d in existing)
    print(f"\nDone! Generated {generated} weishi entries")
    print(f"Final distribution: {dict(final_counts)}")
    print(f"Total: {len(existing)}")


if __name__ == "__main__":
    main()
