#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
佛经数据获取脚本
从 CBETA 获取佛经原文
"""

import requests
import json
import yaml
from pathlib import Path
from typing import List, Dict
import re


def load_config() -> Dict:
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    """清洗文本"""
    # 移除注释编号
    text = re.sub(r"\[\d+\]", "", text)
    # 移除括号注释
    text = re.sub(r"【.*?】", "", text)
    text = re.sub(r"（.*?）", "", text)
    # 移除多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_sutra_from_cbeta(sutra_id: str) -> str:
    """
    从 CBETA API 获取佛经
    注意：实际使用时需要替换为真实的 API 端点
    """
    # CBETA 没有直接的 API，这里提供一个模拟接口
    # 实际使用时，可以从其他来源获取文本

    # 示例：返回占位文本
    return f"[佛经 {sutra_id} 的内容]"


def fetch_sutra_from_local(sutra_id: str, sutra_name: str) -> str:
    """
    从本地或硬编码数据获取佛经
    用于演示目的
    """

    # 硬编码一些佛经片段作为示例
    sample_texts = {
        "T0235": {
            "name": "金刚般若波罗蜜经",
            "content": """
如是我闻。一时，佛在舍卫国祇树给孤独园，与大比丘众千二百五十人俱。
尔时，世尊食时，着衣持钵，入舍卫大城乞食。于其城中，次第乞已，还至本处。饭食讫，收衣钵，洗足已，敷座而坐。
尔时，长老须菩提在大众中即从座起，偏袒右肩，右膝着地，合掌恭敬而白佛言：「希有！世尊！如来善护念诸菩萨，善付嘱诸菩萨。」
""",
        },
        "T0251": {
            "name": "般若波罗蜜多心经",
            "content": """
观自在菩萨，行深般若波罗蜜多时，照见五蕴皆空，度一切苦厄。
舍利子，色不异空，空不异色，色即是空，空即是色，受想行识，亦复如是。
舍利子，是诸法空相，不生不灭，不垢不净，不增不减。
""",
        },
        "T0670": {
            "name": "楞伽阿跋多罗宝经",
            "content": """
如是我闻。一时，婆伽婆在大海中，入娑罗多龙王宫殿，与无量菩萨摩诃萨俱，皆得如来灌顶，住诸菩萨所得三昧，其名曰：海月菩萨、解脱月菩萨、普智菩萨、无边智菩萨。
时，婆伽婆在此海中龙王宫殿，现大神力，震动大海，令龙王宫殿及大海中众生命，皆得安隐。
""",
        },
        "T0676": {
            "name": "解深密经",
            "content": """
如是我闻。一时，薄伽梵在解甚深义意，超过诸菩萨三摩地，名为如来出现，与无量大声闻众、菩萨摩诃萨众俱。
尔时，薄伽婆告最胜子曰：「善哉善哉！汝能请问如来如是甚深义，多所安隐一切众生，令得无上正等菩提。」
""",
        },
        "T0475": {
            "name": "维摩诘所说经",
            "content": """
如是我闻。一时，佛在毗耶离庵罗树园，与大比丘众八千人俱，菩萨三万二千人，众所知识。
尔时，毗耶离大城中有长者，名曰维摩诘，已曾供养无量诸佛，深植善本，得无生忍，辩才无碍，游戏神通。
""",
        },
        "T0842": {
            "name": "大方广圆觉修多罗了义经",
            "content": """
如是我闻。一时婆伽婆入于神通大光明藏三昧正受，一切如来光严住持，是诸众生清净觉地。
身心寂灭平等本际，圆满十方，不二随顺，于不二境，现诸净土。
""",
        },
        "T2008": {
            "name": "六祖大师法宝坛经",
            "content": """
时，大师至宝林，韶州韦刺史与官僚入山请师，出于城中大梵寺讲堂，为众开缘说法。
师曰：「善知识！菩提自性，本来清净，但用此心，直了成佛。」
""",
        },
        "T0099": {
            "name": "杂阿含经",
            "content": """
如是我闻。一时，佛住舍卫国祇树给孤独园。
尔时，世尊告诸比丘：「当观色无常。如是观者，则为正观。正观者，则生厌离；厌离者，喜贪尽；喜贪尽者，说心解脱。」
""",
        },
    }

    if sutra_id in sample_texts:
        return sample_texts[sutra_id]["content"]
    else:
        return f"[佛经 {sutra_name} ({sutra_id}) 的内容将在这里]"


def save_raw_text(sutra_data: Dict, config: Dict):
    """保存原始文本到 raw 目录"""
    raw_dir = Path(__file__).parent.parent / "raw"
    raw_dir.mkdir(exist_ok=True)

    for sutra in sutra_data:
        filename = f"{sutra['id']}_{sutra['short_name']}.txt"
        filepath = raw_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {sutra['name']}\n")
            f.write(f"# ID: {sutra['id']}\n")
            f.write(f"# 分类: {sutra.get('category', '未分类')}\n\n")
            f.write(sutra["content"])
            f.write("\n")

        print(f"✓ 已保存: {filename}")


def main():
    print("=" * 60)
    print("佛经数据获取脚本")
    print("=" * 60)

    # 加载配置
    config = load_config()

    # 合并经和论
    all_texts = []

    # 获取经
    for sutra in config["sutras"]:
        print(f"\n正在获取: {sutra['name']} ({sutra['id']})")

        content = fetch_sutra_from_local(sutra["id"], sutra["short_name"])

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
    if "treatises" in config:
        for treatise in config["treatises"]:
            print(f"\n正在获取: {treatise['name']} ({treatise['id']})")

            content = fetch_sutra_from_local(treatise["id"], treatise["short_name"])

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
    print("保存原始文本...")
    save_raw_text(all_texts, config)

    # 保存元数据
    meta_file = Path(__file__).parent.parent / "raw" / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 元数据已保存: metadata.json")
    print(f"\n✓ 总计获取: {len(all_texts)} 部经典")
    print("=" * 60)


if __name__ == "__main__":
    main()
