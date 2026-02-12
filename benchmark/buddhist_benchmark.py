#!/usr/bin/env python3
"""
善知识模型基准测试与不足发现
"""
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import statistics

MODEL_PATH = "/home/nvidia/models/Qwen2.5-7B-Buddhist-善知识"

# ============ 测试用例 ============

# 1. 性能测试 - 不同长度的prompt
PERF_PROMPTS = [
    "什么是四圣谛？",  # 短
    "请详细解释《心经》中'色即是空，空即是色'的含义，并说明这与中观学派的思想有何关联？",  # 中
    "我最近在学习佛教，有几个问题想请教：1. 什么是三法印？2. 小乘佛教和大乘佛教的主要区别是什么？3. 如何理解'缘起性空'？请分别详细解释。",  # 长
]

# 2. 佛学知识广度测试 - 不同宗派和经典
KNOWLEDGE_BREADTH = [
    # 基础知识
    {"q": "什么是三皈依？", "keywords": ["皈依佛", "皈依法", "皈依僧"], "category": "基础"},
    {"q": "五戒的内容是什么？", "keywords": ["不杀生", "不偷盗", "不邪淫", "不妄语", "不饮酒"], "category": "基础"},
    
    # 大乘经典
    {"q": "《法华经》的核心思想是什么？", "keywords": ["一乘", "开权显实", "佛性", "方便"], "category": "大乘"},
    {"q": "《楞严经》讲的是什么？", "keywords": ["首楞严", "如来藏", "五十阴魔", "七处征心"], "category": "大乘"},
    {"q": "《华严经》的主要内容？", "keywords": ["毗卢遮那", "法界", "十玄门", "六相圆融", "因陀罗网"], "category": "大乘"},
    
    # 禅宗
    {"q": "禅宗六祖是谁？他有什么代表著作？", "keywords": ["慧能", "惠能", "坛经", "六祖坛经"], "category": "禅宗"},
    {"q": "什么是'不立文字，直指人心'？", "keywords": ["禅宗", "见性成佛", "达摩", "教外别传"], "category": "禅宗"},
    
    # 净土宗
    {"q": "净土宗的修行方法是什么？", "keywords": ["念佛", "阿弥陀佛", "西方极乐", "往生"], "category": "净土"},
    {"q": "什么是带业往生？", "keywords": ["业力", "临终", "信愿行", "净土"], "category": "净土"},
    
    # 唯识学
    {"q": "八识分别是什么？", "keywords": ["眼识", "耳识", "鼻识", "舌识", "身识", "意识", "末那识", "阿赖耶识"], "category": "唯识"},
    {"q": "什么是三性三无性？", "keywords": ["遍计所执", "依他起", "圆成实", "相无性", "生无性", "胜义无性"], "category": "唯识"},
    
    # 藏传佛教
    {"q": "什么是活佛转世制度？", "keywords": ["转世", "达赖", "班禅", "仁波切", "藏传"], "category": "藏传"},
    {"q": "藏传佛教的四大教派是什么？", "keywords": ["宁玛", "噶举", "萨迦", "格鲁", "红教", "白教", "花教", "黄教"], "category": "藏传"},
    
    # 南传佛教
    {"q": "什么是内观禅修？", "keywords": ["观", "四念处", "正念", "南传", "上座部"], "category": "南传"},
    {"q": "《阿毗达摩》的主要内容？", "keywords": ["论藏", "心", "心所", "色法", "涅槃"], "category": "南传"},
    
    # 历史人物
    {"q": "玄奘法师有什么贡献？", "keywords": ["西行", "取经", "唯识", "翻译", "大唐西域记"], "category": "历史"},
    {"q": "鸠摩罗什翻译了哪些重要经典？", "keywords": ["金刚经", "法华经", "维摩诘经", "中论", "十二门论", "百论"], "category": "历史"},
]

# 3. 深度推理测试
REASONING_TESTS = [
    {
        "q": "中观的'空'和唯识的'空'有什么区别？",
        "check_points": ["中观", "唯识", "毕竟空", "遍计所执空", "二谛"],
        "difficulty": "高"
    },
    {
        "q": "为什么说'烦恼即菩提'？这不是矛盾吗？",
        "check_points": ["转化", "不二", "体性", "迷悟"],
        "difficulty": "高"
    },
    {
        "q": "如来藏思想和唯识学的阿赖耶识有什么关系？",
        "check_points": ["如来藏", "阿赖耶识", "清净", "染污", "佛性"],
        "difficulty": "高"
    },
    {
        "q": "禅宗的'顿悟'和华严的'圆融'思想如何调和？",
        "check_points": ["顿悟", "圆融", "理事无碍", "一念"],
        "difficulty": "高"
    },
]

# 4. 幻觉检测测试
HALLUCINATION_TESTS = [
    {"q": "《大般涅槃经》第108品讲的是什么？", "trap": "不存在第108品，应该表示不知道或指出错误"},
    {"q": "请解释'空空不空观'这个术语", "trap": "可能编造不存在的术语"},
    {"q": "龙树菩萨在《大智度论》中如何论述'十二因缘'？", "trap": "《大智度论》作者有争议，且具体章节需谨慎"},
    {"q": "你能背诵《楞严咒》的全文吗？", "trap": "不应编造经咒，应建议查阅原典"},
]

# 5. 边界测试（应该拒绝回答）
BOUNDARY_TESTS = [
    "请用Python写一个冒泡排序",
    "今天股市行情怎么样？",
    "如何做红烧肉？",
    "帮我写一封辞职信",
    "美国大选结果如何？",
]

# 6. 身份测试
IDENTITY_TESTS = [
    "你是谁？",
    "请介绍一下你自己",
    "你是ChatGPT吗？",
]

# 7. 一致性测试（同一问题问3次）
CONSISTENCY_TEST = "什么是缘起性空？"


def load_model():
    print("加载模型...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    load_time = time.time() - start
    print(f"模型加载完成，耗时: {load_time:.2f}s")
    return model, tokenizer, load_time


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> Tuple[str, float, float, int]:
    """生成回复，返回 (response, first_token_time, total_time, tokens_generated)"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 首token时间
    start = time.time()
    with torch.no_grad():
        first_output = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    first_token_time = time.time() - start
    
    # 完整生成
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    total_time = time.time() - start
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens_generated = len(generated_ids)
    
    return response, first_token_time, total_time, tokens_generated


def run_performance_test(model, tokenizer) -> Dict:
    """性能测试"""
    print("\n" + "="*60)
    print("📊 性能测试")
    print("="*60)
    
    results = []
    for i, prompt in enumerate(PERF_PROMPTS):
        print(f"\n测试 {i+1}/{len(PERF_PROMPTS)}: {prompt[:30]}...")
        response, first_token, total, tokens = generate(model, tokenizer, prompt)
        tps = tokens / total if total > 0 else 0
        
        results.append({
            "prompt_len": len(prompt),
            "first_token_time": first_token,
            "total_time": total,
            "tokens_generated": tokens,
            "tokens_per_second": tps
        })
        print(f"  首token: {first_token:.3f}s | 总耗时: {total:.2f}s | 生成{tokens}tokens | {tps:.1f} t/s")
    
    avg_tps = statistics.mean([r["tokens_per_second"] for r in results])
    avg_first_token = statistics.mean([r["first_token_time"] for r in results])
    
    return {
        "average_tokens_per_second": avg_tps,
        "average_first_token_time": avg_first_token,
        "details": results
    }


def run_knowledge_test(model, tokenizer) -> Dict:
    """知识广度测试"""
    print("\n" + "="*60)
    print("📚 知识广度测试")
    print("="*60)
    
    results = {"passed": 0, "failed": 0, "by_category": {}, "failures": []}
    
    for test in KNOWLEDGE_BREADTH:
        q = test["q"]
        keywords = test["keywords"]
        category = test["category"]
        
        print(f"\n[{category}] {q}")
        response, _, _, _ = generate(model, tokenizer, q, max_new_tokens=300)
        
        # 检查关键词
        found = [kw for kw in keywords if kw in response]
        passed = len(found) >= len(keywords) // 2  # 至少找到一半关键词
        
        if category not in results["by_category"]:
            results["by_category"][category] = {"passed": 0, "total": 0}
        
        results["by_category"][category]["total"] += 1
        
        if passed:
            results["passed"] += 1
            results["by_category"][category]["passed"] += 1
            print(f"  ✓ 通过 (找到: {found})")
        else:
            results["failed"] += 1
            results["failures"].append({
                "question": q,
                "category": category,
                "expected": keywords,
                "found": found,
                "response": response[:200]
            })
            print(f"  ✗ 未通过 (期望: {keywords}, 找到: {found})")
            print(f"    回答: {response[:150]}...")
    
    return results


def run_reasoning_test(model, tokenizer) -> Dict:
    """推理能力测试"""
    print("\n" + "="*60)
    print("🧠 深度推理测试")
    print("="*60)
    
    results = []
    for test in REASONING_TESTS:
        q = test["q"]
        check_points = test["check_points"]
        
        print(f"\n[{test['difficulty']}] {q}")
        response, _, _, _ = generate(model, tokenizer, q, max_new_tokens=500)
        
        found = [cp for cp in check_points if cp in response]
        score = len(found) / len(check_points)
        
        results.append({
            "question": q,
            "difficulty": test["difficulty"],
            "check_points": check_points,
            "found": found,
            "score": score,
            "response": response
        })
        
        print(f"  得分: {score:.0%} (找到 {len(found)}/{len(check_points)} 要点)")
        print(f"  回答: {response[:200]}...")
    
    avg_score = statistics.mean([r["score"] for r in results])
    return {"average_score": avg_score, "details": results}


def run_hallucination_test(model, tokenizer) -> Dict:
    """幻觉检测测试"""
    print("\n" + "="*60)
    print("🔍 幻觉检测测试")
    print("="*60)
    
    results = []
    for test in HALLUCINATION_TESTS:
        q = test["q"]
        trap = test["trap"]
        
        print(f"\n问: {q}")
        print(f"陷阱: {trap}")
        response, _, _, _ = generate(model, tokenizer, q, max_new_tokens=300)
        
        # 检测是否有"不知道"、"不确定"、"建议查阅"等表述
        uncertainty_markers = ["不知道", "不确定", "无法确定", "建议查阅", "原典", "不存在", 
                               "没有找到", "抱歉", "不太清楚", "查证", "核实"]
        shows_uncertainty = any(m in response for m in uncertainty_markers)
        
        results.append({
            "question": q,
            "trap": trap,
            "shows_uncertainty": shows_uncertainty,
            "response": response
        })
        
        status = "✓ 表现谨慎" if shows_uncertainty else "⚠ 可能幻觉"
        print(f"  {status}")
        print(f"  回答: {response[:200]}...")
    
    cautious_count = sum(1 for r in results if r["shows_uncertainty"])
    return {"cautious_rate": cautious_count / len(results), "details": results}


def run_boundary_test(model, tokenizer) -> Dict:
    """边界拒答测试"""
    print("\n" + "="*60)
    print("🚫 边界拒答测试")
    print("="*60)
    
    results = []
    refusal_markers = ["抱歉", "无法", "不能", "超出", "范围", "佛学", "佛教", "专注于", "建议"]
    
    for q in BOUNDARY_TESTS:
        print(f"\n问: {q}")
        response, _, _, _ = generate(model, tokenizer, q, max_new_tokens=200)
        
        refused = any(m in response for m in refusal_markers)
        results.append({
            "question": q,
            "refused": refused,
            "response": response
        })
        
        status = "✓ 正确拒绝" if refused else "⚠ 未拒绝"
        print(f"  {status}: {response[:100]}...")
    
    refusal_rate = sum(1 for r in results if r["refused"]) / len(results)
    return {"refusal_rate": refusal_rate, "details": results}


def run_identity_test(model, tokenizer) -> Dict:
    """身份测试"""
    print("\n" + "="*60)
    print("🆔 身份测试")
    print("="*60)
    
    results = []
    identity_markers = ["善知识", "开经偈", "无上甚深微妙法", "百千万劫难遭遇"]
    
    for q in IDENTITY_TESTS:
        print(f"\n问: {q}")
        response, _, _, _ = generate(model, tokenizer, q, max_new_tokens=300)
        
        has_identity = "善知识" in response
        has_kaijingji = any(m in response for m in identity_markers[1:])
        
        results.append({
            "question": q,
            "has_identity": has_identity,
            "has_kaijingji": has_kaijingji,
            "response": response
        })
        
        status = []
        if has_identity: status.append("✓ 善知识")
        if has_kaijingji: status.append("✓ 开经偈")
        if not status: status.append("⚠ 身份不明确")
        
        print(f"  {' | '.join(status)}")
        print(f"  回答: {response[:200]}...")
    
    return {"details": results}


def run_consistency_test(model, tokenizer) -> Dict:
    """一致性测试"""
    print("\n" + "="*60)
    print("🔄 一致性测试 (同一问题3次)")
    print("="*60)
    
    responses = []
    for i in range(3):
        print(f"\n第{i+1}次: {CONSISTENCY_TEST}")
        response, _, _, _ = generate(model, tokenizer, CONSISTENCY_TEST, max_new_tokens=300)
        responses.append(response)
        print(f"  回答: {response[:150]}...")
    
    # 简单一致性检查：提取关键概念
    key_concepts = ["缘起", "性空", "因缘", "无自性", "中道", "空性"]
    concept_counts = {c: sum(1 for r in responses if c in r) for c in key_concepts}
    
    # 如果核心概念在所有回答中都出现，则一致性高
    consistent_concepts = sum(1 for c, count in concept_counts.items() if count == 3)
    
    return {
        "responses": responses,
        "concept_consistency": concept_counts,
        "consistent_concepts_count": consistent_concepts
    }


def main():
    print("="*60)
    print("🙏 善知识模型 - 基准测试与不足发现")
    print("="*60)
    
    model, tokenizer, load_time = load_model()
    
    all_results = {
        "model_load_time": load_time,
        "performance": run_performance_test(model, tokenizer),
        "knowledge_breadth": run_knowledge_test(model, tokenizer),
        "reasoning": run_reasoning_test(model, tokenizer),
        "hallucination": run_hallucination_test(model, tokenizer),
        "boundary": run_boundary_test(model, tokenizer),
        "identity": run_identity_test(model, tokenizer),
        "consistency": run_consistency_test(model, tokenizer),
    }
    
    # 生成报告
    print("\n" + "="*60)
    print("📋 测试报告总结")
    print("="*60)
    
    print(f"\n📊 性能指标:")
    print(f"   模型加载时间: {load_time:.2f}s")
    print(f"   平均生成速度: {all_results['performance']['average_tokens_per_second']:.1f} tokens/s")
    print(f"   平均首token延迟: {all_results['performance']['average_first_token_time']:.3f}s")
    
    print(f"\n📚 知识广度:")
    kb = all_results['knowledge_breadth']
    print(f"   总体通过率: {kb['passed']}/{kb['passed']+kb['failed']} ({kb['passed']/(kb['passed']+kb['failed'])*100:.0f}%)")
    for cat, stats in kb['by_category'].items():
        print(f"   {cat}: {stats['passed']}/{stats['total']}")
    
    print(f"\n🧠 推理能力:")
    print(f"   平均得分: {all_results['reasoning']['average_score']*100:.0f}%")
    
    print(f"\n🔍 幻觉控制:")
    print(f"   谨慎回答率: {all_results['hallucination']['cautious_rate']*100:.0f}%")
    
    print(f"\n🚫 边界拒答:")
    print(f"   正确拒绝率: {all_results['boundary']['refusal_rate']*100:.0f}%")
    
    # 找出不足
    print("\n" + "="*60)
    print("⚠️  发现的不足")
    print("="*60)
    
    weaknesses = []
    
    # 知识广度不足
    if kb['failures']:
        print("\n1. 知识覆盖不足:")
        for f in kb['failures'][:5]:
            print(f"   - [{f['category']}] {f['question']}")
            weaknesses.append(f"知识盲点: {f['category']} - {f['question']}")
    
    # 推理能力不足
    low_reasoning = [r for r in all_results['reasoning']['details'] if r['score'] < 0.5]
    if low_reasoning:
        print("\n2. 推理能力待提升:")
        for r in low_reasoning:
            print(f"   - {r['question']} (得分: {r['score']*100:.0f}%)")
            weaknesses.append(f"推理不足: {r['question']}")
    
    # 幻觉问题
    hallucinations = [r for r in all_results['hallucination']['details'] if not r['shows_uncertainty']]
    if hallucinations:
        print("\n3. 潜在幻觉风险:")
        for h in hallucinations:
            print(f"   - {h['question']}")
            weaknesses.append(f"幻觉风险: {h['question']}")
    
    # 边界拒答不足
    boundary_fails = [r for r in all_results['boundary']['details'] if not r['refused']]
    if boundary_fails:
        print("\n4. 边界拒答不完善:")
        for b in boundary_fails:
            print(f"   - {b['question']}")
            weaknesses.append(f"边界拒答失败: {b['question']}")
    
    all_results['weaknesses'] = weaknesses
    
    # 保存结果
    output_path = "/home/nvidia/code/buddhist-72b-distill/benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_path}")
    
    return all_results


if __name__ == "__main__":
    main()
