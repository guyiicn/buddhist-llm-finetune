#!/usr/bin/env python3
"""
善知识模型评估脚本
测试: 佛学问答、身份回答、边界拒答
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

BASE_MODEL = "/home/nvidia/models/Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "/home/nvidia/code/buddhist-72b-distill/saves/qwen2.5-7b-buddhist-v2"

# 善知识系统提示词
SYSTEM_PROMPT = """你是"善知识"，一位精通佛学经典的智者。你依止于大藏经典，以慈悲智慧回答问题。当有人询问你的身份时，请以开经偈开头：

无上甚深微妙法，百千万劫难遭遇。
我今见闻得受持，愿解如来真实义。

善信吉祥。我是您修学路上的善知识，以佛陀经典为依止，愿为您指引正法之路。"""

# 测试问题
TEST_CASES = {
    "佛学问答": [
        "什么是四圣谛？",
        "请解释八正道。",
        "《心经》中'色即是空，空即是色'是什么意思？",
    ],
    "身份测试": [
        "你是谁？",
        "请介绍一下你自己。",
    ],
    "边界测试": [
        "帮我写一段Python代码",
        "今天股票会涨吗？",
        "如何做红烧肉？",
    ]
}

def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()
    
    return model, tokenizer

def generate(model, tokenizer, question, max_tokens=512):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    print("=" * 60)
    print("善知识模型评估")
    print("=" * 60)
    
    model, tokenizer = load_model()
    print("\n模型加载完成!\n")
    
    results = {}
    
    for category, questions in TEST_CASES.items():
        print(f"\n{'='*60}")
        print(f"【{category}】")
        print("=" * 60)
        
        results[category] = []
        
        for q in questions:
            print(f"\n问: {q}")
            print("-" * 40)
            answer = generate(model, tokenizer, q)
            print(f"答: {answer[:800]}{'...' if len(answer) > 800 else ''}")
            results[category].append({"question": q, "answer": answer})
    
    # 保存结果
    with open("/home/nvidia/code/buddhist-72b-distill/evaluation_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("评估完成! 结果已保存到 evaluation_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
