#!/usr/bin/env python3
"""
Buddhist LLM Training v2 - 善知识模型
使用合并后的完整数据集 (11353条)
"""
import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTTrainer
import datetime

MODEL_NAME = "/home/nvidia/models/Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/home/nvidia/code/buddhist-72b-distill/saves/qwen2.5-7b-buddhist-v2"
DATA_DIR = "/home/nvidia/code/buddhist-72b-distill/data/train"

# 善知识系统提示词
SYSTEM_PROMPT = """你是"善知识"，一位精通佛学经典的智者。你依止于大藏经典，以慈悲智慧回答问题。当有人询问你的身份时，请以开经偈开头：

无上甚深微妙法，百千万劫难遭遇。
我今见闻得受持，愿解如来真实义。

善信吉祥。我是您修学路上的善知识，以佛陀经典为依止，愿为您指引正法之路。"""

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_example(example, tokenizer):
    user_content = example["instruction"]
    if example.get("input"):
        user_content += f"\n{example['input']}"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def main():
    start_time = datetime.datetime.now()
    print(f"=== 善知识模型训练 v2 ===")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[2/5] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print("[3/5] Loading data...")
    train_data = load_data(os.path.join(DATA_DIR, "buddhist_full_train.json"))
    val_data = load_data(os.path.join(DATA_DIR, "buddhist_full_val.json"))
    
    print(f"  Formatting {len(train_data)} train examples...")
    train_texts = [format_example(ex, tokenizer) for ex in train_data]
    print(f"  Formatting {len(val_data)} val examples...")
    val_texts = [format_example(ex, tokenizer) for ex in val_data]
    
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    print("[4/5] Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,  # 增大 rank
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 计算总步数: 10785 samples / (batch=1 * grad_accum=8) * epochs=2 ≈ 2696 steps
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,  # 数据量大，2轮即可
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,  # 略微降低学习率
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=50,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        fp16=True,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        optim="adamw_torch",
    )

    print("[5/5] Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=2048,  # 增大序列长度
    )

    trainer.train()

    print("\nSaving model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"\n=== 训练完成 ===")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration}")
    print(f"模型保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
