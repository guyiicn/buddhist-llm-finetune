#!/usr/bin/env python3
"""
Buddhist Sutra Knowledge Model Inference Script
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model: str, adapter_path: str):
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def chat(model, tokenizer, question: str, system_prompt: str = None) -> str:
    if system_prompt is None:
        system_prompt = "你是一位精通佛学的学者，擅长用通俗易懂的语言解释佛教经典。"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return response


def main():
    parser = argparse.ArgumentParser(
        description="Buddhist Sutra Knowledge Model Inference"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="./saves/qwen2.5-3b-buddhist-lora",
        help="LoRA adapter path",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask (if not provided, enters interactive mode)",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )

    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model, args.adapter_path)
    print("\nModel loaded successfully!\n")

    if args.question:
        print(f"Q: {args.question}")
        response = chat(model, tokenizer, args.question)
        print(f"A: {response}")
    elif args.interactive:
        print("=" * 60)
        print("Buddhist Sutra Knowledge Q&A (type 'quit' to exit)")
        print("=" * 60)

        while True:
            question = input("\nYou: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if not question:
                continue

            response = chat(model, tokenizer, question)
            print(f"\nAssistant: {response}")
    else:
        print("Please provide --question or use --interactive mode")
        print("Example: python inference.py --question '什么是四圣谛？'")
        print("Example: python inference.py --interactive")


if __name__ == "__main__":
    main()
