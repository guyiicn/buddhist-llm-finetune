# Buddhist Sutra Knowledge LLM Fine-tuning

基于 Qwen2.5 的佛经知识大模型 LoRA 微调项目。从 CBETA 电子佛典提取 18 部经典原文，使用 Qwen API 生成高质量问答数据，训练专业佛学知识问答模型。

## 项目特点

- 📚 **18 部经典原文** — 佛经十三经 + 5 部重要论典，共 71 万字
- 🤖 **4,386 条高质量 QA** — 使用 Qwen-plus 自动生成，覆盖义理、翻译、应用、辨析等多角度
- ⚡ **LLaMA-Factory LoRA 微调** — 支持 7B/32B 模型，高效训练
- 🎯 **专业佛学问答能力** — 答案平均 387 字，引用原文，通俗易懂

## 数据集统计

| 指标 | 数值 |
|------|------|
| **总 QA 数量** | 4,386 条 |
| **训练集** | 3,947 条 (90%) |
| **验证集** | 439 条 (10%) |
| **覆盖经典** | 18 部 (13经 + 5论) |
| **原文总量** | 713,168 字 |
| **答案平均长度** | 387 字 (范围 270-546) |
| **问题平均长度** | 71 字 (范围 25-171) |

### 数据分布

| 经典 | QA 数量 | 经典 | QA 数量 |
|------|---------|------|---------|
| 成唯识论 | 654 | 大智度论 | 330 |
| 妙法莲华经 | 471 | 瑜伽师地论 | 318 |
| 楞严经 | 465 | 楞伽经 | 285 |
| 杂阿含经 | 393 | 解深密经 | 279 |
| 中论 | 369 | 维摩诘经 | 213 |
| 无量寿经 | 138 | 坛经 | 123 |
| 圆觉经 | 96 | 起信论 | 87 |
| 观经 | 78 | 金刚经 | 45 |
| 阿弥陀经 | 18 | 心经 | 12 |

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/guyiicn/buddhist-llm-finetune.git
cd buddhist-llm-finetune
```

### 2. 安装依赖

```bash
conda create -n buddhist-llm python=3.10 -y
conda activate buddhist-llm

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install llamafactory
```

### 3. 开始训练

数据集已经准备好，可以直接训练：

```bash
# LoRA 微调 Qwen2.5-7B-Instruct
python -m llamafactory.cli train train_config.yaml
```

### 4. 模型推理

```bash
python inference.py --question "什么是四圣谛？"
```

## 项目结构

```
buddhist-llm-finetune/
├── README.md                 # 项目说明
├── requirements.txt          # Python 依赖
├── config.yaml              # 项目配置 (经典列表、QA生成参数)
├── train_config.yaml        # LLaMA-Factory 训练配置
│
├── raw/                     # 原始 CBETA 经文 (18 部)
│   ├── T0235_金刚经.txt      # 6,814 字
│   ├── T0251_心经.txt        # 1,351 字
│   ├── T0262_法华经.txt      # 79,810 字
│   ├── T0360_无量寿经.txt
│   ├── T0365_观经.txt
│   ├── T0366_阿弥陀经.txt
│   ├── T0475_维摩诘经.txt
│   ├── T0670_楞伽经.txt
│   ├── T0676_解深密经.txt
│   ├── T0842_圆觉经.txt
│   ├── T0945_楞严经.txt      # 79,911 字
│   ├── T2008_坛经.txt
│   ├── T0099_杂阿含.txt      # 59,935 字 (节选)
│   ├── T1666_起信论.txt
│   ├── T1564_中论.txt
│   ├── T1579_瑜伽论.txt      # 49,987 字 (节选)
│   ├── T1585_成唯识论.txt    # 102,246 字
│   ├── T1509_大智度论.txt    # 49,971 字 (节选)
│   └── metadata.json
│
├── cleaned/                 # 清洗后的分块数据
│   └── chunks.json          # 1,569 个文本块 (每块约 500 字)
│
├── seeds/                   # 种子数据 (12 条人工标注)
│   └── seed_qa.json
│
├── qa_pairs/               # 生成的 QA 数据
│   └── buddhist_qa.json     # 4,374 条 (Qwen-plus 生成)
│
├── output/                 # 最终训练数据 (Alpaca 格式)
│   ├── buddhist_train_alpaca.json   # 3,947 条
│   ├── buddhist_val_alpaca.json     # 439 条
│   ├── dataset_info.json            # LLaMA-Factory 配置
│   └── dataset_stats.json
│
├── scripts/                 # 数据处理脚本
│   ├── 1_fetch_sutras.py   # 从 CBETA 提取经文
│   ├── 2_clean_text.py     # 文本清洗分块
│   ├── 3_generate_qa.py    # QA 生成 (并发版)
│   └── 4_merge_dataset.py  # 数据集合并
│
└── saves/                  # 训练输出 (LoRA 权重)
```

## 收录经典

### 佛经十三经 (除华严经)

| 经名 | CBETA 编号 | 类别 | 字数 |
|------|------------|------|------|
| 金刚般若波罗蜜经 | T0235 | 般若部 | 6,814 |
| 般若波罗蜜多心经 | T0251 | 般若部 | 1,351 |
| 妙法莲华经 | T0262 | 法华部 | 79,810 |
| 佛说无量寿经 | T0360 | 净土部 | 21,643 |
| 观无量寿佛经 | T0365 | 净土部 | 9,634 |
| 佛说阿弥陀经 | T0366 | 净土部 | 2,587 |
| 维摩诘所说经 | T0475 | 经集部 | 34,510 |
| 楞伽阿跋多罗宝经 | T0670 | 经集部 | 57,222 |
| 解深密经 | T0676 | 经集部 | 39,273 |
| 大方广圆觉修多罗了义经 | T0842 | 经集部 | 14,406 |
| 大佛顶首楞严经 | T0945 | 经集部 | 79,911 |
| 六祖大师法宝坛经 | T2008 | 禅宗 | 33,494 |
| 杂阿含经 | T0099 | 阿含部 | 59,935* |

### 五部重要论典

| 论名 | CBETA 编号 | 作者 | 字数 |
|------|------------|------|------|
| 大乘起信论 | T1666 | 马鸣菩萨 | 13,312 |
| 中论 | T1564 | 龙树菩萨 | 57,062 |
| 瑜伽师地论 | T1579 | 弥勒/无著 | 49,987* |
| 成唯识论 | T1585 | 护法等/玄奘译 | 102,246 |
| 大智度论 | T1509 | 龙树菩萨 | 49,971* |

> *标注为节选，原典过长取前 N 万字。

## 训练配置

```yaml
# train_config.yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target: all

per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 3
bf16: true
gradient_checkpointing: true
```

### 硬件要求

| 模型 | 方法 | 显存需求 | 训练时间 (3947 样本) |
|------|------|----------|----------------------|
| Qwen2.5-7B | LoRA BF16 | ~20 GB | ~3-5 小时 |
| Qwen2.5-7B | QLoRA 4-bit | ~12 GB | ~4-6 小时 |
| Qwen2.5-32B | QLoRA 4-bit | ~45 GB | ~12-20 小时 |

## 使用示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载模型
base_model = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "./saves/qwen2.5-7b-buddhist-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_path)

# 对话
messages = [
    {"role": "system", "content": "你是一位精通佛学的学者。"},
    {"role": "user", "content": "什么是四圣谛？"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)
```

## QA 示例

**问：《中论》如何驳斥"生灭一时俱"的观点？**

> 《中论》以归谬法层层破斥：先假设"灭时生"成立，则"灭时是一有，生时是一有"，即同一刹那中必须同时安立两个实有的"有"——"灭时有"与"生时有"，这违反"一法不二体"的基本理性原则；继而进一步逼问：若真"一时"，则必是"于此阴死，即于此阴生"，意味着同一组五蕴既在死亡又在受生，这显然与经验及教理相违。龙树借此揭示：所谓"灭时生"不过是凡夫将刹那生灭的相似相续错觉为"连续实体"的迷执。其方法论体现典型的中观"破而不立"：不建立任何实有本体或时间模型，而是通过穷尽对方所有可能的逻辑设定，使其自相矛盾...

**问：六祖批评"空心静坐"是邪见，这与传统禅修的"止息妄念"矛盾吗？**

> 这并不矛盾，而是禅宗对修行根本方向的精准勘辨。六祖所破的"空心静坐"，特指以压抑、断灭为手段，刻意追求"百无所思"的枯寂状态，并将此误认为究竟解脱，实则落入"无记空"——一种昏沉、无明、能所双泯却无觉照的死水境界，原文直斥"此一輩人，不可與語"...

## 数据生成流程

1. **提取经文** (`1_fetch_sutras.py`)
   - 从本地 CBETA 文本库读取 18 部经典
   - 去除文件头注释和校勘记
   - 大经节选控制总量

2. **清洗分块** (`2_clean_text.py`)
   - 标准化空格换行
   - 按 500 字切分，段落边界优先
   - 生成 1,569 个文本块

3. **生成 QA** (`3_generate_qa.py`)
   - 5 并发调用 Qwen-plus API
   - 每块生成 3 个多角度 QA
   - 增量保存，支持断点续传

4. **合并数据集** (`4_merge_dataset.py`)
   - 合并种子数据 + 生成数据
   - 去重、90/10 分割
   - 输出 Alpaca 格式

## 许可证

本项目仅供学习研究使用。佛经文本来源于 [CBETA 中华电子佛典协会](https://cbetaonline.dila.edu.tw/)，遵循其开放授权协议。

## 参考资料

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - 高效 LLM 微调框架
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) - 通义千问大模型
- [PEFT](https://github.com/huggingface/peft) - 参数高效微调
- [CBETA 电子佛典](https://cbetaonline.dila.edu.tw/) - 中华电子佛典协会
