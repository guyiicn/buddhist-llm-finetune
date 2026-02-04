# Buddhist Sutra Knowledge LLM Fine-tuning

基于 Qwen2.5 的佛经知识大模型微调项目。使用 LoRA 技术对 Qwen2.5-3B-Instruct 进行微调，使其具备专业的佛学知识问答能力。

## 项目特点

- 📚 收集 18 部经典佛经文本（佛经十三经 + 5 部重要论典）
- 🤖 使用 Qwen API 自动生成高质量 QA 训练数据
- ⚡ 基于 LLaMA-Factory 的 LoRA 高效微调
- 🎯 训练后模型能够准确回答佛学问题

## 环境要求

- Python 3.11+
- CUDA 12.4+
- GPU 显存 >= 12GB（推荐 16GB）
- 约 10GB 磁盘空间

## 快速开始

### 1. 安装依赖

```bash
conda create -n buddhist-llm python=3.11 -y
conda activate buddhist-llm

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 步骤1: 获取佛经原文（从 CBETA 数据库）
python scripts/1_fetch_sutras.py

# 步骤2: 清洗和分块
python scripts/2_clean_text.py

# 步骤3: 生成 QA 数据（需要配置 Qwen API）
# 先在 scripts/3_generate_qa.py 中配置 API key
python scripts/3_generate_qa.py

# 步骤4: 合并数据集
python scripts/4_merge_dataset.py
```

### 3. 模型微调

```bash
# 使用 LLaMA-Factory 进行 LoRA 微调
python -m llamafactory.cli train train_config.yaml
```

### 4. 模型推理

```bash
python inference.py --question "什么是四圣谛？"
```

## 项目结构

```
buddhist_data/
├── README.md                 # 项目说明
├── requirements.txt          # Python 依赖
├── config.yaml              # 项目配置
├── train_config.yaml        # 训练配置
├── dataset_info.json        # LLaMA-Factory 数据集配置
├── inference.py             # 推理脚本
│
├── scripts/                 # 数据处理脚本
│   ├── 1_fetch_sutras.py   # 获取佛经原文
│   ├── 2_clean_text.py     # 文本清洗分块
│   ├── 3_generate_qa.py    # QA 数据生成
│   └── 4_merge_dataset.py  # 数据集合并
│
├── raw/                     # 原始佛经文本
│   ├── T0235_金刚经.txt
│   ├── T0251_心经.txt
│   └── ... (18部经典)
│
├── cleaned/                 # 清洗后的分块数据
│   └── chunks.json
│
├── seeds/                   # 种子数据（人工标注）
│   └── seed_qa.json
│
├── qa_pairs/               # 生成的 QA 数据
│   └── buddhist_qa.json
│
├── output/                 # 最终训练数据
│   ├── buddhist_train_alpaca.json
│   └── buddhist_val_alpaca.json
│
└── saves/                  # 训练输出
    └── qwen2.5-3b-buddhist-lora/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── ...
```

## 收录经典

### 佛经十三经（除华严经）
| 经名 | CBETA 编号 | 类别 |
|------|------------|------|
| 金刚般若波罗蜜经 | T0235 | 般若 |
| 般若波罗蜜多心经 | T0251 | 般若 |
| 妙法莲华经 | T0262 | 法华 |
| 佛说无量寿经 | T0360 | 净土 |
| 观无量寿佛经 | T0365 | 净土 |
| 佛说阿弥陀经 | T0366 | 净土 |
| 维摩诘所说经 | T0475 | 大乘 |
| 楞伽阿跋多罗宝经 | T0670 | 唯识 |
| 解深密经 | T0676 | 唯识 |
| 大方广圆觉修多罗了义经 | T0842 | 禅宗 |
| 大佛顶首楞严经 | T0945 | 禅宗 |
| 六祖大师法宝坛经 | T2008 | 禅宗 |
| 杂阿含经 | T0099 | 阿含 |

### 五部重要论典
| 论名 | CBETA 编号 | 宗派 |
|------|------------|------|
| 大乘起信论 | T1666 | 唯识 |
| 中论 | T1564 | 中观 |
| 瑜伽师地论 | T1579 | 唯识 |
| 成唯识论 | T1585 | 唯识 |
| 大智度论 | T1509 | 般若 |

## 训练配置

```yaml
# train_config.yaml
model_name_or_path: Qwen/Qwen2.5-3B-Instruct
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05

per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 3
bf16: true
gradient_checkpointing: true
```

## 训练结果

| 指标 | 数值 |
|------|------|
| 训练样本 | 56 条 |
| 验证样本 | 7 条 |
| 训练轮次 | 3 epochs |
| 训练时间 | 87 秒 |
| 初始 Loss | 2.24 |
| 最终 Loss | 1.55 |
| 可训练参数 | 59.8M (1.9%) |

## 使用示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载模型
base_model = "Qwen/Qwen2.5-3B-Instruct"
adapter_path = "./saves/qwen2.5-3b-buddhist-lora"

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

## 模型回答示例

**问：什么是四圣谛？**

> 四圣谛是佛陀第一次说法的核心教义，也是佛法的基础。它们分别是：
>
> 1. **苦谛**：生命中存在着生老病死、爱别离、求不得等八苦。这并非指自然界的苦，而是指因贪嗔痴所造业果而带来的身心痛苦。
>
> 2. **集谛**：苦的根源在于无明（无知）和贪瞋（渴求与嗔恨）。无明使我们执着于错误的认知和观念，贪瞋则驱使我们追求虚幻的快乐或逃避真实的痛苦。
>
> 3. **灭谛**：苦是可以被熄灭的。通过彻底断除烦恼和业力，可以达到涅槃——心灵的寂静与解脱状态。
>
> 4. **道谛**：实现灭苦的方法是八正道：正见、正思维、正语、正业、正命、正精进、正念、正定。

## 后续优化方向

1. **扩充数据集**：增加更多 QA 数据到 500+ 条
2. **使用更大模型**：尝试 Qwen2.5-7B 或 14B
3. **QLoRA 量化训练**：减少显存占用
4. **多轮对话微调**：支持连续对话
5. **RLHF 优化**：基于人类反馈强化学习

## 许可证

本项目仅供学习研究使用。佛经文本来源于 [CBETA 中华电子佛典协会](https://cbetaonline.dila.edu.tw/)。

## 参考资料

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5)
- [PEFT](https://github.com/huggingface/peft)
- [CBETA 电子佛典](https://cbetaonline.dila.edu.tw/)
