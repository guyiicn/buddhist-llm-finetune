# 善知识 32B (v5) 训练数据

**生成日期**: 2026-02-21  
**目标模型**: Qwen2.5-32B-Instruct  
**数据格式**: instruction/output pairs (jsonl)

## 数据统计

| 类别 | 条数 | 说明 |
|------|------|------|
| **合并训练数据** | 24,553 | train_all.jsonl (去重后) |
| | | |
| **原始分类数据** | | |
| boundary_v5 | 1,000 | 边界控制（医疗/技术/金融/政治/迷信） |
| distilled | 10,000 | 通用问答补充 |
| dpo | 2,000 | DPO偏好对 |
| guru_reliance | 493 | 上师依止专项 |
| identity_v5 | 496 | 身份认证扩展 |
| koans | 500 | 公案禅话 |
| modern_masters | 798 | 现代高僧问答 |
| multiturn | 2,967 | 多轮对话（展开为3,742对） |
| practice_guide | 1,468 | 修行实践指导 |
| schools | 1,996 | 宗派专项（禅宗/净土/唯识/天台/华严/律宗） |
| sutra_deep | 1,984 | 经典深度解读 |
| terms | 902 | 佛学名词解释 |
| **原始合计** | 24,604 | raw/*.jsonl |

## 数据生成方法

### P0 核心数据 (4,743条)
- **上师依止** (500) + **现代高僧** (800) + **经典深度** (2,000) + **修行实践** (1,500)
- 方法: RAG (佛经/高僧传/灯录) + Qwen-plus API 生成

### P1 扩展数据 (6,459条)
- **边界控制** (1,000) + **身份认证** (500) + **宗派专项** (2,000) + **多轮对话** (3,000)
- 方法: 模板生成 (boundary/identity) + RAG + LLM (schools/multiturn)

### P2 补充数据 (13,402条)
- **名词解释** (1,000) + **公案禅话** (500) + **通用问答** (10,000) + **DPO偏好** (2,000)
- 方法: RAG + 佛学辞典 + Qwen-plus 蒸馏

## 数据格式

### merged/train_all.jsonl
```json
{
  "instruction": "问题",
  "output": "回答",
  "source": "v5-distilled",
  "category": "distilled"
}
```

### raw/*.jsonl
- **标准QA**: `{"instruction": "...", "output": "...", "category": "..."}`
- **多轮对话**: `{"conversations": [{"role": "user", "content": "..."}, ...], "category": "multiturn"}`
- **DPO偏好**: `{"question": "...", "good_response": "...", "bad_response": "...", "category": "dpo"}`

## 数据特点

- ✅ **去重**: 24,604条原始 → 24,553条合并（去除4,724重复）
- ✅ **RAG增强**: 所有佛学内容都基于CBETA RAG检索，确保准确性
- ✅ **引经据典**: 回答包含佛经原文引用
- ✅ **多样化**: 覆盖禅宗/净土/唯识/天台/华严/律宗等六大宗派
- ✅ **边界控制**: 医疗/技术/金融/政治/迷信领域拒答训练
- ✅ **多轮对话**: 支持递进式佛学探讨
- ✅ **DPO准备**: 包含2,000组好/坏回答对比

## 使用方法

```bash
# 直接使用合并数据
python train.py --data_path ./32b/train_all.jsonl --model_name Qwen/Qwen2.5-32B-Instruct

# 或使用原始分类数据
python train.py --data_path ./32b/raw/ --model_name Qwen/Qwen2.5-32B-Instruct
```

## 生成脚本

数据生成脚本位于: `~/code/buddhist-72b-distill/v5_training/scripts/`
- `gen_p0_all.py` - P0 核心数据
- `gen_p1_all.py` - P1 扩展数据
- `gen_p2_all.py` - P2 补充数据
- `gen_boundary_expand.py` - 边界控制扩充
- `merge_data.py` - 数据合并清洗

## 配置

- **RAG API**: http://localhost:8000/v1/search (Qdrant向量库, 205,485条佛学内容)
- **LLM API**: Qwen-plus (dashscope.aliyuncs.com)
- **Embedding**: bge-m3 (Ollama)
