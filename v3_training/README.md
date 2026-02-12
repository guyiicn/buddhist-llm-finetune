# 善知识模型 v3 训练数据生成

## 项目背景

基于 v2 模型基准测试发现的三个关键问题，生成专项训练数据：

1. **幻觉控制** - 模型面对不存在的佛经内容会编造（需"不确定"表达数据）
2. **边界拒答** - 60%非佛学问题未能拒绝（需更强边界数据）
3. **身份丢失** - 完全丢失"善知识"身份和开经偈（需强化身份数据）
4. **唯识学盲区** - 八识、三性等回答不佳（需专项数据）

## 数据生成计划

### 目标数据量

| 类别 | 目标数量 | 实际数量 | 状态 |
|------|----------|----------|------|
| hallucination_defense | 300 | 300 | ✅ 完成 |
| boundary | 500 | 500 | ✅ 完成 |
| identity | 500 | 500 | ✅ 完成 |
| weishi | 200 | 200 | ✅ 完成 |
| **合计** | **1500** | **1500** | ✅ 完成 |

### 数据生成方法

#### 1. 幻觉控制数据 (hallucination_defense)
- **方法**: 构造不存在的佛经名、品名、术语，训练模型回答"我不确定"
- **来源**: 手工设计 + qwen-plus API 生成
- **示例**: 
  - Q: "《大乘无量寿金刚宝忏》是哪部佛经的第七品？"
  - A: "我不确定。目前佛教大藏经及主流目录中未见此经名..."

#### 2. 边界拒答数据 (boundary)
- **方法**: 覆盖12类非佛学问题（科技、医疗、金融、法律、情感、职业、教育、娱乐、生活、政治、伪佛教、迷信）
- **来源**: 模板生成，无需API调用
- **脚本**: `gen_boundary.py`
- **示例**:
  - Q: "股票怎么开户？"
  - A: "阿弥陀佛，贫僧专研佛教典籍，金融理财非吾所能答也..."

#### 3. 身份强化数据 (identity)
- **方法**: 训练模型在被问及身份时，以开经偈开头，自称"善知识"
- **来源**: 模板生成，无需API调用
- **脚本**: `gen_identity.py`
- **开经偈**:
  ```
  无上甚深微妙法，百千万劫难遭遇。
  我今见闻得受持，愿解如来真实义。
  ```

#### 4. 唯识学专项数据 (weishi)
- **方法**: 使用 RAG (CBETA佛经知识库) + qwen-plus API 生成
- **来源**: 调用 http://192.168.50.12:8000/search 获取经论原文，再用 qwen-plus 生成回答
- **脚本**: `gen_weishi.py`, `gen_weishi2.py`
- **覆盖主题**:
  - 八识详解（眼耳鼻舌身意末那阿赖耶）
  - 三性三无性
  - 种子与熏习
  - 转识成智
  - 五位百法
  - 唯识核心概念
  - 唯识经论与祖师

## 文件说明

```
buddhist-v3-data/
├── README.md                 # 本文档
├── gen_boundary.py           # 边界拒答数据生成脚本
├── gen_identity.py           # 身份强化数据生成脚本
├── gen_weishi.py             # 唯识学数据生成脚本 (batch 1)
├── gen_weishi2.py            # 唯识学数据生成脚本 (batch 2)
└── v3_training_data.json     # 生成的训练数据 (1500条)
```

## 数据格式

```json
{
  "instruction": "问题",
  "input": "",
  "output": "回答",
  "category": "hallucination_defense|boundary|identity|weishi"
}
```

## API 配置

```bash
# qwen-plus API
QWEN_API_KEY=sk-8cdb59cb607348ff9ef65478d24d635b
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# CBETA RAG 服务
RAG_URL=http://192.168.50.12:8000/search
```

## 使用方法

### 生成边界数据
```bash
python gen_boundary.py
```

### 生成身份数据
```bash
python gen_identity.py
```

### 生成唯识数据（需要API和RAG服务）
```bash
python gen_weishi.py
python gen_weishi2.py
```

## v3 训练

数据已合并到 192.168.50.12 服务器：
- 训练数据: `~/code/buddhist-72b-distill/data/train/buddhist_v3_train.json` (12285条)
- 验证数据: `~/code/buddhist-72b-distill/data/train/buddhist_v3_val.json` (568条)

训练命令:
```bash
cd ~/code/buddhist-72b-distill
nohup ./run_train_v3.sh > training_v3.log 2>&1 &
```

## 数据分布统计

v3 训练数据总计 12285 条:
- general (v2原数据): 10785
- identity (身份强化): 500
- boundary (边界拒答): 500
- hallucination_defense (幻觉控制): 300
- weishi (唯识学专项): 200
