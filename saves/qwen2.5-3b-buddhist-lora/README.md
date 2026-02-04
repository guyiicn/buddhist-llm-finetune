---
library_name: peft
license: other
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- base_model:adapter:Qwen/Qwen2.5-3B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen2.5-3b-buddhist-lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen2.5-3b-buddhist-lora

This model is a fine-tuned version of [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) on the buddhist_train dataset.
It achieves the following results on the evaluation set:
- Loss: 1.9432

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 2
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.6.0+cu124
- Datasets 4.0.0
- Tokenizers 0.22.2