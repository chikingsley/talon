# Email Signature Extraction Modernization Plan

This document outlines a path to upgrade talon's signature extraction from ~90% accuracy to 99%+ using modern small language models and synthetic data generation.


## The Core Idea

Use Claude or GPT-4 as a "teacher" to label email data, then train a tiny local model (the "student") on that labeled data. The student model runs fast and cheap at scale while maintaining quality.

This is called distillation. Fine-tuning adapts an existing model to your task. Distillation specifically means using a larger model's outputs to train a smaller one. For our purposes they overlap since we'll fine-tune a small model on teacher-generated labels.


## Task Definition

Input: Raw email text
Output: Structured extraction with zones labeled

Zones to identify:
- greeting (Hi John)
- body (main content)
- signature_block (everything after the sign-off)
- quoted_text (previous messages)

From the signature_block, extract structured contact info:
- name
- title
- company
- email
- phone
- social links


## Model Options

Three tiers based on size and capability.

Tier 1 - Tiny (runs anywhere)
- SmolLM2-135M https://huggingface.co/unsloth/SmolLM2-135M
- SmolLM2-360M https://huggingface.co/unsloth/SmolLM2-360M

Tier 2 - Small (fast on your RTX 5070)
- Qwen3-0.6B https://huggingface.co/unsloth/Qwen3-0.6B-unsloth-bnb-4bit
- SmolLM2-1.7B https://huggingface.co/unsloth/SmolLM2-1.7B-Instruct

Tier 3 - Medium (still fits in 12GB)
- SmolLM3-3B https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- Qwen3-1.7B https://huggingface.co/unsloth/Qwen3-1.7B

Note on picoLLM: Their training platform (picoLLM GYM) is enterprise-only. Not accessible for this project. Their inference engine is useful for deployment but we cannot train custom models through them without an enterprise contract.


## Unsloth Resources

Unsloth is the standard for efficient fine-tuning. 2x faster, 70% less VRAM.

Main repo: https://github.com/unslothai/unsloth

Notebooks repo: https://github.com/unslothai/notebooks

Relevant notebooks (swap model name for smaller variants):
- Qwen3 14B conversational: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb
- Qwen2.5 3B GRPO: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb
- Qwen2.5-Coder 1.5B tool calling: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_Coder_(1.5B)-Tool_Calling.ipynb

Documentation for Qwen3: https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune

Key code pattern:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
)
```

Export to GGUF for deployment:
```python
model.save_pretrained_gguf("email-signature-model", tokenizer)
```


## Phase 1 - Quick Validation

Goal: Prove the approach works before investing in full dataset creation.

Steps:
1. Grab 50-100 emails from talon's existing test data
2. Have Claude label them with zone tags and extract contact info
3. Fine-tune SmolLM2-360M or Qwen3-0.6B using Unsloth
4. Compare accuracy against talon's current output

This can be done in a day on your hardware.


## Phase 2 - Synthetic Data Generation

Generate 5-10K labeled examples using Claude as teacher.

Prompt template:
```
Given this email, identify:
1. Which lines are signature vs body (mark each line)
2. Extract any contact information found

Email:
{email_text}

Respond in this JSON format:
{
  "zones": [
    {"line": 1, "text": "Hi John,", "zone": "greeting"},
    {"line": 2, "text": "Thanks for...", "zone": "body"},
    ...
  ],
  "contacts": {
    "name": "...",
    "email": "...",
    "phone": "...",
    "title": "...",
    "company": "..."
  }
}
```

Quality filtering:
- Have Claude review its own outputs for consistency
- Discard examples where labeling is ambiguous
- Mix with any human-validated examples you can gather

Data sources:
- Enron corpus (public): https://www.cs.cmu.edu/~enron/
- Mailgun forge dataset: https://github.com/mailgun/forge
- Generate synthetic emails with varied signature styles


## Phase 3 - Model Training

Use Unsloth on your RTX 5070.

Recommended starting point: Qwen3-0.6B
- Good balance of quality and speed
- Already distilled from larger Qwen models
- 4-bit fits easily in 12GB

Training approach:
- LoRA fine-tuning (only train 1% of parameters)
- 3-5 epochs over your synthetic dataset
- Validate on held-out test set after each epoch

Alternative: SmolLM2-360M if you want maximum speed at inference time.


## Phase 4 - Integration

Add the trained model as an alternative backend to talon.

Keep the existing regex/heuristics as fast path for obvious cases. Use the ML model for ambiguous cases or when higher accuracy is needed.

Deployment options:
- llama.cpp / Ollama for local inference
- ONNX runtime for Python integration
- Direct transformers integration


## Expected Results

Current talon accuracy: ~90% (claimed, needs validation)
2021 research (BiLSTM-CRF): 98% zone prediction
Target with modern approach: 99%+

At 100K emails per day:
- 90% accuracy = 10,000 errors
- 99% accuracy = 1,000 errors

The structured contact extraction is additional value that talon does not currently provide.


## Hardware Notes

Your RTX 5070 (12GB VRAM):
- Can fine-tune any model up to ~3B with 4-bit quantization
- Inference is fast for all listed models
- LoRA training keeps memory usage low

Your 96GB RAM machine:
- Can run larger models if needed
- Useful for batch inference during data generation
- Can handle SmolLM3-3B at full precision if desired


## Distillation vs Fine-tuning Clarification

Fine-tuning: Take a model, train it on task-specific data, model stays same size.

Distillation: Use a large model (teacher) to generate training data or soft labels, then train a smaller model (student) on that data.

What we are doing is both. We use Claude (teacher) to generate labeled data, then fine-tune a small model (student) on that data. This is sometimes called "teaching via data" or synthetic data distillation.

The key insight: The small model learns from the teacher's knowledge without needing the teacher at inference time.


## References

Unsloth documentation: https://unsloth.ai/docs
SmolLM2 paper: https://arxiv.org/abs/2502.02737
Qwen3 technical report: https://arxiv.org/abs/2505.09388
Google distillation guide: https://developers.google.com/machine-learning/crash-course/llm/tuning
Snorkel distillation guide: https://snorkel.ai/blog/llm-distillation-demystified-a-complete-guide/
Predibase distillation playbook: https://github.com/predibase/llm_distillation_playbook
