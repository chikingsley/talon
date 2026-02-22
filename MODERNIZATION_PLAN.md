# Email Signature Extraction Modernization Plan (2026)

This document outlines a path to upgrade talon's signature extraction from ~90% accuracy to 99%+ using modern small language models and synthetic data generation.

**Updated January 2026** with latest model recommendations and Unsloth improvements.

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

## Model Options (2026 Recommendations)

Three tiers based on size, capability, and latest performance data.

**Tier 1 - Tiny** (runs anywhere, <500MB)

- SmolLM2-135M <https://huggingface.co/unsloth/SmolLM2-135M>
- SmolLM2-360M <https://huggingface.co/unsloth/SmolLM2-360M>
- Use case: Mobile/edge deployment with <1GB RAM

**Tier 2 - Recommended for this project** (SOTA 2026, trains on 3.9GB)

- **SmolLM3-3B** <https://huggingface.co/HuggingFaceTB/SmolLM3-3B>
  - Trained on 11T tokens (vs 5T for SmolLM2), best in class for 3B scale
  - Multilingual (6 languages), 128K context with NoPE/YaRN
  - Better instruction-following for structured extraction tasks
  - **With 2026 Unsloth: trains on only 3.9GB VRAM** ✨
- Qwen3-0.6B <https://huggingface.co/Qwen/Qwen3-0.6B>
  - If you need maximum inference speed

**Tier 3 - Maximum accuracy** (uses full 12GB)

- Qwen3-4B <https://huggingface.co/Qwen/Qwen3-4B>
  - Better reasoning, higher accuracy ceiling
  - Still fits 4-bit quantization on RTX 5070

Note on picoLLM: Their training platform (picoLLM GYM) is enterprise-only. Not accessible for this project. Their inference engine is useful for deployment but we cannot train custom models through them without an enterprise contract.

## Unsloth Resources (2026 Update)

Unsloth is the standard for efficient fine-tuning. **2026 improvements: 3× faster, 30% less VRAM** (up from 2× faster, 70% less).

Key 2026 improvements:

- New RoPE + MLP Triton kernels for speed
- Padding-free training reduces overhead
- Uncontaminated packing prevents data leakage
- SmolLM3-3B now trains on just 3.9GB VRAM (vs 12GB previously)

Main repo: <https://github.com/unslothai/unsloth>

Notebooks repo: <https://github.com/unslothai/notebooks>

Relevant notebooks (swap model name for smaller variants):

- SmolLM3-3B SFT: <https://github.com/unslothai/notebooks/blob/main/nb/SmolLM3_(3B)-SFT-Training.ipynb>
- Qwen3-0.6B fine-tuning: <https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_(0.6B)-SFT-Training.ipynb>

Documentation: <https://unsloth.ai/docs/get-started/fine-tuning-llms-guide>

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

```text
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

- Enron corpus (public): <https://www.cs.cmu.edu/~enron/>
- Mailgun forge dataset: <https://github.com/mailgun/forge>
- Generate synthetic emails with varied signature styles

## Phase 3 - Model Training

Use Unsloth on your RTX 5070. With 2026 improvements, we can afford better models.

**Recommended starting point: SmolLM3-3B**

- SOTA at 3B scale (OpenReview 2025)
- 11T token training data gives better generalization
- 128K context window (can process entire email threads)
- Multilingual support (6 languages)
- **Trains on only 3.9GB VRAM with Unsloth 2026** - leaves room for batch inference
- Better instruction-following = higher accuracy on structured extraction

Training approach:

- LoRA fine-tuning with Unsloth (only train 2-4% of parameters)
- 2-3 epochs over synthetic dataset (fewer needed due to better base model)
- Validate on held-out test set after each epoch
- Use GQA (Grouped Query Attention) for efficient inference

Alternative if speed is critical: SmolLM2-360M for deployment <200MB model size.

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

## Distillation Approach (Clarified 2026)

We're using **synthetic data distillation**, which combines teacher knowledge with efficient student training.

**Fine-tuning vs Distillation:**

- **Fine-tuning**: Train model on task-specific data. Model stays same size.
- **Distillation**: Use large model (teacher) to generate training data, then train smaller model (student).

**Our approach: Synthetic data distillation**

1. Claude (teacher) labels raw emails → structured JSON with zones and contacts
2. We collect 5-10K labeled examples (synthetic dataset)
3. We fine-tune SmolLM3-3B (student) on Claude's labels
4. Student learns task-specific patterns without needing teacher at inference

**Why this works (2026 insight):**

- Claude is expensive at scale ($$$), our model is cheap (runs locally, free)
- Teacher knowledge is encoded into the student during training
- Student model is 500x smaller than Claude (3B vs 405B) but specialized for email
- No teacher at inference time = instant, free classification

**Quality mechanism:**

- Claude can make mistakes, but patterns across 5-10K examples reveal true task structure
- LoRA training focuses learning on task-relevant parameters (2-4% of weights)
- Hold-out validation catches degenerate solutions

The key insight: The small model learns general task patterns from the teacher's labels, then specializes through LoRA fine-tuning.

## References (2026 Updated)

**Unsloth & Training:**

- Unsloth documentation (2026 improvements): <https://unsloth.ai/docs>
- Unsloth December 2025 update: <https://unslothai.substack.com/p/unsloth-december-update>
- 🚀 Unsloth Explained (2026 Edition): <https://medium.com/@dewasheesh.rana/unsloth-explained-2026-edition-c2678f23cca3>

**Model Papers:**

- SmolLM2 paper (Feb 2025): <https://arxiv.org/abs/2502.02737>
- SmolLM3 blog (HuggingFace, Nov 2025): <https://huggingface.co/blog/smollm3>
- Qwen3 technical report (May 2025): <https://arxiv.org/abs/2505.09388>

**Distillation & Learning:**

- Google distillation guide: <https://developers.google.com/machine-learning/crash-course/llm/tuning>
- Snorkel distillation guide: <https://snorkel.ai/blog/llm-distillation-demystified-a-complete-guide/>
- Predibase distillation playbook: <https://github.com/predibase/llm_distillation_playbook>

**Data Sources:**

- Enron corpus: <https://www.cs.cmu.edu/~enron/>
- Mailgun forge dataset: <https://github.com/mailgun/forge>
