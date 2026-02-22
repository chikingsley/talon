# Talon Signature Extraction Evals

Evaluation scripts comparing talon's rule-based signature extraction against LLM approaches.

## Files

- `baseline.py` - Runs talon's built-in extraction on test fixtures
- `gemini_eval.py` - Tests Gemini models on the same fixtures
- `prompts.py` - LLM prompts for signature extraction
- `forge/` - Forge email dataset (192 emails with labeled signatures)
- `results/` - JSONL output files (gitignored)

## Usage

```bash
# Talon baseline
uv run evals/baseline.py

# Gemini models
uv run evals/gemini_eval.py --model gemini-2.5-flash-lite
uv run evals/gemini_eval.py --model gemini-2.5-pro --verbose

# Resume interrupted run
uv run evals/gemini_eval.py --resume evals/results/gemini-2.5-pro_*.jsonl
```

## Results

| Model | Exact Match | Partial | FP | FN | Errors |
|-------|-------------|---------|----|----|--------|
| talon baseline | 163/199 (81.9%) | 8 | 2 | 25 | 0 |
| gemini-2.5-flash-lite | 153/199 (76.9%) | 24 | 17 | 2 | 1 |
| gemini-2.5-flash | 141/199 (70.9%) | 38 | 11 | 4 | 1 |
| gemini-2.5-pro | 171/199 (85.9%) | 12 | 13 | 1 | 1 |
| gemini-3-flash-preview | 175/199 (87.9%) | 11 | 11 | 1 | 1 |
| gemini-3-pro-preview | 183/199 (92.0%) | 3 | 12 | 1 | 1 |

**Winner: gemini-3-pro-preview** at 92% accuracy.

## Dataset

- **Stripped fixtures** (6 emails) - Basic signature test cases
- **P fixtures** (1 email) - Signature marker tests
- **Forge dataset** (192 emails) - Real-world emails with labeled signatures

Total: 199 emails
