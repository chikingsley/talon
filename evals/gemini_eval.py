#!/usr/bin/env python3
"""
Evaluate Gemini models on signature extraction.

Tests how modern LLMs perform on the same dataset as talon baseline.

Usage:
    uv run evals/gemini_eval.py
    uv run evals/gemini_eval.py --model gemini-2.5-flash-lite
    uv run evals/gemini_eval.py --limit 20 --verbose
    uv run evals/gemini_eval.py --resume evals/results/gemini-2.5-flash-lite_*.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from baseline import (
    load_forge_dataset,
    load_sig_marker_fixtures,
    load_stripped_fixtures,
    normalize_text,
    texts_match,
)
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from prompts import SIGNATURE_EXTRACTION_PROMPT, SIGNATURE_EXTRACTION_PROMPT_STRICT
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
]


@dataclass
class EvalResult:
    email_id: str
    expected_signature: str | None
    predicted_signature: str | None
    signature_match: bool
    raw_response: str
    error: str | None = None

    def to_dict(self) -> dict[str, str | bool | None]:
        d = {
            "email_id": self.email_id,
            "expected_signature": self.expected_signature,
            "predicted_signature": self.predicted_signature,
            "signature_match": self.signature_match,
        }
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class EvalMetrics:
    total: int = 0
    signature_exact_match: int = 0
    signature_partial_match: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    errors: int = 0
    results: list[EvalResult] = field(default_factory=list)

    @property
    def signature_accuracy(self) -> float:
        return self.signature_exact_match / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "total": self.total,
            "signature_exact_match": self.signature_exact_match,
            "signature_partial_match": self.signature_partial_match,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "errors": self.errors,
            "signature_accuracy": f"{self.signature_accuracy:.2%}",
        }


def is_retryable_error(exc: BaseException) -> bool:
    """Check if exception is a retryable API error (rate limit or transient)."""
    if isinstance(exc, genai_errors.APIError):
        # Rate limit (429) or server errors (500, 502, 503, 504)
        return exc.code in (429, 500, 502, 503, 504)
    return False


@retry(
    retry=retry_if_exception_type((genai_errors.APIError,)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry_error_callback=lambda retry_state: retry_state.outcome.result()
    if retry_state.outcome
    else None,
    before_sleep=lambda retry_state: print(
        f"    Retrying after {retry_state.next_action.sleep:.1f}s...", flush=True
    ),
)
def extract_signature_gemini(
    client: genai.Client, model: str, email_body: str, prompt_template: str = SIGNATURE_EXTRACTION_PROMPT
) -> tuple[str | None, str]:
    """Extract signature using Gemini. Returns (signature, raw_response)."""
    prompt = prompt_template.format(email=email_body)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    raw = response.text.strip()

    # Parse response
    if raw == "NO_SIGNATURE" or raw.upper() == "NO_SIGNATURE":
        return None, raw

    return raw, raw


def load_completed_ids(output_path: Path) -> set[str]:
    """Load email IDs that have already been processed from JSONL file."""
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        result = json.loads(line)
                        completed.add(result["email_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return completed


def write_result_jsonl(f, result: EvalResult) -> None:
    """Write a single result to JSONL file."""
    f.write(json.dumps(result.to_dict()) + "\n")
    f.flush()


def evaluate_gemini(
    client: genai.Client,
    model: str,
    emails: list[dict[str, str]],
    output_path: Path,
    completed_ids: set[str],
    verbose: bool = False,
    prompt_template: str = SIGNATURE_EXTRACTION_PROMPT,
) -> EvalMetrics:
    """Run Gemini extraction on emails and compute metrics."""
    metrics = EvalMetrics()

    # Filter out already completed emails
    pending_emails = [(i, e) for i, e in enumerate(emails) if e["id"] not in completed_ids]
    skipped = len(emails) - len(pending_emails)
    if skipped > 0:
        print(f"  Resuming: skipping {skipped} already processed emails")

    # Open output file in append mode
    with open(output_path, "a") as f:
        for idx, (original_idx, email) in enumerate(pending_emails):
            email_id = email["id"]
            body = email["body"]
            expected_sig = email.get("expected_signature")
            progress_num = skipped + idx + 1
            total_num = len(emails)

            try:
                predicted_sig, raw_response = extract_signature_gemini(client, model, body, prompt_template)
                error = None
            except Exception as e:
                predicted_sig = None
                raw_response = ""
                error = str(e)
                metrics.errors += 1

            # Check matches
            if error:
                sig_match = False
                partial_match = False
            else:
                sig_match = texts_match(expected_sig, predicted_sig)
                partial_match = False
                if not sig_match and expected_sig and predicted_sig:
                    exp_norm = normalize_text(expected_sig)
                    pred_norm = normalize_text(predicted_sig)
                    partial_match = exp_norm in pred_norm or pred_norm in exp_norm

            result = EvalResult(
                email_id=email_id,
                expected_signature=expected_sig,
                predicted_signature=predicted_sig,
                signature_match=sig_match,
                raw_response=raw_response,
                error=error,
            )
            metrics.results.append(result)
            metrics.total += 1

            # Write immediately to JSONL
            write_result_jsonl(f, result)

            if sig_match:
                metrics.signature_exact_match += 1
            elif partial_match:
                metrics.signature_partial_match += 1

            # False positive: predicted signature when none expected
            if not expected_sig and predicted_sig:
                metrics.false_positives += 1

            # False negative: no signature predicted when one expected
            if expected_sig and not predicted_sig:
                metrics.false_negatives += 1

            # Real-time progress with running accuracy
            status = "ERROR" if error else ("PASS" if sig_match else ("PARTIAL" if partial_match else "FAIL"))
            running_acc = metrics.signature_accuracy * 100
            print(f"[{progress_num}/{total_num}] {email_id} - {status} ({running_acc:.1f}%)", flush=True)

            if verbose and not sig_match and not error:
                print(f"       Expected: {repr(expected_sig)[:60]}...")
                print(f"       Got:      {repr(predicted_sig)[:60]}...")

    return metrics


def get_default_output_path(model: str) -> Path:
    """Generate default output path: evals/results/{model}_{timestamp}.jsonl"""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Sanitize model name for filename
    safe_model = model.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_dir / f"{safe_model}_{timestamp}.jsonl"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Gemini on signature extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available models: {', '.join(MODELS)}",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemini-2.5-flash-lite",
        help=f"Gemini model to use (default: gemini-2.5-flash-lite). Options: {', '.join(MODELS)}",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed per-email diffs")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output JSONL file (default: evals/results/{model}_{timestamp}.jsonl)",
    )
    parser.add_argument("--limit", "-l", type=int, help="Limit number of emails to evaluate")
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        help="Resume from existing JSONL file (skips already processed emails)",
    )
    parser.add_argument(
        "--fixtures",
        type=str,
        default="tests/fixtures",
        help="Path to fixtures directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict prompt (requires email/phone/company for signature)",
    )
    args = parser.parse_args()

    # Load .env from project root
    load_dotenv(Path(__file__).parent.parent / ".env")

    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        return 1

    client = genai.Client(api_key=api_key)

    fixtures_dir = Path(args.fixtures)
    if not fixtures_dir.is_absolute():
        fixtures_dir = Path(__file__).parent.parent / fixtures_dir

    evals_dir = Path(__file__).parent

    # Determine output path
    if args.resume:
        output_path = Path(args.resume)
        if not output_path.exists():
            print(f"ERROR: Resume file not found: {output_path}")
            return 1
    elif args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = get_default_output_path(args.model)

    # Load completed IDs if resuming
    completed_ids = load_completed_ids(output_path) if args.resume else set()

    print("=" * 60)
    print("Gemini Signature Extraction Evaluation")
    print(f"Model: {args.model}")
    if args.strict:
        print("Prompt: STRICT (formal business signatures only)")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Load all fixtures (same as baseline)
    emails = []

    print("\nLoading fixtures...")
    stripped = load_stripped_fixtures(fixtures_dir)
    print(f"  Stripped directory: {len(stripped)} emails")
    emails.extend(stripped)

    p_fixtures = load_sig_marker_fixtures(fixtures_dir / "signature" / "emails" / "P", prefix="P/")
    print(f"  P directory: {len(p_fixtures)} emails")
    emails.extend(p_fixtures)

    forge = load_forge_dataset(evals_dir)
    if forge:
        print(f"  Forge dataset: {len(forge)} emails")
        emails.extend(forge)

    if args.limit:
        emails = emails[: args.limit]
        print(f"\nLimited to {args.limit} emails")

    print(f"\nTotal: {len(emails)} emails")
    if completed_ids:
        print(f"Already completed: {len(completed_ids)} emails")

    # Select prompt template
    prompt_template = SIGNATURE_EXTRACTION_PROMPT_STRICT if args.strict else SIGNATURE_EXTRACTION_PROMPT

    # Run evaluation
    print("\nRunning evaluation...")
    if args.strict:
        print("Using STRICT prompt (requires email/phone/company)")
    print("-" * 60)
    metrics = evaluate_gemini(
        client, args.model, emails, output_path, completed_ids, verbose=args.verbose,
        prompt_template=prompt_template
    )

    # Print results
    print("-" * 60)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Total emails evaluated: {metrics.total}")
    print(
        f"Signature exact match:  {metrics.signature_exact_match}/{metrics.total} ({metrics.signature_accuracy:.1%})"
    )
    print(f"Signature partial match: {metrics.signature_partial_match}")
    print(f"False positives:        {metrics.false_positives}")
    print(f"False negatives:        {metrics.false_negatives}")
    print(f"Errors:                 {metrics.errors}")
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
