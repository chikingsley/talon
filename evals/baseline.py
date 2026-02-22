#!/usr/bin/env python3
"""
Baseline evaluation for talon signature extraction.

Measures current accuracy on existing test fixtures to establish baseline
before ML model training.

Usage:
    uv run python evals/baseline.py
    uv run python evals/baseline.py --verbose
    uv run python evals/baseline.py --output results.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from talon import signature
from talon.signature import extract

# Initialize the ML classifier (required before extraction works)
signature.initialize()


@dataclass
class EvalResult:
    """Result of evaluating one email."""

    email_id: str
    sender: str
    body_length: int
    expected_signature: str | None
    predicted_signature: str | None
    expected_body: str | None
    predicted_body: str | None
    signature_match: bool
    body_match: bool

    def to_dict(self) -> dict[str, str | int | bool | None]:
        return {
            "email_id": self.email_id,
            "sender": self.sender,
            "body_length": self.body_length,
            "expected_signature": self.expected_signature,
            "predicted_signature": self.predicted_signature,
            "signature_match": self.signature_match,
            "body_match": self.body_match,
        }


@dataclass
class EvalMetrics:
    """Aggregated metrics from evaluation run."""

    total: int = 0
    signature_exact_match: int = 0
    signature_partial_match: int = 0  # predicted is subset of expected or vice versa
    body_exact_match: int = 0
    false_positives: int = 0  # predicted signature when none expected
    false_negatives: int = 0  # no signature predicted when one expected
    results: list[EvalResult] = field(default_factory=list)

    @property
    def signature_accuracy(self) -> float:
        return self.signature_exact_match / self.total if self.total else 0.0

    @property
    def body_accuracy(self) -> float:
        return self.body_exact_match / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "total": self.total,
            "signature_exact_match": self.signature_exact_match,
            "signature_partial_match": self.signature_partial_match,
            "body_exact_match": self.body_exact_match,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "signature_accuracy": f"{self.signature_accuracy:.2%}",
            "body_accuracy": f"{self.body_accuracy:.2%}",
        }


def normalize_text(text: str | None) -> str:
    """Normalize text for comparison (strip whitespace, normalize newlines)."""
    if text is None:
        return ""
    return text.strip().replace("\r\n", "\n").replace("\r", "\n")


def texts_match(expected: str | None, predicted: str | None) -> bool:
    """Check if two texts match after normalization."""
    return normalize_text(expected) == normalize_text(predicted)


def load_stripped_fixtures(fixtures_dir: Path) -> list[dict[str, str]]:
    """Load email fixtures from the stripped directory format."""
    stripped_dir = fixtures_dir / "signature" / "emails" / "stripped"
    emails = []

    if not stripped_dir.exists():
        return emails

    # Find all _body files and load corresponding _sender and _signature
    for body_file in stripped_dir.glob("*_body"):
        name = body_file.stem.replace("_body", "")
        sender_file = stripped_dir / f"{name}_sender"
        sig_file = stripped_dir / f"{name}_signature"

        if not sender_file.exists():
            continue

        sender = sender_file.read_text(encoding="utf-8").strip()
        body = body_file.read_text(encoding="utf-8")

        # Load signature and strip #sig# markers if present
        signature = None
        if sig_file.exists():
            sig_text = sig_file.read_text(encoding="utf-8")
            sig_lines = [
                line[5:] if line.startswith("#sig#") else line
                for line in sig_text.splitlines()
            ]
            signature = "\n".join(sig_lines)

        emails.append(
            {
                "id": name,
                "sender": sender,
                "body": body,
                "expected_signature": signature,
            }
        )

    return emails


def load_sig_marker_fixtures(directory: Path, prefix: str = "") -> list[dict[str, str]]:
    """Load email fixtures using #sig# inline markers.

    Works for both talon's P fixtures and forge dataset format.
    """
    emails = []

    if not directory.exists():
        return emails

    for body_file in directory.glob("*_body"):
        name = body_file.stem.replace("_body", "")
        sender_file = directory / f"{name}_sender"

        if not sender_file.exists():
            continue

        sender = sender_file.read_text(encoding="utf-8").strip()
        raw_body = body_file.read_text(encoding="utf-8")

        # Parse inline #sig# markers
        sig_lines = []
        body_lines = []
        for line in raw_body.splitlines():
            if line.startswith("#sig#"):
                sig_lines.append(line[5:])  # Remove #sig# prefix
            else:
                body_lines.append(line)

        # Reconstruct body without #sig# markers
        body = "\n".join(body_lines + sig_lines)  # Full email for extraction
        signature = "\n".join(sig_lines) if sig_lines else None

        emails.append(
            {
                "id": f"{prefix}{name}",
                "sender": sender,
                "body": body,
                "expected_signature": signature,
            }
        )

    return emails


def load_forge_dataset(evals_dir: Path) -> list[dict[str, str]]:
    """Load the mailgun/forge dataset (labeled Enron emails)."""
    forge_dir = evals_dir / "forge" / "dataset"
    emails = []

    if not forge_dir.exists():
        return emails

    # P = emails with signatures
    emails.extend(load_sig_marker_fixtures(forge_dir / "P", prefix="forge/P/"))

    # N = emails without signatures (expected_signature will be None)
    emails.extend(load_sig_marker_fixtures(forge_dir / "N", prefix="forge/N/"))

    return emails


def evaluate_talon(emails: list[dict[str, str]], verbose: bool = False) -> EvalMetrics:
    """Run talon extraction on emails and compute metrics."""
    metrics = EvalMetrics()

    for email in emails:
        email_id = email["id"]
        sender = email["sender"]
        body = email["body"]
        expected_sig = email.get("expected_signature")

        # Run talon extraction
        try:
            predicted_body, predicted_sig = extract(body, sender)
        except Exception as e:
            if verbose:
                print(f"  ERROR on {email_id}: {e}")
            predicted_body, predicted_sig = body, None

        # Compute expected body (body minus signature)
        expected_body = None
        if expected_sig:
            sig_normalized = normalize_text(expected_sig)
            body_normalized = normalize_text(body)
            if body_normalized.endswith(sig_normalized):
                expected_body = body_normalized[: -len(sig_normalized)].strip()

        # Check matches
        sig_match = texts_match(expected_sig, predicted_sig)
        body_match = texts_match(expected_body, predicted_body) if expected_body else True

        # Partial match detection
        partial_match = False
        if not sig_match and expected_sig and predicted_sig:
            exp_norm = normalize_text(expected_sig)
            pred_norm = normalize_text(predicted_sig)
            partial_match = exp_norm in pred_norm or pred_norm in exp_norm

        result = EvalResult(
            email_id=email_id,
            sender=sender,
            body_length=len(body),
            expected_signature=expected_sig,
            predicted_signature=predicted_sig,
            expected_body=expected_body,
            predicted_body=predicted_body,
            signature_match=sig_match,
            body_match=body_match,
        )
        metrics.results.append(result)
        metrics.total += 1

        if sig_match:
            metrics.signature_exact_match += 1
        elif partial_match:
            metrics.signature_partial_match += 1

        if body_match:
            metrics.body_exact_match += 1

        # False positive: predicted signature when none expected
        if not expected_sig and predicted_sig:
            metrics.false_positives += 1

        # False negative: no signature predicted when one expected
        if expected_sig and not predicted_sig:
            metrics.false_negatives += 1

        if verbose:
            status = "PASS" if sig_match else ("PARTIAL" if partial_match else "FAIL")
            print(f"  [{status}] {email_id}")
            if not sig_match:
                print(f"       Expected: {repr(expected_sig)[:60]}...")
                print(f"       Got:      {repr(predicted_sig)[:60]}...")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate talon signature extraction")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-email results")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file for results")
    parser.add_argument(
        "--fixtures",
        type=str,
        default="tests/fixtures",
        help="Path to fixtures directory",
    )
    args = parser.parse_args()

    fixtures_dir = Path(args.fixtures)
    if not fixtures_dir.is_absolute():
        fixtures_dir = Path(__file__).parent.parent / fixtures_dir

    print("=" * 60)
    print("Talon Signature Extraction - Baseline Evaluation")
    print("=" * 60)

    # Load all fixtures
    emails = []
    evals_dir = Path(__file__).parent

    print("\nLoading fixtures...")
    stripped = load_stripped_fixtures(fixtures_dir)
    print(f"  Stripped directory: {len(stripped)} emails")
    emails.extend(stripped)

    p_fixtures = load_sig_marker_fixtures(fixtures_dir / "signature" / "emails" / "P", prefix="P/")
    print(f"  P directory: {len(p_fixtures)} emails")
    emails.extend(p_fixtures)

    # Load forge dataset if available
    forge = load_forge_dataset(evals_dir)
    if forge:
        print(f"  Forge dataset: {len(forge)} emails")
        emails.extend(forge)

    if not emails:
        print("ERROR: No evaluation emails found!")
        print(f"Looked in: {fixtures_dir}")
        return 1

    print(f"\nTotal: {len(emails)} emails")

    # Run evaluation
    print("\nRunning evaluation...")
    metrics = evaluate_talon(emails, verbose=args.verbose)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total emails evaluated: {metrics.total}")
    print(
        f"Signature exact match:  {metrics.signature_exact_match}/{metrics.total} ({metrics.signature_accuracy:.1%})"
    )
    print(f"Signature partial match: {metrics.signature_partial_match}")
    print(
        f"Body exact match:       {metrics.body_exact_match}/{metrics.total} ({metrics.body_accuracy:.1%})"
    )
    print(f"False positives:        {metrics.false_positives}")
    print(f"False negatives:        {metrics.false_negatives}")

    # Save JSON output if requested
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "metrics": metrics.to_dict(),
            "results": [r.to_dict() for r in metrics.results],
        }
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
