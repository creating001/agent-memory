from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


RETAINED_DIR = Path("outputs/retained/strong_memory_v1")


@dataclass(frozen=True)
class RetainedMetrics:
    name: str
    path: Path
    total: int
    correct: int
    avg_query_tokens: float
    max_query_tokens: int
    over_8k: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def retained_metrics(name: str, path: Path) -> RetainedMetrics:
    total = 0
    correct = 0
    token_sum = 0
    token_count = 0
    max_query_tokens = 0
    over_8k = 0

    for row in read_jsonl(path):
        total += 1
        if row.get("judge_label") is True:
            correct += 1
        query_tokens = row.get("query_tokens")
        if isinstance(query_tokens, int | float):
            value = int(query_tokens)
            token_sum += value
            token_count += 1
            max_query_tokens = max(max_query_tokens, value)
            over_8k += value > 8000

    return RetainedMetrics(
        name=name,
        path=path,
        total=total,
        correct=correct,
        avg_query_tokens=(token_sum / token_count if token_count else 0.0),
        max_query_tokens=max_query_tokens,
        over_8k=over_8k,
    )


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def format_metrics(metrics: RetainedMetrics) -> str:
    return (
        f"{metrics.name}: {metrics.correct}/{metrics.total} = {metrics.accuracy:.4%}; "
        f"avg_query_tokens={metrics.avg_query_tokens:.1f}; "
        f"max_query_tokens={metrics.max_query_tokens}; over_8k={metrics.over_8k}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize retained judged artifacts.")
    parser.add_argument(
        "--retained-dir",
        type=Path,
        default=RETAINED_DIR,
        help="Directory containing retained judged JSONL artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = {
        "LongMemEval": args.retained_dir / "longmemeval.judge.jsonl",
        "LoCoMo non-adversarial": args.retained_dir / "locomo_non_adversarial.judge.jsonl",
    }
    for name, path in artifacts.items():
        print(format_metrics(retained_metrics(name, path)))


if __name__ == "__main__":
    main()
