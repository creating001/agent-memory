# Strong Memory Method

This project implements a unified long-context memory QA pipeline. The same
prediction algorithm is used for LongMemEval and LoCoMo: routing is based on
question semantics and available memory evidence, not dataset name, sample id,
gold answer, or judge feedback.

## Pipeline

1. Build turn-level memory chunks and Qwen3 embeddings.
2. Retrieve candidate evidence with dense search plus lexical expansion.
3. Apply factual-slot reranking with anchor retention when rerank confidence is
   expected to improve precision without removing high-rank dense evidence.
4. Route questions into general task strategies: factual, temporal,
   multi-evidence, recency-state, or preference/advice.
5. Compile structured evidence for temporal and multi-evidence questions.
6. Generate concise JSON answers from the retrieved context or compiled
   evidence.
7. Apply deterministic evidence finalization for mechanical consistency:
   distinct-count deduplication, additive sums, baseline-plus-delta quantities,
   chronological ordering, insufficient-evidence detail, and endpoint day
   durations when evidence dates are explicit.
8. Apply guardrails for relative-time normalization and target consistency.

LoCoMo adversarial questions are excluded from the retained benchmark result,
matching the project evaluation setting.

## Retained Full Results

| benchmark | split | correct / total | accuracy | avg query tokens |
|---|---:|---:|---:|---:|
| LongMemEval | full | 421 / 500 | 84.20% | 4714.2 |
| LoCoMo | non-adversarial | 1248 / 1540 | 81.04% | 5150.8 |

Retained artifacts:

- `outputs/retained/strong_memory_v1/longmemeval.judge.jsonl`
- `outputs/retained/strong_memory_v1/locomo_non_adversarial.judge.jsonl`
- `outputs/retained/strong_memory_v1/config.yaml`
