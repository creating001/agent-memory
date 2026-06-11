# External Baselines

This directory contains reproduction baselines that are intentionally kept
outside the main `src/agent_memory/baseline` method.

Implemented baselines:

- `full_context`: answers from the full conversation/memory context.
- `naive_rag`: turn-level embedding retrieval with top-k dense search.

Outputs should be written under `external/outputs/`, which is ignored by git.

## Full Context

```bash
PYTHONPATH=src:. python external/baselines/full_context.py \
  --config src/agent_memory/configs/base.yaml \
  --dataset longmemeval \
  --data data/longmemeval_s_cleaned.json \
  --out external/outputs/full_context/longmemeval_predictions.jsonl \
  --overwrite
```

## Naive RAG

```bash
PYTHONPATH=src:. python external/baselines/naive_rag.py \
  --config src/agent_memory/configs/base.yaml \
  --dataset longmemeval \
  --data data/longmemeval_s_cleaned.json \
  --out external/outputs/naive_rag/longmemeval_predictions.jsonl \
  --top-k 10 \
  --overwrite
```

The output JSONL follows the same basic prediction schema used by the main
evaluation scripts, so it can be judged with:

```bash
PYTHONPATH=src python -m agent_memory.evaluation.judge \
  --config src/agent_memory/configs/base.yaml \
  --pred external/outputs/naive_rag/longmemeval_predictions.jsonl \
  --out external/outputs/naive_rag/longmemeval_predictions.judge.jsonl
```
