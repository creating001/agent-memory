# Agent Memory Baseline

Unified long-context memory QA baseline for LongMemEval and LoCoMo. The default
pipeline uses the same prediction algorithm for both datasets and routes by
question semantics rather than dataset labels or sample ids.

## Method

```text
memory build:
  chunk_unit: turn
  memory content: raw dialogue turns
  embedding text: Date + turn text
  build LLM: none

retrieval:
  dense retrieval with Qwen3 embedding
  optional expanded and lexical retrieval views
  BGE-M3 dedicated rerank for factual-slot questions
  anchor retention after rerank to preserve high-rank dense evidence
  route selected from question text only

evidence / answer:
  evidence-table compiler for temporal and multi-evidence tasks
  concise JSON answer generation
  answer-detail requirements to preserve distinguishing evidence
  structured evidence finalizer for deterministic count, sum, order, and duration fixes
  task-aware duration and list evidence requirements
  relative-time and target-consistency guardrails

judge:
  DeepSeek API for evaluation only
```

More details:

- [Method](docs/method.md)

## Project Layout

```text
src/agent_memory/baseline/     production memory pipeline
src/agent_memory/prompts/      stable prompt profiles and templates
src/agent_memory/core/         model clients, config, IO, shared schema
src/agent_memory/datasets/     dataset loaders
src/agent_memory/evaluation/   judge and metrics
scripts/                       local vLLM service helpers
outputs/retained/              retained judged artifacts
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project is run directly from `src` with `PYTHONPATH=src`; no package build
step is required.

## Services

默认端口：

```text
answer:    http://127.0.0.1:8000/v1
embedding: http://127.0.0.1:8001/v1
rerank:    http://127.0.0.1:8002/v1
```

默认 GPU 布局按当前服务器设置为三个服务都可见 `0,1,2,3`：

```text
answer:    CUDA_VISIBLE_DEVICES=0,1,2,3, gpu_memory_utilization=0.70
embedding: CUDA_VISIBLE_DEVICES=0,1,2,3, gpu_memory_utilization=0.10
rerank:    CUDA_VISIBLE_DEVICES=0,1,2,3, gpu_memory_utilization=0.10
```

启动、查看、停止全部服务：

```bash
bash scripts/serve_all.sh start
bash scripts/serve_all.sh status
bash scripts/serve_all.sh stop
```

也可以分别启动：

```bash
bash scripts/serve_answer.sh
bash scripts/serve_embedding.sh
bash scripts/serve_rerank.sh
```

## Run

LongMemEval：

```bash
PYTHONPATH=src python -m agent_memory.run_baseline \
  --config src/agent_memory/configs/base.yaml \
  --dataset longmemeval \
  --data data/longmemeval_s_cleaned.json \
  --out outputs/baseline/longmemeval_predictions.jsonl \
  --store-root outputs/baseline/longmemeval_stores \
  --log-file outputs/logs/longmemeval_run.log \
  --workers 4 \
  --overwrite
```

LoCoMo：

```bash
PYTHONPATH=src python -m agent_memory.run_baseline \
  --config src/agent_memory/configs/base.yaml \
  --dataset locomo \
  --data data/locomo10.json \
  --exclude-question-type adversarial \
  --out outputs/baseline/locomo_predictions.jsonl \
  --store-root outputs/baseline/locomo_stores \
  --log-file outputs/logs/locomo_run.log \
  --workers 2 \
  --overwrite
```

## Evaluation

Judge：

```bash
PYTHONPATH=src python -m agent_memory.evaluation.judge \
  --config src/agent_memory/configs/base.yaml \
  --pred outputs/baseline/longmemeval_predictions.jsonl \
  --out outputs/baseline/longmemeval_predictions.judge.jsonl \
  --workers 8 \
  --overwrite
```

Metrics：

```bash
PYTHONPATH=src python -m agent_memory.evaluation.metrics \
  --pred outputs/baseline/longmemeval_predictions.judge.jsonl \
  --out outputs/baseline/longmemeval_metrics.md
```

Retained result check：

```bash
PYTHONPATH=src python -m agent_memory.evaluation.retained
```

## Retained Result

当前保留的 strong memory v1 结果：

```text
LongMemEval full:
  accuracy: 421 / 500 = 0.8420
  avg query tokens: 4,714.2 / sample

LoCoMo non-adversarial:
  accuracy: 1248 / 1540 = 0.8104
  avg query tokens: 5,150.8 / sample
```

Retained artifacts:

```text
outputs/retained/strong_memory_v1/longmemeval.judge.jsonl
outputs/retained/strong_memory_v1/locomo_non_adversarial.judge.jsonl
outputs/retained/strong_memory_v1/config.yaml
```
