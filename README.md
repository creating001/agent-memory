# Agent Memory Baseline

本项目提供一个简洁、可复现的 Agent Memory baseline，用于长对话记忆构建、检索、回答生成和评测。

方法在预测阶段只使用本地部署的 answer model 和 embedding model，不使用 reference answer、sample id、question type 或 judge label 参与预测。Judge model 只用于最终评测。

## 方法概览

```text
memory build:
  chunk_unit: turn
  memory content: raw dialogue turn
  embedding text: Date + turn text
  build LLM: none

retrieval:
  base: dense semantic retrieval
  query: question + question_date when available
  fusion: RRF over semantic, expanded semantic, lexical, and reflective views
  routing: selected from question text only

answer:
  model: local Qwen/Qwen3-30B-A3B-Instruct-2507
  temperature: 0.0
  format: JSON with answer field
  verification: used for narrow order / first temporal questions

judge:
  model: DeepSeek
  use: evaluation only
```

## 

运行时只根据 question text 选择一个 route：

```text
auto:
  default route

prefer_user:
  for count / preference-like questions that should prefer user-side evidence

duration_temporal:
  for how long / since / days ago / weeks ago / months ago questions

order_reflective:
  for first / order / chronological questions
  uses reflective retrieval and local answer verification

state_history:
  for previous / initial / original / used to / usually questions

recency_auto:
  for most recent / most recently questions
```

这些 route 不读取 benchmark label，也不读取 gold answer。

## Configuration

默认配置文件：

```text
src/agent_memory/configs/base.yaml
```

核心设置：

```yaml
retrieval:
  chunk_unit: turn
  top_k: 40

answer:
  temperature: 0.0
  max_tokens: 8192
  final_max_tokens: 1024

embedding:
  dims: 1024
  normalize: true
  max_input_bytes: 8192
```

## Project Structure

```text
src/agent_memory/
  baseline/
    chunking.py     # turn-level memory chunk construction
    store.py        # chunk / embedding / build stats persistence
    retrieve.py     # dense retrieval, BM25 lexical retrieval, RRF fusion
    routing.py      # route, strategy, and top-k rules
    queries.py      # query text, lexical expansion, retrieved rerank helpers
    guardrails.py   # answer post-processing and narrow guardrails
    pipeline.py     # build / retrieve / answer orchestration
  configs/          # model, retrieval, and path config
  core/             # LLM, embedding, schema, IO utilities
  datasets/         # LongMemEval and LoCoMo loaders
  evaluation/       # judge and metrics
  prompts/          # answer, retrieval, and judge prompts
  run_baseline.py   # command-line entry

data/               # local datasets; see dataset links below
docs/               # method notes and evaluation summaries
outputs/            # local outputs
```

Dataset files:

- `data/longmemeval_s_cleaned.json`: [LongMemEval-S cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json)
- `data/locomo10.json`: [LoCoMo](https://github.com/snap-research/locomo/blob/main/data/locomo10.json)

## Run

LongMemEval：

```bash
PYTHONPATH=src python -m agent_memory.run_baseline \
  --dataset longmemeval \
  --data data/longmemeval_s_cleaned.json \
  --out outputs/baseline/predictions.jsonl \
  --store-root outputs/baseline/stores \
  --log-file outputs/logs/run.log \
  --workers 4 \
  --overwrite
```

LoCoMo：

```bash
PYTHONPATH=src python -m agent_memory.run_baseline \
  --dataset locomo \
  --data data/locomo10.json \
  --out outputs/baseline/locomo_predictions.jsonl \
  --store-root outputs/baseline/locomo_stores \
  --log-file outputs/logs/locomo_run.log \
  --workers 2 \
  --overwrite
```

也可以分阶段运行：

```bash
PYTHONPATH=src python -m agent_memory.run_baseline --mode build ...
PYTHONPATH=src python -m agent_memory.run_baseline --mode query ...
```

## Evaluation Results

LongMemEval：

```text
accuracy:          417 / 500 = 0.8340
f1:                0.4538
bleu:              0.1179
build_tokens:      0
query_tokens:      2,367,025
avg_query_tokens:  4,734.1 / sample
```

LoCoMo：

```text
accuracy:          1489 / 1986 = 0.7497
f1:                0.5381
bleu:              0.2371
query_tokens:      2,520,124
avg_query_tokens:  1,269.0 / sample
```
