# 当前实验结果统计

更新时间：2026-06-10

## 评测设置

- LongMemEval：完整 500 条样本。
- LoCoMo：排除 adversarial 后的 1540 条样本。
- 主方法：`Ours`，LongMemEval 和 LoCoMo 使用同一套算法流程。
- 外部基线：放在 `external/` 下，和主方法代码、输出隔离。
- Judge：当前统计文件中的 `judge_label=None` 数量均为 0。
- Token 统计只计算 answer/build 阶段的 LLM prompt token，embedding 模型输入 token 不计入 `Avg Build Tokens`。

## 模型配置

预测阶段使用本地 vLLM 服务，LongMemEval 和 LoCoMo 使用同一套模型与参数配置。

| 模块 | 模型 |
|---|---|
| Answer LLM | Qwen/Qwen3-30B-A3B-Instruct-2507 |
| Embedding | Qwen/Qwen3-Embedding-0.6B |
| Rerank | BAAI/bge-m3 |

评测阶段使用 `deepseek-v4-flash` 作为 judge；judge 只用于离线打分，不参与预测流程。

## 总体结果

### LongMemEval

| 方法 | 正确数 / 总数 | Accuracy | Avg Build Tokens (K) | Avg Query Tokens (K) |
|---|---:|---:|---:|---:|
| Ours | 421 / 500 | 84.20% | 0.0 | 4.7 |
| Full Context | 84 / 500 | 16.80% | 0.0 | 126.4 |
| Naive RAG | 345 / 500 | 69.00% | 0.0 | 4.7 |

### LoCoMo non-adversarial

| 方法 | 正确数 / 总数 | Accuracy | Avg Build Tokens (K) | Avg Query Tokens (K) |
|---|---:|---:|---:|---:|
| Ours | 1248 / 1540 | 81.04% | 0.0 | 5.2 |
| Full Context | 1033 / 1540 | 67.08% | 0.0 | 41.8 |
| Naive RAG | 1030 / 1540 | 66.88% | 0.0 | 3.0 |

## Ours 按类型结果

### LongMemEval

| 类型 | 样本数 | Accuracy | F1 | BLEU |
|---|---:|---:|---:|---:|
| knowledge-update | 78 | 84.62% | 0.4177 | 0.1216 |
| multi-session | 133 | 79.70% | 0.2972 | 0.0539 |
| single-session-assistant | 56 | 98.21% | 0.5979 | 0.2426 |
| single-session-preference | 30 | 76.67% | 0.1864 | 0.0206 |
| single-session-user | 70 | 92.86% | 0.6203 | 0.1063 |
| temporal-reasoning | 133 | 79.70% | 0.4591 | 0.2257 |

### LoCoMo non-adversarial

| 类型 | 样本数 | Accuracy | F1 | BLEU |
|---|---:|---:|---:|---:|
| multi-hop | 282 | 76.60% | 0.3705 | 0.0594 |
| open-domain | 96 | 69.79% | 0.2281 | 0.0137 |
| single-hop | 841 | 86.68% | 0.5416 | 0.1220 |
| temporal | 321 | 73.52% | 0.6826 | 0.1692 |

## 外部基线按类型结果

### Full Context / LongMemEval

| 类型 | 样本数 | Accuracy | F1 | BLEU |
|---|---:|---:|---:|---:|
| knowledge-update | 78 | 23.08% | 0.1648 | 0.0815 |
| multi-session | 133 | 8.27% | 0.0604 | 0.0276 |
| single-session-assistant | 56 | 46.43% | 0.3143 | 0.1522 |
| single-session-preference | 30 | 20.00% | 0.0746 | 0.0043 |
| single-session-user | 70 | 17.14% | 0.0607 | 0.0096 |
| temporal-reasoning | 133 | 8.27% | 0.0946 | 0.0709 |

### Full Context / LoCoMo non-adversarial

| 类型 | 样本数 | Accuracy | F1 | BLEU |
|---|---:|---:|---:|---:|
| multi-hop | 282 | 61.70% | 0.2769 | 0.0552 |
| open-domain | 96 | 47.92% | 0.1822 | 0.0241 |
| single-hop | 841 | 82.52% | 0.4687 | 0.1392 |
| temporal | 321 | 37.07% | 0.3270 | 0.0456 |

### Naive RAG / LongMemEval

| 类型 | 样本数 | Accuracy | F1 | BLEU |
|---|---:|---:|---:|---:|
| knowledge-update | 78 | 78.21% | 0.4724 | 0.1426 |
| multi-session | 133 | 54.14% | 0.3428 | 0.0435 |
| single-session-assistant | 56 | 91.07% | 0.8166 | 0.5080 |
| single-session-preference | 30 | 40.00% | 0.1357 | 0.0122 |
| single-session-user | 70 | 90.00% | 0.6622 | 0.1288 |
| temporal-reasoning | 133 | 64.66% | 0.4131 | 0.1851 |

### Naive RAG / LoCoMo non-adversarial

| 类型 | 样本数 | Accuracy | F1 | BLEU |
|---|---:|---:|---:|---:|
| multi-hop | 282 | 64.18% | 0.2910 | 0.0576 |
| open-domain | 96 | 54.17% | 0.2192 | 0.0185 |
| single-hop | 841 | 78.83% | 0.4865 | 0.1533 |
| temporal | 321 | 41.74% | 0.4172 | 0.0776 |

## 结果文件

主方法：

- `outputs/retained/strong_memory_v1/longmemeval.judge.jsonl`
- `outputs/retained/strong_memory_v1/locomo_non_adversarial.judge.jsonl`
- `outputs/retained/strong_memory_v1/config.yaml`

外部基线：

- `external/outputs/full_context/longmemeval_judged.jsonl`
- `external/outputs/full_context/locomo_judged.jsonl`
- `external/outputs/naive_rag_top40/longmemeval_judged.jsonl`
- `external/outputs/naive_rag_top40/locomo_judged.jsonl`

LoCoMo 的 `judge_label=None` 样本已经单独重判，重判前备份保留在：

- `external/outputs/rejudge/full_context_locomo_judged.before_none_rejudge.jsonl`
- `external/outputs/rejudge/naive_rag_top40_locomo_judged.before_none_rejudge.jsonl`
