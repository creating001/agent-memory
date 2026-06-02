# LTS 回滚清单

当前 LTS 是跨 benchmark 稳定版本 `v50_narrow_temporal_recency_verify`。
项目代码已经收敛为该方法，不再保留其他 candidate 方法分支。

## LongMemEval

```text
tag: v50_narrow_temporal_recency_verify
accuracy: 417 / 500 = 0.8340
avg_query_tokens: 4734.1 / sample
```

```text
judge/result:
outputs/baseline/final/longmemeval_v50/predictions.judge.jsonl

metrics:
outputs/baseline/final/longmemeval_v50/metrics.md

predictions:
outputs/baseline/final/longmemeval_v50/predictions.jsonl

report:
docs/reports/lts_method_report.md
```

说明：v50 是 83% clean single-branch 版本，预测阶段只使用本地 Qwen answer / embedding 服务，不使用 reference answer、sample id、question_type 或 judge label 做预测决策。

## LoCoMo

```text
v50 是跨 benchmark LTS；LoCoMo 对照结果保留在 `outputs/baseline/locomo/`。
```

```text
metrics:
outputs/baseline/locomo/locomo_metrics.md

predictions:
outputs/baseline/locomo/locomo_predictions.jsonl

judge/result:
outputs/baseline/locomo/locomo_predictions.judge.jsonl
```

## 规则

```text
1. LTS 文件只读保留，不覆盖。
2. 默认运行入口 `PYTHONPATH=src python -m agent_memory.run_baseline` 只运行该 LTS 方法。
3. 若未来重新探索新方法，应先另建分支，不在当前 LTS 主干里混入实验分支。
```
