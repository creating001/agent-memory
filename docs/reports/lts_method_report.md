# LTS 方法简报

## 结论

当前 LTS 是 `v50_narrow_temporal_recency_verify`。项目代码当前只保留该方法。

这是一个 clean single-branch 版本：预测阶段只使用本地 Qwen answer / embedding 服务，不使用 reference answer、sample id、question_type 或 judge label 做任何预测决策。DeepSeek 只用于最终 judge。

## 指标

### LongMemEval

```text
accuracy:          417 / 500 = 0.8340
f1:                0.4538
bleu:              0.1179
build_tokens:      0
query_tokens:      2,367,025
avg_query_tokens:  4,734.1 / sample
```

按问题类型：

```text
knowledge-update             69 / 78   = 0.8846
multi-session               101 / 133  = 0.7594
single-session-assistant     54 / 56   = 0.9643
single-session-preference    24 / 30   = 0.8000
single-session-user          67 / 70   = 0.9571
temporal-reasoning          102 / 133  = 0.7669
```

### LoCoMo 对照

当前已保存的 LoCoMo 对照结果：

```text
accuracy:          1489 / 1986 = 0.7497
f1:                0.5381
bleu:              0.2371
build_tokens:      277,765
query_tokens:      2,520,124
avg_query_tokens:  1,269.0 / sample
```

按问题类型：

```text
adversarial        388 / 446 = 0.8700
multi-hop          201 / 282 = 0.7128
open-domain         60 / 96  = 0.6250
single-hop         671 / 841 = 0.7979
temporal           169 / 321 = 0.5265
```

说明：LongMemEval 的 LTS 文件已固化在 `outputs/baseline/final/longmemeval_v50/`；LoCoMo 这里作为跨 benchmark 稳定性对照。

## 方法

### Memory 构建

```text
chunk_unit:       turn
memory content:   原始对话 turn，不做摘要压缩
embedding text:   Date + turn text
build LLM:        不使用
build_tokens:     0
```

核心思想是尽量保留原始证据，不在 build 阶段提前总结或丢弃信息，把主要判断放到 query-time retrieval 和 answer 阶段。

### 检索

```text
base retrieval:   semantic top-k
evidence:         retrieved original turns
merge:            RRF-style rank fusion when multiple retrieval views are used
date handling:    日期进入 chunk embedding 文本和 answer evidence
```

路由只根据 question text 触发，不使用 benchmark 标注：

```text
prefer_user:
  用于偏好、习惯、多证据类问题
  倾向保留 user-side evidence

duration_temporal:
  用于 how long / since / ago / duration 类问题
  强化时间证据和单位换算

order_reflective:
  用于 first / earliest / chronological / happened first 类问题
  增加反思式检索与答案校验

state_history:
  用于 previous / initial / originally / used to 类历史状态问题

recency_auto:
  用于 most recent / most recently 类问题

auto:
  默认路线
```

### 回答

```text
answer model:      local Qwen
temperature:       0.0
answer format:     JSON answer field
evidence style:    retrieved evidence first, then final answer
verification:      仅在特定时间/顺序问题上使用本地 Qwen 自检
```

回答阶段不访问 reference answer，不调用 DeepSeek，不根据 sample id 或 question_type 写规则。

## 为什么作为 LTS

```text
1. LongMemEval 达到 0.8340，满足 83% 目标。
2. 平均 query token 为 4.7k/sample，低于 8k/sample 约束。
3. 方法保持 general：路由由问题文本触发，不依赖 benchmark 标签。
4. 当前主干不保留其他 candidate 方法，便于复现和回滚。
```

## 回滚文件

```text
predictions:
outputs/baseline/final/longmemeval_v50/predictions.jsonl

judge:
outputs/baseline/final/longmemeval_v50/predictions.judge.jsonl

metrics:
outputs/baseline/final/longmemeval_v50/metrics.md

LTS note:
docs/LTS.md
```
