# 13/14 Train-only 风险研究验收报告

## 结论

`RISK_MODEL_QUALIFICATION_FAIL`。本阶段只读取 Train 8,000 期，未读取 Validation/Test，未启动 PPO。

候选选择使用 Fit 前 6,400 期内的 12 个连续 500 期未来窗口；Calibration 使用 800 期；Qualification 使用独立最后 800 期。

## Qualification 结果

| 事件 | Brier skill | ECE | AUC | 高/低风险发生率差 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 和值 13 | 0.726849% | 0.022057% | 0.473315 | -2.354065% | FAIL |
| 和值 14 | 0.552102% | 1.095964% | 0.503453 | 不可计算（高组 0 行） | FAIL |
| 13 或 14 | 1.439599% | 1.118021% | 0.444702 | -5.522660% | FAIL |

硬门为每个事件 Brier skill `>=1%`、ECE `<=5%`、AUC `>=0.5`、高风险组发生率高于低风险组且两组样本充足。任一事件失败即阻断 PPO。

## 选择与阻断

Fit-only 研究选出的候选为 `shrunk_frequency_3200_theory_1600`。但独立 Qualification 未通过：13、14 的 Brier skill 均不足 1%；13 与联合事件的风险排序方向相反；14 的校准阈值导致高风险组无样本。

因此：

- PPO 风控实验：未执行；
- 动作审计：未执行；
- Validation：未读取；
- Test：未读取；
- Champion：无。

详细机器报告见 `train_only_research.json`。
