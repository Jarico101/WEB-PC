# PC28 组合模拟系统交接说明

## 当前目标与边界

这是 PC28 **虚拟模拟**系统：绝不能连接真实下注、真实资金、投注执行器或自动下单服务。

- 基础项：`小单 / 小双 / 大单 / 大双`；每个实际订单必须恰好选 3 项，或等待。
- 和值边界：小 `1-13`、大 `14-27`；和值 `0` 不命中基础项。
- 赔率含本金：小单/大双 `3.69`；小双/大单 `4.29`。
- 默认正式动作空间：`[5, 1, 3, 3]`；4 个三基础组合 + 等待，默认禁用和值/极值扩展。
- 金额：每项整数 `$30-$300`，每轮总额 `<= $300`。
- 风控：连续 3 个亏损实际订单后等待 10 期；余额、回撤与最低可投注金额为硬门。
- 已确认无返水、手续费或额外派奖。

工作树有大量用户未提交/未跟踪文件。不要回退、删除或覆盖不相关文件；禁止 `git reset --hard`、`git checkout --` 等破坏性操作。

## 数据与防泄漏

唯一正式数据源：`data/history_combo_pc28_latest_10000.jsonl`，连续 10,000 期：

| 切分 | 期数 | 范围 | 用途 |
|---|---:|---|---|
| Train | 8,000 | `3468115-3476114` | 训练、Train 内研究 |
| Validation | 1,000 | `3476115-3477114` | 仅候选评分/选择 |
| Test | 1,000 | `3477115-3478114` | 封存，只确认唯一 Validation 第一名 |

关键文件：

- `data/history_combo_train_8000.jsonl`
- `data/history_combo_validation_1000.jsonl`
- `data/history_combo_test_1000.jsonl`
- `data/history_combo_8_1_1.manifest.json`
- `data/train_chunks_500/train_01_500.jsonl` 至 `train_16_500.jsonl`
- `data/train_chunks_500/train_chunks.manifest.json`

所有特征只可使用当前期之前的数据。禁止随机打乱、把 Validation/Test 用于调参，或让当前结果进入当前决策。

## 正式 PPO/DQN：仍被资格门阻止

正式协议：`pc28-qualified-opportunity-v1`。

Train 内严格分为 Fit 6,400、Calibration 800、独立 Qualification 800。权重、缩放、窗口选择只来自 Fit；温度只来自 Calibration；Qualification 只读评分。

正式 PPO/DQN 启动前必须同时达到：`Brier skill >= 1%`、`ECE <= 5%`。

最新 Train 8,000 期资格：

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| Brier skill | `0.090571%` | `>= 1%` |
| ECE | `4.118299%` | `<= 5%` |
| 结论 | `FAIL` | 正式训练禁止启动 |

Train 内 12 个未来 500 期窗口研究也没有稳定预测信号；理论和值分布与 3200 期收缩频率最接近，但均未稳定通过。报告：`ai_agent/logs/train_internal_probability_research_20260906.json`；实现：`ai_agent/probability_research_combo.py`。

不要启动正式 PPO/DQN、自动调参、Champion 或 Test。固定下注率、负期望收益等只能是 `EXPERIMENTAL_NON_CHAMPION` 隔离实验，绝不能进入 Champion/Test。

## 用户批准的随机模拟脚本（不需要 PPO）

用户已明确的目标：

```text
每期：
1. 从四种合法三基础组合中等概率随机选择一种；
2. 50% 概率随机追加一个和值；
3. 金额随机分配，但每项为整数 $30-$300、总额 <= $300；
4. 始终受止损、止盈、回撤、连亏冷却约束。
```

实现：`ai_agent/experimental_random_combo.py`。

- 固定分类：`EXPERIMENTAL_NON_CHAMPION`。
- 当前追加池：`08 / 09 / 18 / 19`，每期至多一个。
- 默认止损 20%；用户要求的止盈为 **盈利 200%**，即余额达到初始资金 3 倍；最大回撤 35%。
- 入口：`run_experimental_random_backtest(...)`。
- 永远不能声称预测能力、成为 Champion 或读取 Test。

一次 Train 8,000 期运行：`ai_agent/logs/experimental_random_train8000_takeprofit200_20260906.json`。结果：61 单、最终 `$7,978.99`、盈亏 `-$2,021.01`、最大回撤 `21.83%`。

## PPO 实验：动作塌缩，不能当作有效模型

`ai_agent/train_combo.py` 已添加：

- PPO 探索熵 `ent_coef=0.01`；随机起点、最多 200 期 episode。
- 重复基础组合惩罚 `repeat_base_combo_penalty=0.05`。
- 实验性状态条件化奖励：仅根据开奖前最近 100 期四基础项比例，奖励选择历史比例最高的三项。它只验证“随可见状态切换”，不是预测优势。
- 训练后强制动作审计：至少 2 个基础组合、3 个完整动作、单一动作占比 `<=90%`、状态匹配率 `>=40%`、至少覆盖 3 个状态目标组合；否则 `ACTION_COLLAPSED`，不保存、不回测为有效候选、不进 Validation/Champion/Test。

近期拒绝报告：

| 报告 | 参数 | 结果 |
|---|---|---|
| `ai_agent/logs/ppo_combo_experimental_500_2500_rate35_diversity_20260906.json` | 500期 / 2,500 steps / 35%上限 | 单一动作 99.6%，拒绝 |
| `ai_agent/logs/ppo_combo_experimental_500_2500_state_conditioned_20260907.json` | 500期 / 2,500 steps / 状态条件化 | 仅 1 个动作，拒绝 |

被拒绝时，绝不能伪造余额、订单、收益或 Champion。

## 尚未实现：13/14 三层风险系统

用户要求设计“躲避/对冲 13、14”系统。数学事实：

- `13=小单`，`14=大双`。
- 强制选三个基础项时只可排除一项，因此不可能同时躲避 13 和 14。
- 若三位数字独立均匀：`P(13)=7.5%`、`P(14)=7.5%`、`P(13或14)=15%`。遗漏久不改变下一期概率；不能把“欲出”硬编码为下注理由。
- 13/14 单点赔率为 11 倍，盈亏平衡概率 `1/11=9.09%`；无可信个别概率超过此值时，追加 13/14 为负期望。
- 三基础项必须下注时，`小单 + 大单 + 大双` 可提高小单/大双金额来对冲 13/14，但这不是预测躲避。

若继续实现，按此顺序：

1. **第一层，监督风险模型，不是 PPO**：输出 `p13`、`p14`、`p13+p14`。只用因果特征：13/14 单独与合并遗漏、20/50/100/200期频率、间隔/连开、和值统计、基础项比例等。
2. **第二层，独立资格**：延续 Fit/Calibration/Qualification；验证 Brier skill、ECE、高低风险组命中差、AUC、样本覆盖、多个 Train 内未来窗口稳定性。失败则不启用。
3. **第三层，风险动作**：只有第一层通过后，PPO 才能在等待/对冲/原随机策略间决定。低风险用原随机；中风险用小单+大单+大双对冲；高风险等待；仅当个别 `p13` 或 `p14` 经验证稳定高于 9.09%并有安全余量，才可在追加时选择对应 13/14。

用户提议“将 13/14 加入 50% 随机和值池”尚未实现。先确认其含义：

1. 纯娱乐池 `08/09/18/19/13/14`：仅实验、无预测声明；或
2. 仅在通过的 13/14 风险门触发时，按 `p13/p14` 对冲追加：数学上更合理。

## 核心文件

- `PC28_COMBO_TRAINING_PLAN.md`：正式协议。
- `ai_agent/probability_combo.py`：概率模型与资格评分。
- `ai_agent/probability_research_combo.py`：Train 内研究。
- `ai_agent/statistical_combo_policy.py`：概率/收益/风控门。
- `ai_agent/combo_env.py`：环境、冷却、结算前门控、状态条件目标。
- `ai_agent/portfolio_combo.py`：组合、金额、实验追加池。
- `ai_agent/settlement_combo.py`：虚拟逐项结算。
- `ai_agent/backtest_combo.py`：连续账户回测。
- `ai_agent/train_combo.py`：训练入口、资格阻断、动作多样性审计。
- `ai_agent/experimental_random_combo.py`：用户批准的随机脚本。
- `tests/test_combo_agent.py`、`tests/test_probability_qualification.py`：测试。

## 验证状态与汇报要求

专项测试通常运行：

```bash
./.venv/bin/python -m unittest tests.test_probability_qualification tests.test_combo_agent.ComboTrainingTests -q
./.venv/bin/python -m py_compile ai_agent/*.py
git diff --check
```

全量 `unittest discover` 中有两条旧“资格失败仍可下注”的随机基线断言，与当前严格正式资格协议冲突。不要为让旧断言通过而放松正式门槛；如运行全量测试应如实报告该已知冲突。

完成训练/回测时使用：

| 模式 | 下单 | 最终余额 | 盈亏 | 最大回撤 |
|---|---:|---:|---:|---:|

- 金额 `$` 两位小数；盈亏 = 最终余额 - 初始资金；回撤为百分比；按余额降序。
- 表格后仅说明最佳模式、最终 Champion、Test 最终结果、是否通过门槛。
- 如果仍在训练、资格失败或 `ACTION_COLLAPSED`，只报告真实状态和原因，绝不能捏造最终结果。

## 下一位代理的起点

1. 阅读本文件和 `PC28_COMBO_TRAINING_PLAN.md`。
2. 先确认用户想实现 13/14 三层风险系统，还是仅扩展纯随机和值池。
3. 若实现风险系统，先做 Train-only 风险模型资格研究；不得读取 Validation/Test、不得启动 PPO。
4. 仅在风险模型独立资格通过后，才继续第三层 PPO 风控实验。
5. 保持全程虚拟模拟边界。

## PWA / VPS 运行状态（2026-09-11）

- PWA 生产域名为 `https://pc28.startpops.xyz`。浏览器请求同源 `/api/ai-signal`，由 Vercel 重写转发至 VPS API `https://api.startpops.xyz/api/ai-signal`，避免移动端直连 VPS API 失败。
- VPS 的 `pc28-api.service`、`pc28-refresh-worker.service` 与 Nginx 均已启用并运行；最近期号连续性检查通过。
- PWA 的 `BOT SIGNAL` 依赖 API `prediction` 字段。完整历史现超过 31,000 期，不能直接对完整历史运行旧的二次复杂度回放，否则会阻塞 API。
- 已在 VPS 的 `pc28-api.service` 设置 `PC28_API_PREDICTION_ENABLED=1` 及 `PC28_PREDICTION_MAX_DRAWS=1000`。预测区块只使用最近 1,000 期；主统计、遗漏与压力仍使用完整历史。
- `COMBO SIMULATION` 后端未部署：`/api/combo-signal` 与 `/api/combo/metrics` 当前返回 404，PWA 应显示 `OFFLINE`，不得伪造模拟结果。
- PWA 已更正指标展示：理论遗漏按连续未出期望 `(1-p)/p` 显示；策略区分当前遗漏条件率与“出顺后”历史率；BOT 命中率不再标为原始分数；走势图仅显示遗漏变化，不再伪造逐期历史压力评分。
- 移动端已修正走势图触摸提示被裁切、组合模拟统计强制三列的问题。2026-09-11 的生产 HTTP、TLS、DNS 与 API CORS 检查均正常，未见 Vercel 访问保护或 IP 拦截证据。

## 13/14 风险计划执行状态（2026-09-07）

已执行 `PC28_13_14_RISK_EXECUTION_ACCEPTANCE_PLAN.md` 的 Train-only 风险研究与独立资格阶段。报告：`ai_agent/logs/risk_1314/train_only_research.json` 和 `ai_agent/logs/risk_1314/train_only_research.md`。

- 结果：`RISK_MODEL_QUALIFICATION_FAIL`。
- 13 Brier skill `0.726849%`，低于 `1%`；AUC `0.473315`，风险分组方向反转。
- 14 Brier skill `0.552102%`，低于 `1%`；高风险组无样本。
- 13 或 14 Brier skill `1.439599%`，但 AUC `0.444702`，风险分组方向反转。
- Validation/Test 均未读取，PPO、动作审计均未启动；当前无 Champion。
- 已按计划停止，等待下一步指令。

## 13/14 结构性覆盖随机实验（2026-09-07）

用户明确的实验结算：13 仅返还小单本金，14 仅返还大双本金；其他基础项使用用户给定的含本金赔率：小单/大双 `4.6`、小双/大单 `4.2`。这与正式 `markets_combo.py` 赔率冲突，已隔离为 `EXPERIMENTAL_NON_CHAMPION`，不得进入 PPO、Validation、Test 或 Champion。

- 实现：`ai_agent/experimental_1314_hedged.py`。
- 规则：等概率随机选择 `小单+大单+大双` 或 `小单+小双+大双`；基础项每项 `$30`；50% 概率追加 `08/09/18/19` 之一 `$30`；3 连亏后等待 10 期；止损 20%、止盈 200%、最大回撤 35%。
- 首次 Train 尾部 500 期运行（`3475615-3476114`）：处理 334 期、实际 233 单、等待 101 期、最终 `$7,930.00`、盈亏 `-$2,070.00`、最大回撤 `25.19%`；第 334 期触发 20% 止损，故未处理余下 166 期。
- 13/14 实际订单 40 单、合计盈亏 `-$3,000.00`；报告：`ai_agent/logs/experimental_1314_hedged_train16_500.json`。

取消和值的同窗口复跑：`ai_agent/logs/experimental_1314_hedged_no_sums_train16_500.json`。处理 425 期、实际 284 单、正收益 165 单（胜率 `58.10%`）、最终 `$7,972.00`、盈亏 `-$2,028.00`、最大回撤 `24.11%`；第 425 期触发 20% 止损。13/14 实际 49 单，每单 `-$60.00`，合计 `-$2,940.00`。胜率提升不代表盈利，仍为隔离实验。

## 2026-10-25 VPS 恢复完成

- 新 VPS `45.205.22.131`：Ubuntu 24.04、2 vCPU、约 2 GB RAM、29 GB SSD。
- `api.startpops.xyz` A 记录已切换到新 IP；HTTPS 证书有效期至 2027-01-23。
- 后端 `/opt/pc28-ai-board` 已启动：`pc28-api.service`、`pc28-refresh-worker.service`、Nginx，均 enabled/active。
- 采集 worker 每 15 秒运行；最近日志连续 `ok:true`、`missing:0`，最新期号已从 3498416 更新到 3498417。
- API `https://api.startpops.xyz/api/health` 返回 200，`history_count` 60811；`/api/ai-signal` 返回 200、`data_integrity:true`。
- Vercel `https://pc28.startpops.xyz/` 浏览器实测能通过同源 API 显示最新期号和 60812 期全历史。
- 恢复历史候选共 60811 期，无重复；165 个缺号全部匹配既有跳期/已知缺失规则。
- 备份：服务器 `/var/backups/pc28/recovered-history-20261025.tgz`，本地 `.pc28-recovery/recovered-history-20261025.tgz`。
- SSH 临时端口 2222 已启动，22 仍保留；密码曾在对话中明文发送，必须尽快重置并改用 SSH 密钥。
- 组合模拟服务仍未部署，前端显示 `OFFLINE`，不得伪造结果。

## 2026-10-27 历史开奖分类与分页

- 历史状态改为和值组合：0–13 小、14–27 大，奇数单、偶数双；桌面与手机一致。
- 最近 500 期，每页 20 期，上一页/下一页及页码跳转；数据不足时按实际数量显示。
- VPS `/opt/pc28-ai-board/config.json` displayLimit 已由 50 改为 500，重启 pc28-api，公网接口确认返回 500 条。
- `npm run lint`、`npm run build` 通过；本地浏览器连接真实 API，验证分类、翻页、末页 20 条/下一页禁用。
- 前端已发布：Vercel 登录恢复，生产部署 dpl_8FhHjKVTHtREu7171SvfE5hSiHVn 已绑定 https://pc28.startpops.xyz。线上浏览器实测 API 200、500 条，状态显示大小单双组合，第 25 页 20 条且下一页禁用。
