# PC28 动作空间与结算逻辑改造计划

## 1. 目的和边界

本计划只覆盖 PC28 攻防组合模拟器的动作空间、组合约束、虚拟 stake 分配和结算账务改造。它补充总体计划 [PC28_COMBO_AGENT_TASK_PLAN.md](./PC28_COMBO_AGENT_TASK_PLAN.md)，不替代训练、回测、实时同步或前端的完整计划。

边界必须保持明确：系统只生成、结算和评估虚拟模拟决策；不得提交外部订单、连接投注执行器、处理真实资金，或把虚拟余额当成真实余额。

## 2. 已确认的规则契约

### 2.1 开奖和值和基础项

以三位开奖号码和值 `S` 作为唯一结算输入，`S` 的合法范围为 `0-27`。

| 分类 | 规则 |
| --- | --- |
| 小 | `1-13` |
| 大 | `14-27` |
| 单 | `S` 为奇数 |
| 双 | `S` 为偶数 |

基础市场固定为 `小单`、`小双`、`大单`、`大双`，内部 market id 分别为 `small_odd`、`small_even`、`big_odd`、`big_even`。每轮必须恰好选择其中 3 个不同项，不能少选、重复选择，也不能用数字或极值项替代基础项。

| 基础项 | 含本金赔率 |
| --- | ---: |
| 小单 | 3.69 |
| 大双 | 3.69 |
| 小双 | 4.29 |
| 大单 | 4.29 |

### 2.2 扩展项

在三个基础项以外，最多追加 2 项，总选择数为 `3-5`。扩展项仅允许：

- 数字单点 `sum_00` 到 `sum_27`；
- `extreme_small`（极小）；
- `extreme_big`（极大）。

数字单点按已确认的赔率表结算，`00/27` 为 `888` 倍。极小、极大均为 `16` 倍，但它们的和值边界尚未由产品确认；未配置边界时必须从动作空间移除，不能借用旧玩法的边界。

### 2.3 已确认配置

- 和值 `0` 排除四个基础项；数字 `00` 与极小仍按各自市场独立结算。
- 极小闭区间为 `0-5`，极大闭区间为 `22-27`；边界、相邻非边界值和 `0/27` 已有测试覆盖。
- 虚拟 stake 档位由 `StakeAllocationConfig` 集中配置；最大暴露保持为当前虚拟余额的 `1%`，不得因连亏扩大。

## 3. 目标动作空间

采用合法组合优先的四轴 `MultiDiscrete` 编码，而不是让模型自由拼接市场 id 后再修复非法动作：

```text
[base_combo_id, extension_pair_id, stake_profile_id, mode_id]
```

| 轴 | 值域 | 约束和含义 |
| --- | --- | --- |
| `base_combo_id` | `0-3` | 四种“三选一排除”的基础组合。 |
| `extension_pair_id` | 空、单扩展项或两个不同扩展项 | 预枚举的 `0-2` 项组合，不允许重复。极值未配置时不枚举。 |
| `stake_profile_id` | 防守、均衡、进攻 | 只决定虚拟预算内的 stake 比例，不改变最大暴露。 |
| `mode_id` | 防守、均衡、进攻 | 模型动作只产生真实虚拟组合；`shadow` 仅保留给显式离线影子结算。 |

基础组合映射必须在 `portfolio_combo.py`、训练 metadata、推理校验和测试中完全一致：

```text
0: small_odd,  small_even, big_odd   # 排除 big_even
1: small_odd,  small_even, big_even  # 排除 big_odd
2: small_odd,  big_odd,   big_even   # 排除 small_even
3: small_even, big_odd,   big_even   # 排除 small_odd
```

目标版本标识为 `multidiscrete-base4-extension-pair-profile-real-mode-v3`。改变任何轴的顺序、数量、市场顺序、档位定义或极值可用条件时，必须升级动作空间版本并拒绝加载旧模型。

## 4. 结算契约

所有赔率均为含本金返还赔率。每一个 selection 独立命中、独立返还；同一和值可以同时命中一个基础项、一个数字单点和一个极值项。

```text
total_stake = Σ selection.stake
return_i = selection.stake × selection.odds, 当且仅当 selection 命中
total_return = Σ return_i
net_contribution_i = return_i - selection.stake
net_profit = total_return - total_stake
```

执行模式下：

```text
balance_after = balance_before + net_profit
```

影子模式下仍记录 `total_stake`、`total_return`、`net_profit` 和逐项命中，但：

```text
balance_delta = 0
balance_after = balance_before
shadow_net_profit = net_profit
```

金额统一按十进制计算，并在逐项金额和汇总金额处按两位小数、`ROUND_HALF_UP` 取整。结算入口必须再次校验决策，即使决策来自内部模型，以阻止旧模型、API 负载或损坏账本绕过组合与预算约束。

## 5. 改造任务

状态含义：`[ ]` 未开始，`[>]` 进行中，`[x]` 已完成，`[!]` 规则或外部依赖阻塞。

当前基线：市场注册表、四轴组合解码、组合约束、含本金逐项结算、影子结算和基础组合测试已经存在。此次改造的重点是消除环境与通用动作枚举之间的重复定义，完成特征/metadata 版本门禁，补足账本幂等键和端到端回归验证。

### A0. 冻结规则和版本 `[x]`

- [x] 在 `ai_agent/markets_combo.py` 维护唯一的基础市场、数字赔率、极值赔率、和值分类和 `ZeroPolicy`。
- [x] 产品已确认：和值 `0` 排除基础项；极小为 `0-5`，极大为 `22-27`。该配置已写入模型 metadata、训练、推理、评估和实时模拟。
- [x] 定义 `RULE_VERSION`、`ODDS_VERSION`、`ACTION_SPACE_VERSION` 和 `FEATURE_VERSION`；任何不一致的模型均不可用于推理或评估。
- [x] 在 `ai_agent/schemas_combo.py` 固化 `Selection`、`ComboDecision`、`SettlementItem`、`ComboSettlement` 的 JSON 字段与类型。

验收：一个给定的 market id 在环境、训练、推理、API、回测和结算中得到相同赔率、类型和命中结果。

### A1. 清点并隔离旧动作 `[x]`

- [x] 检索 `casino_env.py`、旧训练入口、旧推理/API 与前端，列出顺子/非顺子、对子、豹子、位数、单独大小/单双等旧动作的所有引用。
- [x] 新组合路径只接受四轴动作；旧路径如需保留历史兼容，必须使用独立路由、独立模型目录和独立 API 名称。
- [x] 禁止用旧动作别名映射到新组合，避免旧 checkpoint 看似可加载却产生不同语义。
- [x] 为旧模型和错误 metadata 写明确拒绝错误，而不是静默回退，并补齐 feature version 校验。

验收：任何新组合决策都可被解析为 3 个基础项和不超过 2 个扩展项；旧顺子类模型不能进入组合推理。

### A2. 实现动作解码和组合硬约束 `[x]`

- [x] 在 `ai_agent/portfolio_combo.py` 固化四个基础组合和无重复扩展组合的枚举顺序。
- [x] `decode_action()` 只接受四个整数轴，并拒绝越界、缺失轴和不可表示的扩展组合。
- [x] `build_decision()` 将离散动作转换为 `ComboDecision`，为每项写入 canonical `market_type`、赔率、预测概率和角色。
- [x] `validate_decision()` 强制：基础项恰好三个且不同、扩展项 `0-2` 且不同、总项数 `3-5`、market type/odds 与注册表一致。
- [x] 把最大暴露、最小 stake 和 stake 档位移动到显式配置对象；禁止根据连亏自动放大预算。

验收：穷举基础组合和扩展组合时，不存在少于三基础项、重复项、未知市场、负 stake 或超预算决策。

### A3. 对齐训练、推理和 observation `[x]`

- [x] `ComboCasinoEnv` 使用同一套 `action_nvec()` 和扩展项可用性规则，避免环境自行生成另一种编码。
- [x] PPO 使用原生四轴 `MultiDiscrete`；DQN 使用可逆的混合进制映射，并用当前 `nvec` 生成离散动作数。
- [x] 特征中为 `sum_00` 至 `sum_27` 提供只基于历史期的概率，避免数字单点的预测概率恒为零。
- [x] metadata 写入动作空间、特征列及训练数据截止期号；推理时精确校验这些值。
- [x] 重训所有升级后动作空间或 observation 维度不兼容的模型，不迁移旧权重作为新模型。

验收：PPO 与 DQN 的动作可往返转换；模型只能用当前特征维度推理；所有扩展项的 `probability` 都来自预测时点之前的数据。

### A4. 重构结算器为唯一账务真相 `[x]`

- [x] `settle_combo()` 先调用 `validate_decision()`，再校验和值、虚拟余额和单轮最大暴露。
- [x] 对每个 selection 调用统一的 `market_hits()`；不得在环境、回测或 API 内复制大小/单双、数字和极值的命中规则。
- [x] 逐项保存 `hit`、`return_amount` 和 `net_contribution`，汇总保存成本、返还、净收益、结算前后余额与 shadow 字段。
- [x] 严格区分“净收益”和“余额变化”：影子结算有收益记录但余额变化必须为零。
- [x] 为实时账本加入 `(target_issue, decision_id/model_version)` 幂等键；同一期已结算决策不得二次入账。

验收：环境、回测、在线同步调用同一结算函数；重放相同决策和同一期结果不会重复改变余额。

### A5. 迁移实时模拟和历史数据 `[x]`

- [x] 决策日志先写入不可变 `decision` 事件，再在开奖后写对应 `settlement` 事件，二者由目标期号关联。
- [x] 服务重启时从账本恢复余额、pending 决策和已结算期号，且不根据实时 API 覆盖既有决策。
- [x] 遇到重复期号、倒序期号、非数字期号或中间缺口时写诊断并暂停生成下一期实际模拟决策。
- [x] `/combo-signal`、`/combo-history`、`/combo/metrics` 只返回虚拟模拟数据，且逐项结算字段使用 `ComboSettlement` schema。

验收：连续实时数据可完成“上一期结算 -> 下一期决策”；重复同步和服务重启不会改变历史虚拟余额。

### A6. 建立回归测试和发布门禁 `[x]`

- [x] 市场测试覆盖和值 `0-27`、四种基础命中、全部数字赔率、未配置极值拒绝和已配置极值边界。
- [x] 组合测试覆盖四个基础组合、所有扩展对、非法重复、少选基础项、超两项扩展、错误赔率、错误 market type 和超预算。
- [x] 结算测试覆盖下表向量、分位金额舍入、多个追加项同时命中和 shadow 余额不变。
- [x] 集成测试覆盖 PPO/DQN smoke training、模型 metadata 拒绝、回测和账本幂等恢复。
- [x] 发布前运行组合专项测试、全量 Python 测试、前端 lint/build 与 `git diff --check`。

## 6. 必须通过的结算向量

| 场景 | 和值 | 虚拟选择 | 成本 | 返还 | 净收益 |
| --- | ---: | --- | ---: | ---: | ---: |
| 用户基础示例 | 15（大单） | 小单 30、小双 30、大单 30 | 90.00 | 128.70 | 38.70 |
| 三基础项全部落空 | 15（大单） | 小单 30、小双 30、大双 30 | 90.00 | 0.00 | -90.00 |
| 基础项加数字 | 15（大单） | 小单 30、小双 30、大单 30、15 10 | 100.00 | 258.70 | 158.70 |
| 影子结算 | 15（大单） | 与用户基础示例相同，模式为 `shadow` | 90.00 | 128.70 | 38.70 影子收益，余额变化 0.00 |
| 数字加极值 | 已确认的极值边界内 | 三基础项、命中数字、命中对应极值 | 按 stake | 三项可同时返还 | 按公式计算 |

## 7. 文件职责

| 文件 | 唯一职责 |
| --- | --- |
| `ai_agent/markets_combo.py` | 市场注册表、赔率、和值分类、命中规则和极值配置。 |
| `ai_agent/schemas_combo.py` | 决策、逐项结算、汇总结算和模型元数据 schema。 |
| `ai_agent/portfolio_combo.py` | 动作解码、组合合法性、角色和预算内 stake 分配。 |
| `ai_agent/settlement_combo.py` | 唯一的逐项结算与虚拟余额计算。 |
| `ai_agent/combo_env.py` | Gym 环境、动作/observation 对齐和奖励调用。 |
| `ai_agent/train_combo.py`、`ai_agent/infer_combo.py` | 算法适配、模型 metadata 写入与兼容性拒绝。 |
| `ai_agent/online_combo.py`、`ai_agent/live_combo.py` | 不可变模拟账本、幂等同步和状态恢复。 |
| `tests/test_combo_agent.py` | 规则、动作、结算、训练和实时模拟的回归覆盖。 |

## 8. 实施顺序和完成标准

### 本次执行记录（2026-09-05）

- 已将动作空间升级为 `multidiscrete-base4-extension-pair-profile-real-mode-v3`，并由 `portfolio_combo.py` 作为环境、训练与推理共用的扩展组合枚举来源；确认极值边界后动作维度为 `[4, 466, 3, 3]`，模型不能选择 `shadow`。
- 已启用 `combo-history-v2-sum-rates`：每个数字单点 `sum_00` 至 `sum_27` 使用预测时点前 100 期频率；PPO/DQN 模型 metadata 精确校验特征版本、特征列、动作维度、算法和 `ZeroPolicy`。
- 已生成仅用于本地模拟验证的 `ppo_combo_feature_v2_smoke.zip` 与 `dqn_combo_feature_v2_smoke.zip`，旧 11 维 v1 模型会被拒绝，不能再进入新组合推理。
- 已为追加式账本决策生成稳定 `decision_id`，并保留按目标期号的重复结算保护。
- 已将 stake 档位比例提取为 `StakeAllocationConfig`；分配基于十进制互补预算计算，避免浮点精度导致总暴露少一分或超预算。
- 已清点旧顺子/非顺子、对子、豹子和位数动作：它们仍仅存在于旧 `CasinoEnv`、旧训练/展示路径；组合路由、模型、推理和 API 不会引用或兼容这些动作。
- 已加入 `ai_agent/challenger_combo.py`、`ai_agent/training_protocol_combo.py` 与 `ai_agent/champion_combo.py`：每 100 个已结算模拟期才标记 Challenger 训练任务；最近连续 10,000 期严格按 8,000 Train、1,000 Validation、1,000 封存 Test 切分。PPO、DQN 候选仅以 Validation 五个连续 200 期窗口统一门控和自动排名；只有 Validation 第一名可读取 Test 做最终确认。候选永不自动替换或启用 Champion。
- 首个正式 PPO Challenger 已使用 `3435545-3445544` 训练，并在 `3445545-3446544` 的五窗口盲测中 `FAIL`（0/5 窗口通过），所以未晋级；报告位于 `ai_agent/logs/ppo_combo_challenger_3445544.json`。
- 已通过组合专项及全量 Python 115 项、TypeScript lint、生产构建和 `git diff --check`；对 `api.startpops.xyz/api/ai-signal` 的只读 smoke 已生成合法虚拟组合，未写入账本或调用外部执行器。
- 已新增 `ai_agent/sync_combo_history.py` 以按日、限速从公开源恢复结果，并生成单一来源的最新连续训练窗口。当前窗口为 `data/history_combo_pc28_latest_10000.jsonl`（`3468115-3478114`、10,000 期、零和值 9 期、无缺期），审计清单记录来源和 SHA-256。

按 `A0 -> A1 -> A2 -> A3 -> A4 -> A5 -> A6` 执行。`A0` 的两个产品确认项不会阻碍基础组合、数字单点和影子账务开发，但会阻碍极值动作启用及正式模型晋级。

本次改造完成的定义：

- 新动作空间只产生“恰好三个基础项 + 零至两个扩展项”的合法决策。
- 新旧动作、模型、特征与 API 路径已隔离，版本不匹配会被明确拒绝。
- 所有结算使用含本金赔率、逐项独立命中和统一两位小数账务规则。
- shadow 收益可学习、可审计，但不会改变虚拟余额。
- 决策与结算在实时账本中可追溯、可恢复、不可重复入账。
- 上述结算向量、组合约束和端到端训练/回测/同步测试全部通过。
