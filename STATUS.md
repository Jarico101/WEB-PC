# STATUS

Current branch: `codex/pc28-ppo-agent-plan`

Current DQN state:
- Main active code has been updated in `ai_agent/train_dqn.py`.
- Release checker exists at `ai_agent/dqn_release_check.py`.
- Latest useful model for short-cycle balance-first eval:
  - `ai_agent/models/dqn_balance_100p_max50_lookback_100k.zip`
  - metadata: `ai_agent/models/dqn_balance_100p_max50_lookback_100k.metadata.json`
- Previous balance-first model:
  - `ai_agent/models/dqn_balance_100p_max50_80k.zip`
- Profit-only experiments exist but are not the current preferred direction:
  - `ai_agent/models/dqn_profit_only_80k.zip`
  - `ai_agent/models/dqn_profit_only_100p_max50_80k.zip`

What was verified:
- `dqn_balance_100p_max50_80k` passed the 5-window short-cycle release check.
- Cross-checks on the later continuous slices showed mixed results, with A/C weaker than B/D.
- `dqn_balance_100p_max50_lookback_100k` improved A and D, but B/C still need work.
- `dqn_balance_100p_max50_lookback_v2_100k` regressed; it is not the preferred model.

Important logs:
- `ai_agent/logs/dqn_balance_100p_max50_80k_release_5windows.md`
- `ai_agent/logs/dqn_balance_100p_max50_lookback_100k_abcd_summary.md`
- `ai_agent/logs/dqn_balance_100p_max50_lookback_v2_100k_eval_5windows.md`

Current code posture:
- `train_dqn.py` now supports `profit` and `balance` reward objectives.
- `train_dqn.py` also supports configurable `max_steps` for short-window experiments.
- `dqn_release_check.py` is used for short-cycle validation.

Open issue:
- Short-cycle profitability is still not stable across all recent slices.
- B/C slices remain weaker than A/D for the lookback model family.
- B/C comparison result: C has better-looking history stats in some windows, but its realized balance is worse on the weak windows, especially window 3 and window 5 in the current slice set.
