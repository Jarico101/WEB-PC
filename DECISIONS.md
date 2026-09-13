# DECISIONS

## Accepted directions

- Short-cycle evaluation is the current gate for the DQN effort.
- For the current profitable direction, the useful window shape is `100` steps, not the earlier `200`-step risk windows.
- The useful short-cycle objective is balance-first, not pure bet-count compliance.
- The model should not be judged only by a single window; A/B/C/D slice checks are part of validation.

## Rejected directions

- Pure risk training that forces bet-count targets too hard.
- Pure profit-only training without balance protection.
- Continuing to tune the over-constrained 200-step risk setup as the main path.

## Concrete model choices already made

- `dqn_balance_100p_max50_80k` was accepted as the first short-cycle pass candidate.
- `dqn_balance_100p_max50_lookback_100k` became the better candidate after adding lookback features.
- `dqn_balance_100p_max50_lookback_v2_100k` is not the preferred version because it regressed on the slice checks.

## Validation rules currently in use

- Short-cycle release checks are done with `ai_agent/dqn_release_check.py`.
- The key checks used for the short-cycle balance-first gate are:
  - total PnL > 0
  - profit window rate above the agreed floor
  - max drawdown under the agreed cap
  - minimum final balance above the agreed floor
  - max bets per window within bounds
  - top-profit concentration not too high

## Notes

- Current worktree contains many unrelated modified/untracked files. They were not reverted.
- No training, commit, deployment, or rollback policy was changed by these docs.
