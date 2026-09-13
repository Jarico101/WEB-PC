# WEB-PC / PC-28 dry-run v1 handoff

## Current ownership

- WEB-PC only owns frontend display/build output.
- PC-28/VPS owns backend API, data, worker, scraper, dry-run logs, and any future execution logic.
- Do not change PC-28 backend/data/worker from the WEB-PC window.

## API

- Keep API fixed:
  - `https://api.startpops.xyz/api/ai-signal`
- Local source already defaults to this endpoint.
- `vite.config.ts` still has `http://localhost:3000` only as local dev proxy fallback.
- No `old.startpops` reference was found.

## Frontend status

- Local dev server was started successfully at `http://localhost:5173/` and then stopped.
- API was reachable with elevated network permissions and returned `ok:true`.
- API latest issue during check: `#3448319`.
- `npm run build` passed.
- Latest build output:
  - `dist/index.html`
  - `dist/assets/index-ByoLdNnY.js`
  - `dist/assets/index-CURCEGP1.css`
- Built JS contains the correct API endpoint.

## Settled betting assumptions

Current dry-run discussion is constrained to:

- Direction: only buy non-straight.
- Stake: fixed 10.
- Settlement:
  - Non-straight result: `+10`
  - Straight result: `-170`
  - Wait: `0`
- Non-straight break-even win rate:
  - `170 / (170 + 10) = 94.44%`

## Useful results from this window

Recent 400-issue replay using API `pressure_series`, buying only non-straight and selecting lowest non-straight risk scores:

| quota rate | max bets | wins | losses | win rate | pnl | max losing streak |
|---:|---:|---:|---:|---:|---:|---:|
| 50% | 200 | 193 | 7 | 96.50% | +740 | 1 |
| 40% | 160 | 155 | 5 | 96.88% | +700 | 1 |
| 30% | 120 | 117 | 3 | 97.50% | +660 | 1 |
| 20% | 80 | 77 | 3 | 96.25% | +260 | 1 |
| 10% | 40 | 40 | 0 | 100.00% | +400 | 0 |

Notes:

- These are short-window replay results only, not proof of long-term edge.
- The user chose to start dry-run v1 at `quota_rate = 0.4`.
- At 400 issues/day, 40% means `quota_limit = 160` dry-run bets/day.

## Risk score idea used in replay

Lower score means safer for buying non-straight.

Inputs:

- Prior current miss streak.
- Prior average non-straight gap.
- Prior max miss gap.
- Prior 30/50/100 straight rates.

Replay score shape:

```text
score =
  recent_100_straight_rate * 100
  + recent_50_straight_rate * 70
  + recent_30_straight_rate * 50
  + (current_miss / avg_gap) * 30
  + (current_miss / max_gap) * 20
```

This is a starting risk score, not a final model.

## Backend handoff instruction

Implement dry-run v1 in the PC-28 backend only.

Requirements:

1. Do not change `/api/ai-signal` path.
2. Do not touch real betting worker.
3. Do not place real bets.
4. Add `dry_run_v1` to the existing API payload.
5. Write one dry-run JSONL record per issue.
6. Backfill prior pending issue when its result is known.
7. Default `quota_rate = 0.4`.
8. Estimate daily quota using 400 issues/day:
   - `quota_limit = 160`
9. If quota is used up, output `WAIT`.

Suggested API field:

```json
{
  "dry_run_v1": {
    "enabled": true,
    "issue": "3448320",
    "decision": "BUY_NON_STRAIGHT",
    "stake": 10,
    "quota_rate": 0.4,
    "quota_used": 12,
    "quota_limit": 160,
    "risk_score": 15.2,
    "reasons": [
      "低非顺风险分",
      "今日额度未用完"
    ],
    "result": "pending",
    "pnl": 0,
    "summary": {
      "bets": 12,
      "wins": 11,
      "losses": 1,
      "win_rate": 0.9167,
      "pnl": -60,
      "max_losing_streak": 1
    }
  }
}
```

Suggested log record:

```json
{
  "issue": "3448320",
  "time": "2026-06-23 05:20:00",
  "decision": "BUY_NON_STRAIGHT",
  "stake": 10,
  "quota_rate": 0.4,
  "quota_used": 12,
  "quota_limit": 160,
  "risk_score": 15.2,
  "features": {
    "current_miss": 12,
    "avg_gap": 17.15,
    "recent_30_rate": 0.0333,
    "recent_50_rate": 0.04,
    "recent_100_rate": 0.03,
    "conditional_straight_rate": 0.05
  },
  "reasons": ["低非顺风险分", "今日额度未用完"],
  "result": "pending",
  "pnl": 0
}
```

Backfill:

- If decision was `BUY_NON_STRAIGHT` and result is non-straight:
  - `result = "win"`, `pnl = 10`
- If decision was `BUY_NON_STRAIGHT` and result is straight:
  - `result = "loss"`, `pnl = -170`
- If decision was `WAIT`:
  - `result = "skip"`, `pnl = 0`

## WEB-PC display after backend is ready

WEB-PC should only display backend-provided fields:

- `dry_run_v1.decision`
- `stake`
- `quota_used / quota_limit`
- `risk_score`
- `reasons`
- `summary.bets`
- `summary.win_rate`
- `summary.pnl`
- `summary.max_losing_streak`

No frontend-side decision calculation.

## Caveats

- Do not treat recent 400-issue replay as long-term proof.
- dry-run v1 is a script/algorithmic dry-run base, not AI by itself.
- AI/LLM decision review can be added later as a separate `ai_review` field.
- PPO is not recommended as the main controller at this stage.
