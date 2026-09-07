"""Virtual-only random PC28 experiment with structural 13/14 loss reduction.

This is not a predictive model.  It always covers small-odd and big-even,
which makes a 13 or 14 outcome return only that selected market's principal,
under the user-specified experimental settlement rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
from typing import Any, Sequence

from .data_adapter import clean_combo_history


EXPERIMENTAL_CLASSIFICATION = "EXPERIMENTAL_NON_CHAMPION"
BASE_COMBINATIONS = (
    ("small_odd", "big_odd", "big_even"),
    ("small_odd", "small_even", "big_even"),
)
SUM_POOL = (8, 9, 18, 19)
GROSS_ODDS = {
    "small_odd": 4.6,
    "big_even": 4.6,
    "small_even": 4.2,
    "big_odd": 4.2,
    "sum_08": 22.0,
    "sum_09": 18.0,
    "sum_18": 18.0,
    "sum_19": 22.0,
}


@dataclass(frozen=True)
class Hedged1314Config:
    seed: int = 42
    base_stake: int = 30
    extension_stake: int = 30
    extension_probability: float = 0.50
    stop_loss_ratio: float = 0.20
    take_profit_ratio: float = 2.0
    max_drawdown: float = 0.35
    loss_streak_for_cooldown: int = 3
    cooldown_issues: int = 10

    def __post_init__(self) -> None:
        if self.base_stake != 30 or self.extension_stake != 30:
            raise ValueError("this experimental profile fixes every selected stake at the $30 minimum")
        if not 0 <= self.extension_probability <= 1:
            raise ValueError("extension_probability must be within [0, 1]")
        if not 0 < self.stop_loss_ratio <= 1 or self.take_profit_ratio <= 0 or not 0 < self.max_drawdown <= 1:
            raise ValueError("risk ratios are invalid")
        if self.loss_streak_for_cooldown <= 0 or self.cooldown_issues <= 0:
            raise ValueError("cooldown values must be positive")


def run_hedged_1314_backtest(
    history: Any,
    *,
    initial_balance: float = 10_000.0,
    config: Hedged1314Config = Hedged1314Config(),
    max_steps: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a chronological, random-only virtual simulation.

    A 13 returns only the small-odd stake; a 14 returns only the big-even
    stake.  All other winning base outcomes use the user-specified gross odds.
    Exact-sum extensions retain their existing gross odds and never include
    13 or 14.
    """

    rows = clean_combo_history(history, min_rows=1)
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        rows = rows[:max_steps]
    rng = random.Random(config.seed)
    balance = float(initial_balance)
    peak = balance
    losing_streak = 0
    cooldown = 0
    decisions: list[dict[str, Any]] = []
    termination_reason: str | None = None

    for row in rows:
        drawdown = (peak - balance) / max(peak, 1e-9)
        wait_reason = _wait_reason(balance, initial_balance, drawdown, config)
        if wait_reason is None and cooldown:
            wait_reason = "loss_streak_cooldown"
        if wait_reason is not None:
            decisions.append(_wait_record(row, balance, cooldown, wait_reason))
            if cooldown:
                cooldown -= 1
                if cooldown == 0:
                    losing_streak = 0
            if wait_reason != "loss_streak_cooldown":
                termination_reason = wait_reason
                break
            continue

        base = BASE_COMBINATIONS[rng.randrange(len(BASE_COMBINATIONS))]
        extension = rng.choice(SUM_POOL) if rng.random() < config.extension_probability else None
        selections = [{"market": market, "stake": config.base_stake, "gross_odds": GROSS_ODDS[market]} for market in base]
        if extension is not None:
            selections.append({"market": f"sum_{extension:02d}", "stake": config.extension_stake, "gross_odds": GROSS_ODDS[f"sum_{extension:02d}"]})
        total_stake = float(sum(item["stake"] for item in selections))
        if balance < total_stake:
            decisions.append(_wait_record(row, balance, cooldown, "insufficient_balance"))
            termination_reason = "insufficient_balance"
            break
        total_return, items = _settle(selections, int(row["sum"]))
        net_profit = total_return - total_stake
        before = balance
        balance += net_profit
        peak = max(peak, balance)
        if net_profit < 0:
            losing_streak += 1
            if losing_streak >= config.loss_streak_for_cooldown:
                cooldown = config.cooldown_issues
        else:
            losing_streak = 0
        decisions.append(
            {
                "issue": row["issue"],
                "outcome_sum": int(row["sum"]),
                "wait": False,
                "base_combination": list(base),
                "extension": f"sum_{extension:02d}" if extension is not None else None,
                "selections": items,
                "balance_before": _money(before),
                "total_stake": _money(total_stake),
                "total_return": _money(total_return),
                "net_profit": _money(net_profit),
                "balance": _money(balance),
                "losing_streak": losing_streak,
                "cooldown_remaining": cooldown,
                "settlement_note": "13/14 return only the matching small_odd/big_even principal",
            }
        )

    executed = [item for item in decisions if not item["wait"]]
    waits = [item for item in decisions if item["wait"]]
    special = [item for item in executed if item["outcome_sum"] in (13, 14)]
    result = {
        "schema_version": "pc28-experimental-hedged-1314-v1",
        "classification": EXPERIMENTAL_CLASSIFICATION,
        "champion_eligible": False,
        "prediction_claim": "none_random_selection_only_structural_loss_reduction",
        "validation_accessed": False,
        "test_accessed": False,
        "rule": {
            "base_combinations": [list(item) for item in BASE_COMBINATIONS],
            "base_selection": "equal_probability_random",
            "base_stakes": "$30 each",
            "extension_probability": config.extension_probability,
            "extension_pool": [f"sum_{item:02d}" for item in SUM_POOL],
            "special_13_14_settlement": "matching small_odd or big_even returns principal only",
            "ordinary_base_gross_odds": {"small_odd": 4.6, "big_even": 4.6, "small_even": 4.2, "big_odd": 4.2},
        },
        "config": asdict(config),
        "input": {"rows": len(rows), "first_issue": rows[0]["issue"], "last_issue": rows[-1]["issue"]},
        "summary": {
            "steps": len(decisions),
            "executed_rounds": len(executed),
            "wait_rounds": len(waits),
            "wins": sum(item["net_profit"] > 0 for item in executed),
            "losses": sum(item["net_profit"] < 0 for item in executed),
            "final_balance": _money(balance),
            "net_profit": _money(balance - initial_balance),
            "max_drawdown": _round(_max_drawdown([initial_balance, *[item.get("balance", initial_balance) for item in decisions]])),
            "total_exposure": _money(sum(item["total_stake"] for item in executed)),
            "special_13_14_rounds": len(special),
            "special_13_14_net_profit": _money(sum(item["net_profit"] for item in special)),
            "termination_reason": termination_reason,
        },
        "decisions": decisions,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _settle(selections: Sequence[dict[str, Any]], outcome: int) -> tuple[float, list[dict[str, Any]]]:
    total_return = 0.0
    items: list[dict[str, Any]] = []
    for selection in selections:
        market = str(selection["market"])
        stake = float(selection["stake"])
        if market == "small_odd":
            hit = outcome <= 13 and outcome % 2 == 1
        elif market == "small_even":
            hit = 1 <= outcome <= 13 and outcome % 2 == 0
        elif market == "big_odd":
            hit = outcome >= 14 and outcome % 2 == 1
        elif market == "big_even":
            hit = outcome >= 14 and outcome % 2 == 0
        else:
            hit = outcome == int(market.removeprefix("sum_"))
        special_principal_only = hit and ((outcome == 13 and market == "small_odd") or (outcome == 14 and market == "big_even"))
        returned = stake if special_principal_only else stake * float(selection["gross_odds"]) if hit else 0.0
        total_return += returned
        items.append({**selection, "hit": hit, "return_amount": _money(returned), "net_contribution": _money(returned - stake), "special_principal_only": special_principal_only})
    return _money(total_return), items


def _wait_reason(balance: float, initial: float, drawdown: float, config: Hedged1314Config) -> str | None:
    if balance <= initial * (1 - config.stop_loss_ratio):
        return "stop_loss"
    if balance >= initial * (1 + config.take_profit_ratio):
        return "take_profit"
    if drawdown >= config.max_drawdown:
        return "max_drawdown"
    return None


def _wait_record(row: dict[str, Any], balance: float, cooldown: int, reason: str) -> dict[str, Any]:
    return {"issue": row["issue"], "outcome_sum": int(row["sum"]), "wait": True, "wait_reason": reason, "balance_before": _money(balance), "balance": _money(balance), "total_stake": 0.0, "total_return": 0.0, "net_profit": 0.0, "cooldown_remaining": cooldown}


def _max_drawdown(values: Sequence[float]) -> float:
    peak, result = values[0], 0.0
    for value in values:
        peak = max(peak, value)
        result = max(result, (peak - value) / peak if peak else 0.0)
    return result


def _money(value: float) -> float:
    return round(float(value), 2)


def _round(value: float) -> float:
    return round(float(value), 8)
