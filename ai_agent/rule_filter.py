from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .backtest import BASELINE_NAMES, run_backtest, run_baselines
from .casino_env import ACTION_SPACE_SIZE, DIRECTION_LABELS, CasinoEnv
from .config import DEFAULT_INITIAL_BALANCE, LOG_DIR
from .data_adapter import load_history
from .evaluate_current import _format_table
from .file_policy import resolve_history_path, resolve_log_path


Package = tuple[int, ...]


@dataclass(frozen=True)
class FilterParams:
    window: int
    min_score: float
    max_losing_streak: int
    stake_bin: int = 0
    max_active_bet_rate: float = 0.50


DEFAULT_WINDOWS = (30, 50, 100, 200)
DEFAULT_MIN_SCORES = (0.0, 10.0, 30.0, 50.0, 100.0, 200.0)
DEFAULT_MAX_LOSING_STREAKS = (3, 5, 8, 13)
DEFAULT_MAX_ACTIVE_BET_RATE = 0.50
DEFAULT_MAX_ACCEPTABLE_DRAWDOWN = 0.08
DEFAULT_TARGET_ACTIVE_BET_RATE = 0.30
PACKAGE_LABELS = (
    ("non_straight_only",),
    ("big_even",),
    ("small_even",),
    ("big_odd",),
    ("small_odd",),
    ("non_straight_only", "big_even"),
    ("non_straight_only", "small_even"),
    ("big_even", "small_even"),
    ("big_even", "big_odd"),
    ("big_even", "small_odd"),
    ("small_even", "big_odd"),
    ("small_even", "small_odd"),
    ("big_odd", "small_odd"),
    ("big_even", "small_even", "big_odd"),
    ("big_even", "small_even", "small_odd"),
    ("big_even", "big_odd", "small_odd"),
    ("small_even", "big_odd", "small_odd"),
    ("sum_8", "small_even"),
)


def evaluate_filter(
    history: list[dict[str, Any]],
    params: FilterParams,
    *,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
) -> dict[str, Any]:
    decisions = _build_decisions(history, params)
    cursor = {"index": 0}

    def policy(obs, info):
        index = min(cursor["index"], len(decisions) - 1)
        cursor["index"] += 1
        return decisions[index]

    metrics = run_backtest(history, policy=policy, initial_balance=initial_balance)
    metrics["params"] = asdict(params)
    metrics["top_plays"] = _top_plays(decisions)
    return metrics


def evaluate_package_filter(
    history: list[dict[str, Any]],
    params: FilterParams,
    *,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
) -> dict[str, Any]:
    packages = _default_packages()
    decisions = _build_package_decisions(history, params, packages)
    return _run_package_backtest(history, decisions, params, initial_balance=initial_balance)


def tune_filter(
    train_history: list[dict[str, Any]],
    valid_history: list[dict[str, Any]],
    *,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    min_scores: tuple[float, ...] = DEFAULT_MIN_SCORES,
    max_losing_streaks: tuple[int, ...] = DEFAULT_MAX_LOSING_STREAKS,
    max_active_bet_rate: float = DEFAULT_MAX_ACTIVE_BET_RATE,
    package_mode: bool = False,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    evaluator = evaluate_package_filter if package_mode else evaluate_filter
    for window in windows:
        for min_score in min_scores:
            for max_losing_streak in max_losing_streaks:
                params = FilterParams(
                    window=window,
                    min_score=min_score,
                    max_losing_streak=max_losing_streak,
                    max_active_bet_rate=max_active_bet_rate,
                )
                train = evaluator(train_history, params, initial_balance=initial_balance)
                valid = evaluator(valid_history, params, initial_balance=initial_balance)
                candidates.append(
                    {
                        "params": asdict(params),
                        "train": _compact_metrics(train),
                        "valid": _compact_metrics(valid),
                        "score": _selection_score(valid, max_active_bet_rate=max_active_bet_rate),
                    }
                )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return {"best": candidates[0], "candidates": candidates}


def evaluate_splits(
    *,
    train_path: str | Path = "data/train.jsonl",
    valid_path: str | Path = "data/valid.jsonl",
    test_path: str | Path = "data/test.jsonl",
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    max_active_bet_rate: float = DEFAULT_MAX_ACTIVE_BET_RATE,
    package_mode: bool = False,
) -> dict[str, Any]:
    train_history = load_history(resolve_history_path(train_path))
    valid_history = load_history(resolve_history_path(valid_path))
    test_history = load_history(resolve_history_path(test_path))
    tuning = tune_filter(
        train_history,
        valid_history,
        initial_balance=initial_balance,
        max_active_bet_rate=max_active_bet_rate,
        package_mode=package_mode,
    )
    params = FilterParams(**tuning["best"]["params"])

    report: dict[str, Any] = {
        "strategy": "rolling_package_filter" if package_mode else "rolling_all_play_filter",
        "package_mode": package_mode,
        "initial_balance": initial_balance,
        "selection": {
            "method": "train_grid_valid_best_score_max_active_bet_rate_test_once",
            "best": tuning["best"],
            "candidate_count": len(tuning["candidates"]),
            "max_active_bet_rate": max_active_bet_rate,
        },
        "train": _section(train_history, params, initial_balance=initial_balance, source=train_path, package_mode=package_mode),
        "valid": _section(valid_history, params, initial_balance=initial_balance, source=valid_path, package_mode=package_mode),
        "test": _section(test_history, params, initial_balance=initial_balance, source=test_path, package_mode=package_mode),
    }

    walk_forward = []
    for index in range(1, 6):
        path = Path(f"data/walk_forward_{index}.jsonl")
        if resolve_history_path(path).exists():
            history = load_history(path)
            walk_forward.append(_section(history, params, initial_balance=initial_balance, source=path, package_mode=package_mode))
    report["walk_forward"] = walk_forward
    report["walk_forward_audit"] = _walk_forward_audit(walk_forward, params)
    return report


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    output = resolve_log_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output


def format_report(report: dict[str, Any]) -> str:
    best = report["selection"]["best"]
    rows = []
    for name in ("train", "valid", "test"):
        agent = report[name]["agent"]
        rows.append(_metric_cells(name, "filter", agent))
        for baseline in BASELINE_NAMES:
            rows.append(_metric_cells(name, baseline, report[name]["baselines"][baseline]))
    lines = [
        f"策略：{report.get('strategy', 'rolling_all_play_filter')}",
        f"选择：valid best score，候选数 {report['selection']['candidate_count']}",
        f"活跃下注率上限：{float(report['selection'].get('max_active_bet_rate', DEFAULT_MAX_ACTIVE_BET_RATE)) * 100:.2f}%",
        f"参数：{best['params']}",
        "",
        *_format_table(
                ["数据", "策略", "下注数", "活跃下注率", "下注额", "ROI", "最终余额", "盈亏", "最大回撤"],
            rows,
            right_align={2, 3, 4, 5, 6, 7, 8},
        ),
    ]
    walk = report.get("walk_forward") or []
    if walk:
        lines.append("")
        wf_rows = [_metric_cells(f"WF{index}", "filter", section["agent"]) for index, section in enumerate(walk, start=1)]
        lines.extend(
            _format_table(
                ["Walk", "策略", "下注数", "活跃下注率", "下注额", "ROI", "最终余额", "盈亏", "最大回撤"],
                wf_rows,
                right_align={2, 3, 4, 5, 6, 7, 8},
            )
        )
        audit = report.get("walk_forward_audit")
        if isinstance(audit, dict):
            status = "有效" if audit.get("valid") else "无效"
            reasons = ", ".join(str(reason) for reason in audit.get("reasons", [])) or "none"
            lines.extend(["", f"Walk-forward 审计：{status}（{reasons}）"])
    return "\n".join(lines)


def _section(
    history: list[dict[str, Any]],
    params: FilterParams,
    *,
    initial_balance: float,
    source: str | Path,
    package_mode: bool = False,
) -> dict[str, Any]:
    agent = (
        evaluate_package_filter(history, params, initial_balance=initial_balance)
        if package_mode
        else evaluate_filter(history, params, initial_balance=initial_balance)
    )
    baselines = run_baselines(history, initial_balance=initial_balance)
    return {
        "source": str(resolve_history_path(source)),
        "agent": _compact_metrics(agent),
        "agent_top_plays": agent.get("top_plays", []),
        "agent_top_packages": agent.get("top_packages", []),
        "baselines": {name: _compact_metrics(metrics) for name, metrics in baselines.items()},
    }


def _walk_forward_audit(sections: list[dict[str, Any]], params: FilterParams) -> dict[str, Any]:
    if not sections:
        return {"valid": False, "reasons": ["missing_walk_forward"], "segments": 0, "window": params.window}

    segment_steps = [int(section["agent"]["steps"]) for section in sections if isinstance(section.get("agent"), dict)]
    invalid_segments = sum(1 for steps in segment_steps if params.window >= steps)
    all_watch_segments = sum(1 for section in sections if int(section["agent"]["bets"]) == 0)
    reasons: list[str] = []
    if invalid_segments:
        reasons.append("window_not_smaller_than_segment")
    if all_watch_segments == len(sections):
        reasons.append("all_segments_watch_only")

    return {
        "valid": not reasons,
        "reasons": reasons,
        "segments": len(sections),
        "invalid_segments": invalid_segments,
        "all_watch_segments": all_watch_segments,
        "window": params.window,
        "segment_steps": segment_steps,
    }


def _build_decisions(history: list[dict[str, Any]], params: FilterParams) -> list[list[int]]:
    env = CasinoEnv(history, initial_balance=DEFAULT_INITIAL_BALANCE)
    rows = env.history
    history_stats: dict[int, list[float]] = {play: [] for play in range(1, ACTION_SPACE_SIZE)}
    decisions: list[list[int]] = []
    losing_streaks = {play: 0 for play in range(1, ACTION_SPACE_SIZE)}
    active_bets = 0

    for index, row in enumerate(rows):
        best_play = 0
        best_score = float("-inf")
        for play in range(1, ACTION_SPACE_SIZE):
            recent = history_stats[play][-params.window :]
            if len(recent) < params.window:
                continue
            loss_streak = _tail_losing_streak(recent)
            if loss_streak > params.max_losing_streak:
                continue
            score = sum(recent) - _max_drawdown(recent) - loss_streak * 10.0
            if score > best_score:
                best_score = score
                best_play = play
        selected_play = best_play if best_score >= params.min_score else 0
        if selected_play and _would_exceed_active_bet_rate(
            active_bets=active_bets,
            index=index,
            window=params.window,
            max_active_bet_rate=params.max_active_bet_rate,
        ):
            selected_play = 0
        if selected_play:
            active_bets += 1
        decisions.append([selected_play, params.stake_bin])

        for play in range(1, ACTION_SPACE_SIZE):
            profit_loss = _settle_one(env, play, row, params.stake_bin)
            history_stats[play].append(profit_loss)
            losing_streaks[play] = losing_streaks[play] + 1 if profit_loss < 0 else 0

    return decisions


def _build_package_decisions(
    history: list[dict[str, Any]],
    params: FilterParams,
    packages: tuple[Package, ...] | None = None,
) -> list[Package]:
    packages = packages or _default_packages()
    env = CasinoEnv(history, initial_balance=DEFAULT_INITIAL_BALANCE)
    rows = env.history
    history_stats: dict[Package, list[float]] = {package: [] for package in packages}
    decisions: list[Package] = []
    active_bets = 0

    for index, row in enumerate(rows):
        best_package: Package = ()
        best_score = float("-inf")
        for package in packages:
            recent = history_stats[package][-params.window :]
            if len(recent) < params.window:
                continue
            loss_streak = _tail_losing_streak(recent)
            if loss_streak > params.max_losing_streak:
                continue
            score = _package_window_score(recent, len(package), env.min_stake)
            if score > best_score:
                best_score = score
                best_package = package
        selected_package = best_package if best_score >= params.min_score else ()
        if selected_package and _would_exceed_active_bet_rate(
            active_bets=active_bets,
            index=index,
            window=params.window,
            max_active_bet_rate=params.max_active_bet_rate,
        ):
            selected_package = ()
        if selected_package:
            active_bets += 1
        decisions.append(selected_package)

        for package in packages:
            history_stats[package].append(_settle_package(env, package, row, params.stake_bin))

    return decisions


def _settle_one(env: CasinoEnv, play: int, row: dict[str, Any], stake_bin: int) -> float:
    amount_ratio = env._amount_ratio(stake_bin)
    stake = env.min_stake
    settlement = env._settle_bet(play, stake, row, amount_ratio)
    return float(settlement.profit_loss)


def _settle_package(env: CasinoEnv, package: Package, row: dict[str, Any], stake_bin: int) -> float:
    return sum(_settle_one(env, play, row, stake_bin) for play in package)


def _run_package_backtest(
    history: list[dict[str, Any]],
    decisions: list[Package],
    params: FilterParams,
    *,
    initial_balance: float,
) -> dict[str, Any]:
    env = CasinoEnv(history, initial_balance=initial_balance)
    balance = float(initial_balance)
    equity_curve = [balance]
    bets = 0
    wins = 0
    losses = 0
    losing_streak = 0
    longest_losing_streak = 0
    total_staked = 0.0
    stake_per_play = float(env.min_stake)

    for index, row in enumerate(env.history):
        package = decisions[index] if index < len(decisions) else ()
        profit_loss = 0.0
        if package and balance >= stake_per_play * len(package):
            bets += 1
            total_staked += stake_per_play * len(package)
            profit_loss = _settle_package(env, package, row, params.stake_bin)
            if profit_loss > 0.0:
                wins += 1
                losing_streak = 0
            elif profit_loss < 0.0:
                losses += 1
                losing_streak += 1
                longest_losing_streak = max(longest_losing_streak, losing_streak)
        balance += profit_loss
        if balance <= 0.0:
            balance = 0.0
        equity_curve.append(balance)

    return {
        "steps": len(env.history),
        "bets": bets,
        "bet_rate": round(bets / max(len(env.history), 1), 6),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / max(bets, 1), 6),
        "longest_losing_streak": longest_losing_streak,
        "bankruptcies": 1 if balance <= 0.0 else 0,
        "guard_denies": 0,
        "final_balance": round(balance, 6),
        "net_profit": round(balance - initial_balance, 6),
        "max_drawdown": round(_equity_max_drawdown(equity_curve), 6),
        "total_reward": round(balance - initial_balance, 6),
        "total_staked": round(total_staked, 6),
        "roi": round((balance - initial_balance) / max(total_staked, 1.0), 6),
        "equity_curve": [round(value, 6) for value in equity_curve],
        "params": asdict(params),
        "top_packages": _top_packages(decisions),
    }


def _package_window_score(values: list[float], package_size: int, min_stake: float) -> float:
    staked = max(float(package_size) * float(min_stake) * len(values), 1.0)
    roi = sum(values) / staked
    drawdown_rate = _max_drawdown(values) / staked
    loss_streak = _tail_losing_streak(values)
    losing_rate = sum(1 for value in values if value < 0.0) / max(len(values), 1)
    return roi * 1000.0 - drawdown_rate * 1500.0 - loss_streak * 20.0 - losing_rate * 50.0


def _tail_losing_streak(values: list[float]) -> int:
    streak = 0
    for value in reversed(values):
        if value >= 0:
            break
        streak += 1
    return streak


def _max_drawdown(values: list[float]) -> float:
    balance = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        balance += value
        peak = max(peak, balance)
        drawdown = max(drawdown, peak - balance)
    return drawdown


def _equity_max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 0.0
    max_dd = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0.0:
            max_dd = max(max_dd, (peak - value) / peak)
    return max_dd


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    params = metrics.get("params") if isinstance(metrics.get("params"), dict) else {}
    warmup_steps = int(params.get("window", 0) or 0)
    return {
        "steps": metrics["steps"],
        "bets": metrics["bets"],
        "bet_rate": metrics["bet_rate"],
        "active_steps": _active_steps(metrics["steps"], warmup_steps),
        "active_bet_rate": _active_bet_rate(metrics["bets"], metrics["steps"], warmup_steps),
        "final_balance": metrics["final_balance"],
        "net_profit": metrics["net_profit"],
        "max_drawdown": metrics["max_drawdown"],
        "total_staked": metrics.get("total_staked", 0.0),
        "roi": metrics.get("roi", 0.0),
        "longest_losing_streak": metrics.get("longest_losing_streak", 0),
        "bankruptcies": metrics.get("bankruptcies", 0),
    }


def _selection_score(metrics: dict[str, Any], *, max_active_bet_rate: float = DEFAULT_MAX_ACTIVE_BET_RATE) -> float:
    if metrics["bets"] == 0:
        return -1_000_000.0
    active_bet_rate = float(metrics.get("active_bet_rate", metrics["bet_rate"]))
    max_drawdown = float(metrics["max_drawdown"])
    if active_bet_rate > max_active_bet_rate:
        return -1_000_000.0
    if max_drawdown > DEFAULT_MAX_ACCEPTABLE_DRAWDOWN:
        return -1_000_000.0
    active_penalty = max(0.0, active_bet_rate - DEFAULT_TARGET_ACTIVE_BET_RATE) * 200.0
    streak_penalty = float(metrics.get("longest_losing_streak", 0)) * 0.5
    if float(metrics.get("total_staked", 0.0)) > 0.0:
        return float(metrics.get("roi", 0.0)) * 1000.0 - max_drawdown * 800.0 - streak_penalty - active_penalty
    profit_rate = float(metrics["net_profit"]) / max(DEFAULT_INITIAL_BALANCE, 1.0)
    return profit_rate * 1000.0 - max_drawdown * 800.0 - streak_penalty - active_penalty


def _active_steps(steps: int, warmup_steps: int) -> int:
    return max(0, int(steps) - max(0, int(warmup_steps)))


def _active_bet_rate(bets: int, steps: int, warmup_steps: int) -> float:
    return round(float(bets) / max(_active_steps(steps, warmup_steps), 1), 6)


def _would_exceed_active_bet_rate(
    *,
    active_bets: int,
    index: int,
    window: int,
    max_active_bet_rate: float,
) -> bool:
    active_step = int(index) - int(window) + 1
    if active_step <= 0:
        return True
    return (int(active_bets) + 1) / active_step > float(max_active_bet_rate)


def _top_plays(decisions: list[list[int]]) -> list[dict[str, Any]]:
    counts: dict[int, int] = {}
    for play, _amount in decisions:
        counts[play] = counts.get(play, 0) + 1
    return [
        {"play": play, "label": DIRECTION_LABELS.get(play, "unknown"), "count": count}
        for play, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10]
    ]


def _top_packages(decisions: list[Package]) -> list[dict[str, Any]]:
    counts: dict[Package, int] = {}
    for package in decisions:
        counts[package] = counts.get(package, 0) + 1
    return [
        {"package": list(package), "labels": [DIRECTION_LABELS.get(play, "unknown") for play in package], "count": count}
        for package, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10]
    ]


def _default_packages() -> tuple[Package, ...]:
    return tuple(tuple(_play_for_label(label) for label in labels) for labels in PACKAGE_LABELS)


def _play_for_label(label: str) -> int:
    for play, play_label in DIRECTION_LABELS.items():
        if play_label == label:
            return play
    raise ValueError(f"unknown play label: {label}")


def _metric_cells(section: str, name: str, metrics: dict[str, Any]) -> list[str]:
    return [
        section,
        name,
        str(metrics["bets"]),
        f"{float(metrics.get('active_bet_rate', metrics['bet_rate'])) * 100:.2f}%",
        f"{float(metrics.get('total_staked', 0.0)):.2f}",
        f"{float(metrics.get('roi', 0.0)) * 100:.2f}%",
        f"{float(metrics['final_balance']):.2f}",
        f"{float(metrics['net_profit']):.2f}",
        f"{float(metrics['max_drawdown']) * 100:.2f}%",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a rolling all-play filter without hand-picking plays.")
    parser.add_argument("--initial-balance", type=float, default=DEFAULT_INITIAL_BALANCE)
    parser.add_argument("--max-active-bet-rate", type=float, default=DEFAULT_MAX_ACTIVE_BET_RATE)
    parser.add_argument("--package-mode", action="store_true")
    parser.add_argument("--output", default=str(LOG_DIR / "eval_rule_filter_latest.json"))
    parser.add_argument("--table-output", default=str(LOG_DIR / "eval_rule_filter_latest.md"))
    args = parser.parse_args()

    report = evaluate_splits(
        initial_balance=args.initial_balance,
        max_active_bet_rate=args.max_active_bet_rate,
        package_mode=args.package_mode,
    )
    output = save_report(report, args.output)
    table = format_report(report)
    table_path = resolve_log_path(args.table_output)
    table_path.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\n报告：{output}")


if __name__ == "__main__":
    main()
