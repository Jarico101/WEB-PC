from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .casino_env import BET_SPECS, CasinoEnv, DIRECTION_LABELS
from .config import DEFAULT_INITIAL_BALANCE, LOG_DIR
from .data_adapter import load_history
from .file_policy import resolve_history_path, resolve_log_path
from .state_stats import _is_reverse_straight, _is_straight, _normalize_row


@dataclass(frozen=True)
class StateRule:
    name: str
    zh: str
    play_label: str
    condition: Callable[[list[dict[str, Any]], int], bool]


@dataclass(frozen=True)
class ConservativeConfig:
    name: str = "conservative_combo_v1"
    zh: str = "保守组合v1：反顺后大双，普通顺后小双"
    warmup_steps: int = 100
    min_prior_signals: int = 12
    rolling_signal_window: int = 30
    max_bet_rate: float = 0.08
    pause_steps_after_loss: int = 20


def evaluate_state_rules(
    history: list[dict[str, Any]],
    *,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    segment_size: int = 1000,
) -> dict[str, Any]:
    rows = [_normalize_row(row) for row in history]
    rules = _candidate_rules()
    segments = []
    for segment_index, start in enumerate(range(0, len(rows), segment_size), start=1):
        segment = rows[start : start + segment_size]
        if len(segment) < max(200, segment_size // 2):
            continue
        segments.append(
            {
                "segment": segment_index,
                "start": start,
                "rows": len(segment),
                "rules": {
                    rule.name: _evaluate_rule(segment, rule, initial_balance=initial_balance)
                    for rule in rules
                },
            }
        )

    return {
        "strategy": "state_rule_filter_candidates",
        "initial_balance": initial_balance,
        "segment_size": segment_size,
        "rules": [
            {"name": rule.name, "zh": rule.zh, "play_label": rule.play_label}
            for rule in rules
        ],
        "segments": segments,
        "summary": _summarize_segments(segments, rules),
    }


def evaluate_conservative_strategy(
    history: list[dict[str, Any]],
    *,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    segment_size: int = 1000,
    config: ConservativeConfig = ConservativeConfig(),
) -> dict[str, Any]:
    rows = [_normalize_row(row) for row in history]
    segments = []
    for segment_index, start in enumerate(range(0, len(rows), segment_size), start=1):
        segment = rows[start : start + segment_size]
        if len(segment) < max(200, segment_size // 2):
            continue
        segments.append(
            {
                "segment": segment_index,
                "start": start,
                "rows": len(segment),
                "strategy": _evaluate_conservative_segment(
                    segment,
                    initial_balance=initial_balance,
                    config=config,
                ),
            }
        )

    return {
        "strategy": config.name,
        "strategy_zh": config.zh,
        "initial_balance": initial_balance,
        "segment_size": segment_size,
        "config": {
            "warmup_steps": config.warmup_steps,
            "min_prior_signals": config.min_prior_signals,
            "rolling_signal_window": config.rolling_signal_window,
            "max_bet_rate": config.max_bet_rate,
            "pause_steps_after_loss": config.pause_steps_after_loss,
        },
        "segments": segments,
        "summary": _summarize_strategy_segments(segments),
    }


def format_report(report: dict[str, Any]) -> str:
    if report.get("strategy") == ConservativeConfig.name:
        return format_conservative_report(report)

    lines = [
        "# 顺/反顺状态规则 holdout",
        "",
        f"分段大小：{report['segment_size']}",
        f"初始余额：{report['initial_balance']:.2f}",
        "",
        "## 规则汇总",
        "",
        "| 规则 | 正收益段 | 总段数 | 总下注 | 总下注额 | 总盈亏 | 汇总ROI | 最差回撤 | 平均下注率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['zh']} | {row['positive_segments']} | {row['segments']} | {row['bets']} | {row['total_staked']:.2f} | {row['net_profit']:.2f} | {row['roi'] * 100:.2f}% | {row['worst_drawdown'] * 100:.2f}% | {row['avg_bet_rate'] * 100:.2f}% |"
        )

    lines.extend(["", "## 分段明细", ""])
    for segment in report["segments"]:
        lines.extend(
            [
                f"### 第 {segment['segment']} 段",
                "",
                "| 规则 | 下注 | 下注率 | 盈亏 | ROI | 最大回撤 | 最长连亏 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(segment["rules"].values(), key=lambda item: item["net_profit"], reverse=True):
            lines.append(
                f"| {row['zh']} | {row['bets']} | {row['bet_rate'] * 100:.2f}% | {row['net_profit']:.2f} | {row['roi'] * 100:.2f}% | {row['max_drawdown'] * 100:.2f}% | {row['longest_losing_streak']} |"
            )
        lines.append("")
    return "\n".join(lines)


def format_conservative_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# 保守状态组合 holdout",
        "",
        f"策略：{report['strategy_zh']}",
        f"分段大小：{report['segment_size']}",
        f"初始余额：{report['initial_balance']:.2f}",
        (
            "参数："
            f"warmup={report['config']['warmup_steps']}，"
            f"min_prior_signals={report['config']['min_prior_signals']}，"
            f"rolling_signal_window={report['config']['rolling_signal_window']}，"
            f"max_bet_rate={report['config']['max_bet_rate'] * 100:.2f}%，"
            f"pause_after_loss={report['config']['pause_steps_after_loss']}"
        ),
        "",
        "## 汇总",
        "",
        "| 正收益段 | 总段数 | 总下注 | 总下注额 | 总盈亏 | 汇总ROI | 最差回撤 | 平均下注率 | 最长连亏 | 结论 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {summary['positive_segments']} | {summary['segments']} | {summary['bets']} | "
            f"{summary['total_staked']:.2f} | {summary['net_profit']:.2f} | "
            f"{summary['roi'] * 100:.2f}% | {summary['worst_drawdown'] * 100:.2f}% | "
            f"{summary['avg_bet_rate'] * 100:.2f}% | {summary['longest_losing_streak']} | "
            f"{summary['verdict']} |"
        ),
        "",
        "## 分段明细",
        "",
        "| 段 | 下注 | 下注率 | 盈亏 | ROI | 最大回撤 | 最长连亏 | 大双 | 小双 | 跳过原因Top |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for segment in report["segments"]:
        row = segment["strategy"]
        skip_text = ", ".join(f"{item['reason']}={item['count']}" for item in row["top_skip_reasons"])
        play_counts = row["play_counts"]
        lines.append(
            f"| {segment['segment']} | {row['bets']} | {row['bet_rate'] * 100:.2f}% | "
            f"{row['net_profit']:.2f} | {row['roi'] * 100:.2f}% | "
            f"{row['max_drawdown'] * 100:.2f}% | {row['longest_losing_streak']} | "
            f"{play_counts.get('big_even', 0)} | {play_counts.get('small_even', 0)} | {skip_text} |"
        )
    return "\n".join(lines)


def save_report(report: dict[str, Any], output: str | Path, table_output: str | Path) -> tuple[Path, Path]:
    json_path = resolve_log_path(output)
    md_path = resolve_log_path(table_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(format_report(report) + "\n", encoding="utf-8")
    return json_path, md_path


def _evaluate_rule(
    rows: list[dict[str, Any]],
    rule: StateRule,
    *,
    initial_balance: float,
) -> dict[str, Any]:
    env = CasinoEnv(rows, initial_balance=initial_balance)
    play = _play_for_label(rule.play_label)
    stake = env.min_stake
    amount_ratio = env._amount_ratio(0)
    balance = float(initial_balance)
    equity_curve = [balance]
    bets = 0
    wins = 0
    losses = 0
    total_staked = 0.0
    losing_streak = 0
    longest_losing_streak = 0

    for index, row in enumerate(env.history):
        if rule.condition(rows, index):
            settlement = env._settle_bet(play, stake, row, amount_ratio)
            profit_loss = float(settlement.profit_loss)
            balance += profit_loss
            bets += 1
            total_staked += stake
            if profit_loss > 0.0:
                wins += 1
                losing_streak = 0
            elif profit_loss < 0.0:
                losses += 1
                losing_streak += 1
                longest_losing_streak = max(longest_losing_streak, losing_streak)
        equity_curve.append(balance)

    return {
        "rule": rule.name,
        "zh": rule.zh,
        "play_label": rule.play_label,
        "steps": len(rows),
        "bets": bets,
        "bet_rate": round(bets / max(len(rows), 1), 6),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / max(bets, 1), 6),
        "total_staked": round(total_staked, 6),
        "final_balance": round(balance, 6),
        "net_profit": round(balance - initial_balance, 6),
        "roi": round((balance - initial_balance) / max(total_staked, 1.0), 6),
        "max_drawdown": round(_max_drawdown(equity_curve), 6),
        "longest_losing_streak": longest_losing_streak,
    }


def _evaluate_conservative_segment(
    rows: list[dict[str, Any]],
    *,
    initial_balance: float,
    config: ConservativeConfig,
) -> dict[str, Any]:
    stake = 10.0
    balance = float(initial_balance)
    equity_curve = [balance]
    bets = 0
    wins = 0
    losses = 0
    total_staked = 0.0
    losing_streak = 0
    longest_losing_streak = 0
    pause_remaining = 0
    prior_signal_returns: list[float] = []
    play_counts: dict[str, int] = {}
    skip_reasons: dict[str, int] = {}

    for index, row in enumerate(rows):
        play_label = _conservative_play_label(rows, index)
        selected_label: str | None = None
        if play_label is not None:
            reason = _conservative_skip_reason(
                index=index,
                bets=bets,
                pause_remaining=pause_remaining,
                prior_signal_returns=prior_signal_returns,
                config=config,
            )
            if reason is None:
                selected_label = play_label
            else:
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

        if selected_label is not None:
            profit_loss = _settle_label(selected_label, stake, row)
            balance += profit_loss
            bets += 1
            total_staked += stake
            play_counts[selected_label] = play_counts.get(selected_label, 0) + 1
            if profit_loss > 0.0:
                wins += 1
                losing_streak = 0
            elif profit_loss < 0.0:
                losses += 1
                losing_streak += 1
                longest_losing_streak = max(longest_losing_streak, losing_streak)
                pause_remaining = max(pause_remaining, config.pause_steps_after_loss)

        equity_curve.append(balance)

        if play_label is not None:
            prior_signal_returns.append(_settle_label(play_label, stake, row))
        if pause_remaining > 0:
            pause_remaining -= 1

    return {
        "steps": len(rows),
        "bets": bets,
        "bet_rate": round(bets / max(len(rows), 1), 6),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / max(bets, 1), 6),
        "total_staked": round(total_staked, 6),
        "final_balance": round(balance, 6),
        "net_profit": round(balance - initial_balance, 6),
        "roi": round((balance - initial_balance) / max(total_staked, 1.0), 6),
        "max_drawdown": round(_max_drawdown(equity_curve), 6),
        "longest_losing_streak": longest_losing_streak,
        "play_counts": play_counts,
        "skip_reasons": skip_reasons,
        "top_skip_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(skip_reasons.items(), key=lambda item: item[1], reverse=True)[:5]
        ],
    }


def _settle_label(label: str, stake: float, row: dict[str, Any]) -> float:
    total = int(row["sum"])
    if label == "big_even":
        matched = total >= 14 and total % 2 == 0
    elif label == "small_even":
        matched = total <= 13 and total % 2 == 0
    elif label == "big_odd":
        matched = total >= 14 and total % 2 == 1
    elif label == "small_odd":
        matched = total <= 13 and total % 2 == 1
    elif label.startswith("sum_"):
        matched = total == int(label.removeprefix("sum_"))
    elif label == "big":
        matched = total >= 14
    elif label == "small":
        matched = total <= 13
    elif label == "odd":
        matched = total % 2 == 1
    elif label == "even":
        matched = total % 2 == 0
    else:
        raise ValueError(f"unsupported lightweight settlement label: {label}")
    if not matched:
        return -stake
    return stake * _gross_odds_for_label(label) - stake


def _gross_odds_for_label(label: str) -> float:
    for spec in BET_SPECS.values():
        if spec.label == label:
            return float(spec.gross_odds)
    raise ValueError(f"unknown play label: {label}")


def _conservative_play_label(rows: list[dict[str, Any]], index: int) -> str | None:
    if index <= 0:
        return None
    previous = rows[index - 1]
    if _is_reverse_straight(previous):
        return "big_even"
    if _is_straight(previous):
        return "small_even"
    return None


def _conservative_skip_reason(
    *,
    index: int,
    bets: int,
    pause_remaining: int,
    prior_signal_returns: list[float],
    config: ConservativeConfig,
) -> str | None:
    if index < config.warmup_steps:
        return "warmup"
    if len(prior_signal_returns) < config.min_prior_signals:
        return "not_enough_prior_signals"
    if pause_remaining > 0:
        return "pause_after_loss"
    active_steps = max(index + 1 - config.warmup_steps, 1)
    if (bets + 1) / active_steps > config.max_bet_rate:
        return "bet_rate_cap"
    recent = prior_signal_returns[-config.rolling_signal_window :]
    if recent and sum(recent) < 0.0:
        return "recent_signal_negative"
    return None


def _summarize_segments(segments: list[dict[str, Any]], rules: list[StateRule]) -> list[dict[str, Any]]:
    rows = []
    for rule in rules:
        metrics = [segment["rules"][rule.name] for segment in segments]
        total_staked = sum(float(item["total_staked"]) for item in metrics)
        net_profit = sum(float(item["net_profit"]) for item in metrics)
        rows.append(
            {
                "rule": rule.name,
                "zh": rule.zh,
                "segments": len(metrics),
                "positive_segments": sum(1 for item in metrics if float(item["net_profit"]) > 0.0),
                "bets": sum(int(item["bets"]) for item in metrics),
                "total_staked": round(total_staked, 6),
                "net_profit": round(net_profit, 6),
                "roi": round(net_profit / max(total_staked, 1.0), 6),
                "worst_drawdown": round(max((float(item["max_drawdown"]) for item in metrics), default=0.0), 6),
                "avg_bet_rate": round(sum(float(item["bet_rate"]) for item in metrics) / max(len(metrics), 1), 6),
            }
        )
    rows.sort(key=lambda row: (row["positive_segments"], row["roi"], -row["worst_drawdown"]), reverse=True)
    return rows


def _summarize_strategy_segments(segments: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [segment["strategy"] for segment in segments]
    total_staked = sum(float(item["total_staked"]) for item in metrics)
    net_profit = sum(float(item["net_profit"]) for item in metrics)
    worst_drawdown = max((float(item["max_drawdown"]) for item in metrics), default=0.0)
    positive_segments = sum(1 for item in metrics if float(item["net_profit"]) > 0.0)
    avg_bet_rate = sum(float(item["bet_rate"]) for item in metrics) / max(len(metrics), 1)
    longest_losing_streak = max((int(item["longest_losing_streak"]) for item in metrics), default=0)
    verdict = "继续观察"
    if metrics:
        positive_rate = positive_segments / len(metrics)
        roi = net_profit / max(total_staked, 1.0)
        if positive_rate >= 0.65 and roi >= 0.0 and worst_drawdown <= 0.08 and avg_bet_rate <= 0.08:
            verdict = "可进下一轮holdout，不可直接dry-run"
        elif net_profit < 0.0 or positive_rate < 0.55:
            verdict = "不稳定"
    return {
        "segments": len(metrics),
        "positive_segments": positive_segments,
        "bets": sum(int(item["bets"]) for item in metrics),
        "total_staked": round(total_staked, 6),
        "net_profit": round(net_profit, 6),
        "roi": round(net_profit / max(total_staked, 1.0), 6),
        "worst_drawdown": round(worst_drawdown, 6),
        "avg_bet_rate": round(avg_bet_rate, 6),
        "longest_losing_streak": longest_losing_streak,
        "verdict": verdict,
    }


def _candidate_rules() -> list[StateRule]:
    return [
        StateRule("prev_reverse_non_straight_only", "上一期反顺 -> 顺/反顺只买非顺", "non_straight_only", _previous_reverse),
        StateRule("prev_reverse_big_even", "上一期反顺 -> 大双", "big_even", _previous_reverse),
        StateRule("prev_reverse_even", "上一期反顺 -> 双", "even", _previous_reverse),
        StateRule("prev_reverse_small_even", "上一期反顺 -> 小双", "small_even", _previous_reverse),
        StateRule("prev_reverse_sum_8", "上一期反顺 -> 和值8", "sum_8", _previous_reverse),
        StateRule("reverse_rate_30_high_non_straight_only", "近30反顺率偏高 -> 顺/反顺只买非顺", "non_straight_only", _reverse_rate_high_30),
        StateRule("reverse_rate_30_high_sum_8", "近30反顺率偏高 -> 和值8", "sum_8", _reverse_rate_high_30),
        StateRule("reverse_rate_30_high_sum_19", "近30反顺率偏高 -> 和值19", "sum_19", _reverse_rate_high_30),
        StateRule("reverse_rate_50_high_non_straight_only", "近50反顺率偏高 -> 顺/反顺只买非顺", "non_straight_only", _reverse_rate_high_50),
        StateRule("reverse_rate_50_high_sum_8", "近50反顺率偏高 -> 和值8", "sum_8", _reverse_rate_high_50),
        StateRule("reverse_rate_50_high_sum_9", "近50反顺率偏高 -> 和值9", "sum_9", _reverse_rate_high_50),
        StateRule("reverse_rate_50_high_sum_19", "近50反顺率偏高 -> 和值19", "sum_19", _reverse_rate_high_50),
        StateRule("reverse_rate_50_high_small_even", "近50反顺率偏高 -> 小双", "small_even", _reverse_rate_high_50),
        StateRule("reverse_rate_100_high_non_straight_only", "近100反顺率偏高 -> 顺/反顺只买非顺", "non_straight_only", _reverse_rate_high_100),
        StateRule("reverse_rate_100_high_sum_8", "近100反顺率偏高 -> 和值8", "sum_8", _reverse_rate_high_100),
        StateRule("reverse_rate_100_high_sum_19", "近100反顺率偏高 -> 和值19", "sum_19", _reverse_rate_high_100),
        StateRule("reverse_rate_100_high_small_even", "近100反顺率偏高 -> 小双", "small_even", _reverse_rate_high_100),
        StateRule("reverse_rate_100_high_big_odd", "近100反顺率偏高 -> 大单", "big_odd", _reverse_rate_high_100),
        StateRule("prev_straight_non_straight_only", "上一期顺 -> 顺/反顺只买非顺", "non_straight_only", _previous_straight),
        StateRule("prev_straight_sum_9", "上一期顺 -> 和值9", "sum_9", _previous_straight),
        StateRule("prev_straight_sum_18", "上一期顺 -> 和值18", "sum_18", _previous_straight),
        StateRule("prev_straight_small_even", "上一期顺 -> 小双", "small_even", _previous_straight),
        StateRule("prev_straight_big_even", "上一期顺 -> 大双", "big_even", _previous_straight),
    ]


def _previous_reverse(rows: list[dict[str, Any]], index: int) -> bool:
    return index > 0 and _is_reverse_straight(rows[index - 1])


def _previous_straight(rows: list[dict[str, Any]], index: int) -> bool:
    return index > 0 and _is_straight(rows[index - 1])


def _reverse_rate_high_30(rows: list[dict[str, Any]], index: int) -> bool:
    return _rate(rows, index, 30, _is_reverse_straight) > 0.06


def _reverse_rate_high_50(rows: list[dict[str, Any]], index: int) -> bool:
    return _rate(rows, index, 50, _is_reverse_straight) > 0.06


def _reverse_rate_high_100(rows: list[dict[str, Any]], index: int) -> bool:
    return _rate(rows, index, 100, _is_reverse_straight) > 0.06


def _rate(
    rows: list[dict[str, Any]],
    index: int,
    window: int,
    predicate: Callable[[dict[str, Any]], bool],
) -> float:
    if index < window:
        return 0.0
    sample = rows[index - window : index]
    return sum(1 for row in sample if predicate(row)) / max(len(sample), 1)


def _play_for_label(label: str) -> int:
    for play, play_label in DIRECTION_LABELS.items():
        if play_label == label:
            return play
    raise ValueError(f"unknown play label: {label}")


def _max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 0.0
    max_dd = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0.0:
            max_dd = max(max_dd, (peak - value) / peak)
    return max_dd


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed state-conditioned PC28 rules.")
    parser.add_argument("--history", default="data/pc28_round_20000_raw.jsonl")
    parser.add_argument("--segment-size", type=int, default=1000)
    parser.add_argument("--initial-balance", type=float, default=DEFAULT_INITIAL_BALANCE)
    parser.add_argument("--mode", choices=("rules", "conservative"), default="rules")
    parser.add_argument("--output", default=str(LOG_DIR / "state_filter_latest.json"))
    parser.add_argument("--table-output", default=str(LOG_DIR / "state_filter_latest.md"))
    args = parser.parse_args()

    history = load_history(resolve_history_path(args.history))
    if args.mode == "conservative":
        report = evaluate_conservative_strategy(
            history,
            initial_balance=args.initial_balance,
            segment_size=args.segment_size,
        )
    else:
        report = evaluate_state_rules(history, initial_balance=args.initial_balance, segment_size=args.segment_size)
    json_path, md_path = save_report(report, args.output, args.table_output)
    print(format_report(report))
    print(f"\n报告：{json_path}")
    print(f"表格：{md_path}")


if __name__ == "__main__":
    main()
