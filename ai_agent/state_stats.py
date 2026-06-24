from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from .casino_env import SUM_OUTCOME_COUNTS
from .data_adapter import load_history
from .file_policy import resolve_history_path, resolve_log_path


WINDOWS = (30, 50, 100)
MIN_SAMPLES = 200


def build_state_stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [_normalize_row(row) for row in history]
    plays = _candidate_plays()
    context = _build_state_context(rows)
    state_flags = _state_flags(context)
    play_flags = _play_flags(rows, plays)
    rows_out: list[dict[str, Any]] = []

    for state_name, flags in state_flags.items():
        samples, hits_by_play = _count_next_hits(flags, play_flags, max_index=len(rows) - 1)
        for play_name, play in plays.items():
            hits = hits_by_play[play_name]
            observed = hits / samples if samples else 0.0
            break_even = 1.0 / float(play["odds"])
            edge = observed - break_even
            rows_out.append(
                {
                    "state": state_name,
                    "play": play_name,
                    "play_zh": play["zh"],
                    "odds": play["odds"],
                    "samples": samples,
                    "hits": hits,
                    "observed_rate": round(observed, 6),
                    "theoretical_rate": round(float(play["probability"]), 6),
                    "break_even_rate": round(break_even, 6),
                    "edge_vs_break_even": round(edge, 6),
                    "observed_roi": round(observed * float(play["odds"]) - 1.0, 6),
                    "pass": samples >= MIN_SAMPLES and edge > 0.0,
                }
            )

    rows_out.sort(key=lambda row: (row["pass"], row["observed_roi"], row["samples"]), reverse=True)
    return {
        "rows": len(rows),
        "min_samples": MIN_SAMPLES,
        "windows": WINDOWS,
        "candidate_plays": {
            name: {
                "zh": play["zh"],
                "odds": play["odds"],
                "probability": play["probability"],
            }
            for name, play in plays.items()
        },
        "stats": rows_out,
    }


def format_state_stats(report: dict[str, Any]) -> str:
    lines = [
        "# PC28 状态统计",
        "",
        f"样本行数：{report['rows']}",
        f"最低样本门槛：{report['min_samples']}",
        "",
        "## 候选玩法",
        "",
        "| 玩法 | 赔率 | 理论概率 | 盈亏平衡命中率 | 理论ROI |",
        "|---|---:|---:|---:|---:|",
    ]
    for play in report["candidate_plays"].values():
        theoretical_roi = float(play["probability"]) * float(play["odds"]) - 1.0
        lines.append(
            f"| {play['zh']} | {float(play['odds']):.2f} | {float(play['probability']) * 100:.2f}% | {100 / float(play['odds']):.2f}% | {theoretical_roi * 100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## 状态结果",
            "",
            "| 状态 | 玩法 | 样本 | 命中 | 观察命中率 | 盈亏平衡 | 观察ROI | 结论 |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["stats"]:
        verdict = "候选" if row["pass"] else ""
        lines.append(
            f"| {row['state']} | {row['play_zh']} | {row['samples']} | {row['hits']} | {row['observed_rate'] * 100:.2f}% | {row['break_even_rate'] * 100:.2f}% | {row['observed_roi'] * 100:.2f}% | {verdict} |"
        )
    return "\n".join(lines)


def save_report(report: dict[str, Any], json_path: str | Path, md_path: str | Path) -> tuple[Path, Path]:
    json_output = resolve_log_path(json_path)
    md_output = resolve_log_path(md_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_output.write_text(format_state_stats(report) + "\n", encoding="utf-8")
    return json_output, md_output


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    digits = row.get("digits")
    if not digits and isinstance(row.get("raw"), dict):
        digits = row["raw"].get("digits")
    digits = [int(value) for value in digits] if isinstance(digits, list) else []
    total = row.get("sum")
    if total is None and isinstance(row.get("raw"), dict):
        total = row["raw"].get("sum")
    if total is None:
        total = sum(digits)
    return {"digits": digits, "sum": int(float(total))}


def _candidate_plays() -> dict[str, dict[str, Any]]:
    return {
        "big": {"zh": "大", "odds": 1.98, "probability": 0.5, "predicate": lambda row: row["sum"] >= 14},
        "small": {"zh": "小", "odds": 1.98, "probability": 0.5, "predicate": lambda row: row["sum"] <= 13},
        "odd": {"zh": "单", "odds": 1.98, "probability": 0.5, "predicate": lambda row: row["sum"] % 2 == 1},
        "even": {"zh": "双", "odds": 1.98, "probability": 0.5, "predicate": lambda row: row["sum"] % 2 == 0},
        "small_odd": {
            "zh": "小单",
            "odds": 3.69,
            "probability": _sum_probability_for(lambda total: total <= 13 and total % 2 == 1),
            "predicate": lambda row: row["sum"] <= 13 and row["sum"] % 2 == 1,
        },
        "big_even": {
            "zh": "大双",
            "odds": 3.69,
            "probability": _sum_probability_for(lambda total: total >= 14 and total % 2 == 0),
            "predicate": lambda row: row["sum"] >= 14 and row["sum"] % 2 == 0,
        },
        "small_even": {
            "zh": "小双",
            "odds": 4.29,
            "probability": _sum_probability_for(lambda total: total <= 13 and total % 2 == 0),
            "predicate": lambda row: row["sum"] <= 13 and row["sum"] % 2 == 0,
        },
        "big_odd": {
            "zh": "大单",
            "odds": 4.29,
            "probability": _sum_probability_for(lambda total: total >= 14 and total % 2 == 1),
            "predicate": lambda row: row["sum"] >= 14 and row["sum"] % 2 == 1,
        },
        "sum_8": {"zh": "和值8", "odds": 22.0, "probability": SUM_OUTCOME_COUNTS[8] / 1000.0, "predicate": lambda row: row["sum"] == 8},
        "sum_9": {"zh": "和值9", "odds": 18.0, "probability": SUM_OUTCOME_COUNTS[9] / 1000.0, "predicate": lambda row: row["sum"] == 9},
        "sum_18": {"zh": "和值18", "odds": 18.0, "probability": SUM_OUTCOME_COUNTS[18] / 1000.0, "predicate": lambda row: row["sum"] == 18},
        "sum_19": {"zh": "和值19", "odds": 22.0, "probability": SUM_OUTCOME_COUNTS[19] / 1000.0, "predicate": lambda row: row["sum"] == 19},
    }


def _build_state_context(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    straight_flags = [_is_straight(row) for row in rows]
    reverse_flags = [_is_reverse_straight(row) for row in rows]
    straight_omissions: list[int] = []
    reverse_omissions: list[int] = []
    straight_gap = 0
    reverse_gap = 0
    for is_straight, is_reverse in zip(straight_flags, reverse_flags):
        straight_omissions.append(straight_gap)
        reverse_omissions.append(reverse_gap)
        straight_gap = 0 if is_straight else straight_gap + 1
        reverse_gap = 0 if is_reverse else reverse_gap + 1
    context: dict[str, list[int]] = {
        "straight_flags": straight_flags,
        "reverse_flags": reverse_flags,
        "straight_omissions": straight_omissions,
        "reverse_omissions": reverse_omissions,
    }
    for window in WINDOWS:
        context[f"straight_rate_{window}"] = _rolling_prior_rates(straight_flags, window)
        context[f"reverse_rate_{window}"] = _rolling_prior_rates(reverse_flags, window)
    return context


def _state_flags(context: dict[str, list[int]]) -> dict[str, list[bool]]:
    row_count = len(context["straight_flags"])
    states: dict[str, list[bool]] = {"全部": [True] * row_count}
    for window in WINDOWS:
        straight_rates = context[f"straight_rate_{window}"]
        reverse_rates = context[f"reverse_rate_{window}"]
        states[f"近{window}顺率偏高"] = [rate > 0.06 for rate in straight_rates]
        states[f"近{window}顺率偏低"] = [rate < 0.035 for rate in straight_rates]
        states[f"近{window}反顺率偏高"] = [rate > 0.06 for rate in reverse_rates]
        states[f"近{window}反顺率偏低"] = [rate < 0.035 for rate in reverse_rates]
    states["上一期顺"] = [bool(value) for value in context["straight_flags"]]
    states["上一期反顺"] = [bool(value) for value in context["reverse_flags"]]
    states["顺遗漏>=20"] = [value >= 20 for value in context["straight_omissions"]]
    states["反顺遗漏>=20"] = [value >= 20 for value in context["reverse_omissions"]]
    return states


def _rolling_prior_rates(flags: list[bool], window: int) -> list[float]:
    rates: list[float] = []
    running = 0
    for index, value in enumerate(flags):
        if index >= window:
            rates.append(running / window)
            running -= int(flags[index - window])
        else:
            rates.append(0.0)
        running += int(value)
    return rates


def _play_flags(rows: list[dict[str, Any]], plays: dict[str, dict[str, Any]]) -> dict[str, list[bool]]:
    return {
        play_name: [bool(play["predicate"](row)) for row in rows]
        for play_name, play in plays.items()
    }


def _count_next_hits(
    state_flags: list[bool],
    play_flags: dict[str, list[bool]],
    *,
    max_index: int,
) -> tuple[int, dict[str, int]]:
    hits = {play_name: 0 for play_name in play_flags}
    samples = 0
    for index in range(max_index):
        if not state_flags[index]:
            continue
        samples += 1
        next_index = index + 1
        for play_name, flags in play_flags.items():
            hits[play_name] += int(flags[next_index])
    return samples, hits


def _sum_probability_for(predicate: Callable[[int], bool]) -> float:
    return sum(count for total, count in SUM_OUTCOME_COUNTS.items() if predicate(total)) / 1000.0


def _is_straight(row: dict[str, Any]) -> bool:
    digits = sorted(set(row["digits"]))
    if len(digits) < 3:
        return False
    return any(digits[index + 2] - digits[index] == 2 for index in range(len(digits) - 2))


def _is_reverse_straight(row: dict[str, Any]) -> bool:
    digits = row["digits"]
    return len(digits) == 3 and (digits[0] + 1 == digits[1] + 2 == digits[2] + 3 or digits[0] - 1 == digits[1] - 2 == digits[2] - 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PC28 state-conditioned probabilities for near-fair markets.")
    parser.add_argument("--history", default="data/history.jsonl")
    parser.add_argument("--output", default="ai_agent/logs/state_stats_latest.json")
    parser.add_argument("--table-output", default="ai_agent/logs/state_stats_latest.md")
    args = parser.parse_args()

    history = load_history(resolve_history_path(args.history))
    report = build_state_stats(history)
    json_path, md_path = save_report(report, args.output, args.table_output)
    print(format_state_stats(report))
    print(f"\n报告：{json_path}")
    print(f"表格：{md_path}")


if __name__ == "__main__":
    main()
