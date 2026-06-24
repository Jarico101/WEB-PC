from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib import parse, request

from stable_baselines3 import PPO

from .casino_env import CasinoEnv, DIRECTION_LABELS
from .config import DEFAULT_MAX_STAKE, MODEL_PATH, PROJECT_ROOT
from .data_adapter import clean_history
from .infer import action_to_decision
from .model_policy import resolve_ppo_model_path
from .risk_guard import RuntimeState, evaluate_risk


LIVE_SIGNAL_URL = "https://api.startpops.xyz/api/ai-signal"
SMOKE_MODEL_PATH = PROJECT_ROOT / "ai_agent" / "models" / "ppo_casino_model_smoke.zip"
_MODEL_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class LiveServerConfig:
    api_url: str = LIVE_SIGNAL_URL
    model_path: str = str(SMOKE_MODEL_PATH if SMOKE_MODEL_PATH.exists() else MODEL_PATH)
    host: str = "0.0.0.0"
    port: int = 8787
    timeout: float = 8.0
    balance: float = 1000.0
    max_stake: float = DEFAULT_MAX_STAKE
    initial_balance: float | None = None
    losing_streak: int = 0
    deterministic: bool = True


Predictor = Callable[..., dict[str, Any]]


def fetch_live_payload(api_url: str = LIVE_SIGNAL_URL, *, timeout: float = 8.0) -> dict[str, Any]:
    req = request.Request(api_url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def predict_live_decision(
    payload: dict[str, Any],
    *,
    model_path: str,
    balance: float,
    max_stake: float = DEFAULT_MAX_STAKE,
    deterministic: bool = True,
) -> dict[str, Any]:
    history = clean_history(payload, initial_capital=balance, min_rows=1)
    env = CasinoEnv(history, initial_balance=balance)
    obs, _ = env.reset(options={"start_step": len(history) - 1})
    action, _ = _load_model(model_path).predict(obs, deterministic=deterministic)
    decision = action_to_decision(list(action), balance=balance, max_stake=max_stake)
    decision["issue"] = _extract_issue(payload, history[-1].get("issue"))
    return decision


def build_agent_signal(
    payload: dict[str, Any],
    *,
    model_path: str,
    balance: float = 1000.0,
    max_stake: float = DEFAULT_MAX_STAKE,
    initial_balance: float | None = None,
    losing_streak: int = 0,
    deterministic: bool = True,
    predictor: Predictor = predict_live_decision,
) -> dict[str, Any]:
    pressure_score = _extract_pressure_score(payload)
    issue = str(_extract_issue(payload, ""))
    data_quality = _detect_recent_issue_gap(payload)
    if data_quality["has_gap"]:
        return {
            "ok": True,
            "dry_run": True,
            "issue": issue,
            "direction": 0,
            "direction_label": DIRECTION_LABELS[0],
            "amount_ratio": 0.0,
            "stake": 0.0,
            "allowed": False,
            "guard_reasons": ["data_gap"],
            "pressure_score": pressure_score,
            "ppo": None,
            "data_quality": data_quality,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    ppo = predictor(payload, model_path=model_path, balance=balance, max_stake=max_stake, deterministic=deterministic)
    issue = str(_extract_issue(payload, ppo.get("issue")))
    guard = evaluate_risk(
        direction=int(ppo["direction"]),
        amount_ratio=float(ppo["amount_ratio"]),
        state=RuntimeState(
            issue=issue,
            balance=balance,
            initial_balance=initial_balance or balance,
            losing_streak=losing_streak,
            pressure_score=pressure_score,
            data_age_seconds=0.0,
        ),
    )
    uncapped_amount_ratio = guard.capped_amount_ratio if guard.allowed else 0.0
    direction = guard.direction if guard.allowed else 0
    stake = 0.0 if direction == 0 else min(balance * uncapped_amount_ratio, max_stake)
    amount_ratio = 0.0 if direction == 0 else stake / max(balance, 1e-9)

    return {
        "ok": True,
        "dry_run": True,
        "issue": issue,
        "direction": direction,
        "direction_label": DIRECTION_LABELS.get(direction, "unknown"),
        "amount_ratio": round(amount_ratio, 6),
        "max_stake": round(max_stake, 6),
        "stake": round(stake, 6),
        "allowed": guard.allowed,
        "guard_reasons": list(guard.reasons),
        "pressure_score": pressure_score,
        "ppo": ppo,
        "data_quality": data_quality,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def make_handler(config: LiveServerConfig) -> type[BaseHTTPRequestHandler]:
    class AgentSignalHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = parse.urlparse(self.path)
            if parsed.path == "/healthz":
                self._send_json({"ok": True})
                return
            if parsed.path != "/agent-signal":
                self._send_json({"ok": False, "error": "not_found"}, status=404)
                return

            params = parse.parse_qs(parsed.query)
            try:
                payload = fetch_live_payload(config.api_url, timeout=config.timeout)
                result = build_agent_signal(
                    payload,
                    model_path=config.model_path,
                    balance=_float_param(params, "balance", config.balance),
                    max_stake=_float_param(params, "max_stake", config.max_stake),
                    initial_balance=_optional_float_param(params, "initial_balance", config.initial_balance),
                    losing_streak=_int_param(params, "losing_streak", config.losing_streak),
                    deterministic=config.deterministic,
                )
                self._send_json(result)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return AgentSignalHandler


def serve(config: LiveServerConfig) -> None:
    server = ThreadingHTTPServer((config.host, config.port), make_handler(config))
    print(f"agent-signal listening on http://{config.host}:{config.port}/agent-signal", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve live PPO dry-run signal.")
    parser.add_argument("--api-url", default=os.getenv("LIVE_SIGNAL_URL", LIVE_SIGNAL_URL))
    parser.add_argument("--model", default=os.getenv("PPO_MODEL_PATH", str(SMOKE_MODEL_PATH if SMOKE_MODEL_PATH.exists() else MODEL_PATH)))
    parser.add_argument("--host", default=os.getenv("AGENT_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("AGENT_PORT", "8787")))
    parser.add_argument("--balance", type=float, default=float(os.getenv("AGENT_BALANCE", "1000")))
    parser.add_argument("--max-stake", type=float, default=float(os.getenv("AGENT_MAX_STAKE", str(DEFAULT_MAX_STAKE))))
    parser.add_argument("--initial-balance", type=float, default=None)
    parser.add_argument("--losing-streak", type=int, default=int(os.getenv("AGENT_LOSING_STREAK", "0")))
    args = parser.parse_args()

    serve(
        LiveServerConfig(
            api_url=args.api_url,
            model_path=args.model,
            host=args.host,
            port=args.port,
            balance=args.balance,
            max_stake=args.max_stake,
            initial_balance=args.initial_balance,
            losing_streak=args.losing_streak,
        )
    )


def _load_model(model_path: str):
    key = str(resolve_ppo_model_path(model_path))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = PPO.load(key)
    return _MODEL_CACHE[key]


def _extract_issue(payload: dict[str, Any], fallback: Any) -> str:
    feature = payload.get("feature")
    if isinstance(feature, dict):
        latest = feature.get("latest")
        if isinstance(latest, dict) and latest.get("issue"):
            return str(latest["issue"])
    return str(fallback or "")


def _extract_pressure_score(payload: dict[str, Any]) -> float:
    signal = payload.get("signal")
    if isinstance(signal, dict):
        value = _to_float(signal.get("risk_score"))
        if value is not None:
            return value
    feature = payload.get("feature")
    if isinstance(feature, dict):
        latest = feature.get("latest")
        if isinstance(latest, dict):
            value = _to_float(latest.get("risk_score", latest.get("pressure_score")))
            if value is not None:
                return value
    return 0.0


def _detect_recent_issue_gap(payload: dict[str, Any], *, lookback: int = 80) -> dict[str, Any]:
    feature = payload.get("feature")
    if not isinstance(feature, dict):
        return {"has_gap": False, "checked": 0, "missing_count": 0, "missing_ranges": []}

    issues: list[int] = []
    latest = feature.get("latest")
    if isinstance(latest, dict):
        issue = _to_int(latest.get("issue"))
        if issue is not None:
            issues.append(issue)

    series = feature.get("pressure_series")
    if isinstance(series, list):
        for point in series:
            if isinstance(point, dict):
                issue = _to_int(point.get("issue"))
                if issue is not None:
                    issues.append(issue)

    unique = sorted(set(issues))
    if len(unique) < 2:
        return {"has_gap": False, "checked": len(unique), "missing_count": 0, "missing_ranges": []}

    recent = unique[-lookback:]
    missing_ranges: list[str] = []
    missing_count = 0
    for previous, current in zip(recent, recent[1:]):
        gap = current - previous
        if gap <= 1:
            continue
        start = previous + 1
        end = current - 1
        missing_count += end - start + 1
        missing_ranges.append(str(start) if start == end else f"{start}-{end}")

    return {
        "has_gap": missing_count > 0,
        "checked": len(recent),
        "missing_count": missing_count,
        "missing_ranges": missing_ranges[:8],
    }


def _float_param(params: dict[str, list[str]], name: str, default: float) -> float:
    return _to_float(params.get(name, [default])[0]) or default


def _optional_float_param(params: dict[str, list[str]], name: str, default: float | None) -> float | None:
    if name not in params:
        return default
    return _to_float(params[name][0])


def _int_param(params: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int(params.get(name, [default])[0])
    except (TypeError, ValueError):
        return default


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
