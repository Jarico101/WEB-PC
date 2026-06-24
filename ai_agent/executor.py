from __future__ import annotations

import json
from typing import Any
from urllib import request


DIRECTION_PAYLOAD = {
    1: "straight",
    2: "non_straight",
}


def build_bet_payload(*, issue: str, direction: int, stake: float) -> dict[str, Any]:
    if direction not in DIRECTION_PAYLOAD:
        raise ValueError("executor only accepts bet directions 1 or 2")
    return {
        "issue": str(issue),
        "direction": DIRECTION_PAYLOAD[direction],
        "amount": round(float(stake), 6),
    }


def execute_bet(
    *,
    api_url: str,
    issue: str,
    direction: int,
    stake: float,
    dry_run: bool = True,
    timeout: float = 8.0,
) -> dict[str, Any]:
    payload = build_bet_payload(issue=issue, direction=direction, stake=stake)
    if dry_run:
        return {"ok": True, "dry_run": True, "payload": payload}

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(api_url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return {"ok": 200 <= response.status < 300, "status": response.status, "payload": payload, "response": raw}
    except Exception as exc:
        return {"ok": False, "payload": payload, "error": str(exc)}

