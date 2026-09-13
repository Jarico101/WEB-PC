"""Isolated non-predictive random PC28 simulation mode.

This module is deliberately not a PPO/DQN trainer, auto-tuning input, or
Champion candidate.  It provides a transparent virtual-risk experiment for a
user-requested random three-base portfolio and optional 08/09/18/19 sum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import random

from .backtest_combo import ComboPolicy, run_combo_backtest
from .markets_combo import ZeroPolicy
from .portfolio_combo import BASE_COMBINATIONS, EXPERIMENTAL_SUM_EXTENSION_PAIRS, WAIT_BASE_ACTION_ID
from .statistical_combo_policy import OpportunityGateConfig


EXPERIMENTAL_CLASSIFICATION = "EXPERIMENTAL_NON_CHAMPION"


@dataclass(frozen=True)
class ExperimentalRandomConfig:
    """Controls loss pace only; it does not claim any outcome prediction."""

    seed: int = 42
    sum_extension_probability: float = 0.50
    stop_loss_ratio: float = 0.20
    # 200% profit target: stop when balance reaches 3x the initial balance.
    take_profit_ratio: float = 2.0
    max_drawdown: float = 0.35

    def __post_init__(self) -> None:
        if not 0 <= self.sum_extension_probability <= 1:
            raise ValueError("sum_extension_probability must be within [0, 1]")
        if not 0 < self.stop_loss_ratio <= 1 or not 0 < self.take_profit_ratio:
            raise ValueError("stop-loss must be within (0, 1] and take-profit must be positive")
        if not 0 < self.max_drawdown <= 1:
            raise ValueError("max_drawdown must be within (0, 1]")


def experimental_random_policy(config: ExperimentalRandomConfig = ExperimentalRandomConfig()) -> ComboPolicy:
    """Randomly choose exactly three base markets and zero/one allowed sum.

    The stateful closure uses only prior simulated account information.  It
    never reads the current outcome; the environment separately applies its
    ten-issue cooldown after three losing actual virtual orders.
    """

    rng = random.Random(config.seed)
    initial_balance: float | None = None
    peak_balance: float | None = None

    def policy(_obs: Any, info: dict[str, Any]) -> list[int]:
        nonlocal initial_balance, peak_balance
        balance = float(info.get("balance", 0.0))
        initial_balance = balance if initial_balance is None else initial_balance
        peak_balance = max(peak_balance if peak_balance is not None else balance, balance)
        drawdown = (peak_balance - balance) / max(peak_balance, 1e-9)
        if (
            balance <= initial_balance * (1.0 - config.stop_loss_ratio)
            or balance >= initial_balance * (1.0 + config.take_profit_ratio)
            or drawdown >= config.max_drawdown
        ):
            return [WAIT_BASE_ACTION_ID, 0, 0, 0]
        base_id = rng.randrange(len(BASE_COMBINATIONS))
        extension_id = rng.randrange(1, len(EXPERIMENTAL_SUM_EXTENSION_PAIRS)) if rng.random() < config.sum_extension_probability else 0
        profile_id = rng.randrange(3)
        mode_id = rng.randrange(3)
        return [base_id, extension_id, profile_id, mode_id]

    setattr(policy, "classification", EXPERIMENTAL_CLASSIFICATION)
    setattr(policy, "config", asdict(config))
    return policy


def run_experimental_random_backtest(
    history: Any,
    *,
    initial_balance: float = 10_000.0,
    config: ExperimentalRandomConfig = ExperimentalRandomConfig(),
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
) -> dict[str, Any]:
    """Run an auditable virtual-only random experiment, never a Champion test."""

    policy = experimental_random_policy(config)
    result = run_combo_backtest(
        history,
        policy=policy,
        initial_balance=initial_balance,
        zero_policy=zero_policy,
        allow_extensions=True,
        extension_options=EXPERIMENTAL_SUM_EXTENSION_PAIRS,
        # The environment's cooldown remains active.  This explicit config is
        # metadata for the non-predictive simulation, not a probability gate.
        gate_config=OpportunityGateConfig(
            min_expected_return_ratio=-0.01,
            allow_negative_expected_return=True,
            experimental_non_champion=True,
            require_probability_validation=False,
            max_drawdown=config.max_drawdown,
        ),
    )
    return {
        "classification": EXPERIMENTAL_CLASSIFICATION,
        "champion_eligible": False,
        "test_accessed": False,
        "prediction_claim": "none_random_selection_only",
        "config": asdict(config),
        "allowed_extensions": [list(item) for item in EXPERIMENTAL_SUM_EXTENSION_PAIRS],
        "result": result,
    }
