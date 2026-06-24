from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "ai_agent" / "models" / "ppo_casino_model.zip"
MODEL_DIR = PROJECT_ROOT / "ai_agent" / "models"
LOG_DIR = PROJECT_ROOT / "ai_agent" / "logs"
DEFAULT_INITIAL_BALANCE = 12_000.0
DEFAULT_MAX_STAKE = 10_000.0


@dataclass(frozen=True)
class AgentConfig:
    initial_balance: float = DEFAULT_INITIAL_BALANCE
    max_stake: float = DEFAULT_MAX_STAKE
    straight_profit_multiplier: float = 17.0
    non_straight_profit_multiplier: float = 1.0
    non_straight_loss_multiplier: float = 20.0
    total_timesteps: int = 500_000
    dry_run: bool = True
