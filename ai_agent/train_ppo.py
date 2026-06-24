from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed

from .casino_env import ACTION_SPACE_SIZE, CasinoEnv, GameObjectiveConfig
from .config import DEFAULT_INITIAL_BALANCE, DEFAULT_MAX_STAKE, MODEL_DIR, MODEL_PATH
from .data_adapter import FEATURE_COLUMNS, load_history
from .file_policy import resolve_model_path
from .model_policy import resolve_ppo_model_path
from .training_utils import resolve_device, select_torch_device


@dataclass(frozen=True)
class TrainSettings:
    model_path: Path = MODEL_PATH
    total_timesteps: int = 500_000
    initial_balance: float = DEFAULT_INITIAL_BALANCE
    max_stake: float = DEFAULT_MAX_STAKE
    device: str = "auto"
    seed: int = 42
    n_steps: int = 2048
    batch_size: int = 64
    learning_rate: float = 0.0003
    verbose: int = 1
    check_env: bool = True
    reward_mode: str = "normalized"
    game_objective_enabled: bool = False
    game_max_steps: int = 200
    game_pass_balance_multiplier: float = 1.3
    game_fail_balance_multiplier: float = 0.7
    game_target_bet_rate: float = 0.2
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", resolve_ppo_model_path(self.model_path))
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if self.max_stake <= 0:
            raise ValueError("max_stake must be positive")
        if self.n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.reward_mode not in {"pnl", "normalized"}:
            raise ValueError("reward_mode must be one of: pnl, normalized")
        if self.game_max_steps <= 0:
            raise ValueError("game_max_steps must be positive")
        if self.game_pass_balance_multiplier <= 1.0:
            raise ValueError("game_pass_balance_multiplier must be greater than 1")
        if not 0.0 < self.game_fail_balance_multiplier < 1.0:
            raise ValueError("game_fail_balance_multiplier must be between 0 and 1")
        if not 0.0 <= self.game_target_bet_rate <= 1.0:
            raise ValueError("game_target_bet_rate must be between 0 and 1")


def build_ppo_kwargs(settings: TrainSettings) -> dict[str, Any]:
    return {
        "device": resolve_device(settings.device),
        "seed": settings.seed,
        "n_steps": settings.n_steps,
        "batch_size": settings.batch_size,
        "learning_rate": settings.learning_rate,
        "verbose": settings.verbose,
    }


def build_ppo_game_objective(settings: TrainSettings) -> GameObjectiveConfig:
    return GameObjectiveConfig(
        enabled=settings.game_objective_enabled,
        max_steps=settings.game_max_steps,
        pass_balance_multiplier=settings.game_pass_balance_multiplier,
        fail_balance_multiplier=settings.game_fail_balance_multiplier,
        target_bet_rate=settings.game_target_bet_rate,
    )


def metadata_path_for(model_path: str | Path) -> Path:
    path = resolve_model_path(model_path)
    return path.with_name(f"{path.stem}.metadata.json")


def write_training_metadata(
    settings: TrainSettings,
    *,
    history_rows: int,
    resolved_device: str,
    observation_high: list[float],
) -> Path:
    metadata_path = metadata_path_for(settings.model_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_path": str(settings.model_path),
        "history_rows": history_rows,
        "device": resolved_device,
        "feature_columns": FEATURE_COLUMNS,
        "observation_high": observation_high,
        "action_space": f"MultiDiscrete([{ACTION_SPACE_SIZE}, 100])",
        "settings": {**asdict(settings), "model_path": str(settings.model_path)},
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def train_model(
    history_source,
    *,
    model_path: str | Path = MODEL_PATH,
    total_timesteps: int = 500_000,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    max_stake: float = DEFAULT_MAX_STAKE,
    check: bool = True,
    device: str = "auto",
    seed: int = 42,
    n_steps: int = 2048,
    batch_size: int = 64,
    learning_rate: float = 0.0003,
    verbose: int = 1,
    reward_mode: str = "normalized",
    game_objective_enabled: bool = False,
    game_max_steps: int = 200,
    game_pass_balance_multiplier: float = 1.3,
    game_fail_balance_multiplier: float = 0.7,
    game_target_bet_rate: float = 0.2,
    resume: bool = False,
) -> Path:
    settings = TrainSettings(
        model_path=Path(model_path),
        total_timesteps=total_timesteps,
        initial_balance=initial_balance,
        max_stake=max_stake,
        device=device,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=verbose,
        check_env=check,
        reward_mode=reward_mode,
        game_objective_enabled=game_objective_enabled,
        game_max_steps=game_max_steps,
        game_pass_balance_multiplier=game_pass_balance_multiplier,
        game_fail_balance_multiplier=game_fail_balance_multiplier,
        game_target_bet_rate=game_target_bet_rate,
        resume=resume,
    )
    rows = load_history(history_source)
    env = CasinoEnv(
        rows,
        initial_balance=settings.initial_balance,
        max_stake=settings.max_stake,
        reward_mode=settings.reward_mode,
        game_objective=build_ppo_game_objective(settings),
    )
    if settings.check_env:
        check_env(env, warn=True)

    resolved_device = resolve_device(settings.device)
    set_random_seed(settings.seed)
    if settings.resume:
        if not settings.model_path.exists():
            raise FileNotFoundError(f"cannot resume missing model: {settings.model_path}")
        model = PPO.load(settings.model_path, env=env, device=resolved_device)
    else:
        model = PPO("MlpPolicy", env, **build_ppo_kwargs(settings))
    model.learn(total_timesteps=settings.total_timesteps)

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(settings.model_path)
    write_training_metadata(
        settings,
        history_rows=len(env.history),
        resolved_device=resolved_device,
        observation_high=[float(value) for value in env.observation_space.high.tolist()],
    )
    return settings.model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PC28 PPO model.")
    parser.add_argument("history", help="CSV/JSON/JSONL history file")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--capital-game-model", action="store_true", help="Save to ppo_capital_game_v1.zip and enable the game objective.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--initial-balance", type=float, default=DEFAULT_INITIAL_BALANCE)
    parser.add_argument("--max-stake", type=float, default=DEFAULT_MAX_STAKE)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--reward-mode", choices=["pnl", "normalized"], default="normalized")
    parser.add_argument("--game-objective", action="store_true", help="Enable 200-step pass/fail capital-game reward shaping.")
    parser.add_argument("--game-max-steps", type=int, default=200)
    parser.add_argument("--game-pass-multiplier", type=float, default=1.3)
    parser.add_argument("--game-fail-multiplier", type=float, default=0.7)
    parser.add_argument("--game-target-bet-rate", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true", help="Resume training from the target model path instead of starting fresh.")
    parser.add_argument("--skip-check-env", action="store_true")
    args = parser.parse_args()

    model_path = MODEL_DIR / "ppo_capital_game_v1.zip" if args.capital_game_model else args.model
    game_objective_enabled = bool(args.game_objective or args.capital_game_model)
    output = train_model(
        args.history,
        model_path=model_path,
        total_timesteps=args.timesteps,
        initial_balance=args.initial_balance,
        max_stake=args.max_stake,
        check=not args.skip_check_env,
        device=args.device,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        verbose=args.verbose,
        reward_mode=args.reward_mode,
        game_objective_enabled=game_objective_enabled,
        game_max_steps=args.game_max_steps,
        game_pass_balance_multiplier=args.game_pass_multiplier,
        game_fail_balance_multiplier=args.game_fail_multiplier,
        game_target_bet_rate=args.game_target_bet_rate,
        resume=args.resume,
    )
    print(f"saved model: {output}")
    print(f"saved metadata: {metadata_path_for(output)}")


if __name__ == "__main__":
    main()
