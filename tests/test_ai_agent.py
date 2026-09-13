import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class DataAdapterTests(unittest.TestCase):
    def test_clean_history_builds_training_rows_without_future_leakage(self):
        from ai_agent.data_adapter import FEATURE_COLUMNS, clean_history

        rows = [
            {"issue": "1001", "is_straight": False},
            {"issue": "1002", "is_straight": False},
            {"issue": "1003", "is_straight": True},
            {"issue": "1004", "is_straight": False},
            {"issue": "1005", "is_straight": True},
        ]

        cleaned = clean_history(rows, bot_win_rate=0.945, min_rows=3)

        self.assertEqual([row["issue"] for row in cleaned], ["1001", "1002", "1003", "1004", "1005"])
        self.assertEqual([row["current_omission"] for row in cleaned], [0, 1, 2, 0, 1])
        self.assertEqual(cleaned[2]["window_density_100"], 0.0)
        self.assertAlmostEqual(cleaned[3]["window_density_100"], 1 / 3)
        self.assertTrue(cleaned[2]["outcome_is_straight"])
        self.assertEqual(set(FEATURE_COLUMNS), set(cleaned[0]["features"].keys()))

    def test_history_feature_helpers_require_current_index_guard(self):
        from ai_agent.data_adapter import _conditional_rate, _window_density

        rows = [
            {"outcome_is_straight": False},
            {"outcome_is_straight": True},
            {"outcome_is_straight": True},
        ]

        self.assertEqual(_window_density(rows, index=1, window_size=100, current_index=1), 0.0)
        self.assertEqual(
            _conditional_rate(rows, [0], current_omission=0, fallback=0.5, current_index=1),
            0.0,
        )
        with self.assertRaises(AssertionError):
            _window_density(rows, index=1, window_size=100, current_index=2)
        with self.assertRaises(AssertionError):
            _conditional_rate(rows, [0], current_omission=0, fallback=0.5, current_index=2)

    def test_prediction_row_features_only_use_prior_draws(self):
        from ai_agent.data_adapter import clean_history

        rows = [
            {"issue": "998", "is_straight": True},
            {"issue": "999", "is_straight": False},
            {
                "issue": "1000",
                "is_straight": True,
                "pressure_score": 99,
                "bot_win_rate": 0.01,
            },
        ]

        cleaned = clean_history(rows, bot_win_rate=0.945, min_rows=3)
        target = cleaned[2]

        self.assertTrue(target["outcome_is_straight"])
        self.assertEqual(target["features"]["current_omission"], 1.0)
        self.assertAlmostEqual(target["features"]["window_density_100"], 0.5)
        self.assertEqual(target["features"]["bot_win_rate"], 0.945)
        self.assertNotEqual(target["features"]["pressure_score"], 99)

    def test_clean_history_rejects_result_leakage_inside_features(self):
        from ai_agent.data_adapter import DataAdapterError, clean_history

        with self.assertRaises(DataAdapterError):
            clean_history(
                [
                    {"issue": "1001", "is_straight": False, "features": {"sum": 13}},
                    {"issue": "1002", "is_straight": True},
                ],
                min_rows=1,
            )


class CasinoEnvTests(unittest.TestCase):
    def test_environment_settles_agent_action_when_threshold_allows_it(self):
        from ai_agent.casino_env import CasinoEnv, TrainingGateConfig
        from ai_agent.data_adapter import clean_history

        cleaned = clean_history(
            [
                {"issue": "1001", "is_straight": False, "pressure_score": 90},
                {"issue": "1002", "is_straight": True, "pressure_score": 10},
                {"issue": "1003", "is_straight": False, "pressure_score": 50},
            ],
            min_rows=3,
        )
        env = CasinoEnv(
            cleaned,
            initial_balance=1000.0,
            training_gate=TrainingGateConfig(straight_min_rate=0.05, negative_edge_penalty_multiplier=0.0),
        )
        obs, info = env.reset(seed=7)

        self.assertEqual(obs.shape, (8,))
        obs, reward, terminated, truncated, info = env.step([1, 99])

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["direction"], "straight")
        self.assertEqual(info["stake"], 100.0)
        self.assertEqual(info["profit_loss"], -100.0)
        self.assertEqual(info["bet_cost"], 0.05)
        self.assertEqual(info["balance"], 900.0)
        self.assertEqual(reward, -115.05)
        self.assertIn("high_stake", info["gate_reasons"])

    def test_training_risk_penalizes_gray_zone_without_blocking_action(self):
        from ai_agent.casino_env import CasinoEnv

        env = CasinoEnv(
            [
                {
                    "issue": "1001",
                    "is_straight": False,
                    "features": {"cond_straight_rate": 0.05},
                }
            ],
            initial_balance=1000.0,
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([2, 99])

        self.assertEqual(info["intended_direction"], "non_straight")
        self.assertEqual(info["direction"], "non_straight")
        self.assertFalse(info["action_forced_to_watch"])
        self.assertIn("gray_zone_bet", info["risk_reasons"])
        self.assertEqual(info["stake"], 100.0)
        self.assertEqual(info["profit_loss"], 100.0)
        self.assertLess(reward, info["profit_loss"])

    def test_training_risk_penalizes_cooldown_without_blocking_action(self):
        from ai_agent.casino_env import CasinoEnv, TrainingGateConfig

        env = CasinoEnv(
            [
                {"issue": "1001", "is_straight": True, "features": {"cond_straight_rate": 0.0}},
                {"issue": "1002", "is_straight": False, "features": {"cond_straight_rate": 0.0}},
                {"issue": "1003", "is_straight": False, "features": {"cond_straight_rate": 0.0}},
            ],
            initial_balance=1000.0,
            training_gate=TrainingGateConfig(non_straight_max_rate=0.06, straight_min_rate=0.07),
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, first = env.step([2, 0])
        obs, reward, terminated, truncated, second = env.step([2, 99])

        self.assertFalse(terminated)
        self.assertEqual(first["direction"], "non_straight")
        self.assertEqual(first["profit_loss"], -200.0)
        self.assertEqual(second["direction"], "non_straight")
        self.assertFalse(second["action_forced_to_watch"])
        self.assertIn("cooldown_violation", second["risk_reasons"])
        self.assertLess(reward, second["profit_loss"])

    def test_training_risk_penalizes_drawdown_without_blocking_action(self):
        from ai_agent.casino_env import CasinoEnv

        env = CasinoEnv(
            [{"issue": "1001", "is_straight": False}],
            initial_balance=1000.0,
        )
        obs, info = env.reset(seed=7)
        env.peak_balance = 1000.0
        env.balance = 640.0
        obs, reward, terminated, truncated, info = env.step([2, 99])

        self.assertEqual(info["direction"], "non_straight")
        self.assertFalse(info["action_forced_to_watch"])
        self.assertIn("drawdown_exposure", info["risk_reasons"])
        self.assertGreater(info["risk_penalty"], 0.0)

    def test_training_risk_penalizes_large_non_straight_exposure_without_capping_stake(self):
        from ai_agent.casino_env import CasinoEnv

        env = CasinoEnv(
            [{"issue": "1001", "is_straight": True, "features": {"cond_straight_rate": 0.0}}],
            initial_balance=1000.0,
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([2, 99])

        self.assertEqual(info["direction"], "non_straight")
        self.assertEqual(info["stake"], 100.0)
        self.assertEqual(info["profit_loss"], -2000.0)
        self.assertTrue(terminated)
        self.assertIn("non_straight_exposure_high", info["risk_reasons"])
        self.assertLess(reward, -3000.0)

    def test_environment_caps_stake_at_max_stake(self):
        from ai_agent.casino_env import CasinoEnv

        env = CasinoEnv(
            [{"issue": "1001", "is_straight": False}],
            initial_balance=200000.0,
            max_stake=10000.0,
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([2, 99])

        self.assertEqual(info["amount_ratio"], 0.1)
        self.assertEqual(info["max_stake"], 10000.0)
        self.assertEqual(info["stake"], 10000.0)

    def test_three_part_action_can_watch_with_selected_play_intent(self):
        from ai_agent.casino_env import CasinoEnv

        env = CasinoEnv(
            [{"issue": "1001", "is_straight": False}],
            initial_balance=1000.0,
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([0, 2, 99])

        self.assertFalse(info["should_bet"])
        self.assertEqual(info["intended_direction"], "non_straight")
        self.assertEqual(info["direction"], "watch")
        self.assertEqual(info["stake"], 0.0)
        self.assertEqual(info["profit_loss"], 0.0)

    def test_three_part_action_executes_selected_play_when_enabled(self):
        from ai_agent.casino_env import CasinoEnv

        env = CasinoEnv(
            [{"issue": "1001", "is_straight": False}],
            initial_balance=1000.0,
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([1, 2, 99])

        self.assertTrue(info["should_bet"])
        self.assertEqual(info["direction"], "non_straight")
        self.assertEqual(info["stake"], 100.0)
        self.assertEqual(info["profit_loss"], 100.0)

    def test_two_part_action_remains_backward_compatible(self):
        from ai_agent.casino_env import ACTION_SPACE_SIZE, CasinoEnv

        env = CasinoEnv(
            [{"issue": "1001", "is_straight": False}],
            initial_balance=1000.0,
        )
        self.assertEqual(env.action_space.nvec.tolist(), [ACTION_SPACE_SIZE, 100])
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([2, 99])

        self.assertTrue(info["should_bet"])
        self.assertEqual(info["direction"], "non_straight")
        self.assertEqual(info["stake"], 100.0)

    def test_environment_supports_extended_sum_exact_odds(self):
        from ai_agent.casino_env import CasinoEnv, DIRECTION_LABELS, TrainingGateConfig

        direction = next(key for key, label in DIRECTION_LABELS.items() if label == "sum_13")
        env = CasinoEnv(
            [{"issue": "1001", "digits": [9, 2, 2], "sum": 13, "is_straight": False}],
            initial_balance=1000.0,
            training_gate=TrainingGateConfig(enabled=False),
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([direction, 99])

        self.assertEqual(info["direction"], "sum_13")
        self.assertEqual(info["stake"], 100.0)
        self.assertEqual(info["profit_loss"], 1000.0)
        self.assertEqual(reward, 999.95)

    def test_environment_settles_big_small_13_14_without_special_push(self):
        from ai_agent.casino_env import CasinoEnv, DIRECTION_LABELS, TrainingGateConfig

        big = next(key for key, label in DIRECTION_LABELS.items() if label == "big")
        small_odd = next(key for key, label in DIRECTION_LABELS.items() if label == "small_odd")

        env = CasinoEnv(
            [{"issue": "1001", "digits": [9, 1, 4], "sum": 14, "is_straight": False}],
            initial_balance=20000.0,
            training_gate=TrainingGateConfig(enabled=False),
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([big, 99])
        self.assertEqual(info["profit_loss"], 1960.0)

        env = CasinoEnv(
            [{"issue": "1002", "digits": [9, 1, 4], "sum": 14, "is_straight": False}],
            initial_balance=100000.0,
            training_gate=TrainingGateConfig(enabled=False),
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([big, 99])
        self.assertEqual(info["stake"], 10000.0)
        self.assertEqual(info["profit_loss"], 9800.0)

        env = CasinoEnv(
            [{"issue": "1002", "digits": [9, 2, 2], "sum": 13, "is_straight": False}],
            initial_balance=1000.0,
            training_gate=TrainingGateConfig(enabled=False),
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([small_odd, 99])
        self.assertEqual(info["profit_loss"], 269.0)

    def test_environment_caps_single_bet_payout_by_play_type(self):
        from ai_agent.casino_env import CasinoEnv, DIRECTION_LABELS, TrainingGateConfig

        sum_0 = next(key for key, label in DIRECTION_LABELS.items() if label == "sum_0")
        env = CasinoEnv(
            [{"issue": "1001", "digits": [0, 0, 0], "sum": 0, "is_straight": False}],
            initial_balance=100000.0,
            training_gate=TrainingGateConfig(enabled=False),
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([sum_0, 99])
        self.assertEqual(info["stake"], 10000.0)
        self.assertEqual(info["profit_loss"], 10000.0)

        env = CasinoEnv(
            [{"issue": "1002", "digits": [1, 2, 3], "sum": 6, "is_straight": True}],
            initial_balance=100000.0,
            training_gate=TrainingGateConfig(enabled=False),
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([1, 99])
        self.assertEqual(info["stake"], 10000.0)
        self.assertEqual(info["profit_loss"], 170000.0)

    def test_environment_supports_abc_digit_exact_odds(self):
        from ai_agent.casino_env import CasinoEnv, DIRECTION_LABELS, TrainingGateConfig

        direction = next(key for key, label in DIRECTION_LABELS.items() if label == "A_9")
        env = CasinoEnv(
            [{"issue": "1001", "digits": [9, 2, 2], "sum": 13, "is_straight": False}],
            initial_balance=1000.0,
            training_gate=TrainingGateConfig(enabled=False),
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([direction, 99])

        self.assertEqual(info["direction"], "A_9")
        self.assertEqual(info["profit_loss"], 850.0)

    def test_settlement_only_uses_selected_play_odds(self):
        from ai_agent.casino_env import CasinoEnv, DIRECTION_LABELS, TrainingGateConfig

        pair = next(key for key, label in DIRECTION_LABELS.items() if label == "pair")
        env = CasinoEnv(
            [{"issue": "1001", "digits": [1, 1, 2], "sum": 4, "is_straight": True}],
            initial_balance=1000.0,
            training_gate=TrainingGateConfig(enabled=False),
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([pair, 99])

        self.assertEqual(info["direction"], "pair")
        self.assertEqual(info["selected_gross_odds"], 3.5)
        self.assertEqual(info["profit_loss"], 250.0)

        env = CasinoEnv(
            [{"issue": "1001", "digits": [1, 1, 2], "sum": 4, "is_straight": True}],
            initial_balance=1000.0,
            training_gate=TrainingGateConfig(enabled=False),
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([1, 99])

        self.assertEqual(info["direction"], "straight")
        self.assertEqual(info["selected_gross_odds"], 18.0)
        self.assertEqual(info["profit_loss"], 1700.0)

    def test_non_straight_only_play_uses_fixed_reverse_settlement(self):
        from ai_agent.casino_env import CasinoEnv, DIRECTION_LABELS, TrainingGateConfig

        direction = next(key for key, label in DIRECTION_LABELS.items() if label == "non_straight_only")

        win_env = CasinoEnv(
            [{"issue": "1001", "is_straight": False}],
            initial_balance=1000.0,
            training_gate=TrainingGateConfig(enabled=False),
        )
        obs, info = win_env.reset(seed=7)
        obs, reward, terminated, truncated, win = win_env.step([direction, 0])

        self.assertEqual(win["direction"], "non_straight_only")
        self.assertEqual(win["stake"], 10.0)
        self.assertEqual(win["selected_gross_odds"], 2.0)
        self.assertEqual(win["selected_net_profit_multiplier"], 1.0)
        self.assertEqual(win["selected_loss_multiplier"], 17.0)
        self.assertEqual(win["profit_loss"], 10.0)

        lose_env = CasinoEnv(
            [{"issue": "1002", "is_straight": True}],
            initial_balance=1000.0,
            training_gate=TrainingGateConfig(enabled=False),
        )
        obs, info = lose_env.reset(seed=7)
        obs, reward, terminated, truncated, lose = lose_env.step([direction, 0])

        self.assertEqual(lose["direction"], "non_straight_only")
        self.assertEqual(lose["stake"], 10.0)
        self.assertEqual(lose["profit_loss"], -170.0)

    def test_training_risk_penalizes_watch_starting_at_configured_streak(self):
        from ai_agent.casino_env import CasinoEnv, TrainingGateConfig

        rows = [{"issue": str(1000 + index), "is_straight": False} for index in range(7)]
        env = CasinoEnv(
            rows,
            initial_balance=1000.0,
            training_gate=TrainingGateConfig(watch_penalty_start=5, excessive_watch_penalty=10.0),
        )
        obs, info = env.reset(seed=7)
        infos = []
        rewards = []
        for _ in range(6):
            obs, reward, terminated, truncated, info = env.step([0, 0])
            infos.append(info)
            rewards.append(reward)

        self.assertEqual(infos[3]["watch_streak"], 4)
        self.assertEqual(infos[3]["watch_penalty"], 0.0)
        self.assertEqual(infos[4]["watch_streak"], 5)
        self.assertEqual(infos[4]["watch_penalty"], 10.0)
        self.assertEqual(rewards[4], -10.0)
        self.assertEqual(infos[5]["watch_streak"], 6)
        self.assertEqual(infos[5]["watch_penalty"], 10.0)
        self.assertFalse(infos[5]["action_forced_to_bet"])
        self.assertIn("excessive_watch", infos[5]["risk_reasons"])

    def test_default_watch_penalty_is_small_after_grace_period(self):
        from ai_agent.casino_env import CasinoEnv

        rows = [{"issue": str(1000 + index), "is_straight": False} for index in range(6)]
        env = CasinoEnv(rows, initial_balance=1000.0)
        obs, info = env.reset(seed=7)
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step([0, 0])

        self.assertEqual(info["watch_streak"], 5)
        self.assertEqual(info["watch_penalty"], 0.25)
        self.assertEqual(reward, -0.25)

    def test_normalized_reward_mode_keeps_cash_settlement_unchanged(self):
        from ai_agent.casino_env import CasinoEnv, TrainingGateConfig

        env = CasinoEnv(
            [{"issue": "1001", "is_straight": False}],
            initial_balance=1000.0,
            reward_mode="normalized",
            training_gate=TrainingGateConfig(enabled=False),
        )
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([2, 99])

        self.assertEqual(info["profit_loss"], 100.0)
        self.assertEqual(info["balance"], 1100.0)
        self.assertEqual(info["raw_reward"], 99.95)
        self.assertAlmostEqual(reward, 9.995)
        self.assertEqual(info["reward_mode"], "normalized")

    def test_training_risk_penalizes_bet_rate_above_limit_after_warmup(self):
        from ai_agent.casino_env import CasinoEnv, TrainingGateConfig

        rows = [{"issue": str(1000 + index), "is_straight": False} for index in range(25)]
        env = CasinoEnv(
            rows,
            initial_balance=1000.0,
            training_gate=TrainingGateConfig(
                bet_rate_penalty_min_steps=20,
                max_bet_rate=0.35,
                bet_rate_penalty_multiplier=1.0,
                frequent_bet_penalty_start=0,
                high_stake_penalty_multiplier=0.0,
                negative_edge_penalty_multiplier=0.0,
                threshold_penalty_multiplier=0.0,
            ),
        )
        obs, info = env.reset(seed=7)
        infos = []
        for _ in range(19):
            obs, reward, terminated, truncated, info = env.step([2, 0])
            infos.append(info)

        self.assertNotIn("bet_rate_too_high", infos[-1]["risk_reasons"])
        obs, reward, terminated, truncated, info = env.step([2, 0])

        self.assertEqual(info["episode_bets"], 20)
        self.assertEqual(info["bet_rate"], 1.0)
        self.assertIn("bet_rate_too_high", info["risk_reasons"])
        self.assertGreater(info["risk_penalty"], 0.0)

    def test_training_risk_penalizes_low_bet_rate_when_agent_keeps_watching(self):
        from ai_agent.casino_env import CasinoEnv, TrainingGateConfig

        rows = [{"issue": str(1000 + index), "is_straight": False} for index in range(25)]
        env = CasinoEnv(
            rows,
            initial_balance=1000.0,
            training_gate=TrainingGateConfig(
                watch_penalty_start=0,
                min_bet_rate=0.02,
                low_bet_rate_penalty_min_steps=20,
                low_bet_rate_penalty_multiplier=0.01,
            ),
        )
        obs, info = env.reset(seed=7)
        for _ in range(19):
            obs, reward, terminated, truncated, info = env.step([0, 2, 99])

        self.assertNotIn("bet_rate_too_low", info["risk_reasons"])
        obs, reward, terminated, truncated, info = env.step([0, 2, 99])

        self.assertEqual(info["episode_bets"], 0)
        self.assertEqual(info["bet_rate"], 0.0)
        self.assertIn("bet_rate_too_low", info["risk_reasons"])
        self.assertGreater(info["watch_penalty"], 0.0)
        self.assertLess(reward, 0.0)

    def test_current_omission_observation_high_is_capped_at_300(self):
        from ai_agent.casino_env import CasinoEnv

        env = CasinoEnv(
            [
                {"issue": "1001", "is_straight": False},
                {"issue": "1002", "is_straight": True},
            ],
            initial_balance=1000.0,
        )

        self.assertEqual(float(env.observation_space.high[0]), 300.0)

    def test_game_objective_passes_when_balance_reaches_target(self):
        from ai_agent.casino_env import CasinoEnv, DIRECTION_LABELS, GameObjectiveConfig, TrainingGateConfig

        env = CasinoEnv(
            [{"issue": "1001", "digits": [9, 2, 2], "sum": 13, "is_straight": False}],
            initial_balance=1000.0,
            game_objective=GameObjectiveConfig(enabled=True, pass_reward=300.0, bet_rate_bonus=0.0),
            training_gate=TrainingGateConfig(enabled=False),
        )
        direction = next(key for key, label in DIRECTION_LABELS.items() if label == "sum_13")
        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step([direction, 99])

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["game_objective_reason"], "game_pass")
        self.assertEqual(info["game_objective_reward"], 300.0)
        self.assertGreaterEqual(info["balance"], 1300.0)

    def test_game_objective_fails_when_balance_hits_floor(self):
        from ai_agent.casino_env import CasinoEnv, DIRECTION_LABELS, GameObjectiveConfig, TrainingGateConfig

        env = CasinoEnv(
            [{"issue": "1001", "digits": [0, 0, 0], "sum": 0, "is_straight": False}],
            initial_balance=1000.0,
            game_objective=GameObjectiveConfig(enabled=True, fail_penalty=300.0),
            training_gate=TrainingGateConfig(enabled=False),
        )
        direction = next(key for key, label in DIRECTION_LABELS.items() if label == "sum_1")
        obs, info = env.reset(seed=7)
        env.balance = 710.0
        obs, reward, terminated, truncated, info = env.step([direction, 99])

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["game_objective_reason"], "game_fail")
        self.assertEqual(info["game_objective_reward"], -300.0)
        self.assertLessEqual(info["balance"], 700.0)

    def test_game_objective_truncates_at_configured_max_steps(self):
        from ai_agent.casino_env import CasinoEnv, GameObjectiveConfig, TrainingGateConfig

        rows = [{"issue": str(1000 + index), "is_straight": False} for index in range(5)]
        env = CasinoEnv(
            rows,
            initial_balance=1000.0,
            game_objective=GameObjectiveConfig(enabled=True, max_steps=3, bet_rate_bonus=0.0),
            training_gate=TrainingGateConfig(enabled=False),
        )
        obs, info = env.reset(seed=7)
        for _ in range(2):
            obs, reward, terminated, truncated, info = env.step([0, 0])
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        obs, reward, terminated, truncated, info = env.step([0, 0])

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["episode_steps"], 3)
        self.assertEqual(info["game_objective_reason"], "game_complete")

    def test_game_objective_fails_on_bet_limit_violation(self):
        from ai_agent.casino_env import CasinoEnv, GameObjectiveConfig, TrainingGateConfig

        rows = [{"issue": str(1000 + index), "is_straight": False} for index in range(50)]
        env = CasinoEnv(
            rows,
            initial_balance=1000.0,
            max_stake=10.0,
            game_objective=GameObjectiveConfig(
                enabled=True,
                max_steps=200,
                target_bet_rate=0.2,
                pass_balance_multiplier=99.0,
                fail_balance_multiplier=0.01,
                bet_limit_penalty=300.0,
            ),
            training_gate=TrainingGateConfig(enabled=False),
        )
        obs, info = env.reset(seed=7)
        for _ in range(40):
            obs, reward, terminated, truncated, info = env.step([2, 0])
            self.assertFalse(terminated)
            self.assertFalse(truncated)

        obs, reward, terminated, truncated, info = env.step([2, 0])

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["episode_bets"], 41)
        self.assertEqual(info["game_objective_reason"], "game_bet_limit")
        self.assertEqual(info["game_objective_reward"], -300.0)


class RewardTests(unittest.TestCase):
    def test_reverse_non_straight_settlement_matches_margin_rules(self):
        from ai_agent.reward import settle_action

        win = settle_action(
            direction=2,
            stake=100.0,
            outcome_is_straight=False,
            pressure_score=0.0,
            losing_streak=0,
            amount_ratio=0.1,
            current_omission=0.0,
            omission_avg_ratio=0.0,
            straight_profit_multiplier=19.0,
            non_straight_profit_multiplier=1.0,
            non_straight_loss_multiplier=20.0,
        )
        lose = settle_action(
            direction=2,
            stake=100.0,
            outcome_is_straight=True,
            pressure_score=0.0,
            losing_streak=0,
            amount_ratio=0.1,
            current_omission=0.0,
            omission_avg_ratio=0.0,
            straight_profit_multiplier=19.0,
            non_straight_profit_multiplier=1.0,
            non_straight_loss_multiplier=20.0,
        )

        self.assertEqual(win.profit_loss, 100.0)
        self.assertEqual(win.reward, 100.0)
        self.assertEqual(lose.profit_loss, -2000.0)
        self.assertEqual(lose.reward, -2000.0)

    def test_watch_does_not_increment_losing_streak(self):
        from ai_agent.reward import settle_action

        result = settle_action(
            direction=0,
            stake=0.0,
            outcome_is_straight=True,
            pressure_score=50.0,
            losing_streak=2,
            amount_ratio=0.0,
            current_omission=0.0,
            omission_avg_ratio=0.0,
            straight_profit_multiplier=19.0,
            non_straight_profit_multiplier=1.0,
            non_straight_loss_multiplier=20.0,
        )

        self.assertEqual(result.next_losing_streak, 2)
        self.assertEqual(result.reason, "watch")

    def test_reward_matches_realized_pnl_even_on_heavy_omission(self):
        from ai_agent.reward import settle_action

        result = settle_action(
            direction=2,
            stake=100.0,
            outcome_is_straight=True,
            pressure_score=0.0,
            losing_streak=2,
            amount_ratio=0.1,
            current_omission=35.0,
            omission_avg_ratio=2.5,
            straight_profit_multiplier=19.0,
            non_straight_profit_multiplier=1.0,
            non_straight_loss_multiplier=20.0,
        )

        self.assertEqual(result.profit_loss, -2000.0)
        self.assertEqual(result.next_losing_streak, 3)
        self.assertAlmostEqual(result.reward, -2000.0)
        self.assertEqual(result.reason, "loss")


class RiskGuardTests(unittest.TestCase):
    def test_risk_guard_blocks_real_bet_but_never_rewrites_training_action(self):
        from ai_agent.risk_guard import RuntimeState, evaluate_risk

        decision = evaluate_risk(
            direction=1,
            amount_ratio=0.1,
            state=RuntimeState(issue="1001", balance=490.0, initial_balance=1000.0, losing_streak=0),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.direction, 1)
        self.assertEqual(decision.amount_ratio, 0.1)
        self.assertIn("capital_below_stop", decision.reasons)


class StateFilterTests(unittest.TestCase):
    def test_conservative_strategy_uses_fixed_state_rules_and_pause(self):
        from ai_agent.state_filter import ConservativeConfig, evaluate_conservative_strategy

        rows = []
        for index in range(220):
            if index % 6 == 0:
                digits = [1, 2, 3]
            elif index % 6 == 1:
                digits = [3, 2, 1]
            elif index % 6 == 2:
                digits = [8, 4, 2]
            else:
                digits = [9, 9, 1]
            rows.append(
                {
                    "issue": str(1000 + index),
                    "digits": digits,
                    "sum": sum(digits),
                    "is_straight": len(set(digits)) == 3 and sorted(digits) in ([1, 2, 3], [2, 3, 4]),
                }
            )

        report = evaluate_conservative_strategy(
            rows,
            segment_size=220,
            initial_balance=1000.0,
            config=ConservativeConfig(
                warmup_steps=10,
                min_prior_signals=1,
                rolling_signal_window=5,
                max_bet_rate=0.10,
                pause_steps_after_loss=3,
            ),
        )

        strategy = report["segments"][0]["strategy"]
        self.assertLessEqual(strategy["bet_rate"], 0.10)
        self.assertIn("pause_after_loss", strategy["skip_reasons"])
        self.assertTrue(set(strategy["play_counts"]).issubset({"big_even", "small_even"}))


class ExecutorTests(unittest.TestCase):
    def test_dry_run_executor_returns_payload_without_network(self):
        from ai_agent.executor import execute_bet

        result = execute_bet(
            api_url="https://example.invalid/bet",
            issue="1001",
            direction=2,
            stake=12.5,
            dry_run=True,
        )

        self.assertTrue(result["dry_run"])
        self.assertEqual(result["payload"]["direction"], "non_straight")
        self.assertEqual(result["payload"]["amount"], 12.5)


class BacktestInferTrainTests(unittest.TestCase):
    def test_baseline_policies_cover_minimal_controls(self):
        from ai_agent.backtest import BASELINE_NAMES, baseline_policy

        self.assertEqual(
            set(BASELINE_NAMES),
            {
                "watch_all",
                "always_non_straight_min",
                "always_non_straight_only_min",
                "always_straight_min",
                "random_min",
            },
        )
        self.assertEqual(baseline_policy("watch_all")(None, {}), [0, 0])
        self.assertEqual(baseline_policy("always_non_straight_min")(None, {}), [2, 0])
        self.assertGreater(baseline_policy("always_non_straight_only_min")(None, {})[0], 2)
        self.assertEqual(baseline_policy("always_straight_min")(None, {}), [1, 0])
        self.assertEqual(baseline_policy("random_min", seed=7)(None, {})[1], 0)
        with self.assertRaises(ValueError):
            baseline_policy("missing")

    def test_backtest_runs_policy_and_returns_metrics(self):
        from ai_agent.backtest import run_backtest
        from ai_agent.data_adapter import clean_history

        history = clean_history(
            [
                {"issue": "1001", "is_straight": False},
                {"issue": "1002", "is_straight": True},
                {"issue": "1003", "is_straight": False},
            ],
            min_rows=3,
        )
        result = run_backtest(history, policy=lambda obs, info: [2, 0], initial_balance=1000.0)

        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["bets"], 3)
        self.assertEqual(result["bet_rate"], 1.0)
        self.assertIn("max_drawdown", result)
        self.assertEqual(len(result["equity_curve"]), 4)

    def test_backtest_runs_three_part_policy_without_counting_watch_intent_as_bet(self):
        from ai_agent.backtest import run_backtest
        from ai_agent.data_adapter import clean_history

        history = clean_history(
            [
                {"issue": "1001", "is_straight": False},
                {"issue": "1002", "is_straight": False},
            ],
            min_rows=2,
        )
        result = run_backtest(history, policy=lambda obs, info: [0, 2, 99], initial_balance=1000.0)

        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["bets"], 0)
        self.assertEqual(result["bet_rate"], 0.0)
        self.assertEqual(result["final_balance"], 1000.0)

    def test_infer_action_to_decision_maps_amount_bin_to_stake(self):
        from ai_agent.infer import action_to_decision

        decision = action_to_decision([2, 49], balance=1000.0)

        self.assertTrue(decision["should_bet"])
        self.assertEqual(decision["intended_direction"], 2)
        self.assertEqual(decision["direction"], 2)
        self.assertEqual(decision["direction_label"], "non_straight")
        self.assertEqual(decision["amount_ratio"], 0.05)
        self.assertEqual(decision["min_stake"], 10.0)
        self.assertEqual(decision["stake"], 50.0)

    def test_infer_action_to_decision_supports_three_part_watch_action(self):
        from ai_agent.infer import action_to_decision

        decision = action_to_decision([0, 2, 99], balance=1000.0)

        self.assertFalse(decision["should_bet"])
        self.assertEqual(decision["intended_direction"], 2)
        self.assertEqual(decision["intended_direction_label"], "non_straight")
        self.assertEqual(decision["direction"], 0)
        self.assertEqual(decision["direction_label"], "watch")
        self.assertEqual(decision["amount_ratio"], 0.0)
        self.assertEqual(decision["stake"], 0.0)

    def test_infer_action_to_decision_caps_stake_at_max_stake(self):
        from ai_agent.infer import action_to_decision

        decision = action_to_decision([2, 99], balance=200000.0)

        self.assertEqual(decision["amount_ratio"], 0.1)
        self.assertEqual(decision["max_stake"], 10000.0)
        self.assertEqual(decision["stake"], 10000.0)

    def test_train_device_selector_returns_supported_device(self):
        from ai_agent.training_utils import select_torch_device

        self.assertIn(select_torch_device(), {"mps", "cuda", "cpu"})

    def test_train_settings_builds_reproducible_ppo_kwargs_and_metadata_path(self):
        from ai_agent.train_ppo import TrainSettings, build_ppo_game_objective, build_ppo_kwargs, metadata_path_for
        from ai_agent.training_utils import resolve_device

        settings = TrainSettings(
            model_path=Path("ai_agent/models/ppo_casino_model.zip"),
            total_timesteps=64,
            initial_balance=1000.0,
            device="cpu",
            seed=123,
            n_steps=32,
            batch_size=16,
            learning_rate=0.001,
            verbose=0,
            check_env=False,
            reward_mode="normalized",
            game_objective_enabled=True,
            game_max_steps=200,
            game_pass_balance_multiplier=1.3,
            game_fail_balance_multiplier=0.7,
            game_target_bet_rate=0.2,
        )

        kwargs = build_ppo_kwargs(settings)
        objective = build_ppo_game_objective(settings)

        self.assertEqual(resolve_device("cpu"), "cpu")
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["seed"], 123)
        self.assertEqual(kwargs["n_steps"], 32)
        self.assertEqual(kwargs["batch_size"], 16)
        self.assertEqual(kwargs["learning_rate"], 0.001)
        self.assertTrue(objective.enabled)
        self.assertEqual(objective.max_steps, 200)
        self.assertEqual(objective.pass_balance_multiplier, 1.3)
        self.assertEqual(objective.fail_balance_multiplier, 0.7)
        self.assertEqual(objective.target_bet_rate, 0.2)
        self.assertEqual(metadata_path_for(settings.model_path), Path.cwd() / "ai_agent/models/ppo_casino_model.metadata.json")

    def test_ppo_model_policy_rejects_dqn_paths_and_metadata(self):
        import json
        from ai_agent.model_policy import resolve_ppo_model_path

        with self.assertRaisesRegex(ValueError, "refuses DQN model"):
            resolve_ppo_model_path("ai_agent/models/dqn_capital_game_v1.zip")

        with TemporaryDirectory(dir=Path.cwd() / "ai_agent/models") as tmpdir:
            model_path = Path(tmpdir) / "candidate.zip"
            model_path.write_bytes(b"not a real model")
            model_path.with_name("candidate.metadata.json").write_text(
                json.dumps({"action_space": "Discrete(85)", "actions": []}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "refuses DQN model"):
                resolve_ppo_model_path(model_path)

    def test_write_training_metadata_records_feature_and_space_contract(self):
        import json
        from ai_agent.casino_env import ACTION_SPACE_SIZE
        from ai_agent.train_ppo import TrainSettings, write_training_metadata

        settings = TrainSettings(model_path=Path("ai_agent/models/test_model.zip"), total_timesteps=64, check_env=False)
        path = write_training_metadata(
            settings,
            history_rows=12,
            resolved_device="cpu",
            observation_high=[300.0, 100.0, 1.0, 1.0, 100.0, 100.0, 1.0, 100.0],
        )

        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["history_rows"], 12)
        self.assertEqual(payload["device"], "cpu")
        self.assertEqual(payload["feature_columns"][0], "current_omission")
        self.assertEqual(payload["observation_high"][0], 300.0)
        self.assertEqual(payload["action_space"], f"MultiDiscrete([{ACTION_SPACE_SIZE}, 100])")
        self.assertIn("game_objective_enabled", payload["settings"])

    def test_train_resume_requires_existing_model(self):
        from ai_agent.train_ppo import train_model

        missing_model = Path("ai_agent/models/missing_resume_model.zip")
        self.assertFalse(missing_model.exists())
        with self.assertRaises(FileNotFoundError):
            train_model(
                [{"issue": "1001", "is_straight": False}],
                model_path=missing_model,
                total_timesteps=2,
                check=False,
                resume=True,
            )

    def test_dqn_action_wrapper_maps_discrete_wait_and_buy_actions(self):
        from ai_agent.casino_env import CasinoEnv, TrainingGateConfig
        from ai_agent.train_dqn import (
            DQN_AMOUNT_BINS,
            DQN_LOOKBACK_FEATURES,
            DQN_STAKE_AMOUNTS,
            DqnCapitalEnv,
            build_dqn_game_objective,
        )

        base_env = CasinoEnv(
            [
                {"issue": "1001", "digits": [1, 2, 4], "sum": 7, "is_straight": False},
                {"issue": "1002", "digits": [8, 4, 2], "sum": 14, "is_straight": False},
            ],
            initial_balance=1000.0,
            max_stake=100.0,
            min_stake=1.0,
            training_gate=TrainingGateConfig(enabled=False),
            game_objective=build_dqn_game_objective(),
        )
        env = DqnCapitalEnv(base_env)
        obs, info = env.reset(seed=7)

        self.assertEqual(obs.shape, (14 + len(DQN_LOOKBACK_FEATURES),))
        self.assertEqual(obs[10], 10.0)
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(info["direction"], "watch")
        self.assertEqual(info["stake"], 0.0)

        obs, reward, terminated, truncated, info = env.step(1)
        self.assertEqual(info["dqn_action_label"], "small_odd")
        self.assertEqual(info["dqn_amount_bin"], DQN_AMOUNT_BINS[0])
        self.assertEqual(info["stake"], DQN_STAKE_AMOUNTS[0])

    def test_dqn_lookback_features_use_prior_history_windows(self):
        from ai_agent.casino_env import CasinoEnv, TrainingGateConfig
        from ai_agent.train_dqn import DQN_ACTION_SPECS, DQN_LOOKBACK_FEATURES, DqnCapitalEnv, build_dqn_game_objective

        rows = [
            {"issue": "1001", "digits": [1, 1, 1], "sum": 3, "is_straight": True},
            {"issue": "1002", "digits": [1, 2, 4], "sum": 7, "is_straight": False},
            {"issue": "1003", "digits": [8, 4, 2], "sum": 14, "is_straight": False},
        ]
        env = DqnCapitalEnv(
            CasinoEnv(
                rows,
                initial_balance=1000.0,
                max_stake=100.0,
                min_stake=1.0,
                training_gate=TrainingGateConfig(enabled=False),
                game_objective=build_dqn_game_objective(),
            )
        )

        obs, info = env.reset(seed=7, options={"start_step": 2})
        lookback = obs[-len(DQN_LOOKBACK_FEATURES):]

        self.assertAlmostEqual(float(lookback[0]), 0.5)
        self.assertGreaterEqual(float(lookback[1]), 0.0)
        self.assertLessEqual(float(lookback[1]), 1.0)

        action_index = next(
            index
            for index, spec in enumerate(DQN_ACTION_SPECS)
            if spec.label == "big_even+small_even+big_odd" and spec.stake_per_play == 100.0
        )
        edge = env._historical_edge_score(DQN_ACTION_SPECS[action_index])
        self.assertGreaterEqual(edge, -1.0)
        self.assertLessEqual(edge, 1.0)

    def test_dqn_action_wrapper_can_bet_multi_play_package_once_per_period(self):
        from ai_agent.casino_env import CasinoEnv, TrainingGateConfig
        from ai_agent.train_dqn import DQN_ACTION_SPECS, DqnCapitalEnv, build_dqn_game_objective

        base_env = CasinoEnv(
            [
                {"issue": "1001", "digits": [8, 8, 8], "sum": 24, "is_straight": False},
                {"issue": "1002", "digits": [1, 2, 4], "sum": 7, "is_straight": False},
            ],
            initial_balance=1000.0,
            max_stake=100.0,
            min_stake=10.0,
            training_gate=TrainingGateConfig(enabled=False),
            game_objective=build_dqn_game_objective(),
        )
        env = DqnCapitalEnv(base_env)
        action_index = next(
            index
            for index, spec in enumerate(DQN_ACTION_SPECS)
            if spec.label == "big_even+small_even"
        )

        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step(action_index)

        self.assertEqual(info["direction"], "big_even+small_even")
        self.assertEqual(info["stake"], 60.0)
        self.assertEqual(info["stake_per_play"], 30.0)
        self.assertEqual(info["package_size"], 2)
        self.assertEqual(info["package_profit_losses"], (80.7, -30.0))
        self.assertEqual(info["profit_loss"], 50.7)
        self.assertEqual(info["episode_steps"], 1)
        self.assertEqual(info["episode_bets"], 1)
        self.assertEqual(info["dqn_action_label"], "big_even+small_even")

    def test_dqn_wait_reward_and_hard_violation_penalty_are_configurable(self):
        from ai_agent.casino_env import CasinoEnv, GameObjectiveConfig, TrainingGateConfig
        from ai_agent.train_dqn import DqnCapitalEnv, build_dqn_game_objective

        objective = build_dqn_game_objective()
        self.assertEqual(objective.max_steps, 200)
        self.assertEqual(objective.pass_balance_multiplier, 3.0)
        self.assertEqual(objective.bet_limit_penalty, 20000.0)
        self.assertEqual(objective.target_bet_rate, 0.22)

        base_kwargs = {
            "initial_balance": 1000.0,
            "max_stake": 100.0,
            "min_stake": 1.0,
            "training_gate": TrainingGateConfig(enabled=False),
            "game_objective": GameObjectiveConfig(enabled=True, max_steps=200, target_bet_rate=0.2, bet_rate_bonus=0.0),
        }
        no_bonus_env = DqnCapitalEnv(
            CasinoEnv(
                [
                    {"issue": str(1000 + index), "digits": [1, 2, 4], "sum": 7, "is_straight": False}
                    for index in range(10)
                ],
                **base_kwargs,
            ),
            wait_reward=0.0,
            low_bet_rate_wait_penalty=0.0,
        )
        bonus_env = DqnCapitalEnv(
            CasinoEnv(
                [
                    {"issue": str(1000 + index), "digits": [1, 2, 4], "sum": 7, "is_straight": False}
                    for index in range(10)
                ],
                **base_kwargs,
            ),
            wait_reward=0.01,
            low_bet_rate_wait_penalty=0.0,
        )
        obs, info = no_bonus_env.reset(seed=7)
        obs, no_bonus_reward, terminated, truncated, info = no_bonus_env.step(0)
        obs, info = bonus_env.reset(seed=7)
        obs, bonus_reward, terminated, truncated, info = bonus_env.step(0)
        self.assertAlmostEqual(bonus_reward - no_bonus_reward, 0.01)

    def test_dqn_wait_uses_historical_edge_before_bet_rate_pressure(self):
        from ai_agent.casino_env import CasinoEnv, GameObjectiveConfig, TrainingGateConfig
        from ai_agent.train_dqn import DqnCapitalEnv

        env = DqnCapitalEnv(
            CasinoEnv(
                [
                    {"issue": str(1000 + index), "digits": [1, 2, 4], "sum": 7, "is_straight": False}
                    for index in range(20)
                ],
                initial_balance=1000.0,
                max_stake=100.0,
                min_stake=1.0,
                training_gate=TrainingGateConfig(enabled=False),
                game_objective=GameObjectiveConfig(enabled=True, max_steps=200, target_bet_rate=0.2, bet_rate_bonus=0.0),
            ),
            low_bet_rate_wait_penalty=1.0,
        )

        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertGreater(reward, 0.0)
        first_wait_reward = reward

        obs, reward, terminated, truncated, info = env.step(0)
        self.assertLessEqual(reward, first_wait_reward)

    def test_dqn_profit_lock_cashout_waits_and_allows_bets_to_continue(self):
        from ai_agent.casino_env import CasinoEnv, DIRECTION_LABELS, GameObjectiveConfig, TrainingGateConfig
        from ai_agent.train_dqn import (
            DQN_PROFIT_LOCK_FAILED_SPRINT_PENALTY,
            DQN_PROFIT_LOCK_WAIT_REWARD,
            DqnActionSpec,
            DqnCapitalEnv,
        )
        small_odd = next(key for key, value in DIRECTION_LABELS.items() if value == "small_odd")

        env = DqnCapitalEnv(
            CasinoEnv(
                [
                    {"issue": "1001", "digits": [1, 2, 4], "sum": 7, "is_straight": False}
                    for _index in range(5)
                ],
                initial_balance=1000.0,
                max_stake=100.0,
                min_stake=10.0,
                training_gate=TrainingGateConfig(enabled=False),
                game_objective=GameObjectiveConfig(enabled=True, max_steps=200, target_bet_rate=0.2, bet_rate_bonus=0.0),
            ),
            action_specs=(DqnActionSpec("WAIT_0", (), 0), DqnActionSpec("small_odd", (small_odd,), 0)),
        )

        obs, info = env.reset(seed=7)
        env.env.balance = 2001.0
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, DQN_PROFIT_LOCK_WAIT_REWARD)
        self.assertEqual(info["game_objective_reason"], "dqn_profit_lock_cashout")

        obs, info = env.reset(seed=7)
        env.env.balance = 2001.0
        obs, reward, terminated, truncated, info = env.step(1)
        self.assertFalse(truncated)
        self.assertTrue(info["dqn_profit_lock_seen"])

        obs, info = env.reset(seed=7)
        env.env.balance = 2001.0
        env.env.episode_bets = env.env.game_objective.max_bets
        obs, reward, terminated, truncated, info = env.step(1)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertIn(
            info["game_objective_reason"],
            {"game_pass", "dqn_profit_lock_cashout", "dqn_profit_lock_failed_sprint"},
        )
        if info["balance"] < 2000.0:
            self.assertEqual(reward, -DQN_PROFIT_LOCK_FAILED_SPRINT_PENALTY)

        obs, info = env.reset(seed=7)
        env.env.balance = 1000.0
        obs, reward, terminated, truncated, info = env.step(1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["dqn_profit_lock_active"])

        obs, info = env.reset(seed=7)
        env.env.balance = 2001.0
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(reward, DQN_PROFIT_LOCK_WAIT_REWARD)
        self.assertTrue(info["dqn_profit_lock_active"])

    def test_dqn_reward_uses_pressure_score_as_opportunity_score(self):
        from ai_agent.casino_env import CasinoEnv, DIRECTION_LABELS, GameObjectiveConfig, TrainingGateConfig
        from ai_agent.train_dqn import DqnActionSpec, DqnCapitalEnv
        small_odd = next(key for key, value in DIRECTION_LABELS.items() if value == "small_odd")
        action_specs = (DqnActionSpec("WAIT_0", (), 0), DqnActionSpec("small_odd", (small_odd,), 0))

        base_kwargs = {
            "initial_balance": 1000.0,
            "max_stake": 100.0,
            "min_stake": 1.0,
            "training_gate": TrainingGateConfig(enabled=False),
            "game_objective": GameObjectiveConfig(enabled=True, max_steps=200, target_bet_rate=0.2, bet_rate_bonus=0.0),
        }
        low_opportunity_env = DqnCapitalEnv(
            CasinoEnv(
                [
                    {"issue": "1001", "digits": [1, 2, 4], "sum": 7, "is_straight": False},
                    {"issue": "1002", "digits": [8, 4, 2], "sum": 14, "is_straight": False},
                ],
                **base_kwargs,
            ),
            low_bet_rate_wait_penalty=0.0,
            action_specs=action_specs,
        )
        high_opportunity_env = DqnCapitalEnv(
            CasinoEnv(
                [
                    {"issue": "1001", "digits": [1, 1, 1], "sum": 3, "is_straight": True},
                    {"issue": "1002", "digits": [8, 4, 2], "sum": 14, "is_straight": False},
                    {"issue": "1003", "digits": [1, 2, 4], "sum": 7, "is_straight": False},
                    {"issue": "1004", "digits": [8, 4, 2], "sum": 14, "is_straight": False},
                    {"issue": "1005", "digits": [1, 2, 4], "sum": 7, "is_straight": False},
                ],
                **base_kwargs,
            ),
            low_bet_rate_wait_penalty=0.0,
            action_specs=action_specs,
        )

        obs, info = low_opportunity_env.reset(seed=7)
        obs, low_win_reward, terminated, truncated, low_win = low_opportunity_env.step(1)

        obs, info = high_opportunity_env.reset(seed=7, options={"start_step": 2})
        obs, high_wait_reward, terminated, truncated, high_wait = high_opportunity_env.step(0)
        self.assertEqual(high_wait["pressure_score"], 100.0)
        self.assertAlmostEqual(high_wait_reward, -0.3)

        obs, info = high_opportunity_env.reset(seed=7, options={"start_step": 2})
        obs, high_win_reward, terminated, truncated, high_win = high_opportunity_env.step(1)
        self.assertAlmostEqual(high_win_reward, 3.0 * high_win["profit_loss"] - 2.0)
        self.assertAlmostEqual(low_win_reward, low_win["profit_loss"] - 52.0)

        obs, info = high_opportunity_env.reset(seed=7, options={"start_step": 3})
        obs, high_loss_reward, terminated, truncated, high_loss = high_opportunity_env.step(1)
        self.assertLess(high_loss["profit_loss"], 0.0)
        self.assertLess(high_loss_reward, -0.5 * abs(high_loss["profit_loss"]) - 2.0)

        self.assertEqual(DqnCapitalEnv._wait_opportunity_reward(0.15), 0.5)
        self.assertEqual(DqnCapitalEnv._wait_opportunity_reward(0.3), -0.1)
        self.assertEqual(DqnCapitalEnv._wait_opportunity_reward(0.45), -0.3)

    def test_dqn_opportunity_score_blends_pressure_bet_rate_drawdown_and_loss_streak(self):
        from ai_agent.train_dqn import DqnCapitalEnv

        score = DqnCapitalEnv._opportunity_score(
            {
                "pressure_score": 80.0,
                "bet_rate": 0.1,
                "game_target_bet_rate": 0.2,
                "drawdown": 0.2,
                "losing_streak": 2,
            }
        )

        self.assertAlmostEqual(score, 0.72)

    def test_dqn_bet_reward_penalizes_chasing_and_rewards_sprint_zone(self):
        from ai_agent.casino_env import CasinoEnv, GameObjectiveConfig, TrainingGateConfig
        from ai_agent.train_dqn import DqnCapitalEnv

        env = DqnCapitalEnv(
            CasinoEnv(
                [{"issue": "1001", "digits": [1, 2, 4], "sum": 7, "is_straight": False}],
                initial_balance=1000.0,
                max_stake=100.0,
                min_stake=10.0,
                training_gate=TrainingGateConfig(enabled=False),
                game_objective=GameObjectiveConfig(enabled=True, max_steps=200, target_bet_rate=0.2, bet_rate_bonus=0.0),
            )
        )

        strong_edge = 0.2
        self.assertEqual(env._bet_opportunity_reward({"losing_streak": 0, "balance": 1100.0}, 10.0, 0.5, strong_edge), 28.0)
        self.assertEqual(env._bet_opportunity_reward({"losing_streak": 3, "balance": 1100.0}, 10.0, 0.5, strong_edge), 13.0)
        self.assertEqual(env._bet_opportunity_reward({"losing_streak": 0, "balance": 1160.0}, 10.0, 0.5, strong_edge), 53.0)
        self.assertEqual(env._bet_opportunity_reward({"losing_streak": 5, "balance": 1000.0}, -10.0, 0.5, strong_edge), -37.0)
        self.assertEqual(env._bet_opportunity_reward({"losing_streak": 2, "drawdown": 0.5, "balance": 1000.0}, -10.0, 0.2, -0.1), -130.0)
        self.assertAlmostEqual(env._bet_opportunity_reward({"losing_streak": 0, "balance": 1100.0}, 10.0, 0.2, strong_edge), -20.0)
        self.assertAlmostEqual(env._bet_opportunity_reward({"losing_streak": 0, "balance": 1100.0, "package_size": 3}, 10.0, 0.2, strong_edge), -20.0)
        self.assertEqual(env._bet_opportunity_reward({"losing_streak": 0, "balance": 1150.0, "package_size": 1}, 10.0, 0.5, strong_edge), 53.0)
        self.assertEqual(env._bet_opportunity_reward({"losing_streak": 0, "balance": 1150.0, "package_size": 2}, 10.0, 0.5, strong_edge), 48.0)
        self.assertEqual(env._edge_wait_reward(-0.1), 5.0)
        self.assertEqual(env._edge_wait_reward(0.05), 1.0)
        self.assertEqual(env._edge_wait_reward(0.2), 0.0)
        self.assertEqual(env._edge_bet_cost(-0.1), 40.0)
        self.assertEqual(env._edge_bet_cost(0.05), 5.0)
        self.assertEqual(env._edge_bet_cost(0.2), 0.0)
        self.assertEqual(env._risk_loss_multiplier({"drawdown": 0.5, "losing_streak": 2}, -0.1), 3.0)

    def test_dqn_bet_streak_raises_required_score_for_consecutive_bets(self):
        from ai_agent.train_dqn import DqnCapitalEnv

        self.assertAlmostEqual(DqnCapitalEnv._bet_streak_required_score({"bet_streak": 1}), 0.30)
        self.assertAlmostEqual(DqnCapitalEnv._bet_streak_required_score({"bet_streak": 2}), 0.35)
        self.assertAlmostEqual(DqnCapitalEnv._bet_streak_required_score({"bet_streak": 4}), 0.45)
        self.assertAlmostEqual(DqnCapitalEnv._bet_streak_cost({"bet_streak": 4}, 0.35), 20.0)
        self.assertAlmostEqual(DqnCapitalEnv._bet_streak_cost({"bet_streak": 4}, 0.45), 0.0)

    def test_dqn_unreachable_wait_is_forced_terminal_penalty(self):
        from ai_agent.casino_env import CasinoEnv, GameObjectiveConfig, TrainingGateConfig
        from ai_agent.train_dqn import DqnCapitalEnv

        env = DqnCapitalEnv(
            CasinoEnv(
                [{"issue": "1001", "digits": [1, 2, 4], "sum": 7, "is_straight": False}],
                initial_balance=1000.0,
                max_stake=100.0,
                min_stake=1.0,
                training_gate=TrainingGateConfig(enabled=False),
                game_objective=GameObjectiveConfig(
                    enabled=True,
                    max_steps=200,
                    target_bet_rate=0.0,
                    pass_balance_multiplier=1.3,
                    bet_rate_bonus=0.0,
                ),
            )
        )

        obs, info = env.reset(seed=7)
        obs, reward, terminated, truncated, info = env.step(0)

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, -500.0)
        self.assertEqual(info["game_objective_reason"], "dqn_unreachable_wait")

    def test_dqn_settings_metadata_path_and_action_space_are_discrete(self):
        from ai_agent.train_dqn import (
            DQN_ACTION_SPECS,
            DQN_AMOUNT_BINS,
            DQN_LOOKBACK_FEATURES,
            DQN_MIN_STAKE,
            DQN_NET_ARCH,
            DQN_PASS_BONUS_REWARD,
            DQN_PROFIT_LOCK_FAILED_SPRINT_PENALTY,
            DQN_PROFIT_LOCK_MULTIPLIER,
            DQN_PROFIT_LOCK_WAIT_REWARD,
            DQN_STAKE_AMOUNTS,
            DqnTrainSettings,
            build_dqn_kwargs,
            metadata_path_for,
        )
        from ai_agent.training_utils import resolve_device

        settings = DqnTrainSettings(
            model_path=Path("ai_agent/models/dqn_test_model.zip"),
            total_timesteps=64,
            initial_balance=1000.0,
            max_stake=100.0,
            device="cpu",
            verbose=0,
            check_env=False,
            wait_reward=0.01,
        )
        kwargs = build_dqn_kwargs(settings)

        self.assertEqual(resolve_device("cpu"), "cpu")
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["learning_rate"], settings.learning_rate)
        self.assertEqual(kwargs["buffer_size"], 100_000)
        self.assertEqual(kwargs["train_freq"], 1)
        self.assertEqual(kwargs["gradient_steps"], -1)
        self.assertEqual(kwargs["target_update_interval"], 2000)
        self.assertEqual(kwargs["exploration_initial_eps"], 1.0)
        self.assertEqual(kwargs["max_grad_norm"], 10.0)
        self.assertEqual(kwargs["policy_kwargs"], {"net_arch": DQN_NET_ARCH})
        self.assertEqual(settings.wait_reward, 0.01)
        self.assertEqual(settings.min_stake, DQN_MIN_STAKE)
        self.assertEqual(settings.low_bet_rate_wait_penalty, 1.0)
        self.assertEqual(DQN_PASS_BONUS_REWARD, 3500.0)
        self.assertEqual(DQN_PROFIT_LOCK_MULTIPLIER, 2.0)
        self.assertEqual(DQN_PROFIT_LOCK_WAIT_REWARD, 2000.0)
        self.assertEqual(DQN_PROFIT_LOCK_FAILED_SPRINT_PENALTY, 500.0)
        self.assertEqual(DQN_ACTION_SPECS[0].direction, 0)
        self.assertEqual(len(DQN_LOOKBACK_FEATURES), 8)
        self.assertEqual(sorted(set(spec.amount_bin for spec in DQN_ACTION_SPECS[1:])), list(DQN_AMOUNT_BINS))
        self.assertEqual(sorted(set(spec.stake_per_play for spec in DQN_ACTION_SPECS[1:])), list(DQN_STAKE_AMOUNTS))
        self.assertEqual(max(len(spec.directions) for spec in DQN_ACTION_SPECS), 3)
        self.assertNotIn("small_odd+big_even+small_even+big_odd", [spec.label for spec in DQN_ACTION_SPECS])
        self.assertEqual(len([spec for spec in DQN_ACTION_SPECS if spec.is_wait]), 1)
        self.assertEqual(len(DQN_ACTION_SPECS), 1 + 14 * len(DQN_AMOUNT_BINS))
        self.assertEqual(metadata_path_for(settings.model_path), Path.cwd() / "ai_agent/models/dqn_test_model.metadata.json")

    def test_dqn_preflight_requires_complete_lookback_and_episode_window(self):
        from ai_agent.train_dqn import dqn_required_history_rows, inspect_dqn_history, preflight_dqn_training

        short_rows = [
            {"issue": str(1000 + index), "digits": [1, 2, 4], "sum": 7, "is_straight": False}
            for index in range(800)
        ]
        enough_rows = [
            {"issue": str(2000 + index), "digits": [1, 2, 4], "sum": 7, "is_straight": False}
            for index in range(dqn_required_history_rows())
        ]

        report = inspect_dqn_history(short_rows, label="train")

        self.assertFalse(report.ok)
        self.assertEqual(report.rows, 800)
        self.assertEqual(report.required_rows, 1201)
        with self.assertRaisesRegex(ValueError, "train has 800 rows"):
            preflight_dqn_training(short_rows, eval_history_source=short_rows)
        self.assertTrue(inspect_dqn_history(enough_rows, label="train").ok)
        self.assertEqual(preflight_dqn_training(enough_rows)["train"]["complete_start_min"], 1000)

    def test_dqn_episode_starts_skip_incomplete_lookback_window(self):
        from ai_agent.train_dqn import _episode_starts, _single_reference_window_start

        self.assertEqual(_episode_starts(1201, 1, 200, min_start=1000), [1000])
        self.assertEqual(_episode_starts(800, 5, 200, min_start=1000), [0])
        self.assertEqual(_episode_starts(1401, 3, 200, min_start=1000), [1000, 1100, 1200])
        self.assertEqual(_single_reference_window_start(1200, max_steps=200, lookback=1000), 1000)
        self.assertEqual(_single_reference_window_start(1600, max_steps=200, lookback=1000, start_step=1200), 1200)
        with self.assertRaisesRegex(ValueError, "at least"):
            _single_reference_window_start(1199, max_steps=200, lookback=1000)
        with self.assertRaisesRegex(ValueError, "at least lookback"):
            _single_reference_window_start(1600, max_steps=200, lookback=1000, start_step=999)

    def test_dqn_evaluation_summary_tracks_selected_action_history_reference(self):
        from ai_agent.train_dqn import _summarize_episodes

        summary = _summarize_episodes(
            [
                {
                    "result": "fail",
                    "bets": 2,
                    "bet_rate": 1.0,
                    "wait_rate": 0.0,
                    "wins": 0,
                    "pnl": -400.0,
                    "final_balance": 600.0,
                    "max_drawdown": 0.4,
                    "loss_streak": 2,
                    "history_reference_bets": 2,
                    "history_bad_bets": 2,
                    "history_win_rate_avg": 0.4,
                    "history_avg_pnl": -3.0,
                    "history_max_loss_streak": 8,
                },
                {
                    "result": "pass",
                    "bets": 2,
                    "bet_rate": 1.0,
                    "wait_rate": 0.0,
                    "wins": 2,
                    "pnl": 400.0,
                    "final_balance": 1400.0,
                    "max_drawdown": 0.1,
                    "loss_streak": 0,
                    "history_reference_bets": 2,
                    "history_bad_bets": 0,
                    "history_win_rate_avg": 0.6,
                    "history_avg_pnl": 5.0,
                    "history_max_loss_streak": 2,
                },
            ]
        )

        self.assertEqual(summary["history_reference_bets"], 4)
        self.assertEqual(summary["history_bad_bets"], 2)
        self.assertEqual(summary["history_bad_bet_rate"], 0.5)
        self.assertEqual(summary["history_win_rate_avg"], 0.5)
        self.assertEqual(summary["history_avg_pnl"], 1.0)
        self.assertEqual(summary["history_max_loss_streak"], 8)


class EvaluationTests(unittest.TestCase):
    def test_evaluation_classifies_pass_and_overfit(self):
        from ai_agent.evaluation import classify_evaluation

        good_section = {
            "agent": {"steps": 100, "bets": 30, "net_profit": 100.0, "max_drawdown": 0.1},
            "baselines": {
                "watch": {"net_profit": 0.0},
                "always_reverse": {"net_profit": 50.0},
            },
        }
        pass_report = {
            "train": good_section,
            "valid": good_section,
            "test": good_section,
            "walk_forward": [good_section, good_section, good_section],
        }

        self.assertEqual(classify_evaluation(pass_report).status, "PASS")

        weak_test = {
            "agent": {"steps": 100, "bets": 30, "net_profit": -10.0, "max_drawdown": 0.1},
            "baselines": {"watch": {"net_profit": 0.0}},
        }
        self.assertEqual(classify_evaluation({"train": good_section, "valid": good_section, "test": weak_test}).status, "OVERFIT")

    def test_evaluation_report_includes_bet_rate(self):
        from ai_agent.evaluate_current import _display_width, format_chinese_tables

        report = {
            "verdict": {"status": "FAILED", "reasons": ["x"]},
            "train": {
                "agent": {
                    "steps": 100,
                    "bets": 35,
                    "bet_rate": 0.35,
                    "final_balance": 12000.0,
                    "net_profit": 0.0,
                    "max_drawdown": 0.1,
                },
                "baselines": {
                    "watch_all": {"steps": 100, "bets": 0, "bet_rate": 0.0, "final_balance": 12000.0, "net_profit": 0.0, "max_drawdown": 0.0},
                    "always_non_straight_min": {"steps": 100, "bets": 100, "bet_rate": 1.0, "final_balance": 10000.0, "net_profit": -2000.0, "max_drawdown": 0.2},
                    "always_non_straight_only_min": {"steps": 100, "bets": 100, "bet_rate": 1.0, "final_balance": 10100.0, "net_profit": -1900.0, "max_drawdown": 0.2},
                    "always_straight_min": {"steps": 100, "bets": 100, "bet_rate": 1.0, "final_balance": 13000.0, "net_profit": 1000.0, "max_drawdown": 0.1},
                    "random_min": {"steps": 100, "bets": 50, "bet_rate": 0.5, "final_balance": 12100.0, "net_profit": 100.0, "max_drawdown": 0.1},
                },
            },
        }

        table = format_chinese_tables(report)

        self.assertIn("下注率", table)
        self.assertIn("35.00%", table)
        table_blocks = []
        current_block = []
        for line in table.splitlines():
            if line.startswith("|"):
                current_block.append(line)
            elif current_block:
                table_blocks.append(current_block)
                current_block = []
        if current_block:
            table_blocks.append(current_block)

        self.assertTrue(table_blocks)
        for block in table_blocks:
            self.assertEqual(len({_display_width(line) for line in block}), 1)

    def test_split_history_file_preserves_source_order_under_data(self):
        import json
        from unittest.mock import patch
        import ai_agent.split_data as split_data

        with TemporaryDirectory(dir=Path.cwd() / "data") as tmpdir:
            split_root = Path(tmpdir)
            source = split_root / "test_split_source.jsonl"
            output_root = split_root / "outputs"
            rows = [
                {"issue": "1003", "is_straight": False},
                {"issue": "1001", "is_straight": False},
                {"issue": "1002", "is_straight": True},
                {"issue": "1004", "is_straight": False},
                {"issue": "1005", "is_straight": False},
                {"issue": "1006", "is_straight": True},
            ]
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            with patch.object(split_data, "DATA_DIR", output_root):
                manifest = split_data.split_history_file(source, train_ratio=0.5, valid_ratio=0.25, walk_forward_segments=2)

            self.assertEqual(manifest.train_rows, 3)
            self.assertEqual(manifest.valid_rows, 2)
            self.assertEqual(manifest.test_rows, 1)
            self.assertEqual(manifest.walk_forward_segments, 1)
            train_lines = (output_root / "train.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line)["issue"] for line in train_lines], ["1003", "1001", "1002"])

    def test_split_history_file_uses_expected_1500_round_counts(self):
        import json
        from unittest.mock import patch
        import ai_agent.split_data as split_data

        with TemporaryDirectory(dir=Path.cwd() / "data") as tmpdir:
            split_root = Path(tmpdir)
            source = split_root / "round_1500_source.jsonl"
            output_root = split_root / "outputs"
            rows = [{"issue": str(1000 + index), "is_straight": False} for index in range(1500)]
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            with patch.object(split_data, "DATA_DIR", output_root):
                manifest = split_data.split_history_file(source, train_ratio=0.70, valid_ratio=0.20, walk_forward_segments=5)

            self.assertEqual(manifest.train_rows, 1050)
            self.assertEqual(manifest.valid_rows, 300)
            self.assertEqual(manifest.test_rows, 150)

    def test_rule_filter_uses_only_prior_window_and_all_plays(self):
        from ai_agent.casino_env import ACTION_SPACE_SIZE
        from ai_agent.rule_filter import FilterParams, _build_decisions, evaluate_filter

        rows = [
            {"issue": str(1000 + index), "digits": [index % 10, 0, 0], "sum": index % 10, "is_straight": False}
            for index in range(8)
        ]
        params = FilterParams(window=3, min_score=-999.0, max_losing_streak=99, max_active_bet_rate=1.0)
        decisions = _build_decisions(rows, params)

        self.assertEqual(len(decisions), 8)
        self.assertEqual(decisions[:3], [[0, 0], [0, 0], [0, 0]])
        self.assertTrue(all(0 <= play < ACTION_SPACE_SIZE for play, _amount in decisions))

        metrics = evaluate_filter(rows, params, initial_balance=1000.0)
        self.assertEqual(metrics["steps"], 8)
        self.assertIn("top_plays", metrics)

    def test_rule_filter_marks_walk_forward_invalid_when_window_exceeds_segment(self):
        from ai_agent.rule_filter import FilterParams, _walk_forward_audit

        sections = [
            {"agent": {"steps": 30, "bets": 0}},
            {"agent": {"steps": 30, "bets": 0}},
        ]
        audit = _walk_forward_audit(sections, FilterParams(window=100, min_score=0.0, max_losing_streak=5))

        self.assertFalse(audit["valid"])
        self.assertEqual(audit["invalid_segments"], 2)
        self.assertIn("window_not_smaller_than_segment", audit["reasons"])
        self.assertIn("all_segments_watch_only", audit["reasons"])

    def test_rule_filter_selection_rejects_high_active_bet_rate(self):
        from ai_agent.rule_filter import _selection_score

        metrics = {
            "steps": 1500,
            "bets": 1300,
            "bet_rate": 0.866667,
            "active_steps": 1300,
            "active_bet_rate": 1.0,
            "net_profit": 5000.0,
            "max_drawdown": 0.01,
        }

        self.assertEqual(_selection_score(metrics, max_active_bet_rate=0.5), -1_000_000.0)

    def test_rule_filter_caps_active_bet_rate_in_decisions(self):
        from ai_agent.rule_filter import FilterParams, _build_decisions

        rows = [
            {"issue": str(1000 + index), "digits": [index % 10, 0, 0], "sum": index % 10, "is_straight": False}
            for index in range(13)
        ]
        params = FilterParams(window=3, min_score=-999.0, max_losing_streak=99, max_active_bet_rate=0.5)
        decisions = _build_decisions(rows, params)
        active_decisions = decisions[params.window :]

        self.assertLessEqual(sum(1 for play, _amount in active_decisions if play), len(active_decisions) * 0.5)

    def test_rule_filter_package_settlement_uses_min_stake_per_play(self):
        from ai_agent.casino_env import CasinoEnv
        from ai_agent.rule_filter import _play_for_label, _settle_package

        row = {"issue": "1000", "digits": [8, 8, 8], "sum": 24, "is_straight": False}
        env = CasinoEnv([row], initial_balance=1000.0)
        package = (_play_for_label("big_even"), _play_for_label("small_even"))

        self.assertEqual(_settle_package(env, package, env.history[0], 0), 16.9)

    def test_rule_filter_package_mode_can_select_multi_play_package(self):
        from ai_agent.rule_filter import FilterParams, _build_package_decisions, _play_for_label

        rows = [
            {"issue": str(1000 + index), "digits": [8, 8, 8], "sum": 24, "is_straight": False}
            for index in range(8)
        ]
        package = (_play_for_label("big_even"), _play_for_label("small_even"))
        params = FilterParams(window=3, min_score=-999.0, max_losing_streak=99, max_active_bet_rate=1.0)
        decisions = _build_package_decisions(rows, params, (package,))

        self.assertEqual(decisions[:3], [(), (), ()])
        self.assertIn(package, decisions[3:])

    def test_rule_filter_package_mode_caps_active_bet_rate(self):
        from ai_agent.rule_filter import FilterParams, _build_package_decisions, _play_for_label

        rows = [
            {"issue": str(1000 + index), "digits": [8, 8, 8], "sum": 24, "is_straight": False}
            for index in range(13)
        ]
        package = (_play_for_label("big_even"), _play_for_label("small_even"))
        params = FilterParams(window=3, min_score=-999.0, max_losing_streak=99, max_active_bet_rate=0.5)
        decisions = _build_package_decisions(rows, params, (package,))
        active_decisions = decisions[params.window :]

        self.assertLessEqual(sum(1 for package in active_decisions if package), len(active_decisions) * 0.5)


class GrayRunTests(unittest.TestCase):
    def test_gray_cycle_records_dry_run_decision_without_live_bet(self):
        from ai_agent.gray_run import GrayRunSettings, run_gray_cycle

        log_path = Path("ai_agent/logs/test_gray.jsonl")
        if log_path.exists():
            log_path.unlink()
        settings = GrayRunSettings(
            history_path=Path("data/history.jsonl"),
            model_path=Path("ai_agent/models/model.zip"),
            balance=1000.0,
            cycles=2,
            interval_seconds=0,
            dry_run=True,
            log_path=log_path,
        )
        events = run_gray_cycle(
            settings,
            predictor=lambda _: {"issue": "3446544", "direction": 2, "amount_ratio": 0.058, "stake": 58.0},
            executor=lambda **kwargs: {"ok": True, "dry_run": kwargs["dry_run"], "payload": kwargs},
            sleeper=lambda _: None,
        )

        lines = (Path.cwd() / log_path).read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(len(events), 2)
        self.assertEqual(len(lines), 2)
        self.assertTrue(events[0]["execution"]["dry_run"])
        self.assertEqual(events[0]["decision"]["direction"], 2)


class LiveServerTests(unittest.TestCase):
    def test_live_signal_applies_risk_guard_without_hiding_ppo_decision(self):
        from ai_agent.live_server import build_agent_signal

        result = build_agent_signal(
            {"feature": {"latest": {"issue": "3447542"}}, "signal": {"risk_score": 90}},
            model_path="/tmp/model.zip",
            balance=1000.0,
            predictor=lambda *args, **kwargs: {
                "issue": "3447541",
                "direction": 2,
                "direction_label": "non_straight",
                "amount_bin": 99,
                "amount_ratio": 0.1,
                "stake": 100.0,
            },
        )

        self.assertTrue(result["allowed"])
        self.assertEqual(result["issue"], "3447542")
        self.assertEqual(result["direction"], 2)
        self.assertEqual(result["amount_ratio"], 0.02)
        self.assertEqual(result["stake"], 20.0)
        self.assertIn("high_pressure_capped", result["guard_reasons"])
        self.assertEqual(result["ppo"]["amount_ratio"], 0.1)

    def test_live_signal_blocks_when_recent_issues_have_gap(self):
        from ai_agent.live_server import build_agent_signal

        def fail_predictor(*args, **kwargs):
            raise AssertionError("predictor should not run on gapped data")

        result = build_agent_signal(
            {
                "feature": {
                    "latest": {"issue": "1005"},
                    "pressure_series": [
                        {"issue": "1001", "is_straight": False},
                        {"issue": "1002", "is_straight": False},
                        {"issue": "1005", "is_straight": False},
                    ],
                },
                "signal": {"risk_score": 70},
            },
            model_path="/tmp/model.zip",
            balance=1000.0,
            predictor=fail_predictor,
        )

        self.assertFalse(result["allowed"])
        self.assertEqual(result["direction"], 0)
        self.assertEqual(result["stake"], 0.0)
        self.assertIn("data_gap", result["guard_reasons"])
        self.assertEqual(result["data_quality"]["missing_count"], 2)

    def test_live_signal_caps_final_stake_at_max_stake(self):
        from ai_agent.live_server import build_agent_signal

        result = build_agent_signal(
            {"feature": {"latest": {"issue": "3447542"}}, "signal": {"risk_score": 10}},
            model_path="/tmp/model.zip",
            balance=200000.0,
            predictor=lambda *args, **kwargs: {
                "issue": "3447542",
                "direction": 2,
                "direction_label": "non_straight",
                "amount_bin": 99,
                "amount_ratio": 0.1,
                "max_stake": 10000.0,
                "stake": 10000.0,
            },
        )

        self.assertTrue(result["allowed"])
        self.assertEqual(result["amount_ratio"], 0.05)
        self.assertEqual(result["max_stake"], 10000.0)
        self.assertEqual(result["stake"], 10000.0)


if __name__ == "__main__":
    unittest.main()
