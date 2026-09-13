import unittest


class ComboMarketTests(unittest.TestCase):
    def test_base_markets_and_odds_match_confirmed_contract(self):
        from ai_agent.markets_combo import BASE_ODDS, BaseMarket, base_market_for_sum, market_odds

        self.assertEqual(base_market_for_sum(13), BaseMarket.SMALL_ODD)
        self.assertEqual(base_market_for_sum(14), BaseMarket.BIG_EVEN)
        self.assertEqual(base_market_for_sum(15), BaseMarket.BIG_ODD)
        self.assertEqual(base_market_for_sum(0), None)
        self.assertEqual(BASE_ODDS[BaseMarket.SMALL_ODD], 3.69)
        self.assertEqual(BASE_ODDS[BaseMarket.BIG_EVEN], 3.69)
        self.assertEqual(market_odds("small_even"), 4.29)
        self.assertEqual(market_odds("big_odd"), 4.29)
        self.assertEqual(market_odds("sum_15"), 13.0)

    def test_extreme_requires_explicit_boundary_configuration(self):
        from ai_agent.markets_combo import ComboMarketError, market_hits

        with self.assertRaises(ComboMarketError):
            market_hits("extreme_small", 3)

    def test_configured_extreme_ranges_are_inclusive_and_non_overlapping(self):
        from ai_agent.markets_combo import ExtremeRanges, market_hits

        ranges = ExtremeRanges(0, 5, 22, 27)
        self.assertTrue(market_hits("extreme_small", 0, extreme_ranges=ranges))
        self.assertTrue(market_hits("extreme_small", 5, extreme_ranges=ranges))
        self.assertFalse(market_hits("extreme_small", 6, extreme_ranges=ranges))
        self.assertTrue(market_hits("extreme_big", 22, extreme_ranges=ranges))
        self.assertTrue(market_hits("extreme_big", 27, extreme_ranges=ranges))
        self.assertFalse(market_hits("extreme_big", 21, extreme_ranges=ranges))

    def test_all_nonzero_sums_have_exactly_one_base_market(self):
        from ai_agent.markets_combo import base_market_for_sum

        self.assertTrue(all(base_market_for_sum(total) is not None for total in range(1, 28)))


class ComboPortfolioTests(unittest.TestCase):
    def test_every_action_base_combo_has_exactly_three_distinct_base_markets(self):
        from ai_agent.portfolio_combo import BASE_COMBINATIONS

        self.assertEqual(len(BASE_COMBINATIONS), 4)
        for markets in BASE_COMBINATIONS:
            self.assertEqual(len(markets), 3)
            self.assertEqual(len(set(markets)), 3)

    def test_build_decision_never_substitutes_extensions_for_base_markets(self):
        from ai_agent.portfolio_combo import EXTENSION_PAIRS, PortfolioLimits, build_decision

        pair_id = EXTENSION_PAIRS.index(("sum_15", "extreme_big"))
        decision = build_decision(
            action=(0, pair_id, 1, 0),
            balance=20_000,
            probabilities={"small_odd": 0.2, "small_even": 0.2, "big_odd": 0.3, "sum_15": 0.1},
            limits=PortfolioLimits(max_exposure_ratio=0.01),
            extension_options=EXTENSION_PAIRS,
        )

        self.assertEqual(len(decision.selections), 5)
        self.assertEqual(sum(item.market_type == "base" for item in decision.selections), 3)
        self.assertEqual(sum(item.market_type != "base" for item in decision.selections), 2)
        self.assertLessEqual(sum(item.stake for item in decision.selections), 200.0)

    def test_action_encoding_has_no_duplicate_extension_variant(self):
        from ai_agent.markets_combo import ExtremeRanges
        from ai_agent.portfolio_combo import EXTENSION_PAIRS, action_nvec, extension_pairs

        self.assertTrue(all(len(pair) == len(set(pair)) for pair in EXTENSION_PAIRS))
        self.assertTrue(all(len(pair) <= 2 for pair in EXTENSION_PAIRS))
        self.assertEqual(action_nvec(), (5, 1, 3, 3))
        self.assertEqual(
            action_nvec(extreme_ranges=ExtremeRanges(0, 5, 22, 27), extensions_enabled=True),
            (5, len(EXTENSION_PAIRS), 3, 3),
        )

    def test_learned_actions_cannot_select_shadow_mode(self):
        from ai_agent.portfolio_combo import PortfolioError, TRAINING_MODE_IDS, decode_action

        self.assertEqual(TRAINING_MODE_IDS, ("defense", "balanced", "attack"))
        with self.assertRaises(PortfolioError):
            decode_action((0, 0, 0, 3))

    def test_decode_action_rejects_negative_indices(self):
        from ai_agent.portfolio_combo import PortfolioError, decode_action

        for action in ((-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1)):
            with self.assertRaises(PortfolioError):
                decode_action(action)

    def test_stake_profile_allocation_is_explicit_and_budget_bounded(self):
        from ai_agent.portfolio_combo import EXTENSION_PAIRS, PortfolioLimits, StakeAllocationConfig, build_decision

        limits = PortfolioLimits(
            stake_allocations=StakeAllocationConfig(
                defense_base_share=0.8,
                balanced_base_share=0.6,
                attack_base_share=0.4,
            )
        )
        decision = build_decision(action=(0, 1, 0, 0), balance=20_000, limits=limits, extension_options=EXTENSION_PAIRS)
        base_stake = sum(item.stake for item in decision.selections if item.market_type == "base")
        extension_stake = sum(item.stake for item in decision.selections if item.market_type != "base")
        self.assertEqual(base_stake + extension_stake, 240.0)
        self.assertTrue(all(30 <= item.stake <= 300 and item.stake.is_integer() for item in decision.selections))

    def test_profile_and_mode_change_virtual_exposure_and_concentration(self):
        from ai_agent.portfolio_combo import build_decision

        defense = build_decision(action=(0, 0, 0, 0), balance=10_000, issue="10")
        balanced = build_decision(action=(0, 0, 1, 1), balance=10_000, issue="10")
        attack = build_decision(action=(0, 0, 2, 2), balance=10_000, issue="10")

        self.assertEqual(sum(item.stake for item in defense.selections), 150.0)
        self.assertEqual(sum(item.stake for item in balanced.selections), 225.0)
        self.assertEqual(sum(item.stake for item in attack.selections), 300.0)
        self.assertEqual(len({item.stake for item in defense.selections}), 1)
        self.assertGreater(attack.selections[0].stake, max(item.stake for item in attack.selections[1:]))


class ComboSettlementTests(unittest.TestCase):
    @staticmethod
    def _user_example_decision(mode="defense"):
        from ai_agent.markets_combo import market_odds, market_type
        from ai_agent.schemas_combo import ComboDecision, Selection

        return ComboDecision(
            mode=mode,
            selections=(
                Selection("small_odd", market_type("small_odd").value, "defense", 30, market_odds("small_odd"), 0.2),
                Selection("small_even", market_type("small_even").value, "defense", 30, market_odds("small_even"), 0.2),
                Selection("big_odd", market_type("big_odd").value, "primary", 30, market_odds("big_odd"), 0.3),
            ),
        )

    def test_user_example_settles_as_gross_odds_with_stake_included(self):
        from ai_agent.settlement_combo import settle_combo

        result = settle_combo(self._user_example_decision(), outcome_sum=15, balance_before=10_000)

        self.assertEqual(result.total_stake, 90.0)
        self.assertEqual(result.total_return, 128.7)
        self.assertEqual(result.net_profit, 38.7)
        self.assertEqual(result.balance_after, 10038.7)
        self.assertEqual([item.hit for item in result.items], [False, False, True])

    def test_exact_sum_can_return_with_base_market(self):
        from ai_agent.markets_combo import market_odds, market_type
        from ai_agent.schemas_combo import ComboDecision, Selection
        from ai_agent.settlement_combo import settle_combo

        base = self._user_example_decision().selections
        decision = ComboDecision(
            mode="attack",
            selections=base + (
                Selection("sum_15", market_type("sum_15").value, "attack", 30, market_odds("sum_15"), 0.1),
            ),
        )
        result = settle_combo(decision, outcome_sum=15, balance_before=12_000)

        self.assertEqual(result.total_stake, 120.0)
        self.assertEqual(result.total_return, 518.7)
        self.assertEqual(result.net_profit, 398.7)

    def test_shadow_settlement_preserves_actual_balance(self):
        from ai_agent.settlement_combo import settle_combo

        result = settle_combo(self._user_example_decision(mode="shadow"), outcome_sum=15, balance_before=10_000)

        self.assertTrue(result.is_shadow)
        self.assertEqual(result.shadow_net_profit, 38.7)
        self.assertEqual(result.balance_delta, 0.0)
        self.assertEqual(result.balance_after, 10_000.0)

    def test_combo_reward_uses_shadow_pnl_without_changing_balance(self):
        from ai_agent.reward import calculate_combo_reward
        from ai_agent.settlement_combo import settle_combo

        settlement = settle_combo(self._user_example_decision(mode="shadow"), outcome_sum=15, balance_before=10_000)
        reward = calculate_combo_reward(settlement, initial_balance=10_000, drawdown=0.0, losing_streak=0)
        self.assertEqual(settlement.balance_after, 10_000.0)
        self.assertEqual(reward.realized_pnl_component, 0.0)
        self.assertGreater(reward.shadow_pnl_component, 0.0)
        self.assertGreater(reward.calibration_error, 0.0)

    def test_settlement_rejects_external_decision_above_budget(self):
        from ai_agent.settlement_combo import settle_combo

        result = settle_combo(self._user_example_decision(), outcome_sum=15, balance_before=1_000)
        self.assertEqual(result.total_stake, 90.0)


class ComboProbabilityTests(unittest.TestCase):
    @staticmethod
    def _constant_history(start: int, count: int):
        return [
            {"issue": str(start + index), "digits": [2, 2, 2], "sum": 6}
            for index in range(count)
        ]

    def test_probability_model_is_a_normalized_sum_distribution_and_derives_markets(self):
        from ai_agent.data_adapter import clean_combo_history
        from ai_agent.markets_combo import CONFIRMED_EXTREME_RANGES
        from ai_agent.probability_combo import ProbabilityTrainingConfig, fit_combo_probability_model

        model = fit_combo_probability_model(
            self._constant_history(1, 120),
            config=ProbabilityTrainingConfig(epochs=10),
        )
        features = clean_combo_history(self._constant_history(121, 1), min_rows=1)[0]["features"]
        probabilities = model.predict_markets(features, extreme_ranges=CONFIRMED_EXTREME_RANGES)
        sums = model.predict_sum_probabilities(features)

        self.assertAlmostEqual(sum(sums), 1.0)
        self.assertGreaterEqual(probabilities["extreme_small"], 0.0)
        self.assertGreater(probabilities["small_even"], 0.0)
        self.assertEqual(model.training_end_issue, "96")
        self.assertEqual(model.calibration_end_issue, "120")

    def test_statistical_gate_waits_when_probability_validation_fails(self):
        from ai_agent.markets_combo import CONFIRMED_EXTREME_RANGES
        from ai_agent.probability_combo import ProbabilityTrainingConfig, fit_combo_probability_model
        from ai_agent.statistical_combo_policy import OpportunityGateConfig, choose_statistical_opportunity

        history = self._constant_history(1, 120)
        model = fit_combo_probability_model(history, config=ProbabilityTrainingConfig(epochs=10))
        row_features = {key: 0.0 for key in model.feature_columns}
        opportunity = choose_statistical_opportunity(
            model,
            features=row_features,
            balance=10_000,
            drawdown=0.0,
            losing_streak=0,
            probability_validation={"passed": False},
            extreme_ranges=CONFIRMED_EXTREME_RANGES,
            config=OpportunityGateConfig(require_probability_validation=True),
        )

        self.assertTrue(opportunity.wait)
        self.assertEqual(opportunity.reason, "probability_validation_not_predictive")
        self.assertIsNone(opportunity.decision)

    def test_statistical_training_keeps_validation_out_of_probability_fitting(self):
        from ai_agent.probability_combo import ProbabilityTrainingConfig
        from ai_agent.train_statistical_combo import train_and_evaluate_statistical_combo

        report = train_and_evaluate_statistical_combo(
            self._constant_history(1, 120),
            self._constant_history(121, 20),
            training_config=ProbabilityTrainingConfig(epochs=10),
        )

        self.assertEqual(report["model"]["calibration_end_issue"], "120")
        self.assertEqual(report["validation"]["steps"], 20)
        self.assertFalse(report["test_accessed"])


class ComboEnvironmentTests(unittest.TestCase):
    def test_combo_history_uses_live_api_recent_results_before_pressure_series(self):
        from ai_agent.data_adapter import clean_combo_history

        rows = clean_combo_history(
            {
                "feature": {
                    "pressure_series": [{"issue": "1", "risk_score": 50}],
                    "recent_results": [
                        {"issue": "100", "digits": [1, 2, 3], "sum": 6},
                        {"issue": "101", "digits": [2, 3, 4], "sum": 9},
                    ],
                }
            },
            min_rows=2,
        )
        self.assertEqual([row["issue"] for row in rows], ["100", "101"])

    def test_zero_is_not_labeled_as_small_when_zero_policy_excludes_it(self):
        from ai_agent.data_adapter import clean_combo_history

        rows = clean_combo_history(
            [{"issue": "1", "digits": [0, 0, 0], "sum": 0}],
            min_rows=1,
        )
        self.assertIsNone(rows[0]["base_label"])
        self.assertIsNone(rows[0]["size"])

    def test_environment_uses_four_axis_action_and_settles_legal_combo(self):
        from ai_agent.combo_env import ComboCasinoEnv

        history = [
            {"issue": str(index), "digits": [1, 2, 3], "sum": 6}
            for index in range(1, 5)
        ]
        env = ComboCasinoEnv(history, initial_balance=10_000)
        self.assertEqual(env.action_space.nvec.tolist()[0], 5)
        self.assertEqual(len(env.action_space.nvec), 4)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, env.observation_space.shape)
        obs, reward, terminated, truncated, info = env.step([0, 0, 1, 0])
        self.assertEqual(len(info["decision"]["selections"]), 3)
        self.assertEqual(sum(item["selection"]["market_type"] == "base" for item in info["items"]), 3)
        self.assertFalse(terminated)

    def test_environment_excludes_extreme_extensions_until_configured(self):
        from ai_agent.combo_env import ComboCasinoEnv

        env = ComboCasinoEnv([{"issue": str(index), "digits": [1, 2, 3], "sum": 6} for index in range(3)])
        self.assertNotIn(("extreme_small",), env.extension_sets)
        self.assertNotIn(("extreme_big",), env.extension_sets)

    def test_environment_keeps_extensions_disabled_by_default_even_with_extreme_ranges(self):
        from ai_agent.combo_env import ComboCasinoEnv
        from ai_agent.markets_combo import ExtremeRanges

        env = ComboCasinoEnv(
            [{"issue": str(index), "digits": [1, 2, 3], "sum": 6} for index in range(3)],
            extreme_ranges=ExtremeRanges(0, 5, 22, 27),
        )
        self.assertEqual(env.extension_sets, ((),))

    def test_environment_can_explicitly_enable_configured_extreme_extension(self):
        from ai_agent.combo_env import ComboCasinoEnv
        from ai_agent.markets_combo import ExtremeRanges

        env = ComboCasinoEnv(
            [{"issue": str(index), "digits": [1, 2, 3], "sum": 6} for index in range(3)],
            extreme_ranges=ExtremeRanges(0, 5, 22, 27),
            allow_extensions=True,
        )
        self.assertIn(("extreme_small",), env.extension_sets)

    def test_insufficient_balance_ends_episode_without_creating_an_order(self):
        from ai_agent.combo_env import ComboCasinoEnv

        env = ComboCasinoEnv(
            [{"issue": "1", "digits": [1, 2, 3], "sum": 6}],
            initial_balance=2.99,
        )
        env.reset()
        _obs, reward, terminated, truncated, info = env.step([0, 0, 0, 0])

        self.assertEqual(reward, -1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "insufficient_balance")
        self.assertFalse(info["order_executed"])
        self.assertEqual(info["total_stake"], 0.0)
        self.assertEqual(info["balance"], 2.99)

    def test_wait_action_creates_no_order_and_preserves_the_account(self):
        from ai_agent.combo_env import ComboCasinoEnv
        from ai_agent.portfolio_combo import WAIT_BASE_ACTION_ID

        env = ComboCasinoEnv([{"issue": "1", "digits": [1, 2, 3], "sum": 6}], initial_balance=10_000)
        env.reset()
        _obs, reward, terminated, truncated, info = env.step([WAIT_BASE_ACTION_ID, 0, 0, 0])

        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["wait"])
        self.assertFalse(info["order_executed"])
        self.assertEqual(info["total_stake"], 0.0)
        self.assertEqual(info["balance"], 10_000.0)

    def test_losing_streak_cooldown_expires_after_ten_issues_even_if_model_waits(self):
        from ai_agent.combo_env import ComboCasinoEnv
        from ai_agent.portfolio_combo import WAIT_BASE_ACTION_ID

        history = [{"issue": str(index), "digits": [0, 0, 0], "sum": 0} for index in range(1, 15)]
        env = ComboCasinoEnv(history, initial_balance=10_000)
        env.reset()
        for _ in range(3):
            _obs, _reward, _terminated, _truncated, info = env.step([0, 0, 0, 0])
        self.assertEqual(info["losing_streak"], 3)
        self.assertEqual(info["cooldown_remaining"], 10)

        for expected_remaining in range(9, -1, -1):
            _obs, _reward, _terminated, _truncated, info = env.step([WAIT_BASE_ACTION_ID, 0, 0, 0])
            self.assertTrue(info["wait"])
            self.assertEqual(info["wait_reason"], "losing_streak_cooldown")
            self.assertEqual(info["cooldown_remaining"], expected_remaining)

        _obs, _reward, _terminated, _truncated, info = env.step([0, 0, 0, 0])
        self.assertTrue(info["order_executed"])
        self.assertEqual(info["losing_streak"], 1)

    def test_negative_expected_return_requires_an_explicit_experimental_opt_in(self):
        from ai_agent.statistical_combo_policy import OpportunityGateConfig

        with self.assertRaises(ValueError):
            OpportunityGateConfig(min_expected_return_ratio=-0.01)
        config = OpportunityGateConfig(
            min_expected_return_ratio=-0.01,
            allow_negative_expected_return=True,
            target_bet_rate=0.4,
        )
        self.assertEqual(config.target_bet_rate, 0.4)

    def test_random_start_episode_horizon_and_combo_context_are_explicit(self):
        from ai_agent.combo_env import ComboCasinoEnv

        history = [{"issue": str(index), "digits": [1, 2, 3], "sum": 6} for index in range(1, 11)]
        env = ComboCasinoEnv(
            history,
            random_start_on_reset=True,
            episode_horizon=2,
            include_previous_combo_observation=True,
        )
        _obs, info = env.reset(seed=7)
        self.assertEqual(env.observation_space.shape[0], 40)
        self.assertLessEqual(info["step"], 8)
        _obs, _reward, _terminated, truncated, _info = env.step([0, 0, 0, 0])
        self.assertFalse(truncated)
        _obs, _reward, _terminated, truncated, _info = env.step([0, 0, 0, 0])
        self.assertTrue(truncated)

    def test_environment_probability_gate_turns_bet_into_wait(self):
        from ai_agent.combo_env import ComboCasinoEnv
        from ai_agent.probability_combo import ProbabilityTrainingConfig, fit_combo_probability_model
        from ai_agent.statistical_combo_policy import OpportunityGateConfig

        history = [{"issue": str(index), "digits": [2, 2, 2], "sum": 6} for index in range(1, 121)]
        model = fit_combo_probability_model(history, config=ProbabilityTrainingConfig(epochs=10))
        env = ComboCasinoEnv(
            history,
            probability_model=model,
            probability_validation={"passed": False},
            gate_config=OpportunityGateConfig(require_probability_validation=True),
        )
        env.reset()
        _obs, reward, terminated, truncated, info = env.step([0, 0, 0, 0])

        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["wait"])
        self.assertEqual(info["wait_reason"], "probability_validation_not_predictive")
        self.assertFalse(info["order_executed"])
        self.assertTrue(info["probability_gate"]["wait"])
        self.assertEqual(info["probability_gate"]["reason"], "probability_validation_not_predictive")

    def test_random_baseline_risk_mode_audits_failed_probability_without_blocking_on_it(self):
        from ai_agent.combo_env import ComboCasinoEnv
        from ai_agent.probability_combo import ProbabilityTrainingConfig, fit_combo_probability_model

        history = [{"issue": str(index), "digits": [2, 2, 2], "sum": 6} for index in range(1, 121)]
        model = fit_combo_probability_model(history, config=ProbabilityTrainingConfig(epochs=10))
        env = ComboCasinoEnv(
            history,
            probability_model=model,
            probability_validation={"passed": False},
        )
        env.reset()
        _obs, _reward, _terminated, _truncated, info = env.step([0, 0, 0, 0])

        self.assertNotEqual(info.get("wait_reason"), "probability_validation_not_predictive")
        self.assertNotEqual(info["probability_gate"]["reason"], "probability_validation_not_predictive")


class ComboTrainingTests(unittest.TestCase):
    def test_dqn_projection_round_trips_every_action_axis(self):
        from ai_agent.train_combo import combo_action_to_dqn_index, dqn_index_to_combo_action

        nvec = (4, 407, 3, 3)
        action = (3, 406, 2, 2)
        index = combo_action_to_dqn_index(action, nvec)
        self.assertEqual(dqn_index_to_combo_action(index, nvec), action)

    def test_combo_backtest_and_shadow_baseline_preserve_contract(self):
        from ai_agent.backtest_combo import combo_baseline_policy, run_combo_backtest

        history = [{"issue": str(index), "digits": [1, 2, 3], "sum": 6} for index in range(1, 6)]
        result = run_combo_backtest(history, policy=combo_baseline_policy("shadow"))
        self.assertEqual(result["executed_rounds"], 0)
        self.assertEqual(result["final_balance"], 10_000.0)
        self.assertEqual(len(result["decisions"]), 5)

    def test_continuous_backtest_keeps_one_balance_across_all_validation_rows(self):
        from ai_agent.backtest_combo import combo_baseline_policy, run_combo_backtest

        history = [{"issue": str(index), "digits": [1, 2, 3], "sum": 6} for index in range(1, 6)]
        result = run_combo_backtest(history, policy=combo_baseline_policy("fixed_defense"))
        self.assertEqual(result["steps"], 5)
        self.assertEqual(result["executed_rounds"], 5)
        self.assertEqual(len(result["equity_curve"]), 6)
        self.assertNotEqual(result["final_balance"], 10_000.0)

    def test_backtest_does_not_count_insufficient_balance_as_an_order(self):
        from ai_agent.backtest_combo import combo_baseline_policy, run_combo_backtest

        result = run_combo_backtest(
            [{"issue": "1", "digits": [1, 2, 3], "sum": 6}],
            policy=combo_baseline_policy("fixed_defense"),
            initial_balance=2.99,
        )

        self.assertEqual(result["steps"], 1)
        self.assertEqual(result["executed_rounds"], 0)
        self.assertEqual(result["insufficient_balance_events"], 1)
        self.assertEqual(result["termination_reason"], "insufficient_balance")

    def test_backtest_records_waits_without_counting_orders(self):
        from ai_agent.backtest_combo import run_combo_backtest
        from ai_agent.portfolio_combo import WAIT_BASE_ACTION_ID

        result = run_combo_backtest(
            [{"issue": "1", "digits": [1, 2, 3], "sum": 6}],
            policy=lambda _obs, _info: [WAIT_BASE_ACTION_ID, 0, 0, 0],
        )

        self.assertEqual(result["wait_rounds"], 1)
        self.assertEqual(result["executed_rounds"], 0)
        self.assertEqual(result["final_balance"], 10_000.0)

    def test_default_combo_stakes_are_integer_and_at_least_thirty(self):
        from ai_agent.portfolio_combo import build_decision

        decision = build_decision(action=(0, 0, 0, 0), balance=10_000)
        self.assertTrue(all(item.stake.is_integer() for item in decision.selections))
        self.assertTrue(all(item.stake >= 30 for item in decision.selections))

    def test_probability_gate_forces_wait_when_qualification_is_false(self):
        from ai_agent.backtest_combo import probability_gated_policy, run_combo_backtest
        from ai_agent.probability_combo import ProbabilityTrainingConfig, fit_combo_probability_model
        from ai_agent.statistical_combo_policy import OpportunityGateConfig

        history = [{"issue": str(index), "digits": [2, 2, 2], "sum": 6} for index in range(1, 121)]
        model = fit_combo_probability_model(history, config=ProbabilityTrainingConfig(epochs=10))
        policy = probability_gated_policy(
            lambda _obs, _info: [0, 0, 0, 0],
            model,
            probability_validation={"passed": False},
            extreme_ranges=None,
            gate_config=OpportunityGateConfig(require_probability_validation=True),
        )
        result = run_combo_backtest(history, policy=policy)
        self.assertEqual(result["wait_rounds"], 120)
        self.assertEqual(result["executed_rounds"], 0)

    def test_probability_baseline_obeys_three_base_market_constraint(self):
        from ai_agent.backtest_combo import combo_baseline_policy

        action = combo_baseline_policy("probability")([0.1, 0.2, 0.4, 0.3], {})
        self.assertEqual(len(action), 4)
        self.assertIn(action[0], range(4))
        self.assertEqual(action[1:], [0, 0, 0])

    def test_walk_forward_factory_receives_only_prior_rows(self):
        from ai_agent.evaluation_combo import run_combo_walk_forward

        history = [{"issue": str(index), "digits": [1, 2, 3], "sum": 6} for index in range(1, 11)]
        training_sizes: list[int] = []

        def factory(train_rows):
            training_sizes.append(len(train_rows))
            return lambda _obs, _info: [0, 0, 0, 0]

        report = run_combo_walk_forward(history, policy_factory=factory, window_size=2, windows=3)
        self.assertEqual(training_sizes, [4, 6, 8])
        self.assertEqual(report["walk_forward"][0]["train_end_issue"], "4")
        self.assertEqual(report["walk_forward"][0]["test_start_issue"], "5")

    def test_walk_forward_keeps_prior_context_but_only_settles_window_rows(self):
        from ai_agent.evaluation_combo import run_combo_walk_forward

        history = [{"issue": str(index), "digits": [1, 2, 3], "sum": 6} for index in range(1, 11)]
        report = run_combo_walk_forward(history, policy=lambda _obs, _info: [0, 0, 0, 0], window_size=2, windows=3)
        self.assertEqual([item["agent"]["steps"] for item in report["walk_forward"]], [2, 2, 2])

    def test_combo_model_evaluation_rejects_test_period_at_or_before_training_cutoff(self):
        from tempfile import TemporaryDirectory
        import json

        from ai_agent.evaluation_combo import evaluate_combo_model

        history = [{"issue": str(index), "digits": [1, 2, 3], "sum": 6} for index in range(1, 11)]
        with TemporaryDirectory() as directory:
            model = f"{directory}/candidate.zip"
            metadata = f"{directory}/candidate.metadata.json"
            with open(metadata, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "algorithm": "PPO",
                        "rule_version": "pc28-combo-v1",
                        "odds_version": "user-odds-1",
                        "action_space_version": "multidiscrete-base4-extension-profile-mode-v1",
                        "training_data_end_issue": "8",
                        "action_space": {"nvec": [4, 407, 3, 4]},
                    },
                    handle,
                )
            with self.assertRaises(ValueError):
                evaluate_combo_model(history, model_path=model, algorithm="ppo", window_size=2, windows=3)

    def test_combo_inference_rejects_extensions_by_default_and_requires_explicit_opt_in(self):
        from ai_agent.infer_combo import action_to_combo_decision

        with self.assertRaises(ValueError):
            action_to_combo_decision([0, 1, 0, 0], balance=10_000, extreme_ranges=None)
        decision = action_to_combo_decision(
            [0, 29, 0, 0],
            balance=20_000,
            extreme_ranges=None,
            allow_extensions=True,
        )
        markets = [item["market"] for item in decision["selections"]]
        self.assertIn("sum_00", markets)
        self.assertIn("sum_01", markets)

    def test_combo_model_metadata_rejects_legacy_action_schema(self):
        from tempfile import TemporaryDirectory

        from ai_agent.infer_combo import validate_combo_model_metadata

        with TemporaryDirectory() as directory:
            model = f"{directory}/legacy.zip"
            metadata = f"{directory}/legacy.metadata.json"
            with open(metadata, "w", encoding="utf-8") as handle:
                handle.write('{"algorithm":"PPO","rule_version":"legacy","odds_version":"old","action_space_version":"old"}')
            with self.assertRaises(ValueError):
                validate_combo_model_metadata(model, algorithm="ppo")

    def test_combo_model_metadata_rejects_missing_feature_contract(self):
        from tempfile import TemporaryDirectory
        import json

        from ai_agent.infer_combo import validate_combo_model_metadata

        with TemporaryDirectory() as directory:
            model = f"{directory}/legacy-features.zip"
            metadata = f"{directory}/legacy-features.metadata.json"
            with open(metadata, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "algorithm": "PPO",
                        "rule_version": "pc28-combo-v1",
                        "odds_version": "user-odds-1",
                        "action_space_version": "multidiscrete-base4-extension-pair-profile-mode-v2",
                        "action_space": {"type": "MultiDiscrete", "nvec": [4, 407, 3, 4]},
                        "feature_version": "combo-history-v1",
                        "feature_columns": [],
                        "zero_policy": "exclude",
                    },
                    handle,
                )
            with self.assertRaises(ValueError):
                validate_combo_model_metadata(model, algorithm="ppo")

    def test_combo_model_metadata_rejects_mismatched_extreme_ranges(self):
        from ai_agent.infer_combo import validate_combo_model_metadata
        from ai_agent.markets_combo import CONFIRMED_EXTREME_RANGES

        with self.assertRaises(ValueError):
            validate_combo_model_metadata(
                "ai_agent/models/ppo_combo_feature_v2_smoke.zip",
                algorithm="ppo",
                extreme_ranges=CONFIRMED_EXTREME_RANGES,
            )

    def test_combo_prediction_targets_the_next_numeric_issue(self):
        from ai_agent.infer_combo import _next_issue

        self.assertEqual(_next_issue("000099"), "000100")
        self.assertEqual(_next_issue("3478003"), "3478004")

    def test_combo_prediction_probabilities_come_from_prior_features(self):
        from ai_agent.infer_combo import _selection_probabilities

        probabilities = _selection_probabilities({"base_rate_small_odd_100": 0.25, "base_rate_big_even_100": 0.3})
        self.assertEqual(probabilities["small_odd"], 0.25)
        self.assertEqual(probabilities["big_even"], 0.3)
        self.assertEqual(probabilities["big_odd"], 0.0)

    def test_live_combo_signal_is_dry_run_only(self):
        from ai_agent.live_server import build_combo_signal

        payload = {
            "rows": [{"issue": str(index), "digits": [1, 2, 3], "sum": 6} for index in range(1, 6)]
        }
        with self.assertRaises(ValueError):
            build_combo_signal(
                payload,
                model_path="ai_agent/models/ppo_combo_feature_v2_smoke.zip",
                algorithm="ppo",
                balance=10_000,
            )

    def test_combo_log_history_and_metrics_are_read_only(self):
        from tempfile import TemporaryDirectory

        from ai_agent.live_server import combo_metrics, read_combo_history
        from ai_agent.online_combo import ComboOnlineLedger

        with TemporaryDirectory() as directory:
            path = f"{directory}/combo.jsonl"
            ledger = ComboOnlineLedger(balance=10_000, log_path=path)
            ledger.submit(self._decision_with_issue("20260905002"))
            ledger.settle("20260905002", 15)
            history = read_combo_history(path)
            metrics = combo_metrics(path)
            self.assertEqual(len(history), 1)
            self.assertEqual(metrics["executed_rounds"], 1)
            self.assertEqual(metrics["executed_net_profit"], 38.7)

    def test_online_ledger_settles_once_and_restores_balance(self):
        from tempfile import TemporaryDirectory

        from ai_agent.online_combo import ComboOnlineLedger, OnlineLedgerError

        with TemporaryDirectory() as directory:
            path = f"{directory}/combo.jsonl"
            ledger = ComboOnlineLedger(balance=10_000, log_path=path)
            decision = self._decision_with_issue("20260905001")
            ledger.submit(decision)
            self.assertIsNotNone(ledger.pending["20260905001"].decision_id)
            result = ledger.settle("20260905001", 15)
            self.assertEqual(result.net_profit, 38.7)
            self.assertEqual(ledger.balance, 10038.7)
            with self.assertRaises(OnlineLedgerError):
                ledger.settle("20260905001", 15)

            restored = ComboOnlineLedger(balance=0, log_path=path)
            restored.restore()
            self.assertEqual(restored.balance, 10038.7)
            self.assertIn("20260905001", restored.settled_issues)
            self.assertEqual(len(restored.decision_ids), 1)

    def test_online_ledger_restores_unsettled_pending_decision(self):
        from tempfile import TemporaryDirectory

        from ai_agent.online_combo import ComboOnlineLedger

        with TemporaryDirectory() as directory:
            path = f"{directory}/combo.jsonl"
            ledger = ComboOnlineLedger(balance=10_000, log_path=path)
            ledger.submit(self._decision_with_issue("20260905003"))
            restored = ComboOnlineLedger(balance=10_000, log_path=path)
            restored.restore()
            self.assertIn("20260905003", restored.pending)
            self.assertEqual(restored.balance, 10_000)

    def test_live_synchronizer_settles_pending_then_creates_next_simulation(self):
        from ai_agent.live_combo import ComboLiveSynchronizer
        from ai_agent.online_combo import ComboOnlineLedger

        ledger = ComboOnlineLedger(balance=10_000)
        ledger.submit(self._decision_with_issue("102"))
        source = [
            {"issue": "101", "digits": [1, 2, 3], "sum": 6},
            {"issue": "102", "digits": [6, 5, 4], "sum": 15},
        ]

        def predictor(_source, **_kwargs):
            decision = self._decision_with_issue("103").to_dict()
            return {"issue": "103", "decision": decision}

        result = ComboLiveSynchronizer(ledger, "unused", predictor=predictor).process_rows(source, prediction_source=source)
        self.assertEqual(result["settled_issues"], ["102"])
        self.assertIn("103", ledger.pending)
        self.assertNotIn("102", ledger.pending)
        self.assertEqual(ledger.balance, 10038.7)

    def test_live_synchronizer_refuses_new_prediction_when_source_has_gap(self):
        from ai_agent.live_combo import ComboLiveSynchronizer
        from ai_agent.online_combo import ComboOnlineLedger

        result = ComboLiveSynchronizer(ComboOnlineLedger(), "unused", predictor=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError())).process_rows(
            [
                {"issue": "101", "digits": [1, 2, 3], "sum": 6},
                {"issue": "103", "digits": [6, 5, 4], "sum": 15},
            ],
            prediction_source=[],
        )
        self.assertTrue(result["diagnostics"]["has_gap"])
        self.assertIsNone(result["prediction"])

    def test_challenger_is_scheduled_only_at_settlement_milestones(self):
        from ai_agent.challenger_combo import ChallengerConfig, challenger_due

        config = ChallengerConfig(cadence_settlements=100)
        self.assertFalse(challenger_due(0, config=config))
        self.assertFalse(challenger_due(99, config=config))
        self.assertTrue(challenger_due(100, config=config))
        self.assertFalse(challenger_due(101, config=config))

    def test_chronological_8_1_1_split_validates_without_reading_sealed_test(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        from ai_agent.config import DATA_DIR
        from ai_agent.training_protocol_combo import ComboSplitConfig, create_combo_splits, validate_combo_split_manifest

        rows = [
            {"issue": str(index), "time": f"2026-01-01 00:00:{index:02d}", "digits": [1, 2, 3], "sum": 6}
            for index in range(1, 11)
        ]
        with TemporaryDirectory(dir=DATA_DIR) as directory:
            result = create_combo_splits(rows, output_dir=directory, config=ComboSplitConfig(total_rows=10, train_rows=8, validation_rows=1, test_rows=1))
            manifest = validate_combo_split_manifest(result["manifest_path"], include_test=False)
            self.assertEqual(manifest["splits"]["train"]["last_issue"], "8")
            self.assertEqual(manifest["splits"]["validation"]["first_issue"], "9")
            Path(result["paths"]["test"]).unlink()
            validate_combo_split_manifest(result["manifest_path"], include_test=False)
            with self.assertRaises(FileNotFoundError):
                validate_combo_split_manifest(result["manifest_path"], include_test=True)

    def test_train_is_split_into_chronological_five_hundred_row_tuning_chunks(self):
        from tempfile import TemporaryDirectory

        from ai_agent.config import DATA_DIR
        from ai_agent.training_protocol_combo import (
            ComboSplitConfig,
            TrainChunkConfig,
            create_combo_splits,
            create_train_tuning_chunks,
            validate_train_tuning_chunks,
        )

        rows = [
            {"issue": str(index), "time": f"2026-01-01 00:{index // 60:02d}:{index % 60:02d}", "digits": [1, 2, 3], "sum": 6}
            for index in range(1, 1_001)
        ]
        with TemporaryDirectory(dir=DATA_DIR) as directory:
            split = create_combo_splits(
                rows,
                output_dir=directory,
                config=ComboSplitConfig(total_rows=1_000, train_rows=800, validation_rows=100, test_rows=100),
            )
            chunks = create_train_tuning_chunks(
                split["manifest_path"],
                output_dir=f"{directory}/chunks",
                config=TrainChunkConfig(train_rows=800, chunk_rows=100, min_training_chunks=2),
            )
            verified = validate_train_tuning_chunks(chunks["manifest_path"])

        self.assertEqual(len(verified["chunks"]), 8)
        self.assertEqual(verified["chunks"][0]["first_issue"], "1")
        self.assertEqual(verified["chunks"][-1]["last_issue"], "800")
        self.assertEqual(verified["folds"][0]["training_rows"], 200)
        self.assertEqual(verified["folds"][0]["evaluation_chunk_index"], 3)
        self.assertFalse(verified["validation_accessed"])
        self.assertFalse(verified["test_accessed"])

    def test_auto_tune_runs_train_validation_only_and_retains_best_candidate(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import json

        from ai_agent.auto_tune_combo import AutoTuneConfig, ExperimentDirection, run_combo_auto_tuning
        from ai_agent.config import DATA_DIR
        from ai_agent.training_protocol_combo import ComboSplitConfig, create_combo_splits

        rows = [
            {"issue": str(index), "time": f"2026-01-01 00:00:{index:02d}", "digits": [1, 2, 3], "sum": 6}
            for index in range(1, 11)
        ]
        directions = {
            "ppo": (
                ExperimentDirection("first_direction", ({"learning_rate": 0.1}, {"learning_rate": 0.2}, {"learning_rate": 0.3})),
                ExperimentDirection("next_direction", ({"learning_rate": 0.4},)),
            )
        }
        config = AutoTuneConfig(
            total_timesteps=10,
            checkpoint_interval_steps=5,
            no_improvement_patience=2,
            algorithms=("ppo",),
            directions=directions,
        )
        outcomes = iter((("PASS", 10), ("PASS", 9), ("FAIL", 100), ("PASS", 20)))
        calls: list[dict] = []

        def trainer(history, **kwargs):
            self.assertEqual(len(history), 8)
            self.assertNotIn("test", kwargs)
            path = Path(kwargs["model_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake checkpoint")
            calls.append(kwargs)
            return path

        def evaluator(model_path, **kwargs):
            status, profit = next(outcomes)
            self.assertEqual(kwargs["algorithm"], "ppo")
            return {
                "phase": "validation",
                "model_path": str(model_path),
                "algorithm": "ppo",
                "evaluation": {
                    "verdict": {"status": status},
                    "walk_forward": [{"agent": {"net_profit": profit, "max_drawdown": 0.1, "max_exposure_ratio": 0.01, "extension_positive_profit_share": 0.2}, "baselines": {"fixed": {"net_profit": 0}}}],
                },
            }

        with TemporaryDirectory(dir=DATA_DIR) as split_dir, TemporaryDirectory() as output_dir:
            split = create_combo_splits(rows, output_dir=split_dir, config=ComboSplitConfig(total_rows=10, train_rows=8, validation_rows=1, test_rows=1))
            Path(split["paths"]["test"]).unlink()
            result = run_combo_auto_tuning(
                manifest_path=split["manifest_path"],
                config=config,
                model_dir=Path(output_dir) / "models",
                log_dir=Path(output_dir) / "logs",
                run_id="unit-test",
                trainers={"ppo": trainer},
                validation_evaluator=evaluator,
            )
            state = json.loads(Path(result["candidate_champion_path"]).read_text(encoding="utf-8"))

        self.assertEqual(len(calls), 4)
        self.assertEqual([trial["hyperparameters"]["learning_rate"] for trial in result["trials"]], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(result["trials"][2]["direction_stop_reason"], "no_improvement_patience_reached")
        self.assertEqual(result["candidate_champion"]["selected"]["model_path"], result["trials"][3]["model_path"])
        self.assertFalse(result["test_accessed"])
        self.assertFalse(state["test_accessed"])
        self.assertTrue(all(trial["checkpoint_paths"] for trial in result["trials"]))

    def test_combo_train_hyperparameters_reject_unknown_keys(self):
        from ai_agent.train_combo import _dqn_hyperparameters, _ppo_hyperparameters

        self.assertEqual(_ppo_hyperparameters(100, {"learning_rate": 0.001})["learning_rate"], 0.001)
        self.assertEqual(_dqn_hyperparameters(100, 10, {"batch_size": 16})["batch_size"], 16)
        with self.assertRaises(ValueError):
            _ppo_hyperparameters(100, {"not_a_ppo_parameter": 1})
        with self.assertRaises(ValueError):
            _dqn_hyperparameters(100, 10, {"not_a_dqn_parameter": 1})

    def test_validation_champion_selection_is_automatic_and_excludes_failed_candidates(self):
        from ai_agent.champion_combo import select_validation_champion

        def report(path, relative_profit, status="PASS"):
            return {
                "phase": "validation",
                "model_path": path,
                "algorithm": "ppo",
                "evaluation": {
                    "verdict": {"status": status},
                    "walk_forward": [
                        {
                            "agent": {
                                "net_profit": relative_profit,
                                "max_drawdown": 0.1,
                                "max_exposure_ratio": 0.01,
                                "extension_positive_profit_share": 0.2,
                            },
                            "baselines": {"fixed": {"net_profit": 0}},
                        }
                    ],
                },
            }

        result = select_validation_champion([report("weak.zip", 2), report("strong.zip", 3), report("failed.zip", 10, "FAIL")])
        self.assertEqual(result["status"], "SELECTED_PENDING_TEST")
        self.assertEqual(result["selected"]["model_path"], "strong.zip")
        self.assertEqual(result["rejected"][0]["model_path"], "failed.zip")

    def test_combo_evaluation_gate_requires_stability_after_zero_policy_confirmation(self):
        from ai_agent.evaluation_combo import ComboEvaluationThresholds, classify_combo_evaluation

        agent = {
            "net_profit": 100,
            "max_drawdown": 0.1,
            "max_exposure_ratio": 0.01,
            "executed_rounds": 30,
            "extension_positive_profit_share": 0.5,
            "unconfirmed_zero_outcomes": 0,
        }
        baseline = {"net_profit": 10}
        report = {"walk_forward": [{"window": index, "agent": agent, "baselines": {"fixed": baseline}} for index in range(1, 6)]}
        verdict = classify_combo_evaluation(report)
        self.assertEqual(verdict["status"], "PASS")

        report["walk_forward"][0]["agent"] = {**agent, "unconfirmed_zero_outcomes": 1}
        confirmed_zero = classify_combo_evaluation(report)
        self.assertEqual(confirmed_zero["status"], "PASS")

        strict = ComboEvaluationThresholds(max_extension_positive_profit_share=0.4)
        report["walk_forward"][0]["agent"] = agent
        high_extension = classify_combo_evaluation(report, thresholds=strict)
        self.assertEqual(high_extension["status"], "FAIL")

    def _decision_with_issue(self, issue):
        decision = ComboSettlementTests._user_example_decision()
        from ai_agent.schemas_combo import ComboDecision

        return ComboDecision(
            selections=decision.selections,
            mode=decision.mode,
            issue=issue,
        )
