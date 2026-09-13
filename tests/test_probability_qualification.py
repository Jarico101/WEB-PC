import unittest


class ProbabilityQualificationTests(unittest.TestCase):
    @staticmethod
    def _history(count=200):
        return [
            {"issue": str(index), "digits": [index % 10, (index + 1) % 10, (index + 2) % 10]}
            for index in range(1, count + 1)
        ]

    def test_qualification_is_a_later_independent_tail(self):
        from ai_agent.probability_combo import fit_and_qualify_combo_probability_model

        model, report = fit_and_qualify_combo_probability_model(self._history())
        slices = report["slices"]
        self.assertTrue(report["independent_qualification"])
        self.assertLess(int(slices["fit"]["last_issue"]), int(slices["calibration"]["first_issue"]))
        self.assertLess(int(slices["calibration"]["last_issue"]), int(slices["qualification"]["first_issue"]))
        self.assertEqual(slices["qualification"]["rows"], 20)
        self.assertEqual(model.training_end_issue, slices["fit"]["last_issue"])
        self.assertEqual(model.calibration_end_issue, slices["calibration"]["last_issue"])

    def test_formal_gate_rejects_forced_rate_and_negative_ev(self):
        from ai_agent.statistical_combo_policy import OpportunityGateConfig

        with self.assertRaises(ValueError):
            OpportunityGateConfig(target_bet_rate=0.35)
        experimental = OpportunityGateConfig(
            target_bet_rate=0.35,
            min_expected_return_ratio=-0.01,
            allow_negative_expected_return=True,
        )
        self.assertTrue(experimental.experimental_non_champion)

    def test_formal_ppo_refuses_failed_qualification_before_training(self):
        from ai_agent.train_combo import train_ppo_combo

        with self.assertRaisesRegex(ValueError, "qualification failed"):
            train_ppo_combo(self._history(120), total_timesteps=2, check=False)

    def test_train_only_research_uses_future_folds_without_test_access(self):
        import json
        import tempfile
        from pathlib import Path

        from ai_agent.probability_research_combo import ProbabilityResearchConfig, run_train_internal_probability_research

        rows = self._history(3_000)
        with tempfile.TemporaryDirectory(dir="data") as directory:
            root = Path(directory)
            import hashlib

            chunks = []
            for index in range(6):
                chunk_path = root / f"train_{index + 1:02d}_500.jsonl"
                chunk_path.write_text(
                    "\n".join(json.dumps(row) for row in rows[index * 500 : (index + 1) * 500]) + "\n",
                    encoding="utf-8",
                )
                chunks.append({"index": index + 1, "path": str(chunk_path), "sha256": hashlib.sha256(chunk_path.read_bytes()).hexdigest(), "rows": 500})
            manifest = {
                "protocol": "pc28-combo-train-chunks-chronological-v1",
                "config": {"train_rows": 3000, "chunk_rows": 500, "min_training_chunks": 2},
                "chunks": chunks,
                "validation_accessed": False,
                "test_accessed": False,
            }
            manifest_path = root / "train_chunks.manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = run_train_internal_probability_research(
                manifest_path,
                config=ProbabilityResearchConfig(first_training_rows=1000, evaluation_rows=500, include_softmax=False),
            )
            self.assertFalse(report["input"]["validation_accessed"])
            self.assertFalse(report["input"]["test_accessed"])
            self.assertEqual(report["folds"][0]["training_last_issue"], "1000")
            self.assertEqual(report["folds"][0]["score_first_issue"], "1001")

    def test_experimental_random_mode_is_isolated_and_uses_only_allowed_sums(self):
        from ai_agent.experimental_random_combo import (
            EXPERIMENTAL_CLASSIFICATION,
            ExperimentalRandomConfig,
            run_experimental_random_backtest,
        )

        history = [
            {"issue": str(index), "digits": [index % 10, (index * 2) % 10, (index * 3) % 10]}
            for index in range(1, 31)
        ]
        report = run_experimental_random_backtest(
            history,
            config=ExperimentalRandomConfig(seed=7, sum_extension_probability=1.0, stop_loss_ratio=0.9, take_profit_ratio=0.9),
        )
        self.assertEqual(report["classification"], EXPERIMENTAL_CLASSIFICATION)
        self.assertFalse(report["champion_eligible"])
        self.assertFalse(report["test_accessed"])
        for decision in report["result"]["decisions"]:
            selections = decision["selections"]
            if not selections:
                continue
            base = [item for item in selections if item["market_type"] == "base"]
            extensions = [item for item in selections if item["market_type"] != "base"]
            self.assertEqual(len(base), 3)
            self.assertLessEqual(len(extensions), 1)
            self.assertTrue(all(item["market"] in {"sum_08", "sum_09", "sum_18", "sum_19"} for item in extensions))
            self.assertTrue(all(item["stake"].is_integer() and 30 <= item["stake"] <= 300 for item in selections))
            self.assertLessEqual(sum(item["stake"] for item in selections), 300)

    def test_experimental_random_default_take_profit_is_two_hundred_percent(self):
        from ai_agent.experimental_random_combo import ExperimentalRandomConfig

        config = ExperimentalRandomConfig()
        self.assertEqual(config.take_profit_ratio, 2.0)

    def test_ppo_defaults_include_exploration_and_diversity_controls(self):
        from ai_agent.reward import ComboRewardConfig
        from ai_agent.train_combo import _ppo_hyperparameters

        self.assertGreater(_ppo_hyperparameters(500, None)["ent_coef"], 0.0)
        reward = ComboRewardConfig(repeat_base_combo_penalty=0.05, state_combo_alignment_bonus=0.25)
        self.assertGreater(reward.repeat_base_combo_penalty, 0.0)
        self.assertGreater(reward.state_combo_alignment_bonus, 0.0)

    def test_state_target_combo_reads_prior_features_not_current_outcome(self):
        from ai_agent.combo_env import ComboCasinoEnv

        history = [
            {"issue": str(index), "digits": [1, 2, 3]}
            for index in range(1, 110)
        ]
        env = ComboCasinoEnv(history)
        row = env.history[-1]
        target = env._state_target_base_combo(row)
        changed = {**row, "sum": 27, "digits": [9, 9, 9]}
        self.assertEqual(target, env._state_target_base_combo(changed))


if __name__ == "__main__":
    unittest.main()
