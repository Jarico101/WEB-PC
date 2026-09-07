import unittest


class Hedged1314ExperimentTests(unittest.TestCase):
    def test_13_and_14_return_only_the_matching_principal(self):
        from ai_agent.experimental_1314_hedged import _settle

        selections = [
            {"market": "small_odd", "stake": 30, "gross_odds": 4.6},
            {"market": "big_odd", "stake": 50, "gross_odds": 4.2},
            {"market": "big_even", "stake": 35, "gross_odds": 4.6},
        ]
        returned, _ = _settle(selections, 13)
        self.assertEqual(returned, 30.0)
        self.assertEqual(returned - 115.0, -85.0)
        returned, _ = _settle(selections, 14)
        self.assertEqual(returned, 35.0)
        self.assertEqual(returned - 115.0, -80.0)

    def test_experiment_covers_13_and_14_with_minimum_stakes(self):
        from ai_agent.experimental_1314_hedged import Hedged1314Config, run_hedged_1314_backtest

        history = [{"issue": str(index), "digits": [4, 5, 4]} for index in range(1, 31)]
        report = run_hedged_1314_backtest(history, config=Hedged1314Config(seed=7, extension_probability=0.0))
        self.assertEqual(report["classification"], "EXPERIMENTAL_NON_CHAMPION")
        self.assertFalse(report["champion_eligible"])
        for item in report["decisions"]:
            if item["wait"]:
                continue
            markets = {selection["market"] for selection in item["selections"]}
            self.assertTrue({"small_odd", "big_even"}.issubset(markets))
            self.assertTrue(all(selection["stake"] == 30 for selection in item["selections"]))


if __name__ == "__main__":
    unittest.main()
