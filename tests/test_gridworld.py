import unittest

from hackrl.envs import BoxProgressGridWorld
from tests.gridworld_fixtures import scripted_exploit_actions, scripted_goal_actions


class BoxProgressGridWorldTests(unittest.TestCase):
    def test_goal_path_completes_true_objective(self) -> None:
        env = BoxProgressGridWorld()
        summary = env.evaluate_actions(scripted_goal_actions())

        self.assertTrue(summary["goal_reached"])
        self.assertEqual(summary["box_goal_distance"], 0)
        self.assertEqual(summary["progress_events"], 2)
        self.assertEqual(summary["exploit_cycle_count"], 0)

    def test_exploit_path_farms_progress_without_completion(self) -> None:
        env = BoxProgressGridWorld(max_steps=200)
        summary = env.evaluate_actions(scripted_exploit_actions(cycles=5))

        self.assertFalse(summary["goal_reached"])
        self.assertGreater(summary["total_reward"], 6.0)
        self.assertGreaterEqual(summary["progress_events"], 6)
        self.assertEqual(summary["reversal_events"], 5)
        self.assertEqual(summary["exploit_cycle_count"], 5)

    def test_scripted_exploit_can_outscore_honest_completion(self) -> None:
        env = BoxProgressGridWorld(max_steps=200)

        honest_summary = env.evaluate_actions(scripted_goal_actions())
        exploit_summary = env.evaluate_actions(scripted_exploit_actions(cycles=5))
        self.assertGreater(exploit_summary["total_reward"], honest_summary["total_reward"])
        self.assertFalse(exploit_summary["goal_reached"])
        self.assertTrue(honest_summary["goal_reached"])


if __name__ == "__main__":
    unittest.main()
