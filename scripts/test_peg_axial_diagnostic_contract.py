"""CPU-only reset compatibility boundary tests; no simulator imports or calls."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import run_peg_axial_bounded_diagnostic_retry as runner


class ContractBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.binding = {"source_hash": "a" * 64, "files": {"x": "b" * 64}}
        obs = np.zeros(20, dtype=float)
        obs[:7] = [0.52, 0.15, 0.60, 1.0, 0.0, 0.0, 0.0]
        obs[7:13] = [1, 2, 3, 4, 5, 6]
        obs[13:16] = [0.04, -0.15, -0.074]
        obs[16:20] = [1.0, 0.0, 0.0, 0.0]
        self.reference = dict(
            observation_contract_version=runner.V2, obs=obs.tolist(),
            actual_ee_pose=obs[:7].tolist(), target_pose=obs[:7].tolist(),
            raw_sensordata=[0.0] * 12, sim_time=0.02, sensors=[],
            metrics=dict(lateral_error=0.1, axial_error=0.1, orientation_error=0.0))
        self.live = copy.deepcopy(self.reference)
        self.guard = Mock(return_value=set())

    def gate(self, **overrides):
        args = dict(
            reset_seed=runner.RESET_SEED, reset_protocol=runner.RESET_PROTOCOL,
            current_bindings=self.binding, frozen_bindings=self.binding,
            link01_exclusion_exact=True, action_contract_unchanged=True)
        args.update(overrides)
        return runner.validate_post_exclusion_invariants(self.live, self.reference, **args)

    def boundary(self):
        return runner.run_pre_action_boundary(
            self.live, self.reference, self.out,
            reset_seed=runner.RESET_SEED, reset_protocol=runner.RESET_PROTOCOL,
            current_bindings=self.binding, frozen_bindings=self.binding,
            link01_exclusion_exact=True, action_contract_unchanged=True,
            safety_gate=self.guard)

    def test_identical_reset_pass(self):
        self.assertEqual(self.gate()["status"], "PASS")

    def test_force_torque_only_delta_pass(self):
        self.live["obs"][7:13] = [10, 20, 30, 40, 50, 60]
        result = self.gate()
        self.assertEqual(result["force_torque_new"], [10, 20, 30, 40, 50, 60])
        self.assertEqual(result["force_torque_policy"], "OBS_7_13_RECORDED_NOT_EQUALITY_GATED")

    def test_ee_pose_within_tolerance_pass(self):
        self.live["actual_ee_pose"][0] += runner.EE_POSITION_ATOL_M * 0.5
        self.live["obs"][0] = self.live["actual_ee_pose"][0]
        self.live["obs"][13] -= runner.EE_POSITION_ATOL_M * 0.5
        self.assertEqual(self.gate()["status"], "PASS")

    def test_ee_pose_outside_tolerance_fail(self):
        self.live["actual_ee_pose"][0] += runner.EE_POSITION_ATOL_M * 1.1
        self.live["obs"][0] = self.live["actual_ee_pose"][0]
        self.live["obs"][13] -= runner.EE_POSITION_ATOL_M * 1.1
        with self.assertRaisesRegex(ValueError, "EE reset pose"):
            self.gate()

    def test_goal_mismatch_fail(self):
        self.live["obs"][13] += runner.GOAL_POSITION_ATOL_M * 2
        with self.assertRaisesRegex(ValueError, "fixed goal"):
            self.gate()

    def test_contract_version_mismatch_fail(self):
        self.live["observation_contract_version"] = "PEG_OBS_CONTRACT_V1_LEGACY"
        with self.assertRaisesRegex(ValueError, "V2"):
            self.gate()

    def test_shape_mismatch_fail(self):
        self.live["obs"] = self.live["obs"][:-1]
        with self.assertRaisesRegex(ValueError, "shape"):
            self.gate()

    def test_nan_inf_fail(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.live = copy.deepcopy(self.reference)
                self.live["obs"][8] = value
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.gate()

    def test_binding_mismatch_fail(self):
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            self.gate(frozen_bindings={"source_hash": "c" * 64})

    def test_persistence_failure_blocks_guard_and_action(self):
        action = Mock()
        with patch.object(runner, "durable_save", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                self.boundary()
        self.guard.assert_not_called()
        action.assert_not_called()

    def test_persistence_precedes_compatibility_and_guard(self):
        events = []
        real_save = runner.durable_save
        def save(path, value):
            events.append("persist:" + Path(path).name)
            return real_save(path, value)
        def guard():
            events.append("guard")
            return set()
        self.guard.side_effect = guard
        with patch.object(runner, "durable_save", side_effect=save):
            self.boundary()
        self.assertEqual(events, ["persist:live_reset_snapshot.json", "persist:post_exclusion_invariant_gate.json", "guard"])

    def test_all_cpu_tests_dispatch_zero_actions(self):
        action = Mock()
        self.boundary()
        action.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
