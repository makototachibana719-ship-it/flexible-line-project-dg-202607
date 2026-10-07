#!/usr/bin/env python3
"""Zero-training preflight for the Peg F3 V2 minimal HIL-SERL port."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys
from typing import Any

import numpy as np


EXPECTED_DEMO_SHA = "4a266b512d4a3e9b2e7546102227821c249b24f29b03f7df27e9e2d0e8cc3878"
EXPECTED_D_DEMO = 7.41773975814
EXPECTED = {
    "sac": "fd14dd24683fc897b0066b0caa36d63b96ecf1913cbcc448b33c18b05d8c83d8",
    "replay_buffer": "f4ea8301db6bf1c77821f8f1f580b17112bafec12272cd94e20942b038fbafe7",
    "peg_replay_adapter": "c64e273f89ba2d63e4fd3dcdc994284d6032201350b1fd255eece955e8d3d2b5",
    "agentlace_package": "483944d6bea9753ec73473df31aa53e9ff8bfc5d69425346c115f3cd520e1248",
    "c0": "24515269aaa0415772facb1d08a660a54e17a21899dc748297b8210cdb6711b5",
    "c1": "a1ce26d303d707918cc1419dadd4b1b2ba0f63d2dadf313edc0ffb04345bf860",
    "c2": "55f5660ecd27f11cae3553b7a45a8027c88e782979e8ddb0fd22559c948855ce",
    "monotonic_result": "f26e3549d07b4a2eb2c9cef2742dfe9d8fdd527c3bc876fc2c74c4ac43bb11fc",
    "monotonic_evidence": "72317880eefcdc0a5713330f59af0e9d417f8afedd79f2f8d7bc9eed7a181821",
    "production_demo": EXPECTED_DEMO_SHA,
    "async_entry_before": "7ccbecdb8e610a886edf820a3687d54b5d5c188cf8caaf2f1261ad24459a107b",
}
REPLAY_KEYS = {
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "masks",
    "dones",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix != ".pyc"
    )
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class MemoryStore:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def insert(self, transition: dict[str, Any]) -> None:
        self.items.append(transition)


class DeterministicEnv:
    def __init__(self) -> None:
        self.step_call_count = 0
        self.actions: list[np.ndarray] = []

    def step(self, action: Any):
        action = np.asarray(action, dtype=np.float32)
        self.step_call_count += 1
        self.actions.append(action.copy())
        next_obs = np.concatenate(
            [action, action, action[:6]], axis=0
        ).astype(np.float32)
        reward = np.float32(np.dot(action, np.arange(1, 8, dtype=np.float32)))
        terminated = bool(action[6] > 0.5)
        truncated = bool(action[5] < -0.5)
        return next_obs, reward, terminated, truncated, {"source": "fake_env"}


class FakeProvider:
    def __init__(self, *, intervened: bool, action: Any = None) -> None:
        self.intervened = intervened
        self.action = action
        self.call_count = 0

    def decide(self, *, obs20: np.ndarray, policy_action: np.ndarray):
        self.call_count += 1
        return InterventionDecision(
            self.intervened,
            self.action if self.intervened else None,
            {
                "np_float32": np.float32(1.25),
                "np_float64": np.float64(2.5),
                "np_int": np.int64(3),
                "np_bool": np.bool_(True),
                "ndarray": np.asarray([4, 5], dtype=np.int32),
            },
        )


def find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def call_signatures(function: ast.FunctionDef, attr: str) -> list[str]:
    return [
        ast.dump(node, include_attributes=False)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]


def subprocess_text(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return (completed.stdout + completed.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/home/makoto/flexible_line_project"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--async-before",
        type=Path,
        default=Path(
            "external/packages/peg_insertion/peg_insertion/code/example/"
            "async_rlpd_peg_sim/async_sac_state_sim.py."
            "before_hil_serl_intervention_v1_20260909_224459"
        ),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_dir or (
        root
        / "results/validation/peg"
        / f"f3_v2_hil_serl_minimal_port_preflight_{stamp}"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite result directory: {output}")
    output.mkdir(parents=True)
    sidecar_dir = (
        root / "results/datasets" / f"c3_hil_serl_intervention_preflight_{stamp}"
    )
    if sidecar_dir.exists():
        raise FileExistsError(f"refusing to overwrite sidecar directory: {sidecar_dir}")
    sidecar_dir.mkdir(parents=True)

    sys.path.insert(0, str(root))
    global InterventionDecision
    from training.peg_insertion.hil_serl.intervention_adapter_v1 import (
        HILSERLInterventionAdapterV1,
        InterventionDecision,
        assert_legal_aubo_action7,
        execute_intervention_control_step,
        transition_identity,
    )

    adapter_path = root / "training/peg_insertion/hil_serl/intervention_adapter_v1.py"
    async_path = root / (
        "external/packages/peg_insertion/peg_insertion/code/example/"
        "async_rlpd_peg_sim/async_sac_state_sim.py"
    )
    async_before = args.async_before
    if not async_before.is_absolute():
        async_before = root / async_before
    paths = {
        "sac": root / "external/serl/serl_launcher/serl_launcher/agents/continuous/sac.py",
        "replay_buffer": root / "external/serl/serl_launcher/serl_launcher/data/replay_buffer.py",
        "peg_replay_adapter": root / "training/peg_insertion/peg_replay_adapter.py",
        "c0": root / "results/final_packages/peg_f3_v2_monotonic_obs20_option_20260909_001/components/c0_c1_source_checkpoint.msgpack",
        "c1": root / "results/final_packages/peg_f3_v2_monotonic_obs20_option_20260909_001/components/c1_recovery_option.msgpack",
        "c2": root / "results/final_packages/peg_f3_v2_monotonic_obs20_option_20260909_001/components/c2_checkpoint.msgpack",
        "monotonic_result": root / "results/diagnostics/peg/f3_v2_c3_monotonic_obs20_option_matched_20260909_001/result.json",
        "monotonic_evidence": root / "results/diagnostics/peg/f3_v2_c3_monotonic_obs20_option_matched_20260909_001/evidence.json",
        "production_demo": root / "results/datasets/peg_v2_accepted_11_20260906/peg_v2_accepted_11.pkl",
    }
    agentlace_spec = importlib.util.find_spec("agentlace")
    if agentlace_spec is None or agentlace_spec.submodule_search_locations is None:
        raise RuntimeError("Agentlace package not found")
    agentlace_path = Path(next(iter(agentlace_spec.submodule_search_locations)))

    source_before = {
        **EXPECTED,
        "adapter": None,
        "adapter_init": None,
        "preflight": None,
    }
    source_after = {key: sha256_file(path) for key, path in paths.items()}
    source_after["agentlace_package"] = sha256_tree(agentlace_path)
    source_after["async_entry_before"] = sha256_file(async_before)
    source_after["async_entry_after"] = sha256_file(async_path)
    source_after["adapter"] = sha256_file(adapter_path)
    source_after["adapter_init"] = sha256_file(adapter_path.parent / "__init__.py")
    source_after["preflight"] = sha256_file(Path(__file__).resolve())
    json_dump(output / "source_hashes_before.json", source_before)

    log: list[str] = []
    passed: dict[str, bool] = {}

    def check(name: str, operation) -> None:
        try:
            operation()
        except Exception as exc:  # Preserve all failure evidence.
            passed[name] = False
            log.append(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            passed[name] = True
            log.append(f"PASS {name}")

    obs = np.linspace(-0.5, 0.5, 20, dtype=np.float32)
    policy = np.asarray([0.1, -0.2, 0.3, -0.4, 0.2, 0.0, -0.1], dtype=np.float32)
    expert = np.asarray([-0.7, 0.6, -0.5, 0.4, -0.3, -0.8, 0.9], dtype=np.float32)
    state: dict[str, Any] = {}

    def test_a() -> None:
        provider = FakeProvider(intervened=False)
        adapter = HILSERLInterventionAdapterV1(
            mode="TRAINING", intervention_enabled=True, provider=provider,
            provenance_path=sidecar_dir / "no_intervention.jsonl",
        )
        env, online, intvn = DeterministicEnv(), MemoryStore(), MemoryStore()
        result = execute_intervention_control_step(
            env=env, online_store=online, intervention_store=intvn,
            adapter=adapter, obs20=obs, policy_action=policy,
            c3_has_authority=True, episode=0, step=0,
        )
        assert np.array_equal(result.executed_action, policy)
        assert len(online.items) == 1 and len(intvn.items) == 0
        assert np.array_equal(online.items[0]["actions"], policy)
        assert adapter.intervention_count == 0
        assert not (sidecar_dir / "no_intervention.jsonl").exists()
        assert env.step_call_count == 1
        state["a"] = result

    check("A_NO_INTERVENTION", test_a)

    def test_b_c_d() -> None:
        provider = FakeProvider(intervened=True, action=expert)
        adapter = HILSERLInterventionAdapterV1(
            mode="TRAINING", intervention_enabled=True, provider=provider,
            provenance_path=sidecar_dir / "provenance.jsonl",
            checkpoint_identity={"sha256": np.str_("synthetic-checkpoint")},
            expert_identity={"version": np.int32(1)},
        )
        env, online, intvn = DeterministicEnv(), MemoryStore(), MemoryStore()
        result = execute_intervention_control_step(
            env=env, online_store=online, intervention_store=intvn,
            adapter=adapter, obs20=obs, policy_action=policy,
            c3_has_authority=True, episode=2, step=7,
        )
        assert np.array_equal(result.executed_action, expert)
        assert np.array_equal(result.transition["actions"], expert)
        assert not np.array_equal(policy, result.transition["actions"])
        assert len(online.items) == 1 and len(intvn.items) == 1
        assert online.items[0] is intvn.items[0]
        assert env.step_call_count == 1
        expected_next = np.concatenate([expert, expert, expert[:6]])
        expected_reward = np.float32(np.dot(expert, np.arange(1, 8, dtype=np.float32)))
        assert np.array_equal(result.next_obs, expected_next)
        assert result.reward == expected_reward
        assert result.transition["masks"] == np.float32(0.0)
        assert result.transition["dones"] is True
        assert np.array_equal(result.info["intervene_action"], expert)
        state["b"] = result

    check("B_INTERVENTION_AND_DUAL_ROUTING", test_b_c_d)
    passed["C_CAUSALITY"] = passed["B_INTERVENTION_AND_DUAL_ROUTING"]
    passed["D_SINGLE_STEP"] = (
        passed["A_NO_INTERVENTION"] and passed["B_INTERVENTION_AND_DUAL_ROUTING"]
    )
    log.append(("PASS" if passed["C_CAUSALITY"] else "FAIL") + " C_CAUSALITY")
    log.append(("PASS" if passed["D_SINGLE_STEP"] else "FAIL") + " D_SINGLE_STEP")

    def test_e() -> None:
        invalid = [
            np.zeros(6), np.zeros(8),
            np.asarray([np.nan, 0, 0, 0, 0, 0, 0]),
            np.asarray([np.inf, 0, 0, 0, 0, 0, 0]),
            np.asarray([-np.inf, 0, 0, 0, 0, 0, 0]),
            np.asarray([1.01, 0, 0, 0, 0, 0, 0]),
            np.asarray([-1.01, 0, 0, 0, 0, 0, 0]),
            np.asarray([object(), 0, 0, 0, 0, 0, 0], dtype=object),
            np.asarray(["bad", "0", "0", "0", "0", "0", "0"]),
        ]
        for bad in invalid:
            provider = FakeProvider(intervened=True, action=bad)
            adapter = HILSERLInterventionAdapterV1(
                mode="TRAINING", intervention_enabled=True, provider=provider
            )
            env, online, intvn = DeterministicEnv(), MemoryStore(), MemoryStore()
            try:
                execute_intervention_control_step(
                    env=env, online_store=online, intervention_store=intvn,
                    adapter=adapter, obs20=obs, policy_action=policy,
                    c3_has_authority=True, episode=0, step=0,
                )
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f"invalid action accepted: {bad!r}")
            assert env.step_call_count == 0
            assert len(online.items) == 0 and len(intvn.items) == 0
        edge = assert_legal_aubo_action7(
            np.asarray([1.0 + 2e-7, -1.0 - 2e-7, 0, 0, 0, 0, 0])
        )
        assert np.max(edge) <= 1.0 and np.min(edge) >= -1.0

    check("E_ACTION7_FAIL_CLOSED", test_e)

    def test_f() -> None:
        provider = FakeProvider(intervened=True, action=expert)
        adapter = HILSERLInterventionAdapterV1(
            mode="TRAINING", intervention_enabled=True, provider=provider
        )
        env, online, intvn = DeterministicEnv(), MemoryStore(), MemoryStore()
        result = execute_intervention_control_step(
            env=env, online_store=online, intervention_store=intvn,
            adapter=adapter, obs20=obs, policy_action=policy,
            c3_has_authority=False, episode=0, step=0,
        )
        assert provider.call_count == 0 and not result.decision.intervened
        assert np.array_equal(result.executed_action, policy)
        assert not adapter.enabled

    check("F_C3_AUTHORITY", test_f)

    def test_g() -> None:
        provider = FakeProvider(intervened=True, action=expert)
        adapter = HILSERLInterventionAdapterV1(
            mode="TRAINING", intervention_enabled=True, provider=provider
        )
        adapter.disable()
        env, online, intvn = DeterministicEnv(), MemoryStore(), MemoryStore()
        for step in range(3):
            execute_intervention_control_step(
                env=env, online_store=online, intervention_store=intvn,
                adapter=adapter, obs20=obs, policy_action=policy,
                c3_has_authority=True, episode=0, step=step,
            )
        assert provider.call_count == 0 and adapter.intervention_count == 0
        assert len(online.items) == 3 and len(intvn.items) == 0

    check("G_DISABLE_PERMANENT", test_g)

    def test_h() -> None:
        provider = FakeProvider(intervened=True, action=expert)
        try:
            HILSERLInterventionAdapterV1(
                mode="FORMAL_EVAL", intervention_enabled=True, provider=provider
            )
        except ValueError:
            pass
        else:
            raise AssertionError("formal provider construction did not fail closed")
        formal = HILSERLInterventionAdapterV1(
            mode="FORMAL_EVAL", intervention_enabled=False, provider=None
        )
        decision = formal.decide(
            obs20=obs, policy_action=policy, c3_has_authority=True
        )
        assert not decision.intervened and provider.call_count == 0

    check("H_FORMAL_MODE_HARD_OFF", test_h)

    def test_i() -> None:
        for key in ("a", "b"):
            assert set(state[key].transition) == REPLAY_KEYS
        assert "intervention_flag" not in state["b"].transition
        assert "policy_action" not in state["b"].transition

    check("I_REPLAY_KEYS", test_i)

    def test_j() -> None:
        records = [
            json.loads(line)
            for line in (sidecar_dir / "provenance.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert len(records) == 1
        record = records[0]
        assert record["policy_action"] == policy.tolist()
        assert record["executed_action"] == expert.tolist()
        assert record["intervention_action"] == expert.tolist()
        assert record["transition_identity"] == transition_identity(
            state["b"].transition
        )
        assert record["provider_provenance"]["np_bool"] is True
        assert record["provider_provenance"]["np_int"] == 3
        assert record["provider_provenance"]["ndarray"] == [4, 5]
        before = pickle.dumps(state["b"].transition, protocol=5)
        disabled_writer = HILSERLInterventionAdapterV1(
            mode="TRAINING", intervention_enabled=False, provider=None
        )
        assert disabled_writer.append_provenance_sidecar(
            episode=0, step=0, policy_action=policy,
            executed_action=expert, intervention_action=expert,
            observations=obs, next_observations=state["b"].next_obs,
            reward=state["b"].reward, terminated=True, truncated=False,
            transition=state["b"].transition,
        ) is None
        assert pickle.dumps(state["b"].transition, protocol=5) == before
        line_count = len(records)
        bad_writer = HILSERLInterventionAdapterV1(
            mode="TRAINING", intervention_enabled=False, provider=None,
            provenance_path=sidecar_dir / "unknown_object.jsonl",
        )
        try:
            bad_writer.append_provenance_sidecar(
                episode=0, step=0, policy_action=policy,
                executed_action=expert, intervention_action=expert,
                observations=obs, next_observations=state["b"].next_obs,
                reward=state["b"].reward, terminated=True, truncated=False,
                transition=state["b"].transition,
                provider_provenance={"bad": object()},
            )
        except TypeError:
            pass
        else:
            raise AssertionError("unknown provenance object was stringified")
        assert line_count == 1

    check("J_PROVENANCE_SIDECAR", test_j)

    def test_k() -> None:
        old_source = async_before.read_text(encoding="utf-8")
        new_source = async_path.read_text(encoding="utf-8")
        old_tree, new_tree = ast.parse(old_source), ast.parse(new_source)
        old_actor, new_actor = find_function(old_tree, "actor"), find_function(new_tree, "actor")
        old_learner, new_learner = find_function(old_tree, "learner"), find_function(new_tree, "learner")
        old_main, new_main = find_function(old_tree, "main"), find_function(new_tree, "main")
        assert call_signatures(old_actor, "sample_actions") == call_signatures(new_actor, "sample_actions")
        assert len(call_signatures(new_actor, "step")) == 1
        assert call_signatures(old_learner, "update_high_utd") == call_signatures(new_learner, "update_high_utd")
        assert call_signatures(old_main, "save_checkpoint") == call_signatures(new_main, "save_checkpoint")
        assert '"actor_env"' in new_source and '"actor_env_intvn"' in new_source
        assert "dones=terminated or truncated" in new_source
        assert "masks=np.float32(" in new_source
        flag = [
            node for node in ast.walk(new_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "DEFINE_boolean"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "hil_serl_intervention"
        ]
        assert len(flag) == 1
        assert isinstance(flag[0].args[1], ast.Constant) and flag[0].args[1].value is False
        assert "else:\n            actor(agent, data_store, env, sampling_rng)" in new_source
        assert "intervention_demo_buffer=demo_buffer" in new_source
        assert "half_batch_size = (\n            mixed_batch_size // 2" in new_source
        assert "agent.update_high_utd(batch, utd_ratio=FLAGS.utd_ratio)" in new_source

    check("K_DEFAULT_OFF_REGRESSION", test_k)

    check("L_SAC_UNCHANGED", lambda: (
        None if source_after["sac"] == EXPECTED["sac"]
        else (_ for _ in ()).throw(AssertionError("SAC hash changed"))
    ))

    def test_m() -> None:
        assert source_after["replay_buffer"] == EXPECTED["replay_buffer"]
        assert source_after["peg_replay_adapter"] == EXPECTED["peg_replay_adapter"]
        assert source_after["agentlace_package"] == EXPECTED["agentlace_package"]

    check("M_REPLAY_AND_AGENTLACE_UNCHANGED", test_m)

    def test_n() -> None:
        assert source_after["c0"] == EXPECTED["c0"]
        assert source_after["c1"] == EXPECTED["c1"]
        assert source_after["c2"] == EXPECTED["c2"]

    check("N_LOWER_SKILLS_PROTECTED", test_n)

    def test_o() -> None:
        assert source_after["monotonic_result"] == EXPECTED["monotonic_result"]
        assert source_after["monotonic_evidence"] == EXPECTED["monotonic_evidence"]
        baseline = json.loads(paths["monotonic_result"].read_text(encoding="utf-8"))
        assert baseline["CANONICAL_SUCCESS"] == "3/3"
        assert baseline["RECOVERY_CORE_ENTRY"] == "3/3"
        assert baseline["CONTACT_HEALTH"] == "PASS"
        assert baseline["TRAINING_RUNNING"] is False

    check("O_MONOTONIC_POSITIVE_CONTROL", test_o)

    def test_p() -> None:
        assert source_after["production_demo"] == EXPECTED_DEMO_SHA
        with paths["production_demo"].open("rb") as stream:
            assert len(pickle.load(stream)) == 1921

    check("P_PRODUCTION_DEMO", test_p)

    def test_q() -> None:
        package_manifest = json.loads(
            (root / "results/final_packages/peg_f3_v2_monotonic_obs20_option_20260909_001/MANIFEST.json").read_text(encoding="utf-8")
        )
        baseline = json.loads(paths["monotonic_result"].read_text(encoding="utf-8"))
        assert package_manifest["online_admission_threshold"] == EXPECTED_D_DEMO
        assert baseline["ONLINE_ADMISSION_THRESHOLD"] == EXPECTED_D_DEMO
        assert baseline["ONLINE_ADMISSION_THRESHOLD_CHANGED"] == "NO"

    check("Q_D_DEMO_UNCHANGED", test_q)

    residual_processes: list[str] = []
    process_evidence: dict[str, str] = {}

    def test_r() -> None:
        nonlocal residual_processes
        pgrep = subprocess_text(["pgrep", "-af", "async_sac_state_sim.py"])
        tmux = subprocess_text(
            ["bash", "-lc", "tmux list-panes -a -F '#{pane_pid} #{pane_current_command}' 2>&1 || true"]
        )
        gpu = subprocess_text(
            ["bash", "-lc", "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1 || true"]
        )
        process_evidence.update({"pgrep": pgrep, "tmux": tmux, "gpu": gpu})
        residual_processes = [
            line for line in pgrep.splitlines()
            if re.search(r"--(?:actor|learner)(?:=|\s|$)", line)
        ]
        assert not residual_processes

    check("R_NO_TRAINING_RESIDUAL", test_r)

    with (sidecar_dir / "transitions.pkl").open("wb") as stream:
        pickle.dump([state["b"].transition] if "b" in state else [], stream, protocol=5)
    sidecar_manifest = {
        "dataset_kind": "SYNTHETIC_PREFLIGHT_ONLY",
        "real_expert_intervention_run": False,
        "training_started": False,
        "transition_count": 1 if "b" in state else 0,
        "provenance_count": (
            len((sidecar_dir / "provenance.jsonl").read_text(encoding="utf-8").splitlines())
            if (sidecar_dir / "provenance.jsonl").exists() else 0
        ),
        "replay_keys": sorted(REPLAY_KEYS),
        "action_contract": "PEG_F3_V2_AUBO_NORMALIZED_ACTION7_V1",
    }
    json_dump(sidecar_dir / "manifest.json", sidecar_manifest)

    after_demo_sha = sha256_file(paths["production_demo"])
    source_after["production_demo_after_tests"] = after_demo_sha
    json_dump(output / "source_hashes_after.json", source_after)
    protected_manifest = {
        key: {
            "path": str(paths[key]) if key in paths else str(agentlace_path),
            "expected_sha256": EXPECTED[key],
            "before_sha256": EXPECTED[key],
            "after_sha256": source_after[key],
            "unchanged": source_after[key] == EXPECTED[key],
        }
        for key in (
            "sac", "replay_buffer", "peg_replay_adapter", "agentlace_package",
            "c0", "c1", "c2", "monotonic_result", "monotonic_evidence",
            "production_demo",
        )
    }
    protected_manifest["async_entry"] = {
        "path": str(async_path),
        "backup_path": str(async_before),
        "before_sha256": EXPECTED["async_entry_before"],
        "backup_sha256": source_after["async_entry_before"],
        "after_sha256": source_after["async_entry_after"],
        "authorized_guarded_change": True,
    }
    json_dump(output / "protected_manifest.json", protected_manifest)
    (output / "tests.log").write_text(
        "\n".join(log)
        + "\n\nPROCESS_EVIDENCE\n"
        + json.dumps(process_evidence, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    all_pass = all(passed.values()) and after_demo_sha == EXPECTED_DEMO_SHA
    status = "PASS" if all_pass else "FAIL"
    result = {
        "HIL_SERL_MINIMAL_PORT_IMPLEMENTATION": status,
        "HIL_SERL_MINIMAL_PORT_PREFLIGHT": status,
        "ADAPTER_IMPORT": "PASS" if adapter_path.exists() else "FAIL",
        "ADAPTER_PATH": str(adapter_path),
        "ASYNC_ENTRY_PATH": str(async_path),
        "ASYNC_ENTRY_BACKUP_PATH": str(async_before),
        "FEATURE_FLAG_DEFAULT_OFF": "YES" if passed.get("K_DEFAULT_OFF_REGRESSION") else "NO",
        "ACTION_FRAME": "AUBO_ACTION7_UNCHANGED",
        "ACTION7_VALIDATION": "PASS" if passed.get("E_ACTION7_FAIL_CLOSED") else "FAIL",
        "NO_INTERVENTION_PATH": "PASS" if passed.get("A_NO_INTERVENTION") else "FAIL",
        "INTERVENTION_PATH": "PASS" if passed.get("B_INTERVENTION_AND_DUAL_ROUTING") else "FAIL",
        "EXECUTED_ACTION_STORAGE": "PASS" if passed.get("B_INTERVENTION_AND_DUAL_ROUTING") else "FAIL",
        "SINGLE_ENV_STEP": "PASS" if passed.get("D_SINGLE_STEP") else "FAIL",
        "ONLINE_ROUTING": "PASS" if passed.get("B_INTERVENTION_AND_DUAL_ROUTING") else "FAIL",
        "INTERVENTION_DEMO_ROUTING": "PASS" if passed.get("B_INTERVENTION_AND_DUAL_ROUTING") else "FAIL",
        "DUAL_ROUTING": "PASS" if passed.get("B_INTERVENTION_AND_DUAL_ROUTING") else "FAIL",
        "INTERVENTION_TRANSITION_CAUSALITY": "PASS" if passed.get("C_CAUSALITY") else "FAIL",
        "REPLAY_SCHEMA_UNCHANGED": "YES" if passed.get("I_REPLAY_KEYS") else "NO",
        "PROVENANCE_SIDECAR": "PASS" if passed.get("J_PROVENANCE_SIDECAR") else "FAIL",
        "C3_ONLY_INTERVENTION": "PASS" if passed.get("F_C3_AUTHORITY") and passed.get("G_DISABLE_PERMANENT") else "FAIL",
        "FORMAL_INTERVENTION_DISABLE": "PASS" if passed.get("H_FORMAL_MODE_HARD_OFF") else "FAIL",
        "FORMAL_INTERVENTION_DISABLE_COMPATIBLE": "YES" if passed.get("H_FORMAL_MODE_HARD_OFF") else "NO",
        "DEFAULT_OFF_RUNTIME_REGRESSION": "PASS" if passed.get("K_DEFAULT_OFF_REGRESSION") else "FAIL",
        "SAC_UNCHANGED": "YES" if passed.get("L_SAC_UNCHANGED") else "NO",
        "REPLAY_IMPLEMENTATION_UNCHANGED": "YES" if passed.get("M_REPLAY_AND_AGENTLACE_UNCHANGED") else "NO",
        "AGENTLACE_PACKAGE_UNCHANGED": "YES" if passed.get("M_REPLAY_AND_AGENTLACE_UNCHANGED") else "NO",
        "C0_PROTECTED": "PASS" if passed.get("N_LOWER_SKILLS_PROTECTED") else "FAIL",
        "C1_PROTECTED": "PASS" if passed.get("N_LOWER_SKILLS_PROTECTED") else "FAIL",
        "C2_PROTECTED": "PASS" if passed.get("N_LOWER_SKILLS_PROTECTED") else "FAIL",
        "MONOTONIC_C3_BASELINE_PRESERVED": "YES" if passed.get("O_MONOTONIC_POSITIVE_CONTROL") else "NO",
        "PRODUCTION_DEMO_SHA": after_demo_sha,
        "PRODUCTION_DEMO_SHA_UNCHANGED": "YES" if passed.get("P_PRODUCTION_DEMO") and after_demo_sha == EXPECTED_DEMO_SHA else "NO",
        "ONLINE_ADMISSION_THRESHOLD": EXPECTED_D_DEMO,
        "ONLINE_ADMISSION_THRESHOLD_CHANGED": "NO" if passed.get("Q_D_DEMO_UNCHANGED") else "YES",
        "LEARNER_UPDATE_COUNT": 0,
        "OPTIMIZER_UPDATE_COUNT": 0,
        "TRAINING_STARTED": False,
        "REAL_EXPERT_INTERVENTION_RUN": "NO",
        "BOUNDED_PILOT_STARTED": False,
        "RESIDUAL_PROCESSES": "NONE" if not residual_processes else residual_processes,
        "SYNTHETIC_SIDECAR_PATH": str(sidecar_dir),
        "TESTS": {name: "PASS" if ok else "FAIL" for name, ok in passed.items()},
        "NEXT": "BOUNDED_HIL_SERL_PILOT_REVIEW" if all_pass else "STOP_PORT_REQUIRES_MAJOR_CHANGE",
        "EXPERIMENT_STATUS": status,
        "EXPERIMENT_NEXT": "COMPLETE" if all_pass else "ENGINEERING_REVIEW",
        "PROJECT_STATUS": "ACTIVE" if all_pass else "BLOCKED",
        "PROJECT_NEXT": "BOUNDED_HIL_SERL_PILOT_REVIEW" if all_pass else "STOP_PORT_REQUIRES_MAJOR_CHANGE",
        "TRAINING_AUTHORIZED": "NO",
    }
    json_dump(output / "result.json", result)

    implementation_report = f"""# Peg F3 V2 HIL-SERL Minimal Port V1 — Implementation Report

## Outcome

- Implementation: {status}
- Feature flag: `--hil_serl_intervention=false` by default
- Action frame: Aubo normalized Action7 unchanged; no Franka/SpaceMouse wrapper or frame transform
- Training, learner updates, optimizer updates, real expert rollout, and bounded pilot: not started

## Touch points

- New adapter: `{adapter_path}`
- New preflight: `{Path(__file__).resolve()}`
- Guarded entry integration: `{async_path}`
- Required backup: `{async_before}`

## Runtime contract

The enabled branch decides the executed action before one environment step, stores that executed action in the canonical six-field transition, inserts every transition once into online replay, and additionally routes intervention transitions to `actor_env_intvn`. The learner registers that name against the already production-Demo-seeded demo buffer. The 50/50 iterators and SAC update remain unchanged.

The adapter disables permanently when C3 loses authority. Formal mode requires both a disabled flag and no provider. The CLI cannot activate a real expert or scientific trigger without a separately approved launcher injecting both objects.
"""
    (output / "IMPLEMENTATION_REPORT.md").write_text(implementation_report, encoding="utf-8")
    preflight_report = f"""# Peg F3 V2 HIL-SERL Minimal Port V1 — Zero-Training Preflight

## Outcome

- Preflight: {status}
- Synthetic-only sidecar: `{sidecar_dir}`
- Positive control: preserved read-only (canonical 3/3, recovery-core 3/3, contact PASS)
- Production Demo SHA: `{after_demo_sha}`
- D_DEMO: `{EXPECTED_D_DEMO}` (unchanged)
- Residual scoped actor/learner processes: {"none" if not residual_processes else residual_processes}

## Tests A–R

""" + "\n".join(
        f"- {name}: {'PASS' if ok else 'FAIL'}" for name, ok in passed.items()
    ) + "\n\nNo training, formal evaluation, real expert takeover, or policy/optimizer update was executed.\n"
    (output / "PREFLIGHT_REPORT.md").write_text(preflight_report, encoding="utf-8")
    print(json.dumps({"status": status, "result_dir": str(output)}, sort_keys=True))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
