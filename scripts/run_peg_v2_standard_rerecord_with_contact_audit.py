"""Launch the unchanged V2 pilot recorder with a passive, bound contact sidecar."""
import argparse
import builtins
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.peg_insertion.axial_telemetry import AxialTelemetry
from training.peg_insertion.v2_recorder import PKG


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_line(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


class ContactAuditProxy:
    """Delegate exactly one step to the proven passive observer."""

    def __init__(self, env, output):
        self._delegate = env
        self._output = Path(output)
        self._output.mkdir(parents=True, exist_ok=False)
        self._observer = AxialTelemetry(env, self._output / "telemetry")
        self._contacts = (self._output / "contacts.jsonl").open("x")
        self.phase = "setup"
        self.closed = False
        self.steps = 0
        self.link01_seen = False
        self.unexpected_seen = False
        (self._output / "binding.json").write_text(json.dumps({
            "launcher_sha256": sha(__file__),
            "recorder_sha256": sha(ROOT / "scripts/record_peg_v2_demo.py"),
            "v2_recorder_sha256": sha(ROOT / "training/peg_insertion/v2_recorder.py"),
            "telemetry_sha256": sha(ROOT / "training/peg_insertion/axial_telemetry.py"),
            "role": "passive contact sidecar; no training data",
        }, indent=2))

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def set_phase(self, phase):
        self.phase = phase

    def _enrich(self, contacts):
        model = self._delegate.env.model
        result = []
        for source in contacts:
            item = dict(source)
            tokens = []
            bodies = []
            for side in (1, 2):
                geom_id = int(item[f"geom{side}_id"])
                body_id = int(model.geom_bodyid[geom_id])
                data_id = int(model.geom_dataid[geom_id])
                body_name = model.body(body_id).name
                mesh_name = model.mesh(data_id).name if int(model.geom_type[geom_id]) == 7 and data_id >= 0 else None
                item[f"geom{side}_body_id"] = body_id
                item[f"geom{side}_body_name"] = body_name
                item[f"geom{side}_mesh_name"] = mesh_name
                bodies.append(body_name)
                tokens.extend((item.get(f"geom{side}_name"), body_name, mesh_name))
            item["link01"] = frozenset(bodies) == frozenset(("base_link", "shoulder_Link"))
            item["expected_task_contact"] = any("hole" in str(token).lower() for token in tokens if token is not None)
            item["unexpected"] = not item["expected_task_contact"]
            result.append(item)
        return result

    def step(self, action):
        self._observer.set_context(phase=self.phase)
        returned = self._observer.step(action)
        self.steps += 1
        last = self._observer.last
        if last is None or not last.get("complete") or last.get("observer_errors") or last.get("step_exception"):
            raise RuntimeError("passive contact telemetry incomplete")
        contacts = self._enrich(last["post"]["contacts"])
        link01 = any(item["link01"] for item in contacts)
        unexpected = any(item["unexpected"] for item in contacts)
        self.link01_seen = self.link01_seen or link01
        self.unexpected_seen = self.unexpected_seen or unexpected
        summary = {
            "step": self.steps,
            "phase": self.phase,
            "contacts": contacts,
            "link01_contact": link01,
            "unexpected_contact": unexpected,
            "training_data": False,
        }
        json_line(self._contacts, summary)
        print("CONTACT_AUDIT " + json.dumps({
            "step": self.steps,
            "count": len(contacts),
            "link01": link01,
            "unexpected": unexpected,
            "pairs": [[item["geom1_body_name"], item["geom2_body_name"]] for item in contacts],
        }), flush=True)
        return returned

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._observer.close()
        self._contacts.close()
        (self._output / "summary.json").write_text(json.dumps({
            "steps": self.steps,
            "link01_contact_seen": self.link01_seen,
            "unexpected_contact_seen": self.unexpected_seen,
            "training_data": False,
        }, indent=2))
        self._delegate.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-audit-output", type=Path, required=True)
    parser.add_argument("--episode-type", choices=(
        "standard", "lateral_plus_y", "lateral_minus_y",
        "orientation_roll", "orientation_pitch",
    ), required=True)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--authorize-recording", action="store_true")
    args = parser.parse_args()
    if not args.record or not args.authorize_recording:
        raise ValueError("both --record and --authorize-recording are required")
    if args.contact_audit_output.exists():
        raise ValueError("contact audit output must be new")

    sys.path[:0] = [str(PKG / "code/example/async_rlpd_peg_sim"), str(PKG / "code/env")]
    import peg_env_adapter
    original_make = peg_env_adapter.make_peg_state_env
    holder = {}

    def audited_make(*positional, **keywords):
        proxy = ContactAuditProxy(original_make(*positional, **keywords), args.contact_audit_output)
        holder["proxy"] = proxy
        return proxy

    original_input = builtins.input

    def audited_input(prompt=""):
        command = original_input(prompt)
        parts = command.strip().split()
        if parts and parts[0] != "q" and "proxy" in holder:
            holder["proxy"].set_phase(parts[0])
        return command

    peg_env_adapter.make_peg_state_env = audited_make
    builtins.input = audited_input
    try:
        from scripts import record_peg_v2_demo
        sys.argv = [
            str(ROOT / "scripts/record_peg_v2_demo.py"),
            "--output", str(args.output),
            "--episode-type", args.episode_type,
            "--record",
            "--authorize-recording",
        ]
        record_peg_v2_demo.main()
    finally:
        builtins.input = original_input
        peg_env_adapter.make_peg_state_env = original_make
        if "proxy" in holder:
            holder["proxy"].close()


if __name__ == "__main__":
    main()
