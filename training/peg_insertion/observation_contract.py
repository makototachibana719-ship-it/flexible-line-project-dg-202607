"""CPU-only, fail-closed observation compatibility. No legacy bytes are rewritten."""
import hashlib
import json
from pathlib import Path

V1 = 'PEG_OBS_CONTRACT_V1_LEGACY'
V2 = 'PEG_OBS_CONTRACT_V2_SENSOR_FIXED'
ENV_CONTRACT = V2
ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT/'configs/peg_insertion/legacy_observation_contracts.json'

def known(version):
    if version not in (V1, V2):
        raise ValueError('missing/unknown observation contract')
    return version

def require_match(artifact, environment=V2, *, compatibility_diagnostic=False):
    known(artifact); known(environment)
    if type(compatibility_diagnostic) is not bool:
        raise ValueError('diagnostic permission must be explicit boolean')
    mismatch = artifact != environment
    if mismatch and not compatibility_diagnostic:
        raise ValueError(f'observation contract mismatch: {artifact} != {environment}')
    return dict(checkpoint_contract=artifact, env_contract=environment,
                compatibility_diagnostic=compatibility_diagnostic,
                formal_skill_result=not compatibility_diagnostic and not mismatch)

def validate_config(cfg, *, for_training=False):
    version = known(cfg.get('observation', {}).get('contract_version'))
    require_match(version, ENV_CONTRACT)
    demo = cfg.get('demo', {})
    registered = json.loads(REGISTRY.read_text()).get('demo_sha256', {}).get(demo.get('sha256'))
    if registered and demo.get('observation_contract_version') != registered:
        raise ValueError('legacy Demo contract cannot be relabeled')
    if for_training:
        require_match(cfg.get('demo', {}).get('observation_contract_version'), version)
    return version

def checkpoint_version(cp):
    cp = Path(cp)
    if not cp.is_file():
        raise ValueError('checkpoint file missing')
    digest = hashlib.sha256(cp.read_bytes()).hexdigest()
    registry = json.loads(REGISTRY.read_text())
    registered = registry.get('checkpoint_sha256', {}).get(digest)
    mp = cp.with_name(cp.name+'_resume_metadata.json')
    meta = json.loads(mp.read_text()) if mp.is_file() else {}
    explicit = meta.get('observation_contract_version')
    nested = meta.get('observation', {}).get('contract_version')
    if explicit and nested and explicit != nested:
        raise ValueError('conflicting observation contract metadata')
    if registered:
        if (explicit and explicit != registered) or (nested and nested != registered):
            raise ValueError('legacy checkpoint contract cannot be relabeled')
        return known(registered)
    if not explicit or not nested or explicit != nested or meta.get('model_sha256') != digest:
        raise ValueError('unregistered checkpoint lacks hash-bound version metadata')
    return known(explicit)

def checkpoint_preflight(cp, *, env_contract=V2, compatibility_diagnostic=False):
    return require_match(checkpoint_version(cp), env_contract,
                         compatibility_diagnostic=compatibility_diagnostic)
