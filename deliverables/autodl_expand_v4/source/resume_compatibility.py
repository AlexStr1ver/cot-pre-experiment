"""Explicit provenance bridge for the audited v3 -> v3.2 compatibility fix.

This changes no old metadata and never presents changed code as the old code.
The one permitted legacy identity includes the frozen weights, protocol,
prefix plan, tokenizer metadata and all six original code hashes.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


LEGACY_FINGERPRINT = 'bf70f0a22ca788257b5a305a843f12942cdab3d553ef625401c440e60c220858'
LEGACY_COORDINATOR_SHA256 = '3cc9f570b8ba8e06c4452134602838f0128a3e9958bbdf27f819997e968aaff1'
APPROVED_COORDINATOR_SHA256 = '64fd3fd448b5a76c87e06429a6ce934e39b3ba3c400da5708e16c4b55c80cced'
UNCHANGED_CODE_FILES = ('actions.py', 'decoder.py', 'collect_phase2.py', 'v1_core.py', 'prepare_plan.py')
ADDED_CODE_FILES = ('hardware_compatibility.py', 'resume_compatibility.py')
CURRENT_CODE_FILES = ('paired_experiment.py', *UNCHANGED_CODE_FILES, *ADDED_CODE_FILES)
BRIDGE_POLICY = 'known_v3_identity_to_v3_2_capacity_compatibility_v1'


def identity_sha(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def local_code_hashes():
    package = Path(__file__).resolve().parent
    return {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in CURRENT_CODE_FILES}


def checked_identity(config, label):
    identity = config.get('experiment_identity')
    if not isinstance(identity, dict) or config.get('experiment_fingerprint') != identity_sha(identity):
        raise ValueError(label + ' experiment identity does not match its fingerprint')
    if config.get('protocol') != identity.get('protocol'):
        raise ValueError(label + ' protocol differs from its fingerprinted identity')
    if not isinstance(identity.get('code_sha256'), dict):
        raise ValueError(label + ' lacks fingerprinted code hashes')
    return identity


def fingerprint_bridge(prior, current):
    """Return a truthful compatibility audit or reject unknown code/science changes."""
    old_identity = checked_identity(prior, 'Prior')
    new_identity = checked_identity(current, 'Current')
    actual_code = local_code_hashes()
    if actual_code['paired_experiment.py'] != APPROVED_COORDINATOR_SHA256:
        raise ValueError('Executing coordinator is not the audited v3.2 compatibility revision')
    if new_identity['code_sha256'] != actual_code:
        raise ValueError('Current identity code hashes differ from the executing v3.2 files')
    report = {
        'policy': BRIDGE_POLICY,
        'prior_experiment_fingerprint': prior['experiment_fingerprint'],
        'current_experiment_fingerprint': current['experiment_fingerprint'],
        'current_code_sha256': actual_code,
        'prior_metadata_modified': False,
    }
    if old_identity == new_identity:
        report.update(mode='identical_v3_2_identity', changed_code_files=[])
        return report
    if prior['experiment_fingerprint'] != LEGACY_FINGERPRINT:
        raise ValueError('Prior code/protocol/weights/prefix fingerprint is not an approved legacy identity')
    old_code = old_identity['code_sha256']
    if (old_code.get('paired_experiment.py') != LEGACY_COORDINATOR_SHA256 or
            set(old_code) != {'paired_experiment.py', *UNCHANGED_CODE_FILES}):
        raise ValueError('Approved legacy coordinator/code manifest differs')
    expected_identity = copy.deepcopy(old_identity)
    expected_code = expected_identity['code_sha256']
    for name in ('paired_experiment.py', *ADDED_CODE_FILES):
        expected_code[name] = actual_code[name]
    if expected_identity != new_identity:
        raise ValueError('Legacy bridge only permits coordinator/helper changes; scientific identity differs')
    report.update(
        mode='known_legacy_v3_to_v3_2',
        changed_code_files=['paired_experiment.py'],
        added_code_files=list(ADDED_CODE_FILES),
        unchanged_scientific_code_files=list(UNCHANGED_CODE_FILES),
        prior_coordinator_sha256=old_code['paired_experiment.py'],
        current_coordinator_sha256=actual_code['paired_experiment.py'],
        unchanged_identity_fields=sorted(key for key in old_identity if key != 'code_sha256'),
    )
    return report
