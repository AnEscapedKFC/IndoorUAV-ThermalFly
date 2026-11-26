"""Audit the published local weights and protected simulator inputs for the filmed runs."""
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent.parent
sys.path.insert(0, str(BRIDGE))
from audit_results import audit_inputs

OUT = HERE/'results/room_ac_rgbt'
FROZEN = ['model_worker.py', 'sim_coordinates.py', 'grounding_study/policy.py',
          'grounding_study/metric_flight/control.py', 'grounding_study/metric_flight/object_goal.py',
          'grounding_study/metric_flight/tracking.py']


def stream_hash(file):
    digest = hashlib.sha256()
    for block in iter(lambda: file.read(4 * 1024**2), b''):
        digest.update(block)
    return digest.hexdigest()


def sha(path):
    with path.open('rb') as handle:
        return stream_hash(handle)


def policy_hashes():
    return {name:sha(BRIDGE/name) for name in FROZEN}


def main():
    prov = json.loads((BRIDGE/'model_provenance.json').read_text())
    assets = BRIDGE/'assets'
    params = Path(os.environ.get('INDOORUAV_CHECKPOINT', assets/'29999')).resolve()/'params'
    assert params == assets/'29999/params'
    archive = assets/prov['checkpoint_archive']
    assert sha(archive) == prov['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        members = [n for n in z.namelist() if n.startswith('29999/params/') and not n.endswith('/')]
        assert set(members) == {str(p.relative_to(assets)) for p in params.rglob('*') if p.is_file()}
        for name in members:
            with z.open(name) as stream:
                assert stream_hash(stream) == sha(assets/name), name
    assert sha(assets/'norm_stats.json') == prov['normalizer_sha256']
    assert sha(BRIDGE/'model_worker.py') == prov['worker_sha256']
    detector = sha(HERE.parent/'assets/model.safetensors')
    assert detector == '1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3'
    result = dict(published_archive_sha256=prov['archive_sha256'], checkpoint=29999,
                  params_match_archive=len(members), normalizer_matches=True,
                  detector_sha256=detector, frozen_sources=policy_hashes(), protected=audit_inputs())
    (OUT/'audit.json').write_text(json.dumps(result, indent=2))
    print('Verified checkpoint files:', len(members))

if __name__ == '__main__':
    main()
