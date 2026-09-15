"""Check recorded per-control truth against restored simulator states."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path.cwd() / 'scripts/engineering'))
from completion_controls import make_env, restore_episode
from completion_v2 import save, sha
from actionstream.completion_labels import StableTruth, simulator_facts

base = Path('/root/autodl-tmp/actionstream-agent-20260915')
work = base / 'development-controls'
out = work / 'state-replay-audit'
out.mkdir(exist_ok=False)
args = argparse.Namespace(base=base)
protocol = json.loads((work / 'collection/protocol.json').read_text())
env = make_env(args, out)
records = []
try:
    entries = json.loads((work / 'collection/manifest.json').read_text())['entries']
    for stage in ('collection', 'live-development-validation'):
        for entry in entries:
            name = entry['episode']
            path = work / stage / (name + '.npz')
            if not path.exists():
                continue
            restore_episode(env, work / 'collection', name)
            raw = json.loads((work / stage / (name + ('_truth.json' if stage == 'collection' else '_private_truth.json'))).read_text())
            recorded = raw if stage == 'collection' else raw['controls']
            truth = StableTruth(protocol['truth'])
            changes, facts = [], []
            max_speed_delta = 0
            with np.load(path, allow_pickle=False) as d:
                assert len(d['states']) == len(recorded)
                for i, state in enumerate(d['states']):
                    env.set_state(state)
                    env.sim.forward()
                    f = simulator_facts(env)
                    f['strict_complete'] = truth.update(i, f)
                    facts.append(f)
                    fields = [k for k in ('inside', 'finger_contact', 'basket_contact', 'strict_complete') if f[k] != recorded[i][k]]
                    if fields:
                        changes.append(dict(control=i, fields=fields, recorded=recorded[i], restored=f))
                    max_speed_delta = max(max_speed_delta, *(abs(f[k] - recorded[i][k]) for k in ('linear_speed', 'angular_speed')))
            save(out / (stage + '_' + name + '.json'), facts)
            records.append(dict(stage=stage, episode=name, controls=len(facts), changed_controls=changes, max_speed_delta=max_speed_delta, image_state_sha256=sha(path)))
finally:
    env.close()
save(out / 'summary.json', dict(records=records, source_sha256=sha(__file__)))
(out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(episodes=len(records), controls=sum(r['controls'] for r in records), changed_controls=sum(len(r['changed_controls']) for r in records))))
