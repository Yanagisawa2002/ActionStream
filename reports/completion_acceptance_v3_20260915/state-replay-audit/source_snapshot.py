"""Restore every heldout collection/live/post-stop state and audit scorer truth."""
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
work = base / 'completion-acceptance-v3'
out = work / 'state-replay-audit'
out.mkdir(exist_ok=False)
protocol = json.loads((work / 'protocol.json').read_text())
env = make_env(argparse.Namespace(base=base), out)
records = []
try:
    entries = json.loads((work / 'collection/manifest.json').read_text())['entries']
    for stage in ('collection', 'live'):
        for entry in entries:
            name = entry['episode']
            restore_episode(env, work / stage, name)
            before = json.loads((work / stage / (name + '_truth.json')).read_text())
            after = json.loads((work / stage / (name + '_post.json')).read_text())
            recorded = before + after
            truth = StableTruth(protocol['truth'])
            changes, facts = [], []
            path = work / stage / (name + '.npz')
            with np.load(path, allow_pickle=False) as d:
                states = np.concatenate([d['states'], d['post_states']]) if len(d['post_states']) else d['states']
                assert len(states) == len(recorded)
                for i, state in enumerate(states):
                    env.set_state(state)
                    env.sim.forward()
                    f = simulator_facts(env)
                    f['strict_complete'] = truth.update(i, f)
                    facts.append(f)
                    fields = [k for k in ('inside', 'finger_contact', 'basket_contact', 'strict_complete') if f[k] != recorded[i][k]]
                    if fields:
                        changes.append(dict(control=i, fields=fields, recorded=recorded[i], restored=f))
                prefix = None
                if stage == 'live':
                    with np.load(work / 'collection' / (name + '.npz'), allow_pickle=False) as ref:
                        prefix = dict(rgb_equal=np.array_equal(d['rgb'], ref['rgb'][:len(d['rgb'])]), max_state_delta=float(np.max(np.abs(d['states'] - ref['states'][:len(d['states'])]))))
            save(out / (stage + '_' + name + '.json'), facts)
            records.append(dict(stage=stage, episode=name, controls=len(before), post_controls=len(after), changed_controls=changes, reference_prefix=prefix, image_state_sha256=sha(path)))
finally:
    env.close()
save(out / 'summary.json', dict(records=records, source_sha256=sha(__file__), freeze_sha256=sha(work / 'freeze.json')))
(out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(episodes=len(records), controls=sum(r['controls']+r['post_controls'] for r in records), changed_controls=sum(len(r['changed_controls']) for r in records), changed_strict=sum('strict_complete' in c['fields'] for r in records for c in r['changed_controls']))))
