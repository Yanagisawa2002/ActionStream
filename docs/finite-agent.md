# Finite language-to-VLA Agent

The `actionstream-agent` entrypoint connects real Qwen language parsing, whole-utterance authorization, native X-VLA actions, the frozen adapted temporal RGB verifier, continuous confirmation, and a two-second physical stop continuation. It supports the single LIBERO Object task: tomato sauce into the basket.

The model and confirmation policy match the [previous independent completion acceptance](../reports/completion_acceptance_v3_20260915/README.md). That result does not automatically certify this new integration. A separate [predeclared acceptance protocol](../configs/finite_agent_acceptance_v1.json) evaluates the complete entrypoint with new language and layouts.

## Behavior

1. Parse the original request once with pinned Qwen. Recompute the whole-utterance permission before constructing the execution port. Unsupported language, malformed responses and incomplete authorization produce an explicit blocked/error outcome.
2. Execute official X-VLA chunks, with 300 policy controls maximum. Native containment is private scoring information and does not terminate released-and-stable observation.
3. At every control, collect both RGB views. The verifier consumes controls 10/5/0 in its rolling history, with unchanged 192-pixel preprocessing and class thresholds (complete >= 0.95, incomplete >= 0.90).
4. Require eleven consecutive positive decisions spanning 0.5 simulated seconds. Unknown/incomplete clears confirmation; a missing control also clears RGB history. Different requests/revisions and reused/out-of-order observations are rejected.
5. If still unconfirmed after 300 controls, execute the same 60-control hold/open settling extension as the frozen combination while continuing RGB checks. At the 360-control horizon return `unconfirmed_horizon`, never success.
6. A confirmed stop discards queued VLA actions, holds the current arm pose, opens the gripper, and executes forty further controls. The independent scorer checks release, support, containment and stability at every step. A failed post-stop physical call remains an error.

This is synchronous simulation integration. It does not yet use `ActionStreamInferenceEngine`, enforce a 50 ms wall-clock deadline, recover from a failed placement, support arbitrary language/tasks or control real hardware. All timing reported here describes observed wall time; simulated 20 Hz is not a real-time guarantee.

## Run one request

Install the root and LeRobot plugin with the existing locked installation procedure. The CLI accepts explicit asset and checkpoint paths:

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
actionstream-agent \
  --request 'Please put the tomato sauce in the basket.' \
  --seed 12345 \
  --assets /path/to/assets \
  --checkpoint /path/to/completion-adapt-epoch1.pt \
  --language-config configs/finite_agent_language.json \
  --asset-manifest configs/completion_runtime_assets.json \
  --output /path/to/new-single-use-output
```

The package also supports `python -m actionstream.llm_vla.agent_cli` with the same arguments. The assets directory must contain the pinned X-VLA, Qwen and BART assets from the manifest, and BART must also be available in the configured Hugging Face cache. The existing LIBERO assets and dependencies are required. Full model weights and simulator assets are not redistributed in Git.

The only accepted completion checkpoint SHA-256 is `28482b40e470d932dfe44312d2dbcb7fa732146a2b2f9c7ebf7683b47cdfa7f9`. Changing the path does not bypass this identity check.

Each new output directory retains original text/model response/authorization, runtime events, action chunks and physical actions, observation identities, private truth, state/RGB arrays and independent scoring. Exit 0 requires a confirmed, independently supported stable completion. Blocked, unconfirmed, unsafe or failed runs return a nonzero status. A standalone CLI run is a diagnostic episode, not automatically an independent acceptance set; arbitrary caller seeds have no novelty guarantee.

## New integration acceptance

The frozen protocol contains twenty supported instructions assigned procedural seeds `2026092000` through `2026092019`, plus twenty unsupported language-only challenges. Only original text and opaque request ID enter Qwen; evaluator labels and seeds are absent from its prompt. Language-only cases are declared before inference and are never robot execution requests; any false accept fails the language gate.

Gates: at least 18/20 supported requests accepted, zero unsupported accepts, zero model/schema errors, at least 16/20 requested tasks safely completed, zero premature stops, zero missed completed events, maximum safe confirmation delay at most two seconds, and all forty post-stop controls stable. Language refusals and physical task failures remain in the overall twenty-request denominator. Missing required execution records or engineering errors cannot produce GO.

The collector rejects duplicate layouts against all fifty original demonstrations, the consumed v2 holdout, development controls, v3 holdout and other current episodes. Freeze source/config/weights before inference, preserve the first results, and never select replacement instructions/seeds or tune thresholds after seeing acceptance output.

```bash
PYTHONPATH=src python scripts/engineering/evaluate_finite_agent.py freeze \
  --base /path/to/existing-experiment-assets \
  --protocol configs/finite_agent_acceptance_v1.json \
  --output /path/to/new-acceptance
PYTHONPATH=src python scripts/engineering/evaluate_finite_agent.py run \
  --base /path/to/existing-experiment-assets \
  --protocol configs/finite_agent_acceptance_v1.json \
  --output /path/to/new-acceptance
```

The acceptance runner reuses the same parsing and execution boundary as the CLI. Its base directory requires the previous experiment manifests and original demonstrations solely to reject reused layouts. The freeze smoke uses an already-consumed development RGB clip. No training is part of this protocol.

## CPU validation

`tests/test_finite_agent.py` checks authorization before construction, forged stored accept flags, correct temporal offsets, warmup, interrupted confirmation, missing controls, foreign/reused observations, chunk discard, post-stop errors, all-unknown finite termination, bad action shapes, clean per-request history, nonfinite probabilities and non-vacuous acceptance denominators. These tests exercise software behavior; the GPU report records measured model/native performance separately.
