# 50 ms control work budget

The original [2026-09-15 timing NO-GO](../reports/async_agent_20260915/README.md)
remains unchanged. This follow-up profiles only consumed seeds, then freezes a
separate implementation and twenty new cases before acceptance.

The [frozen follow-up report](../reports/async_budget_20260916/README.md) now passes:
10/10 normal safe completions, 4/2,207 work misses (0.181%, every case under 1%),
and 10/10 eligible physical failures recovered. Independent state/RGB audit passes.

## Measured development findings

The original 2,187 controls averaged 34.78 ms of work. The 102 misses averaged
54.44 ms, including 39.01 ms in native simulation and 4.12 ms in the recorded
RGB inference span. New nested profiling separates simulation, action control,
sensor/render work, immutable image preparation, RGB submission/transfers,
private state recording and journal writes. Nested spans overlap and cannot be
summed; CUDA event duration includes contention and is not exclusive GPU time.

On the first detailed consumed case, mean RGB host-to-device transfer was
0.137 ms, result readback 0.047 ms and journal write 0.095 ms. These were small
compared with native simulation. Single-thread BLAS, single-thread Torch and a
1 ms Python switch interval each left development cases above the miss limit.
Those experimental settings are not deployed.

## Validated execution

- Async/recovery CLI execution uses `IsolatedAsyncSimulationPort`. The production
  ActionStream engine still owns the request mailbox, queue and reset epochs.
  Its worker sends only public RGB/proprioception and the authorized instruction
  to a spawned VLA process. All simulator writes stay on the control thread.
- A process owns one episode. It loads the same pinned X-VLA and official
  processors. Reset retains the warmed process and RNG stream; it does not
  reload the model or reset the simulator. Close reaps the task-owned child.
- Each immutable observation caches its unchanged 192-pixel RGB preprocessing.
  The recorder and verifier share those immutable pixels. A new observation or
  revision cannot inherit another observation's cache.
- The exact stateless official output processors operate on flattened batch/time
  rows. Every action retains its order and 7D conversion. Unknown processor
  pipelines are rejected. CPU tests and the actual first GPU warmup chunk both
  matched the former per-timestep output exactly.

The checkpoint, classifier thresholds, eleven-observation confirmation,
150 ms observation freshness, one-second action freshness, queue invalidation,
bounded retry and physical continuation are unchanged. CUDA request deadlines
remain advisory in the engine. Process startup and additional model residency
must be reported separately from steady control timing.

## Reproduce development

Use the locked Linux installation and verified asset store from
[the delivery runbook](agent-delivery.md). Keep outputs in new directories.

```bash
python scripts/engineering/profile_async_budget.py --store "$STORE" \
  --output "$OUTPUT" --seeds 2026092100 2026092103 2026092107 \
  --isolate-inference
python scripts/engineering/summarize_async_profile.py "$OUTPUT"
```

The profiler accepts only the already consumed original asynchronous seeds.
It reauthorizes a retained real Qwen response and preserves complete trajectories,
source snapshots, timings and independent scores. Development rates are not
heldout acceptance rates.

## New acceptance

`configs/async_agent_acceptance_v2.json` retains every original numerical gate,
excludes all 142 historical layouts, and declares ten new normal seeds and ten
other recovery seeds. No replacement, performance retry, fitting or gate
relaxation is allowed. Run a current installed CLI delivery check first, then:

```bash
python scripts/engineering/evaluate_async_budget.py freeze \
  --store "$STORE" --output "$RUN" --source-commit "$COMMIT"
python scripts/engineering/evaluate_async_budget.py normal \
  --store "$STORE" --output "$RUN" --source-commit "$COMMIT" \
  --delivery-receipt "$DELIVERY_RECEIPT"
python scripts/engineering/evaluate_async_budget.py recovery \
  --store "$STORE" --output "$RUN" --source-commit "$COMMIT" \
  --delivery-receipt "$DELIVERY_RECEIPT"
```

The native results and independent state/RGB audit, rather than CI success,
determine the new GO/NO-GO verdict.

These listed seeds have now been consumed. Reusing them is reproduction or
development; another implementation requires a separate freeze and untouched
acceptance cases. The current installed delivery used an independently reinstalled
root/plugin with the verified dependency clone; fresh network installation attempts
remain incomplete and documented in the report. See the report's separate warmup
times and GPU residency sample when provisioning this native process configuration.
