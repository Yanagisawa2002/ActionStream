# M7-G0 result figures

These figures describe the frozen `ros_cpp_test_plant` holdout only. They are
not Isaac Sim task results. Each plot is emitted as a 300-dpi PNG and a vector
PDF from version-controlled raw JSON/JSONL artifacts.

## Captions and provenance

1. **Paired task success.** `m7_success_comparison` shows exact successes out
   of 36 paired seeds for `sync_hold`, `naive_async`, and `aligned_async` under
   both frozen profiles. Source:
   `outputs/m7_g0/holdout/analysis.json`, which is derived from the 216 entries
   in `outputs/m7_g0/holdout/manifest.json`.
2. **Profile B efficiency.** `m7_efficiency_comparison` reports median
   wall-clock completion time and mean inference hold time for all three
   methods. Source: the same independently analyzed holdout summaries. The
   larger aligned-than-naive hold bar is retained; the figure is not selected
   to make aligned execution appear uniformly better.
3. **Auditable asynchronous timeline.** `m7_async_timeline` uses the aligned
   Profile B episode at seed `2026080434`. It contains six requests, seven
   delivered responses, three atomic queue rebuilds, 57 commands, and four
   explicit chunk rejections. The rejection reasons are two duplicate
   responses and two responses arriving after episode termination. No stale
   generation was observed in this episode, so none is implied by the plot.
   Source:
   `outputs/m7_g0/holdout/episodes/profile_b/seed_2026080434/aligned_async.events.jsonl`.

Regenerate the six files after analysis with:

```bash
python -m action_stream_benchmark.cli figures \
  --analysis outputs/m7_g0/holdout/analysis.json \
  --example-event-log outputs/m7_g0/holdout/episodes/profile_b/seed_2026080434/aligned_async.events.jsonl \
  --output-dir outputs/m7_g0/figures
```
