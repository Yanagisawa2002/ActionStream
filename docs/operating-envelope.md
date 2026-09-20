# Operating-envelope benchmark plan

Status: **DRAFT / NO_RUN**.

This document defines the next systems experiment direction. It does not report a
new GPU result and is not yet a frozen preregistration. Before execution, the
runtime commit, identities/tasks, hardware, fault placement, repetition count,
and acceptance criteria must be locked.

## Question

Instead of asking whether one scheduler wins at one latency point, map where the
runtime remains supplied, recoverable, and task-effective as latency and faults
increase.

The intended output is an operating envelope: regions where queue supply,
recovery, compute cost, and task outcome are jointly acceptable.

## Planned axes

- controlled response/delivery delay: 50, 100, 250, 500, 1000, 2000 ms;
- transport: direct, child process, TCP;
- startup state: cold and warm;
- TCP disconnect cadence: none, every 10 requests, every 5 requests;
- TCP executor policy:
  - persistent executor — implemented today;
  - reconstruct-on-reconnect — comparison mechanism **not implemented yet**.

The delay axis must be described by where it is injected. An application response
delay is not automatically equivalent to real network RTT or jitter.

## Metrics

At minimum record:

- task success and completion steps;
- inference requests / second and inference calls per episode;
- queue depth and depletion fraction over time;
- action age at dequeue and stale-prefix discard count;
- request round-trip, server queue wait, server compute, and service latency;
- reconnects, deadlines, transport errors, reset failures, and demonstrated
  recoveries;
- cold-start and reconnect-first latency;
- GPU memory and utilization when a GPU policy is actually used.

A high task-success rate must not by itself define a healthy operating region.
Queue supply and recovery are separate dimensions.

## Comparison discipline

Direct, process, and TCP modes must use the same policy checkpoint, controller
budget, observation stream, task identities, and application delay definition
where those concepts are comparable.

Cold and warm measurements must be labelled separately. A reconnect-first request
must not silently be grouped with a model cold start.

Persistent versus reconstruct-on-reconnect is not executable until the comparison
implementation exists. The matrix planner marks those rows unavailable rather than
pretending the current runtime supports them.

## Execution gates

Before any GPU run:

1. freeze a runtime commit after the current modularization work;
2. choose untouched task/layout identities and record overlap checks;
3. decide whether the study is simulation-only or includes real network shaping;
4. fix fault schedules and repetition count before outcomes;
5. predeclare the operating-envelope acceptance rule;
6. keep raw traces outside ordinary Git and commit only compact replayable evidence.

Generate the current draft matrix with:

```bash
uv run python scripts/engineering/plan_operating_envelope.py \
  --config configs/operating_envelope_v1.json \
  --output /tmp/actionstream-operating-envelope.jsonl
```

The planner only expands protocol rows. It does not run LIBERO, load a model, or
make a GPU claim.
