# CPU-only examples

These examples exercise ActionStream without LIBERO, X-VLA, model weights, or a
GPU. They are intended as a fast review surface for runtime behavior.

Run the local lifecycle demo:

```bash
uv run python examples/local_async_policy.py
```

It blocks one inference call while the control step advances, demonstrates
expired-prefix discard, then resets during another in-flight call and shows that
the old generation cannot repopulate the queue.

Run the self-contained TCP fault demo:

```bash
uv run python examples/fault_recovery.py
```

It uses real loopback sockets, injects a deterministic disconnect, performs an
acknowledged reset, and verifies that the persistent executor is not reconstructed.

For a two-terminal client/server walkthrough, start:

```bash
uv run python examples/tcp_worker.py
```

Then, in another terminal:

```bash
uv run python examples/tcp_client.py
```

The example server is unauthenticated and defaults to loopback. It is not a
production deployment recipe.
