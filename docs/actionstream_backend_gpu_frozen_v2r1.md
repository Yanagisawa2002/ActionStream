# ActionStream backend GPU benchmark v2r1: frozen replacement

The initial v2 freeze is retained but invalidated before any model load or
scored episode because its policy seeds exceeded NumPy/LeRobot's uint32 range.
No result or holdout label was observed and no scientific gate was relaxed.
The exact invalidation receipt is
`configs/actionstream_backend_gpu_v2_invalidated.json`.

v2r1 changes only policy and network seeds and adds a fail-fast seed-range
check. It keeps candidate runtime logic, three task families, states 45--49,
five comparable operating points, runtime matrix, warmup count, GPU sampling,
paired contrasts, and the fixed-950 decision rule unchanged. Its candidate is
commit `a00f580a0956474612b3639e5eccc3d0cf61db8b`.

The full engineering, measurement, fairness, and interpretation contract is
the initial v2 document plus the v2r1 registry. v6 selector and Arena remain
paused. Holdout remains sealed until the complete v2r1 canary passes.
