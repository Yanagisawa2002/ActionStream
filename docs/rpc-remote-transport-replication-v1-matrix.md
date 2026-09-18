# Frozen two-host transport replication run matrix

Source: `configs/rpc_remote_transport_replication_v1.json`. Status: **PRE_REGISTERED_NOT_EXECUTED**.

N = no injected fault; J50 = application delivery 50 +/- 20 ms (fault seed 2026091701); J250 = 250 +/- 100 ms (2026091702); J950 = 950 +/- 250 ms (2026091703); D7 = pre-inference disconnect every 7 requests (2026091704).

Run 1 is the formal smoke/canary and counts once. Each row requires a fresh GPU server process. Fault profiles are server-side application faults; client delivery scheduling is disabled. No row has been executed.

| Run | Suite | Task | State | Policy seed | Condition |
|---:|---|---:|---:|---:|---|
| 1 | libero_object | 5 | 25 | 2026091725 | N |
| 2 | libero_object | 5 | 26 | 2026091726 | N |
| 3 | libero_object | 5 | 27 | 2026091727 | N |
| 4 | libero_object | 5 | 28 | 2026091728 | N |
| 5 | libero_object | 5 | 29 | 2026091729 | N |
| 6 | libero_spatial | 7 | 25 | 2026091725 | N |
| 7 | libero_spatial | 7 | 26 | 2026091726 | N |
| 8 | libero_spatial | 7 | 27 | 2026091727 | N |
| 9 | libero_spatial | 7 | 28 | 2026091728 | N |
| 10 | libero_spatial | 7 | 29 | 2026091729 | N |
| 11 | libero_goal | 2 | 25 | 2026091725 | N |
| 12 | libero_goal | 2 | 26 | 2026091726 | N |
| 13 | libero_goal | 2 | 27 | 2026091727 | N |
| 14 | libero_goal | 2 | 28 | 2026091728 | N |
| 15 | libero_goal | 2 | 29 | 2026091729 | N |
| 16 | libero_object | 5 | 25 | 2026091725 | J50 |
| 17 | libero_object | 5 | 25 | 2026091725 | J250 |
| 18 | libero_object | 5 | 25 | 2026091725 | J950 |
| 19 | libero_object | 5 | 25 | 2026091725 | D7 |
| 20 | libero_object | 5 | 26 | 2026091726 | J50 |
| 21 | libero_object | 5 | 26 | 2026091726 | J250 |
| 22 | libero_object | 5 | 26 | 2026091726 | J950 |
| 23 | libero_object | 5 | 26 | 2026091726 | D7 |
| 24 | libero_object | 5 | 27 | 2026091727 | J50 |
| 25 | libero_object | 5 | 27 | 2026091727 | J250 |
| 26 | libero_object | 5 | 27 | 2026091727 | J950 |
| 27 | libero_object | 5 | 27 | 2026091727 | D7 |
| 28 | libero_spatial | 7 | 25 | 2026091725 | J50 |
| 29 | libero_spatial | 7 | 25 | 2026091725 | J250 |
| 30 | libero_spatial | 7 | 25 | 2026091725 | J950 |
| 31 | libero_spatial | 7 | 25 | 2026091725 | D7 |
| 32 | libero_spatial | 7 | 26 | 2026091726 | J50 |
| 33 | libero_spatial | 7 | 26 | 2026091726 | J250 |
| 34 | libero_spatial | 7 | 26 | 2026091726 | J950 |
| 35 | libero_spatial | 7 | 26 | 2026091726 | D7 |
| 36 | libero_spatial | 7 | 27 | 2026091727 | J50 |
| 37 | libero_spatial | 7 | 27 | 2026091727 | J250 |
| 38 | libero_spatial | 7 | 27 | 2026091727 | J950 |
| 39 | libero_spatial | 7 | 27 | 2026091727 | D7 |
| 40 | libero_goal | 2 | 25 | 2026091725 | J50 |
| 41 | libero_goal | 2 | 25 | 2026091725 | J250 |
| 42 | libero_goal | 2 | 25 | 2026091725 | J950 |
| 43 | libero_goal | 2 | 25 | 2026091725 | D7 |
| 44 | libero_goal | 2 | 26 | 2026091726 | J50 |
| 45 | libero_goal | 2 | 26 | 2026091726 | J250 |
| 46 | libero_goal | 2 | 26 | 2026091726 | J950 |
| 47 | libero_goal | 2 | 26 | 2026091726 | D7 |
| 48 | libero_goal | 2 | 27 | 2026091727 | J50 |
| 49 | libero_goal | 2 | 27 | 2026091727 | J250 |
| 50 | libero_goal | 2 | 27 | 2026091727 | J950 |
| 51 | libero_goal | 2 | 27 | 2026091727 | D7 |
