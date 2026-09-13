# Embodied development status — 2026-09-13

This update combines the compact v1.1 candidate, its byte-identical public
LeRobot dependency source, and four completed implementation stages. The
runtime fixes and experimental evaluation tooling are delivered together;
the language and visual quality gates remain **FAIL**.

| Stage | Commit | Observed result |
|---|---|---|
| Episode ownership and transport lifecycle | `9a923533f522aa648749f1033cfbcdf5f600789d` | All 11 reproduced lifecycle failures pass; 46 CPU contracts passed |
| Fixed native X-VLA / LIBERO | `e47d6256c1fd9e595efc3694a92f89325b7ba1da` | One smoke and three development resets succeeded; 514 control steps and 20 actual inference calls; 6 new CPU contracts |
| Finite language and observation interfaces | `ab1413bba6eb0bd4ee30ab027a05ec0badff98be` | 12/12 supported requests accepted, 2/12 negative requests incorrectly admitted; 13 CPU contracts; vision and integrated execution not run |
| Original-request evidence binding | `eac8bd8d9baef154487d30fe7caff700209867bf` | Paired language comparison and 20 actual dual-camera model calls; 13 relevant CPU contracts; both quality gates failed |

These test counts have different and partly overlapping scopes. They must not
be summed into a single suite result. The merge validation and GitHub CI test
the combined source, including the older PR's independent lifecycle regressions.

## Native baseline

The fixed task is LIBERO object task 5: tomato sauce into the basket. Init
indices 45, 0, 1 and 2 succeeded under the recorded 20 Hz controller and
300-control-step cap. This is four development examples, not a generalization
rate, a full benchmark or an improvement over the pretrained policy.

The executor uses X-VLA revision
`12e8783e996944f5c97e490d37d4c145484ed70a` and the official processor chain.
Raw `[1,30,20]` predictions become `[1,30,7]` native actions without truncating
dimensions. CUDA inference ran on a local RTX 4090. EGL used Mesa llvmpipe:
roughly 50 ms native steps are software-rendering evidence, not NVIDIA hardware
EGL acceptance or a stable 20 Hz wall-clock rate.

## Language completeness remains unresolved

On the same new 48-case development challenge, v1 admitted 23/24 supported
requests and incorrectly admitted 8/24 negative requests. V2 admitted 24/24
supported and 1/24 negative requests. On the already-consumed 24-case regression
set, v2 admitted 12/12 supported and 1/12 negative requests.

V2 binds entity quotes to the immutable original utterance and constructs the
canonical skill call on the host. It blocks unsupported inferred references,
but authentic entity mentions do not establish that every requested action or
object was preserved. It still dropped an extra drawer-closing obligation and
an additional orange-juice object. The latter is a regression from v1.

New-set model-call P50 was 1.788309 s for v1 and 0.276689 s for v2, whose output
is shorter. No component ablation or robot-control speedup is claimed.

## Visual evidence is not yet reliable

All 20 checker outputs violated the frozen JSON-only contract by adding code
fences. Accepted coverage on the 16 labelled observations was therefore 0%.
Separately, literal raw text claimed completion on 19/20 inputs, including all
12 authoritative negative observations. Zero *admitted* false completions
means the outputs were blocked; it is not evidence of an accurate checker.

Simulator truth belongs only to the independent scorer. It is not substituted
for the runtime observation checker. The two high-level stages ran no new
native episode, reset, physical control or recovery. Their interface tests
must not be presented as a working language/vision closed loop.

## Integration and next research

The full lifecycle implementation is retained when combining the older PR's
overlapping queue-commit fix. The older PR's tests and public upstream archive
URL remain included. The latter resolves the same LeRobot commit and checksum;
it does not change model or dependency versions.

The next research scope is complete instruction coverage and verifiable
visual evidence: enumerate all actions, objects and constraints; test image
input sensitivity; establish observable identity and containment evidence
with abstention. Failed experiments remain explicit. The historical H1/H2
records, including H2 NO-GO, are unchanged.

Machine-readable curated evidence is in
[`reports/embodied_development_20260913`](../reports/embodied_development_20260913).
It contains selected measured fields and hashes of the source receipts, not
model weights, machine-specific environments or the complete local raw archive.
The native and high-level reproduction documents describe the required local
assets; CI exercises CPU contracts, not GPU or robot acceptance.
