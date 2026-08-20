# SmolVLA RTC v4 closeout

Status: **NO-GO at the frozen sync gate; formal RTC was not run.**

The development rule selected Object task 3, Spatial task 0, and Goal task 5
before canary outcomes were opened. On new states 29 and 30, Object passed 2/2,
Spatial passed 1/2, and Goal passed 1/2. Overall sync success was 4/6, below the
required 6/6.

`canary_content_frames.png` contains inspected mid/final frames from the first
captured episode in each family. Object and Goal show successful manipulation;
Spatial runs to step 299 without completing the task, consistent with the
recorded failure.

The upstream SmolVLA checkpoint did expose RTC support and all three sync
processes produced valid actual-worker warmup, GPU residency/utilization, and
request-rate evidence. The formal RTC holdout remains sealed because system
compatibility is not a substitute for a stable sync policy gate.
