# Lifecycle repair — 2026-09-13

The final queue commit now checks the request generation and shutdown state while holding the condition lock followed by the queue lock, matching reset. A reset between the earlier acceptance check and merge cannot refill the new episode. Delivery accounting is committed at that boundary. Old-generation inference failures, latency and recovery counters cannot poison a new episode; fatal escalation also checks the generation under the lock.

Run `python -m pytest tests/test_lifecycle_generation.py tests/test_lerobot_inference.py` after the locked Python 3.12 installation in the README. The new tests use real worker/delivery threads with barriers at the previous race window and cover reset/stop with direct/scheduled delivery, plus a failing old request followed by a successful new episode. They are CPU tests, not robot acceptance.

This branch starts from the compact v1.1 review candidate `e80afac6cc8a563783b6f2c71614c01e337bb728`. The default branch's v1.0 release, old large Draft PR and the compact candidate are different revisions. Review the revision linked by the project card. No upstream merge is claimed: LeRobot PR #4466 is closed and unmerged.

H1-R2 remains a historical fixed-950-ms delivery experiment on RTX 5090: 10/15 to 13/15 successes, 11.43x paired request-supply ratio, and 159 to 1,261 inference calls (7.93x). The success interval crosses zero and one task family regresses. H2 remains NO-GO. These tests do not revalidate those GPU runs or establish general inference speedup.
