# Full H20 workflow functional revalidation

Use the existing v14 actor-classification fix. Preserve the v13 FAILED evidence and strict registered PID/start identity gate. The image documents TORCHINDUCTOR_COMPILE_THREADS=1 to disable asynchronous compilation pools; use this only for functional verification, with no performance claim. Old24 PID roles remain unknown because command/parent data were not saved.

Retain bounded startup PID/PPID/PGRP/TGID/name/argv/task metadata for owned processes. Change the ready-phase timeout from120 to300s to cover the1489-token longest workflow plus frontend/prefill time; retain2s normal sampling,10s error burst,1200s global window and180s cleanup. All10 calls, exact output usage, solver/worker joins and both complete workflows must pass; parents must complete before children are submitted. No migration, model download, paid GPU, unrelated-service cleanup or automatic retries.

CPU TDD: reproduce missing compiler environment and identity metadata, then minimally implement and run the actor/functional/bounded-observer regressions. Freeze a clean pin and one new recipe, report hashes and fresh8-GPU/port inventory, then directly execute the authorized one bounded revalidation. Preserve exact-owner cleanup and report either complete success or the concrete remaining blocker.
