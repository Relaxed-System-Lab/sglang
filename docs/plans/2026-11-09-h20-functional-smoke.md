# H20 compatible solver functional smoke implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prepare one bounded real-GPU functional smoke for two fixed ShareGPT workflows, proving solver-to-router-to-worker scheduling and complete P/D generation/transfer, before the explicit launch checkpoint.

**Architecture:** Reuse the reviewed V10 owned container/startup/observer/replay/cleanup support and cached70B TP2 preset. Add one functional-only execution scope and minimal bounded worker dequeue evidence. The route uses existing HexAGenT per-request compatible schedule_request with fail_closed and admission queue disabled; actor batch-owner/credit behavior stays unchanged.

**Tech Stack:** Existing Python asyncio supervisor/replay, SGLang Python workers, existing Rust gateway ELF, CPU-only pytest for preparation, and the already cached image/model. No image/model pull or rentals.

## Frozen functional scope

Base2d0f6b83306e7cd1b7422508376c513657b5d61f. Main explicitly froze2P2D/TP2/70B on8H20, --hexgent-router without --hexgent-admission-queue, --hexgent-fallback-policy fail_closed. Report this as the compatible per-request solver route, not actor pending-set/batch-owner validation or paper-performance evidence. Leave all credit/next_grant_index/actor code untouched. Native baseline endpoints can exist for startup but execute no baseline workload.

Use only first two fixed Cal ShareGPT workflows1HuVef4_0 and2wtLdZm_0,10calls. Reuse exact accepted V9 token proof, preserving requests/messages/parents/output targets and overriding served alias explicitly; accepted input5176 tokens,output targets1952,max input+target1687. Derive a separately pinned tiny runtime copy, not a new trace/tokenization pipeline. Two roots provide placement choice but not guaranteed queue contention or replica use; check actual evidence and report what occurred. No references,calibration comparison,Req95/99 matrix,additionalmodel,or adaptive extra burst is included.

Keep existing CPUs4–15/quota12,512GiB memory=swap,12GiB tmpfs,2GiB shm,2700s total including180s owned cleanup,readiness900s clipped to the original work deadline. Persistent evidence stays256MiB: workers64,gateway/planner32,observer64,raw/telemetry/QC64,reserve32. Recheck all assigned GPU UUIDs idle,ports free,availableRAM/disk,image/model/source/input pins immediately before launch. Do not disturb the two unrelated existing Docker containers. Keep all failures and incomplete records; no automatic repeat or budget reset.

## Task1: RED tests and source evidence

Write CPU-only tests before implementation. Worker-observer tests cover actual request metadata serialization,disabled-by-default behavior,event/record byte bounds,no prompt content and selected-versus-candidate identity. Place hook calls only at actual P and D selection/removal sites and keep queue selection unmodified. Supervisor tests cover functional scope omitting admissionqueue,fail_closed,exact10callpin,no reference/baseline execution,owned cleanup on success/failure and olddefaultscope parity. Run only bounded existing focused suites plus new tests on H20 cores1/3,threads1,4GiB AS,60CPU/120wall; never discover GPU test suites.

## Task2: Minimal worker execution evidence

Add a small pure metadata logging helper and calls at scheduler.py actual P can_run_list/waiting removal and disaggregation/decode.py actual D can_run_list/waiting replacement. Read Req fields already installed by the real router/tokenizer path: rid/workflow/call,actual priority,two planned ranks,schedule revision,target and worker role/port. Include actual waiting and selected lists; no client-invented mutation receipt is an execution proof. Gate logging only for this owned functional run with the explicit run namespace and rank0 only; preserve built-in server startup warmup and all raw logs. Emit only when an actual selected Req matches the namespace, retaining complete waiting/selected lists. Use a small fixed record cap that is visible if reached. Keep it within the existing worker log budget; add no logging framework or policy modifications.

## Task3: Functional-only supervisor and QC

Extend or thinly reuse h20_transport_window.py startup,RuntimeChildren,observer/C1 budgets and workflow replay. Functional scope runs READY checks then the one10callcompatible-solver workload and QC,finally owned teardown. Existing startup-only and full2p2d UNREADY/performance gates remain unchanged. Reuse the existing wrapper; do not duplicate a general supervisor or require a new H100 denominator for functionality-only work.

QC must distinguish real solver result,router assignment/priority injection,worker actual dequeue,KV transfer completion,server usage/generation and whole-workflow completion. Require positive planner revision and no fallback for every actual request. Compare chosen P/D URLs/ranks/revisions to actual worker Req records and generated response/usage. Preserve raw request-attempt denominators,workflow IDs,all errors and artifact caps. A run with onlystartup,credit bypass,missing records or incomplete calls is not a successful functional smoke. Report actual replica choices and whether nontrivial queue contention occurred,without claiming priority-performance effects.

## Task4: Source review and launch checkpoint

Finish exact source commit,CPU-focused tests,independent SPEC/QUALITY reviews and fresh source/input/command/pin manifest. Render a new recipe with fresh deadline anchor/approval path and unique run/CID/label identities; preserve old evidence and expired recipes. Submit exact commands/resource peaks/evidence/only-owned cleanup to Main. GPU launch is still blocked until Main explicitly approves this new binding. Later cleanup signals only this supervisor's setsid groups and verified exact CID; verify container absence,process exit and GPU release. No broad kill/name cleanup,model/image downloads,permission changes or deletion of old data.
