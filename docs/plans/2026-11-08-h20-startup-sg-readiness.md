# H20 startup and SG readiness implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prepare the approved startup-only H20 checkpoint, preserve and sanitize the explicit SG75 calls, and produce one reviewed tokenizer CPU recipe before any new runtime launch.

**Architecture:** Keep the accepted V7 observer and canonical a064 policy. Extend the existing C1 entry points with an approval-bound startup-only scope and guaranteed owned cleanup; extend the existing subset preparer with explicit ID lists and the unchanged sanitizer. A standalone tokenizer checker compares exact IDs along the existing preparation and normal-server rendering paths, importing providers only inside its future explicitly approved CPU entry.

**Tech Stack:** Python standard library, existing C1 launcher/runtime, pytest mocks for external process boundaries; future pinned A4 image and local Llama tokenizer files.

The main chat already approved same-session implementation, SG extraction and offline TDD. Root executes every test and the real SG preparation; agents edit disjoint files. No additional workflow-choice question is needed. Native/GPU/tokenizer-container execution remains outside this approval.

### Task 1: Freeze approval and existing inputs

Files: V8 artifact `main-approval.json`; existing frozen old-C1 runtime-inputs-v2 JSONL files; no changes to those inputs.

Read and hash the exact old three constructor inputs from the existing configuration. Preserve the original resource geometry, inherited 2700-second deadline including 180-second cleanup, source/model/ELF identities, and pinned A4 image. Keep the accepted V7 source/test hashes as the baseline. Main independently verified the 193-test baseline.

### Task 2: Startup-only entry

Modify `scripts/hexgent_real_trace_collectors/h20_transport_window.py` and `test/unit/utils/test_hexgent_h20_transport_window.py`.

1. Write failing tests for plan/CLI scope binding, unchanged constructor input hashes, no calibration/reference/replay/experimental warmup after startup, startup failure and deadline cleanup, original deadline inheritance, and exact owned child/fullCID cleanup plus absence evidence.
2. Root runs the new selectors under the offline guard and records RED.
3. Add the minimal startup-only plan/argv/supervisor path and `C1RuntimeOperations.execute_startup_only()`. It calls the existing startup and cleanup logic. A missing or mismatched flag cannot broaden a startup-only approved plan into full execution. Cleanup failures do not skip remaining owned resources. Built-in server startup warmup remains part of startup.
4. Root verifies GREEN plus the accepted observer/window regression suite. Default full-run and dry-run behavior retain their prior contract.

### Task 3: Explicit SG selection

Modify `scripts/hexgent_real_trace_collectors/prepare_h20_subset.py`; add `test/unit/utils/test_hexgent_h20_explicit_subset.py`.

1. Tests cover explicit Cal8/32 and Eval8/43 IDs, all whole-workflow parents/messages/length targets/arrival nulls, 13 assistant-leading histories, exact raw selected-row hashes, fresh output directory, and old default-selection parity.
2. Root records RED before implementation.
3. Add optional `workflow_ids` and `--workflow-ids-json`, selecting only listed datasets. Keep `_sanitize` unchanged and preserve supplied ID order. Reject overlap/unknown/repeated IDs before output writes. Record archive/member/raw-row/preparer/output hashes.
4. Root verifies GREEN, then performs one authorized new-directory SG artifact preparation at V8/sg-inputs. Do not describe shared-prefix workflow records as independent trajectories. No BFCL or tokenizer action occurs.

### Task 4: Tokenizer CPU preparation

Add `scripts/hexgent_real_trace_collectors/prepare_h20_tokenizer_check.py` and `test/unit/utils/test_hexgent_h20_tokenizer_check.py`; metadata and recipe go under V8/tokenizer-preparation.

1. Inspect the exact server loading, empty-string special-token probe, normal render and encode paths without importing providers.
2. Write fake-tokenizer tests for exact token-ID equality, both server special-token branches, all75-call/ID/target/context checks, failure without clipping or altered histories, and no provider import at module load. Root records RED.
3. Implement the narrow comparison and future explicit CLI. It records counts and hashes, never prompt/tokenizer payloads.
4. Freeze RO resolved tokenizer/config file paths and hashes, image/source/input identities, exact command, UID1002, CPUs1,3/quota2,4GiB,120 seconds,8MiB total evidence, networknone and no GPU/weights/downloads.
5. Individual installed-provider hashes are not currently available from existing evidence. The recipe must expose that gap and propose immutable-image pinning plus stdlib pre-import identity capture for main review; it must not invent provider identity or launch a metadata container.

### Task 5: Consolidated verification and review

Root runs the relevant combined guarded suite once all changed sources are frozen, checks the complete diff, then arranges independent SPEC followed by QUALITY review. Confirm canonical a064 and accepted V7 observation behavior remain intact. Preserve every failed run. Freeze source commit, exact recipes, old constructor inputs, SG artifacts, source hashes, command arguments, planned output paths and original deadlines in one packet. Main reviews the consolidated packet before any CPU tokenizer container or GPU startup execution. Real subset metrics and matched simulator error remain subsequent goal requirements.
