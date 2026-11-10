# H20 Bounded Observer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire the approved private smoke observer cadence and a120s total post-ready limit without GPU execution.
**Architecture:** Preserve the global1200s/180s cleanup clock. Add fixed functional-smoke observation settings to the frozen plan, propagate them to the existing collector, and bound the existing asynchronous post-ready work. Keep public planner and legacy collection unchanged.
**Tech Stack:** Existing Python asyncio driver, collector, pytest fixtures; no new dependency.

1. Add regressions for2s normal/10s error burst, immutable global versus120s active deadline, propagation into the actual collector, and cancellation/drain cleanup; record expected failures before source edits.
2. Patch only `scripts/hexgent_real_trace_collectors/h20_transport_window.py` and `h20_transport_observer.py`, reusing existing timeout and sampling mechanisms. Preserve64MiB observer,512KiB summary,256MiB persistent and exact owner cleanup.
3. Run the new tests plus the focused existing functional-window/collector regression cases, not a complete suite. Review the bounded cancellation behavior independently.
4. Commit a clean local source pin, render the real fixed2P2D/10call configuration with11:00UTC proposed anchor, and update review/config/plan/recipe hashes. Record any expired window as nonlaunchable; do not advance it automatically.

GPU start remains a separate explicit authorization plus source/config review and fresh conflict check. No retry, new workflow, algorithm edit, model download or environment installation.
