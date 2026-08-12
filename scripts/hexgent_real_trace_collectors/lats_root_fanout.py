"""Collect LATS HotpotQA root fanout calls through an OpenAI-compatible proxy.

This intentionally does not run the full LATS search tree. It records the first
`n_generate_sample` candidate trajectories for each HotpotQA workflow so service
profile tables can cover more workflow IDs quickly while preserving fanout
metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict


JsonDict = Dict[str, Any]


def _post_json(url: str, payload: JsonDict, timeout_s: int) -> JsonDict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_lats_modules(lats_hotpot_dir: Path):
    sys.path.insert(0, str(lats_hotpot_dir))
    os.chdir(lats_hotpot_dir)
    from hotpotqa import HotPotQATask  # type: ignore
    from lats import env  # type: ignore

    return HotPotQATask, env


def collect_root_fanout(
    *,
    lats_hotpot_dir: Path,
    endpoint: str,
    model: str,
    start_index: int,
    end_index: int,
    n_generate_sample: int,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> None:
    HotPotQATask, env = _load_lats_modules(lats_hotpot_dir)
    task = HotPotQATask()
    url = endpoint.rstrip("/") + "/chat/completions"

    for idx in range(start_index, end_index):
        workflow_id = f"hotpotqa_{idx}"
        question = env.reset(idx=idx)
        prompt = task.cot_prompt_wrap(question, "")
        payload: JsonDict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n_generate_sample,
            "stop": task.stops[0],
            "metadata": {
                "hexgent": {
                    "workflow_id": workflow_id,
                    "call_id": f"{workflow_id}-call-0",
                    "turn_index": 0,
                    "fanout_group_id": f"{workflow_id}-fanout-0",
                    "fanout_index": 0,
                    "fanout_width": n_generate_sample,
                    "logical_call_count": n_generate_sample,
                    "split_batched_choices": True,
                    "root_fanout_only": True,
                }
            },
        }
        response = _post_json(url, payload, timeout_s)
        print(
            f"{idx} workflow={workflow_id} choices={len(response.get('choices') or [])} "
            f"usage={response.get('usage')}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lats-hotpot-dir", type=Path, default=Path("/workspace/LanguageAgentTreeSearch/hotpot"))
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible proxy endpoint, e.g. http://127.0.0.1:18194/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-start-index", type=int, default=0)
    parser.add_argument("--task-end-index", type=int, default=20)
    parser.add_argument("--n-generate-sample", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--timeout-s", type=int, default=1200)
    args = parser.parse_args()
    collect_root_fanout(
        lats_hotpot_dir=args.lats_hotpot_dir,
        endpoint=args.endpoint,
        model=args.model,
        start_index=args.task_start_index,
        end_index=args.task_end_index,
        n_generate_sample=args.n_generate_sample,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
    )


if __name__ == "__main__":
    main()
