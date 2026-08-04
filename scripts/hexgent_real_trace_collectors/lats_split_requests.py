from __future__ import annotations

import os
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional


JsonDict = Dict[str, Any]
IMPORT_LINE = (
    "from hexgent_real_trace_collectors.lats_split_requests import "
    "build_chat_completion_requests  # HEXGENT_SPLIT_REQUESTS_PATCH\n"
)
ROLLOUT_DEPTH_MARKER = "HEXGENT_LATS_ROLLOUT_DEPTH_PATCH"
EXPAND_DEPTH_MARKER = "HEXGENT_LATS_EXPAND_DEPTH_PATCH"
EMPTY_EXPANSION_MARKER = "HEXGENT_LATS_EMPTY_EXPANSION_PATCH"
OLD_COUNTER_LINE = (
    '            os.environ["HEXGENT_CALL_COUNTER"] = '
    "str(call_index + len(request_kwargs_list))"
)
LOGICAL_COUNTER_BLOCK = '''            logical_call_count = sum(
                int(
                    ((request_kwargs.get("metadata") or {}).get("hexgent") or {}).get(
                        "logical_call_count", 1
                    )
                )
                for request_kwargs in request_kwargs_list
            )
            os.environ["HEXGENT_CALL_COUNTER"] = str(call_index + logical_call_count)'''


def build_chat_completion_requests(
    base_request_kwargs: JsonDict,
    *,
    workflow_id: str,
    call_index_start: int,
    requested_n: int,
) -> List[JsonDict]:
    """Build one or more OpenAI-compatible requests for a LATS expansion.

    LATS asks the model for ``n`` candidate trajectories at a tree node. The
    default OpenAI API path encodes this as one request with ``n > 1``. For
    HexAgenT scheduling experiments we need each candidate trajectory to be a
    separate logical LLM call, so ``HEXGENT_SPLIT_N_REQUESTS=1`` converts that
    batched request into ``n`` independent requests with fan-out metadata.
    """

    if requested_n <= 1:
        request = deepcopy(base_request_kwargs)
        _attach_hexgent_metadata(
            request,
            workflow_id=workflow_id,
            call_index=call_index_start,
        )
        return [request]

    if _record_choices_enabled() and not _split_enabled():
        request = deepcopy(base_request_kwargs)
        _attach_hexgent_metadata(
            request,
            workflow_id=workflow_id,
            call_index=call_index_start,
            fanout_group_id=f"{workflow_id}-fanout-{call_index_start}",
            fanout_index=0,
            fanout_width=requested_n,
            logical_call_count=requested_n,
            split_batched_choices=True,
        )
        return [request]

    if not _split_enabled():
        request = deepcopy(base_request_kwargs)
        _attach_hexgent_metadata(
            request,
            workflow_id=workflow_id,
            call_index=call_index_start,
        )
        return [request]

    fanout_group_id = f"{workflow_id}-fanout-{call_index_start}"
    requests = []
    for fanout_index in range(requested_n):
        request = deepcopy(base_request_kwargs)
        request["n"] = 1
        _attach_hexgent_metadata(
            request,
            workflow_id=workflow_id,
            call_index=call_index_start + fanout_index,
            fanout_group_id=fanout_group_id,
            fanout_index=fanout_index,
            fanout_width=requested_n,
        )
        requests.append(request)
    return requests


def _split_enabled() -> bool:
    return os.getenv("HEXGENT_SPLIT_N_REQUESTS", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _record_choices_enabled() -> bool:
    return os.getenv("HEXGENT_RECORD_N_CHOICES", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _attach_hexgent_metadata(
    request: JsonDict,
    *,
    workflow_id: str,
    call_index: int,
    fanout_group_id: Optional[str] = None,
    fanout_index: Optional[int] = None,
    fanout_width: Optional[int] = None,
    logical_call_count: Optional[int] = None,
    split_batched_choices: bool = False,
) -> None:
    if not workflow_id:
        return
    hexgent = {
        "workflow_id": workflow_id,
        "call_id": f"{workflow_id}-call-{call_index}",
        "turn_index": call_index,
    }
    if fanout_group_id is not None:
        parent_call_id = (
            f"{workflow_id}-call-{call_index - fanout_index - 1}"
            if fanout_index is not None and call_index - fanout_index > 0
            else None
        )
        hexgent.update(
            {
                "fanout_group_id": fanout_group_id,
                "fanout_index": fanout_index,
                "fanout_width": fanout_width,
            }
        )
        if logical_call_count is not None:
            hexgent["logical_call_count"] = logical_call_count
        if split_batched_choices:
            hexgent["split_batched_choices"] = True
        if parent_call_id is not None:
            hexgent["parent_call_id"] = parent_call_id
    request["metadata"] = {"hexgent": hexgent}


def patch_lats_models_source(source: str) -> str:
    """Patch LATS hotpot/models.py so n-way samples become independent calls.

    The patch is intentionally small and idempotent: it keeps the LATS API
    unchanged while replacing the per-chunk ChatCompletion call with a list of
    one or more requests generated by ``build_chat_completion_requests``.
    """

    if "HEXGENT_SPLIT_REQUESTS_PATCH" in source:
        return source.replace(OLD_COUNTER_LINE, LOGICAL_COUNTER_BLOCK, 1)

    if IMPORT_LINE.strip() not in source:
        source = _insert_import(source)

    old = '''        workflow_id = os.getenv("HEXGENT_WORKFLOW_ID", "")
        if workflow_id:
            call_index = int(os.getenv("HEXGENT_CALL_COUNTER", "0") or "0")
            os.environ["HEXGENT_CALL_COUNTER"] = str(call_index + 1)
            request_kwargs["metadata"] = {
                "hexgent": {
                    "workflow_id": workflow_id,
                    "call_id": f"{workflow_id}-call-{call_index}",
                    "turn_index": call_index,
                }
            }
        res = completions_with_backoff(**request_kwargs)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]'''
    new = '''        workflow_id = os.getenv("HEXGENT_WORKFLOW_ID", "")
        call_index = int(os.getenv("HEXGENT_CALL_COUNTER", "0") or "0")
        request_kwargs_list = build_chat_completion_requests(
            request_kwargs,
            workflow_id=workflow_id,
            call_index_start=call_index,
            requested_n=cnt,
        )
        if workflow_id:
            logical_call_count = sum(
                int(
                    ((request_kwargs.get("metadata") or {}).get("hexgent") or {}).get(
                        "logical_call_count", 1
                    )
                )
                for request_kwargs in request_kwargs_list
            )
            os.environ["HEXGENT_CALL_COUNTER"] = str(call_index + logical_call_count)
        for request_kwargs in request_kwargs_list:
            res = completions_with_backoff(**request_kwargs)
            outputs.extend([choice["message"]["content"] for choice in res["choices"]])
            # log completion tokens
            completion_tokens += res["usage"]["completion_tokens"]
            prompt_tokens += res["usage"]["prompt_tokens"]'''
    if old not in source:
        raise ValueError("Could not find the expected LATS chatgpt request block")
    return source.replace(old, new, 1)


def patch_lats_models_file(models_path: Path) -> None:
    source = models_path.read_text(encoding="utf-8")
    patched = patch_lats_models_source(source)
    if patched == source:
        return
    backup_path = models_path.with_suffix(models_path.suffix + ".hexgent_bak")
    if not backup_path.exists():
        backup_path.write_text(source, encoding="utf-8")
    models_path.write_text(patched, encoding="utf-8")


def patch_lats_rollout_depth_source(source: str) -> str:
    """Patch LATS hotpot/lats.py so search depth is environment-configurable.

    The upstream HotpotQA LATS loop has two independent sources of runaway
    work: rollout depth and repeated expansions that produce no parseable
    children. Patch both so profiling can bound each workflow.
    """

    if "import os" not in source:
        source = _insert_plain_import(source, "import os\n")

    if ROLLOUT_DEPTH_MARKER not in source:
        old = (
            "reward, terminal_node = rollout(max(node.children, key=lambda child: "
            "child.value), args, task, idx, max_depth=4)"
        )
        new = (
            'rollout_max_depth = int(os.getenv("HEXGENT_LATS_ROLLOUT_MAX_DEPTH", '
            'os.getenv("HEXGENT_LATS_MAX_DEPTH", "4")) or "4")\n'
            "        reward, terminal_node = rollout(max(node.children, key=lambda child: "
            f"child.value), args, task, idx, max_depth=rollout_max_depth)  # {ROLLOUT_DEPTH_MARKER}"
        )
        if old not in source:
            raise ValueError("Could not find the expected LATS rollout call")
        source = source.replace(old, new, 1)

    if EXPAND_DEPTH_MARKER not in source:
        old = '''def expand_node(node, args, task):
    if node.depth >= 7:
        logging.info("Depth limit reached")
        print("Depth limit reached")
        node.is_terminal = True
        return
    new_nodes = generate_new_states(node, args, task, args.n_generate_sample)
    node.children.extend(new_nodes)'''
        new = f'''def expand_node(node, args, task):
    lats_max_depth = int(os.getenv("HEXGENT_LATS_MAX_DEPTH", os.getenv("HEXGENT_LATS_ROLLOUT_MAX_DEPTH", "7")) or "7")
    if node.depth >= lats_max_depth:  # {EXPAND_DEPTH_MARKER}
        logging.info("Depth limit reached")
        print("Depth limit reached")
        node.is_terminal = True
        node.exhausted = True
        return
    new_nodes = generate_new_states(node, args, task, args.n_generate_sample)
    if not new_nodes:
        logging.info("No valid child states generated")
        node.exhausted = True
        return
    node.children.extend(new_nodes)'''
        if old not in source:
            raise ValueError("Could not find the expected LATS expand_node block")
        source = source.replace(old, new, 1)

    if EMPTY_EXPANSION_MARKER not in source:
        old = '''        while node.is_terminal or not node.children:
            logging.info(f"Depth limit node found at iteration {i + 1}, reselecting...")
            node = select_node(root)
            expand_node(node, args, task)'''
        new = f'''        empty_expansion_retries = int(os.getenv("HEXGENT_LATS_MAX_EMPTY_EXPANSION_RETRIES", "2") or "2")
        empty_expansion_count = 0
        while node.is_terminal or not node.children:
            if node is None or empty_expansion_count >= empty_expansion_retries:  # {EMPTY_EXPANSION_MARKER}
                logging.info(f"No expandable node found after {{empty_expansion_count}} retries at iteration {{i + 1}}. Ending search.")
                all_nodes_list = collect_all_nodes(root)
                best_child = max(all_nodes_list, key=lambda x: x.reward) if all_nodes_list else root
                return best_child.state, best_child.value, all_nodes, best_child.reward, best_child.em
            logging.info(f"Depth limit or empty expansion found at iteration {{i + 1}}, reselecting...")
            node = select_node(root)
            if node is None:
                empty_expansion_count += 1
                continue
            expand_node(node, args, task)
            if node.is_terminal or not node.children:
                empty_expansion_count += 1'''
        if old not in source:
            raise ValueError("Could not find the expected LATS empty-expansion loop")
        source = source.replace(old, new, 1)

    return source


def patch_lats_rollout_depth_file(lats_path: Path) -> None:
    source = lats_path.read_text(encoding="utf-8")
    patched = patch_lats_rollout_depth_source(source)
    if patched == source:
        return
    backup_path = lats_path.with_suffix(lats_path.suffix + ".hexgent_rollout_bak")
    if not backup_path.exists():
        backup_path.write_text(source, encoding="utf-8")
    lats_path.write_text(patched, encoding="utf-8")


def _insert_import(source: str) -> str:
    return _insert_plain_import(source, IMPORT_LINE)


def _insert_plain_import(source: str, import_line: str) -> str:
    lines = source.splitlines(keepends=True)
    insert_at = 0
    for index, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = index + 1
    lines.insert(insert_at, import_line)
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-path", required=True)
    parser.add_argument("--lats-path")
    args = parser.parse_args()
    patch_lats_models_file(Path(args.models_path))
    if args.lats_path:
        patch_lats_rollout_depth_file(Path(args.lats_path))


if __name__ == "__main__":
    main()
