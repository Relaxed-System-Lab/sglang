import os

from scripts.hexgent_real_trace_collectors.lats_split_requests import (
    build_chat_completion_requests,
    patch_lats_rollout_depth_source,
    patch_lats_models_source,
)


def test_build_chat_completion_requests_keeps_single_batched_request_by_default(monkeypatch):
    monkeypatch.delenv("HEXGENT_SPLIT_N_REQUESTS", raising=False)
    base = {"model": "m", "messages": [{"role": "user", "content": "q"}], "n": 3}

    requests = build_chat_completion_requests(
        base,
        workflow_id="hotpotqa_0",
        call_index_start=7,
        requested_n=3,
    )

    assert len(requests) == 1
    assert requests[0]["n"] == 3
    assert requests[0]["metadata"]["hexgent"] == {
        "workflow_id": "hotpotqa_0",
        "call_id": "hotpotqa_0-call-7",
        "turn_index": 7,
    }
    assert "metadata" not in base


def test_build_chat_completion_requests_can_keep_batch_and_mark_choice_split(monkeypatch):
    monkeypatch.delenv("HEXGENT_SPLIT_N_REQUESTS", raising=False)
    monkeypatch.setenv("HEXGENT_RECORD_N_CHOICES", "1")
    base = {"model": "m", "messages": [{"role": "user", "content": "q"}], "n": 5}

    requests = build_chat_completion_requests(
        base,
        workflow_id="hotpotqa_0",
        call_index_start=12,
        requested_n=5,
    )

    assert len(requests) == 1
    assert requests[0]["n"] == 5
    assert requests[0]["metadata"]["hexgent"] == {
        "workflow_id": "hotpotqa_0",
        "call_id": "hotpotqa_0-call-12",
        "turn_index": 12,
        "fanout_group_id": "hotpotqa_0-fanout-12",
        "fanout_width": 5,
        "fanout_index": 0,
        "logical_call_count": 5,
        "split_batched_choices": True,
        "parent_call_id": "hotpotqa_0-call-11",
    }


def test_build_chat_completion_requests_splits_n_into_fanout_requests(monkeypatch):
    monkeypatch.setenv("HEXGENT_SPLIT_N_REQUESTS", "1")
    base = {"model": "m", "messages": [{"role": "user", "content": "q"}], "n": 5}

    requests = build_chat_completion_requests(
        base,
        workflow_id="hotpotqa_0",
        call_index_start=12,
        requested_n=5,
    )

    assert len(requests) == 5
    fanout_group_ids = {
        request["metadata"]["hexgent"]["fanout_group_id"] for request in requests
    }
    assert fanout_group_ids == {"hotpotqa_0-fanout-12"}
    assert [request["n"] for request in requests] == [1, 1, 1, 1, 1]
    assert [
        request["metadata"]["hexgent"]["call_id"] for request in requests
    ] == [f"hotpotqa_0-call-{i}" for i in range(12, 17)]
    assert [
        request["metadata"]["hexgent"]["fanout_index"] for request in requests
    ] == [0, 1, 2, 3, 4]
    assert {request["metadata"]["hexgent"]["fanout_width"] for request in requests} == {
        5
    }
    assert {
        request["metadata"]["hexgent"]["parent_call_id"] for request in requests
    } == {"hotpotqa_0-call-11"}


def test_patch_lats_models_source_delegates_chat_requests_to_split_helper():
    source = '''
def chatgpt(messages, model="gpt-3.5-turbo", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        request_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": cnt,
            "stop": stop,
        }
        workflow_id = os.getenv("HEXGENT_WORKFLOW_ID", "")
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
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs
'''

    patched = patch_lats_models_source(source)

    assert "build_chat_completion_requests" in patched
    assert "for request_kwargs in request_kwargs_list:" in patched
    assert "HEXGENT_CALL_COUNTER" in patched
    assert "logical_call_count" in patched
    assert patched == patch_lats_models_source(patched)


def test_patch_lats_models_source_upgrades_old_split_patch_counter_logic():
    source = '''
import os
from hexgent_real_trace_collectors.lats_split_requests import build_chat_completion_requests  # HEXGENT_SPLIT_REQUESTS_PATCH

def chatgpt(messages, model="gpt-3.5-turbo", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        request_kwargs = {"model": model, "messages": messages, "n": cnt}
        workflow_id = os.getenv("HEXGENT_WORKFLOW_ID", "")
        call_index = int(os.getenv("HEXGENT_CALL_COUNTER", "0") or "0")
        request_kwargs_list = build_chat_completion_requests(
            request_kwargs,
            workflow_id=workflow_id,
            call_index_start=call_index,
            requested_n=cnt,
        )
        if workflow_id:
            os.environ["HEXGENT_CALL_COUNTER"] = str(call_index + len(request_kwargs_list))
        for request_kwargs in request_kwargs_list:
            completions_with_backoff(**request_kwargs)
'''

    patched = patch_lats_models_source(source)

    assert "logical_call_count" in patched
    assert "len(request_kwargs_list)" not in patched
    assert patched == patch_lats_models_source(patched)


def test_patch_lats_rollout_depth_source_uses_env_depth_and_is_idempotent():
    source = '''import itertools
import numpy as np

def lats_search(args, task, idx):
    for i in range(iterations):
        expand_node(node, args, task)
        while node.is_terminal or not node.children:
            logging.info(f"Depth limit node found at iteration {i + 1}, reselecting...")
            node = select_node(root)
            expand_node(node, args, task)
        reward, terminal_node = rollout(max(node.children, key=lambda child: child.value), args, task, idx, max_depth=4)
    return reward, terminal_node

def expand_node(node, args, task):
    if node.depth >= 7:
        logging.info("Depth limit reached")
        print("Depth limit reached")
        node.is_terminal = True
        return
    new_nodes = generate_new_states(node, args, task, args.n_generate_sample)
    node.children.extend(new_nodes)
'''

    patched = patch_lats_rollout_depth_source(source)

    assert "import os" in patched
    assert "HEXGENT_LATS_ROLLOUT_MAX_DEPTH" in patched
    assert "HEXGENT_LATS_MAX_DEPTH" in patched
    assert "HEXGENT_LATS_MAX_EMPTY_EXPANSION_RETRIES" in patched
    assert 'os.getenv("HEXGENT_LATS_ROLLOUT_MAX_DEPTH",' in patched
    assert "max_depth=rollout_max_depth" in patched
    assert "max_depth=4" not in patched
    assert "node.depth >= lats_max_depth" in patched
    assert "node.depth >= 7" not in patched
    assert "No valid child states generated" in patched
    assert "empty_expansion_count >= empty_expansion_retries" in patched
    assert patched == patch_lats_rollout_depth_source(patched)


def test_patch_lats_rollout_depth_source_upgrades_old_rollout_only_patch():
    source = '''import itertools
import os

def lats_search(args, task, idx):
    for i in range(iterations):
        expand_node(node, args, task)
        while node.is_terminal or not node.children:
            logging.info(f"Depth limit node found at iteration {i + 1}, reselecting...")
            node = select_node(root)
            expand_node(node, args, task)
        rollout_max_depth = int(os.getenv("HEXGENT_LATS_ROLLOUT_MAX_DEPTH", "4") or "4")
        reward, terminal_node = rollout(max(node.children, key=lambda child: child.value), args, task, idx, max_depth=rollout_max_depth)  # HEXGENT_LATS_ROLLOUT_DEPTH_PATCH
    return reward, terminal_node

def expand_node(node, args, task):
    if node.depth >= 7:
        logging.info("Depth limit reached")
        print("Depth limit reached")
        node.is_terminal = True
        return
    new_nodes = generate_new_states(node, args, task, args.n_generate_sample)
    node.children.extend(new_nodes)
'''

    patched = patch_lats_rollout_depth_source(source)

    assert "HEXGENT_LATS_ROLLOUT_DEPTH_PATCH" in patched
    assert "HEXGENT_LATS_EXPAND_DEPTH_PATCH" in patched
    assert "HEXGENT_LATS_EMPTY_EXPANSION_PATCH" in patched
    assert "node.depth >= 7" not in patched
    assert patched == patch_lats_rollout_depth_source(patched)
