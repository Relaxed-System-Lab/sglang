"""HTTP body-binding regressions with the real BaseReq/msgspec schemas.

These synthetic tests exercise FastAPI routing and validation through TestClient.
GPU runtime imports are avoided by extracting the actual source definitions.
Only tokenizer IPC replies are substituted; no handler is called directly.
"""

import ast
import copy
import importlib.util
import json
import logging
import os
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, List, Optional

import msgspec
import pytest
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel


_ROOT = Path(__file__).resolve().parents[3]
_PYTHON = _ROOT / "python" / "sglang" / "srt"
_HTTP_SOURCE = Path(os.environ.get(
    "HEXGENT_HTTP_SERVER_SOURCE", str(_PYTHON / "entrypoints" / "http_server.py")
))


def _exec_definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        copy.deepcopy(node) for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    assert {node.name for node in selected} == set(names)
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)


def _http_app(outcome):
    utility_path = _PYTHON / "utils" / "msgspec_utils.py"
    spec = importlib.util.spec_from_file_location("hexgent_http_msgspec_utils", utility_path)
    utility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utility)
    namespace = {
        "__name__": __name__,
        "msgspec": msgspec,
        "Optional": Optional,
        "List": List,
        "Any": Any,
        "Annotated": Annotated,
        "msgspec_struct_pydantic_core_schema": utility.msgspec_struct_pydantic_core_schema,
    }
    _exec_definitions(_PYTHON / "managers" / "io_struct.py", [
        "BaseReq",
        "HexgentUpdateWaitingPriorityReqInput", "HexgentReassignWaitingReqInput",
        "HexgentUpdateWaitingPriorityReqOutput", "HexgentReassignWaitingReqOutput",
    ], namespace)
    # Retain the real validation error envelope as well as the real routes.
    namespace["BaseModel"] = BaseModel
    _exec_definitions(_PYTHON / "entrypoints" / "openai" / "protocol.py",
                      ["ErrorResponse"], namespace)
    calls = []

    def replies(obj, output_name):
        calls.append(obj)
        output = namespace[output_name]
        first = dict(
            success=False, message="rank0 rejected", reason_code=outcome if outcome != "success" else "stale_revision",
            current_revision=None if outcome == "not_found" else 8,
        )
        second = dict(
            success=outcome == "success", message="rank1 applied" if outcome == "success" else "rank1 rejected",
            reason_code=outcome, current_revision=None if outcome == "not_found" else 9,
            applied_revision=9 if outcome == "success" else None,
        )
        if output_name == "HexgentUpdateWaitingPriorityReqOutput":
            first["queue_name"] = second["queue_name"] = obj.queue_name
        else:
            first["target_queue_name"] = second["target_queue_name"] = obj.target_queue_name
            first["source_queue_names"] = second["source_queue_names"] = ["decode_waiting"]
        return [output(**first), output(**second)]

    async def update(obj):
        return replies(obj, "HexgentUpdateWaitingPriorityReqOutput")

    async def reassign(obj):
        return replies(obj, "HexgentReassignWaitingReqOutput")

    manager = SimpleNamespace(
        server_args=SimpleNamespace(port=37100),
        update_hexgent_waiting_priority=update,
        reassign_hexgent_waiting=reassign,
    )
    namespace.update({
        "app": FastAPI(), "Body": Body, "Request": Request,
        "RequestValidationError": RequestValidationError, "HTTPException": HTTPException,
        "HTTPStatus": HTTPStatus,
        "ORJSONResponse": ORJSONResponse, "json": json,
        "logger": logging.getLogger("hexgent_worker_http_body_test"),
        "_global_state": SimpleNamespace(tokenizer_manager=manager),
        "msgspec_to_builtins": utility.msgspec_to_builtins,
    })
    _exec_definitions(_HTTP_SOURCE, [
        "validation_exception_handler", "_log_hexgent_mutation_outcome",
        "update_hexgent_waiting_priority", "reassign_hexgent_waiting",
    ], namespace)
    return namespace["app"], namespace, calls, utility.msgspec_to_builtins


_ENDPOINTS = [
    ("/hexgent/update_waiting_priority", "HexgentUpdateWaitingPriorityReqInput",
     {"queue_name": "prefill_waiting"}, "priority_update", "prefill_waiting"),
    ("/hexgent/reassign_waiting", "HexgentReassignWaitingReqInput",
     {"target_queue_name": "decode_waiting", "requeue_locally": False},
     "reassign_waiting", "decode_waiting"),
]


def _request_body(extra, request_id="synthetic-request"):
    return dict(
        request_id=request_id, priority=7, schedule_revision=9, expected_revision=8,
        planned_decode_instance="d0", rid="synthetic-ipc-rid", http_worker_ipc=None,
        **extra,
    )


@pytest.mark.parametrize("endpoint,input_name,extra,action,queue", _ENDPOINTS)
@pytest.mark.parametrize("outcome", ["success", "stale_revision", "not_found"])
def test_hexgent_mutation_json_reaches_typed_request_and_preserves_tp_reply_and_log(
    endpoint, input_name, extra, action, queue, outcome, caplog,
):
    app, namespace, calls, to_builtins = _http_app(outcome)
    body = _request_body(extra, "missing-business-id" if outcome == "not_found" else "synthetic-request")
    with caplog.at_level(logging.INFO, logger="hexgent_worker_http_body_test"):
        with TestClient(app) as client:
            response = client.post(endpoint, json=body)
    assert response.status_code == 200, response.text
    assert len(calls) == 1
    obj = calls[0]
    assert type(obj) is namespace[input_name]
    assert isinstance(obj, namespace["BaseReq"])
    for field, value in body.items():
        assert getattr(obj, field) == value
    # Recreate the schema's expected TP response objects independently of HTTP.
    output_name = input_name.replace("Input", "Output")
    output_class = namespace[output_name]
    first = dict(success=False, message="rank0 rejected",
                 reason_code=outcome if outcome != "success" else "stale_revision",
                 current_revision=None if outcome == "not_found" else 8)
    second = dict(success=outcome == "success",
                  message="rank1 applied" if outcome == "success" else "rank1 rejected",
                  reason_code=outcome, current_revision=None if outcome == "not_found" else 9,
                  applied_revision=9 if outcome == "success" else None)
    if "queue_name" in extra:
        first["queue_name"] = second["queue_name"] = queue
    else:
        first["target_queue_name"] = second["target_queue_name"] = queue
        first["source_queue_names"] = second["source_queue_names"] = ["decode_waiting"]
    expected = {
        "results": [to_builtins(output_class(**first)), to_builtins(output_class(**second))],
        "success": outcome == "success",
    }
    assert response.json() == expected
    records = [record.getMessage() for record in caplog.records
               if record.getMessage().startswith("HexgentMutationOutcome ")]
    assert len(records) == 1
    assert json.loads(records[0].split(" ", 1)[1]) == {
        "request_id": body["request_id"], "action_type": action, "queue": queue,
        "schedule_revision": 9, "expected_revision": 8, "worker_port": 37100,
        **expected,
    }


@pytest.mark.parametrize("endpoint,input_name,extra,action,queue", _ENDPOINTS)
@pytest.mark.parametrize("invalid", ["missing_required", "invalid_type"])
def test_hexgent_mutation_invalid_json_is_rejected_before_ipc(
    endpoint, input_name, extra, action, queue, invalid, caplog,
):
    app, namespace, calls, _ = _http_app("success")
    body = _request_body(extra)
    if invalid == "missing_required":
        del body["request_id"]
    else:
        body["priority"] = "not-an-integer"
    with caplog.at_level(logging.INFO, logger="hexgent_worker_http_body_test"):
        with TestClient(app) as client:
            response = client.post(endpoint, json=body)
    assert 400 <= response.status_code < 500
    assert calls == []
    assert not any(record.getMessage().startswith("HexgentMutationOutcome ")
                   for record in caplog.records)
