#!/usr/bin/env python3
"""Check frozen ShareGPT tokenization without running an LLM server or model."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path


def _ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def loader_readiness(evidence):
    """Prove the selected input-affecting server post-load fixes are no-ops."""
    reasons = []
    pair = evidence.get("backend_pair") or {}
    paired_backend = (
        pair.get("alias_identity_verified") is True
        and pair.get("distinct_instances") is True
        and pair.get("provider_identity_verified") is True
        and pair.get("provider_version") == "5.12.1"
        and pair.get("server_loader_source_sha256") == _SERVER_LOADER_SHA
        and evidence.get("actual_tokenizer_class") == "TokenizersBackend"
        and pair.get("server", {}).get("actual_tokenizer_class") == "TokenizersBackend"
    )
    if (evidence.get("declared_tokenizer_class") != "PreTrainedTokenizerFast"
            or evidence.get("is_fast") is not True
            or not (paired_backend or evidence.get("actual_tokenizer_class") == "PreTrainedTokenizerFast")):
        reasons.append("selected_generic_fast_loader_not_proven")
    views = [evidence] + ([pair["server"]] if paired_backend else [])
    for view in views:
        raw, loaded = view.get("raw_pre_tokenizer_type"), view.get("loaded_pre_tokenizer_type")
        if not raw or not loaded:
            reasons.append("server_component_fix_noop_not_proven")
        elif raw != loaded:
            reasons.append("server_component_fix_required")
        if (view.get("special_tokens_pattern") == "cls_sep"
                and (view.get("cls_token_id") is None or view.get("sep_token_id") is None)):
            reasons.append("server_special_pattern_fix_required")
    return {"status": "TOKENIZER_LOADER_UNKNOWN" if reasons else "TOKENIZER_LOADER_READY",
            "ready": not reasons, "reasons": reasons, "evidence": copy.deepcopy(evidence)}


def compare_calls(workflows, tokenizer, *, server_tokenizer=None, expected_calls=75, max_context_tokens=8192):
    """Compare exact IDs for the frozen text-only, final-user ShareGPT requests."""
    calls = [(workflow["workflow_id"], call) for workflow in workflows for call in workflow["calls"]]
    if len(calls) != expected_calls:
        raise ValueError(f"expected {expected_calls} calls, got {len(calls)}")
    import time
    started = time.time()
    server_tokenizer = tokenizer if server_tokenizer is None else server_tokenizer
    probe_error = None
    try:
        auto_specials = len(server_tokenizer.encode("")) > 0
    except Exception as error:
        auto_specials = True
        probe_error = type(error).__name__
    encode_kwargs = {"add_special_tokens": False} if auto_specials else {}
    rows = []
    for workflow_id, call in calls:
        messages = call["request"]["messages"]
        local = list(tokenizer.apply_chat_template(copy.deepcopy(messages), tokenize=True,
            add_generation_prompt=True, return_dict=False))
        rendered = server_tokenizer.apply_chat_template(copy.deepcopy(messages), tokenize=False,
            add_generation_prompt=True, tools=None, return_dict=False)
        server = list(server_tokenizer.encode(rendered, **encode_kwargs))
        target = call["output_len"]
        total = len(server) + target
        rows.append({"workflow_id": workflow_id, "call_id": call["call_id"],
            "local_input_tokens": len(local), "server_input_tokens": len(server),
            "output_tokens": target, "total_tokens": total,
            "token_ids_equal": local == server, "context_within_limit": total <= max_context_tokens,
            "local_ids_sha256": _ids_hash(local), "server_ids_sha256": _ids_hash(server)})
    failed = sum(not row["token_ids_equal"] or not row["context_within_limit"] for row in rows)
    return {"status": "TOKENIZER_CHECK_FAILED" if failed else "TOKENIZER_CHECK_PASSED",
        "expected_calls": expected_calls, "checked_calls": len(rows), "failed_calls": failed,
        "max_context_tokens": max_context_tokens, "server_auto_adds_specials": auto_specials,
        "encode_empty_probe_error_type": probe_error, "calls": rows,
        "comparison_started_unix_s": started, "comparison_completed_unix_s": time.time()}



def _fingerprint(path, expected=None):
    path = Path(path)
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            total += len(block); digest.update(block)
    result = {"path": str(path.resolve()), "bytes": total, "sha256": digest.hexdigest()}
    if expected is not None and any(result[key] != expected[key] for key in ("bytes", "sha256")):
        raise ValueError("frozen input fingerprint differs: " + path.name)
    return result




def _provider_metadata():
    """Hash installed distribution bytes using stdlib, without importing them."""
    from importlib import metadata
    wanted = {
        "transformers": {"transformers/__init__.py", "transformers/models/auto/tokenization_auto.py",
            "transformers/tokenization_utils_base.py", "transformers/tokenization_utils_fast.py",
            "transformers/tokenization_utils_tokenizers.py", "transformers/utils/chat_template_utils.py"},
        "tokenizers": {"tokenizers/__init__.py"},
        "jinja2": {"jinja2/__init__.py", "jinja2/environment.py", "jinja2/sandbox.py", "jinja2/runtime.py"}}
    observed = {}
    for name, selected in wanted.items():
        dist = metadata.distribution(name)
        files = {}
        for relative in dist.files or ():
            value = str(relative)
            dist_info = ".dist-info/" in value and value.rsplit("/", 1)[-1] in {"METADATA", "RECORD"}
            extension = name == "tokenizers" and value.startswith("tokenizers/tokenizers.") and value.endswith(".so")
            if value in selected or dist_info or extension:
                files[value] = _fingerprint(Path(dist.locate_file(relative)).resolve(strict=True))
        if name + "/__init__.py" not in files:
            raise ValueError("installed provider path not identified: " + name)
        if name == "tokenizers" and not any(key.endswith(".so") for key in files):
            raise ValueError("installed tokenizer extension not identified")
        observed[name] = {"name": dist.metadata["Name"], "version": dist.version, "files": files}
    return {"status": "CAPTURED_PREIMPORT", "distributions": observed}


_BACKEND_MODULE = "transformers.tokenization_utils_tokenizers"
_SERVER_LOADER_SOURCE = "python/sglang/srt/utils/hf_transformers/tokenizer.py"
_SERVER_LOADER_SHA = "ba68d62e62d8d34c762b3d25b2ff8c51dd568e80a5c2a18c0047203a5d5e9096"


def _tokenizer_state(tokenizer, raw, code, expected_template):
    template = tokenizer.chat_template
    if (not isinstance(template, str)
            or hashlib.sha256(template.encode()).hexdigest() != expected_template):
        raise ValueError("loaded template differs from the frozen embedded template")
    backend = getattr(tokenizer, "_tokenizer", None)
    def kind(value):
        return type(value).__name__ if value is not None else None
    return {
        "actual_tokenizer_class": type(tokenizer).__name__,
        "actual_tokenizer_module": type(tokenizer).__module__,
        "actual_tokenizer_code": code, "is_fast": getattr(tokenizer, "is_fast", False),
        "raw_pre_tokenizer_type": kind(raw.pre_tokenizer),
        "loaded_pre_tokenizer_type": kind(getattr(backend, "pre_tokenizer", None)),
        "raw_decoder_type": kind(raw.decoder),
        "loaded_decoder_type": kind(getattr(backend, "decoder", None)),
        "special_tokens_pattern": getattr(tokenizer, "special_tokens_pattern", None),
        "cls_token_id": getattr(tokenizer, "cls_token_id", None),
        "sep_token_id": getattr(tokenizer, "sep_token_id", None),
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "template_source": "tokenizer_config.json#chat_template",
    }


def _load_tokenizer_pair(directory, providers, metadata):
    """The frozen 5.12.1 generic alias path, not a compatibility resolver."""
    if providers != metadata.get("provider_identity_expected"):
        raise ValueError("frozen installed provider identity differs")
    package = providers["distributions"]["transformers"]
    if (package["version"] != "5.12.1"
            or metadata["reviewed_server_sources"].get(_SERVER_LOADER_SOURCE) != _SERVER_LOADER_SHA):
        raise ValueError("reviewed server/provider source differs")
    config = json.loads((Path(directory) / "tokenizer_config.json").read_bytes())
    if config.get("tokenizer_class") != "PreTrainedTokenizerFast":
        raise ValueError("selected declared tokenizer class differs")

    # Actual provider imports stay after the flushed pre-import event in main.
    from transformers import AutoTokenizer, PreTrainedTokenizerFast, TokenizersBackend
    from tokenizers import Tokenizer as RawTokenizer
    if (PreTrainedTokenizerFast is not TokenizersBackend
            or TokenizersBackend.__module__ != _BACKEND_MODULE
            or TokenizersBackend.__name__ != "TokenizersBackend"):
        raise ValueError("frozen fast/backend alias identity differs")
    code = _fingerprint(sys.modules[_BACKEND_MODULE].__file__)
    if code != package["files"]["transformers/tokenization_utils_tokenizers.py"]:
        raise ValueError("loaded backend code differs from pre-import identity")

    # Exact local preparer call: prepare_h20_runtime_inputs.py:43.
    local = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    # Exact manager/get_tokenizer kwargs for the frozen local model path.
    common = dict(trust_remote_code=False, tokenizer_revision=None,
        clean_up_tokenization_spaces=False, revision=None, use_fast=True)
    try:
        server = AutoTokenizer.from_pretrained(str(directory), **common)
    except TypeError as error:
        raise RuntimeError("server initial tokenizer load failed") from error
    steps = [{"stage": "auto", "class": type(server).__name__}]
    if type(server).__name__ == "TokenizersBackend":
        common = {**common, "use_fast": False}
        try:
            server = AutoTokenizer.from_pretrained(str(directory), **common)
        except (ValueError, TypeError, OSError, ImportError, RuntimeError) as error:
            raise RuntimeError("server tokenizer retry failed") from error
        steps.append({"stage": "retry_use_fast_false", "class": type(server).__name__})
        if type(server).__name__ == "TokenizersBackend":
            # The frozen declared class is the actual provider's alias.
            try:
                declared = PreTrainedTokenizerFast.from_pretrained(str(directory), **common)
            except (OSError, ValueError, TypeError, ImportError):
                declared = None
            server = declared or server
            steps.append({"stage": "declared_alias", "class": type(server).__name__,
                          "kept_retry": declared is None})

    if type(local) is not TokenizersBackend or type(server) is not TokenizersBackend or local is server:
        raise ValueError("distinct frozen backend instances not proven")
    raw = RawTokenizer.from_file(str(Path(directory) / "tokenizer.json"))
    template = metadata["chat_template"]["sha256"]
    local_state = _tokenizer_state(local, raw, code, template)
    server_state = _tokenizer_state(server, raw, code, template)
    # The generic declared class skips BOS/EOS restoration and class-specific
    # Mistral/Kimi patches. Input-affecting fixes must be no-ops below.
    # The remaining stop-ID attachment is metadata only (common.py:455-461).
    added = server.get_added_vocab()
    server.additional_stop_token_ids = {added["<|eom_id|>"]} if "<|eom_id|>" in added else None
    evidence = {
        "declared_tokenizer_class": config["tokenizer_class"], **local_state,
        "backend_pair": {
            "alias_identity_verified": True, "distinct_instances": True,
            "provider_identity_verified": True, "provider_version": package["version"],
            "server_loader_source_sha256": _SERVER_LOADER_SHA,
            "server_resolution_steps": steps, "server": server_state,
        },
    }
    return local, server, evidence


def main(argv=None, *, tokenizer_factory=None, metadata_reader=None, output=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--trace", action="append", default=[])
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--metadata-file")
    parser.add_argument("--metadata-sha256")
    parser.add_argument("--expected-calls", type=int, default=75)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    args = parser.parse_args(argv)
    if not args.execute:
        (output or sys.stdout).write(json.dumps({"mode": "plan", "tokenizer_loaded": False,
            "expected_calls": args.expected_calls, "max_context_tokens": args.max_context_tokens}) + "\n")
        return 0
    report = {"status": "TOKENIZER_CHECK_UNKNOWN", "checked_calls": 0, "calls": [],
              "phase": "input_fingerprints", "tokenizer_loaded": False}
    alarm = False
    emitted = 0
    sink = output or sys.stdout
    def emit(value, *, reserve=0):
        nonlocal emitted
        raw = json.dumps(value, sort_keys=True) + "\n"
        size = len(raw.encode())
        if emitted + size > 8 * 1024**2 - reserve:
            raise ValueError("aggregate JSONL output cap")
        emitted += size
        if sink.write(raw) != len(raw):
            raise OSError("short JSONL evidence write")
        sink.flush()
    try:
        if tokenizer_factory is None and metadata_reader is None:
            import resource, signal
            resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
            def expired(signum, frame):
                raise TimeoutError("original tokenizer CPU deadline reached")
            signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, 90); alarm = True
            for runtime_path in ("/runtime/home", "/runtime/tmp", "/runtime/hf"):
                Path(runtime_path).mkdir(parents=True, exist_ok=True)
        raw_metadata = Path(args.metadata_file).read_bytes()
        if hashlib.sha256(raw_metadata).hexdigest() != args.metadata_sha256:
            raise ValueError("frozen metadata identity differs")
        metadata = json.loads(raw_metadata)
        actual_driver = _fingerprint(Path(__file__))
        if actual_driver["sha256"] != metadata["driver_sha256"]:
            raise ValueError("frozen driver identity differs")
        directory = Path(args.tokenizer_dir)
        inputs = {}
        for name, expected in metadata["tokenizer_files"].items():
            inputs["tokenizer/" + name] = _fingerprint(directory / name, expected)
        workflows = []
        for filename in args.trace:
            path = Path(filename)
            expected = metadata["traces"][path.name]
            inputs["trace/" + path.name] = _fingerprint(path, expected)
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if sum(len(row["calls"]) for row in rows) != expected["calls"]:
                raise ValueError("frozen per-file call coverage differs")
            workflows.extend(rows)
        # The selected SG requests are text-only final-user messages. The normal
        # serving path neither flattens parts nor converts a final assistant.
        for workflow in workflows:
            for call in workflow["calls"]:
                request = call["request"]; messages = request["messages"]
                if (not messages or messages[-1]["role"] != "user"
                        or any(not isinstance(row["content"], str) or row["role"] not in ("system", "user", "assistant") for row in messages)
                        or any(request.get(key) for key in ("tools", "chat_template_kwargs", "continue_final_message", "reasoning_effort", "input_ids"))):
                    raise ValueError("selected request leaves the reviewed normal SG rendering path")
        report.update(input_files=inputs, driver_source=actual_driver,
            reviewed_server_sources=metadata["reviewed_server_sources"],
            provider_identity_before_launch=metadata["provider_identity_before_launch"],
            image_pin=metadata.get("immutable_image"), source_scope="reviewed source fingerprints, not a live server")
        report["phase"] = "provider_metadata"
        providers = (metadata_reader or _provider_metadata)()
        report["provider_metadata"] = providers
        emit({"event": "tokenizer_preimport", **report}, reserve=65536)
        report["phase"] = "tokenizer_load"
        if tokenizer_factory is None:
            tokenizer, server_tokenizer, evidence = _load_tokenizer_pair(directory, providers, metadata)
        else:
            # Explicit offline fixture injection preserves the prior unit API.
            tokenizer, evidence = tokenizer_factory(directory)
            server_tokenizer = None
        report["tokenizer_loaded"] = True
        readiness = loader_readiness(evidence)
        report["loader_readiness"] = readiness
        if readiness["ready"]:
            report["phase"] = "comparison"
            report.update(compare_calls(workflows, tokenizer, server_tokenizer=server_tokenizer,
                expected_calls=args.expected_calls, max_context_tokens=args.max_context_tokens))
            report["phase"] = "complete"
    except Exception as error:
        report["status"] = "TOKENIZER_CHECK_UNKNOWN"
        report["failure_type"] = type(error).__name__
    finally:
        if alarm:
            import signal
            signal.setitimer(signal.ITIMER_REAL, 0)
    encoded = json.dumps(report, sort_keys=True) + "\n"
    if emitted + len(encoded.encode()) > 8 * 1024**2:
        report = {"status": "TOKENIZER_CHECK_UNKNOWN", "checked_calls": 0,
                  "failure_type": "OutputCapReached"}
    emit({"event": "tokenizer_final", **report})
    return 0 if report["status"] == "TOKENIZER_CHECK_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
