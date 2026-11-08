"""Offline contracts for the V8 tokenizer-only comparison; no providers loaded."""
import builtins
import copy
import importlib.util
import io
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/hexgent_real_trace_collectors/prepare_h20_tokenizer_check.py"


def load():
    spec = importlib.util.spec_from_file_location("v8_tokenizer_check_fixture", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def workflows(count=2, output_len=8):
    return [{"workflow_id": "wf", "calls": [
        {"call_id": "call-" + str(index), "input_len": 99, "output_len": output_len,
         "request": {"messages": ([{"role": "assistant", "content": "PRIVATE_PRIOR_ASSISTANT"}] if index == 0 else []) + [{"role": "user", "content": "PRIVATE_PROMPT_PAYLOAD"}]}}
        for index in range(count)]}]


class FakeTokenizer:
    def __init__(self, *, auto_specials=True, probe_error=False, server_ids=None):
        self.auto_specials = auto_specials
        self.probe_error = probe_error
        self.server_ids = [11, 12] if server_ids is None else list(server_ids)
        self.template_calls, self.encode_calls = [], []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((copy.deepcopy(messages), dict(kwargs)))
        return [11, 12] if kwargs["tokenize"] else "PRIVATE_RENDERED_PROMPT"

    def encode(self, text, **kwargs):
        self.encode_calls.append((text, dict(kwargs)))
        if text == "":
            if self.probe_error:
                raise RuntimeError("synthetic encode-empty failure")
            return [128000] if self.auto_specials else []
        if self.auto_specials and kwargs.get("add_special_tokens", True):
            return [128000] + self.server_ids
        return list(self.server_ids)


def test_exact_server_render_and_encode_avoids_double_bos_and_preserves_inputs():
    module = load(); original = workflows(); before = copy.deepcopy(original)
    tokenizer = FakeTokenizer()
    result = module.compare_calls(original, tokenizer, expected_calls=2)
    assert result["status"] == "TOKENIZER_CHECK_PASSED"
    assert result["checked_calls"] == 2 and result["failed_calls"] == 0
    assert tokenizer.encode_calls == [("", {}), ("PRIVATE_RENDERED_PROMPT", {"add_special_tokens": False}),
                                      ("PRIVATE_RENDERED_PROMPT", {"add_special_tokens": False})]
    for index, (messages, kwargs) in enumerate(tokenizer.template_calls):
        assert messages == before[0]["calls"][index // 2]["request"]["messages"]
        assert kwargs == ({"tokenize": True, "add_generation_prompt": True, "return_dict": False}
                          if index % 2 == 0 else
                          {"tokenize": False, "add_generation_prompt": True, "tools": None, "return_dict": False})
    for row in result["calls"]:
        assert row["local_input_tokens"] == row["server_input_tokens"] == 2
        assert row["output_tokens"] == 8 and row["total_tokens"] == 10
        assert row["token_ids_equal"] and row["context_within_limit"]
        assert row["local_ids_sha256"] == row["server_ids_sha256"]
    assert original == before
    assert "PRIVATE_PROMPT" not in json.dumps(result) and "PRIVATE_RENDERED" not in json.dumps(result)


def test_server_without_auto_specials_omits_add_special_tokens_keyword():
    module = load(); tokenizer = FakeTokenizer(auto_specials=False)
    result = module.compare_calls(workflows(1), tokenizer, expected_calls=1)
    assert result["status"] == "TOKENIZER_CHECK_PASSED"
    assert tokenizer.encode_calls == [("", {}), ("PRIVATE_RENDERED_PROMPT", {})]


def test_encode_empty_exception_uses_the_server_specials_default_once():
    module = load(); tokenizer = FakeTokenizer(probe_error=True)
    result = module.compare_calls(workflows(1), tokenizer, expected_calls=1)
    assert result["status"] == "TOKENIZER_CHECK_PASSED"
    assert tokenizer.encode_calls == [("", {}), ("PRIVATE_RENDERED_PROMPT", {"add_special_tokens": False})]


def test_same_length_different_token_ids_fail_the_equivalence_check():
    module = load(); tokenizer = FakeTokenizer(server_ids=[11, 13])
    result = module.compare_calls(workflows(1), tokenizer, expected_calls=1)
    assert result["status"] == "TOKENIZER_CHECK_FAILED" and result["failed_calls"] == 1
    row = result["calls"][0]
    assert row["local_input_tokens"] == row["server_input_tokens"] == 2
    assert row["token_ids_equal"] is False
    assert row["local_ids_sha256"] != row["server_ids_sha256"]


def test_context_cap_is_inclusive_and_never_clips_the_original_target():
    module = load()
    for target, within in ((8190, True), (8191, False)):
        original = workflows(1, output_len=target); before = copy.deepcopy(original)
        result = module.compare_calls(original, FakeTokenizer(), expected_calls=1)
        row = result["calls"][0]
        assert row["output_tokens"] == target and row["total_tokens"] == target + 2
        assert row["context_within_limit"] is within
        assert (result["status"] == "TOKENIZER_CHECK_PASSED") is within
        assert original == before


def test_default_contract_covers_all_seventy_five_calls():
    module = load(); result = module.compare_calls(workflows(75), FakeTokenizer())
    assert result["status"] == "TOKENIZER_CHECK_PASSED"
    assert result["expected_calls"] == result["checked_calls"] == len(result["calls"]) == 75
    assert [row["call_id"] for row in result["calls"]] == ["call-" + str(index) for index in range(75)]


def test_missing_call_cannot_be_reported_as_a_complete_seventy_five_call_check():
    with pytest.raises(ValueError, match="75"):
        load().compare_calls(workflows(74), FakeTokenizer())


def test_default_cli_is_plan_only_and_never_loads_a_provider_or_input():
    output = io.StringIO()
    def forbidden(*args, **kwargs):
        raise AssertionError("default CLI must not load provider or metadata")
    result = load().main([], tokenizer_factory=forbidden, metadata_reader=forbidden, output=output)
    assert result == 0
    plan = json.loads(output.getvalue())
    assert plan["mode"] == "plan" and plan["tokenizer_loaded"] is False
    assert plan["expected_calls"] == 75 and plan["max_context_tokens"] == 8192


def test_import_is_inert_without_transformers_tokenizers_jinja_or_sglang(monkeypatch):
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"transformers", "tokenizers", "jinja2", "sglang", "torch"}:
            raise AssertionError("forbidden provider import: " + name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    module = load()
    assert callable(module.compare_calls) and callable(module.main)


def loader_evidence():
    return {"declared_tokenizer_class": "PreTrainedTokenizerFast",
        "actual_tokenizer_class": "PreTrainedTokenizerFast", "is_fast": True,
        "raw_pre_tokenizer_type": "Sequence", "loaded_pre_tokenizer_type": "Sequence",
        "raw_decoder_type": "ByteLevel", "loaded_decoder_type": "ByteLevel",
        "special_tokens_pattern": None, "cls_token_id": None, "sep_token_id": None}


def test_loader_readiness_requires_the_selected_server_fixes_to_be_noops():
    module=load(); evidence=loader_evidence(); original=copy.deepcopy(evidence)
    result=module.loader_readiness(evidence)
    assert result["status"] == "TOKENIZER_LOADER_READY" and result["ready"] is True
    assert result["reasons"] == [] and evidence == original


@pytest.mark.parametrize("fix", ["component", "special_pattern", "backend_resolution"])
def test_needed_server_postload_fix_is_unknown_and_never_claims_readiness(fix):
    evidence=loader_evidence()
    if fix == "component": evidence["loaded_pre_tokenizer_type"] = "Metaspace"
    elif fix == "special_pattern": evidence["special_tokens_pattern"] = "cls_sep"
    else: evidence["actual_tokenizer_class"] = "TokenizersBackend"
    result=load().loader_readiness(evidence)
    assert result["status"] == "TOKENIZER_LOADER_UNKNOWN" and result["ready"] is False
    assert result["reasons"]


def entry_fixture(tmp_path):
    directory=tmp_path/"tokenizer";directory.mkdir()
    assets={}
    for name in ("config.json","generation_config.json","tokenizer_config.json","special_tokens_map.json","tokenizer.json"):
        data=b"{}"; (directory/name).write_bytes(data)
        assets[name]={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
    trace=tmp_path/"frozen.jsonl"
    data=(json.dumps(workflows()[0])+"\n").encode();trace.write_bytes(data)
    meta={"tokenizer_files":assets,"traces":{trace.name:{"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"calls":2}},
        "driver_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),"reviewed_server_sources":{},
        "provider_identity_before_launch":"unavailable; immutable image plus pre-import capture required"}
    metadata=tmp_path/"metadata.json";raw=json.dumps(meta).encode();metadata.write_bytes(raw)
    argv=["--execute","--trace",str(trace),"--tokenizer-dir",str(directory),"--metadata-file",str(metadata),
          "--metadata-sha256",hashlib.sha256(raw).hexdigest(),"--expected-calls","2"]
    return argv


def test_explicit_entry_records_metadata_before_loading_and_compares_all_frozen_calls(tmp_path):
    events=[];output=io.StringIO();tokenizer=FakeTokenizer()
    def metadata_reader():
        events.append("metadata")
        return {"status":"CAPTURED_PREIMPORT","distributions":{"fixture-provider":{"version":"offline","files":{}}}}
    def tokenizer_factory(directory):
        events.append("tokenizer")
        return tokenizer,loader_evidence()
    code=load().main(entry_fixture(tmp_path),metadata_reader=metadata_reader,tokenizer_factory=tokenizer_factory,output=output)
    result=json.loads(output.getvalue().splitlines()[-1])
    assert code==0 and result["status"]=="TOKENIZER_CHECK_PASSED"
    assert events==["metadata","tokenizer"]
    assert result["checked_calls"]==2 and result["loader_readiness"]["ready"] is True
    assert result["provider_metadata"]["status"]=="CAPTURED_PREIMPORT"
    assert len(result["input_files"])==1+5
    assert "PRIVATE_PROMPT" not in output.getvalue() and "PRIVATE_PRIOR" not in output.getvalue()


def test_explicit_entry_stops_unknown_before_comparison_when_a_server_fix_is_required(tmp_path):
    tokenizer=FakeTokenizer();evidence=loader_evidence();evidence["loaded_pre_tokenizer_type"]="Metaspace"
    output=io.StringIO()
    code=load().main(entry_fixture(tmp_path),metadata_reader=lambda:{"status":"CAPTURED_PREIMPORT"},
        tokenizer_factory=lambda directory:(tokenizer,evidence),output=output)
    result=json.loads(output.getvalue().splitlines()[-1])
    assert code!=0 and result["status"]=="TOKENIZER_CHECK_UNKNOWN"
    assert result["checked_calls"]==0 and result["loader_readiness"]["ready"] is False
    assert tokenizer.encode_calls==[] and tokenizer.template_calls==[]


def test_explicit_entry_metadata_failure_never_loads_the_tokenizer(tmp_path):
    output=io.StringIO()
    def failed_metadata():raise LookupError("fixture missing provider")
    def forbidden(directory):raise AssertionError("metadata failure must precede tokenizer load")
    code=load().main(entry_fixture(tmp_path),metadata_reader=failed_metadata,tokenizer_factory=forbidden,output=output)
    result=json.loads(output.getvalue().splitlines()[-1])
    assert code!=0 and result["status"]=="TOKENIZER_CHECK_UNKNOWN"
    assert result["checked_calls"]==0 and result["phase"]=="provider_metadata"


def test_provider_metadata_hashes_actual_resolved_files_before_any_provider_import(monkeypatch,tmp_path):
    from importlib import metadata
    specs={
        "transformers":{"transformers/__init__.py":b"TRANSFORMERS_REAL_BYTES",
            "transformers/models/auto/tokenization_auto.py":b"AUTO_REAL_BYTES"},
        "tokenizers":{"tokenizers/__init__.py":b"TOKENIZERS_REAL_BYTES",
            "tokenizers/tokenizers.abi3.so":b"EXTENSION_REAL_BYTES"},
        "jinja2":{"jinja2/__init__.py":b"JINJA_REAL_BYTES",
            "jinja2/environment.py":b"ENVIRONMENT_REAL_BYTES"}}
    calls=[]; expected={}
    class Distribution:
        def __init__(self,name):
            self.version="fixture-1.2.3";self.metadata={"Name":name}
            self.files=[];self.directory=tmp_path/name
            payloads={**specs[name],name+"-fixture.dist-info/METADATA":b"Name: fixture\n",
                name+"-fixture.dist-info/RECORD":b"UNTRUSTED_DECLARED_HASH_NOT_ACTUAL_BYTES"}
            expected[name]=payloads
            for relative,data in payloads.items():
                path=self.directory/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                self.files.append(Path(relative))
        def locate_file(self,relative):return self.directory/relative
    distributions={name:Distribution(name) for name in specs}
    def distribution(name):calls.append(name);return distributions[name]
    monkeypatch.setattr(metadata,"distribution",distribution)
    original_import=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split(".")[0] in {"transformers","tokenizers","jinja2","torch","sglang"}:
            raise AssertionError("fingerprinting imported a provider: "+name)
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded)
    result=load()._provider_metadata()
    assert result["status"]=="CAPTURED_PREIMPORT"
    assert calls==["transformers","tokenizers","jinja2"]
    for name,payloads in expected.items():
        actual=result["distributions"][name]
        assert actual["name"]==name and actual["version"]=="fixture-1.2.3"
        assert set(actual["files"])==set(payloads)
        for relative,data in payloads.items():
            row=actual["files"][relative]
            assert row["path"]==str((tmp_path/name/relative).resolve())
            assert row["bytes"]==len(data)
            assert row["sha256"]==hashlib.sha256(data).hexdigest()


def test_preimport_metadata_is_emitted_and_flushed_before_a_fatal_tokenizer_load(tmp_path):
    class Capture(io.StringIO):
        flush_count=0
        def flush(self):self.flush_count+=1;return super().flush()
    class FatalLoad(BaseException):pass
    output=Capture()
    metadata={"status":"CAPTURED_PREIMPORT","distributions":{"fixture-provider":{"version":"actual-fixture","files":{"code":{"sha256":"a"*64}}}}}
    def fatal_factory(directory):
        lines=output.getvalue().splitlines()
        assert len(lines)==1 and output.flush_count>=1
        event=json.loads(lines[0])
        assert event["event"]=="tokenizer_preimport"
        assert event["provider_metadata"]==metadata
        assert event["driver_source"]["sha256"]==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        assert len(event["input_files"])==6
        raise FatalLoad()
    with pytest.raises(FatalLoad):
        load().main(entry_fixture(tmp_path),metadata_reader=lambda:metadata,
            tokenizer_factory=fatal_factory,output=output)
    retained=json.loads(output.getvalue().splitlines()[0])
    assert retained["provider_metadata"]["status"]=="CAPTURED_PREIMPORT"
    assert "PRIVATE_PROMPT" not in output.getvalue()


# V9: provider outcomes below are synthetic I/O. The alias and resolver calls
# are grounded in the five separately captured Transformers 5.12.1 source files.
_BACKEND_MODULE = "transformers.tokenization_utils_tokenizers"
_SERVER_LOADER_SOURCE = "python/sglang/srt/utils/hf_transformers/tokenizer.py"
_SERVER_LOADER_SHA = "ba68d62e62d8d34c762b3d25b2ff8c51dd568e80a5c2a18c0047203a5d5e9096"
_SERVER_KWARGS = {
    "trust_remote_code": False, "tokenizer_revision": None,
    "clean_up_tokenization_spaces": False, "revision": None, "use_fast": True,
}


def backend_pair_fixture(monkeypatch, tmp_path):
    import sys
    import types
    module = load()
    state = types.SimpleNamespace(events=[], retry_error=None, declared_error=None)
    Sequence = type("Sequence", (), {})
    ByteLevel = type("ByteLevel", (), {})
    template = "{{ messages }}"

    class TokenizersBackend(FakeTokenizer):
        def __init__(self):
            super().__init__()
            self.chat_template, self.is_fast = template, True
            self._tokenizer = types.SimpleNamespace(pre_tokenizer=Sequence(), decoder=ByteLevel())
            self.special_tokens_pattern = self.cls_token_id = self.sep_token_id = None

        def get_added_vocab(self):
            return {"<|eom_id|>": 128009}

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            state.events.append(("declared", str(path), dict(kwargs)))
            if state.declared_error is not None:
                raise state.declared_error
            return state.objects["declared"]

    TokenizersBackend.__module__ = _BACKEND_MODULE
    state.backend_class = TokenizersBackend
    state.objects = {name: TokenizersBackend() for name in ("local", "initial", "retry", "declared")}

    class Auto:
        @staticmethod
        def from_pretrained(path, **kwargs):
            names = ("local", "initial", "retry")
            name = names[sum(event[0] != "declared" for event in state.events)]
            state.events.append((name, str(path), dict(kwargs)))
            if name == "retry" and state.retry_error is not None:
                raise state.retry_error
            return state.objects[name]

    provider_code = tmp_path / "tokenization_utils_tokenizers.py"
    provider_code.write_bytes(b"# synthetic provider bytes; never imported\n")
    code = {"path": str(provider_code.resolve()), "bytes": provider_code.stat().st_size,
            "sha256": hashlib.sha256(provider_code.read_bytes()).hexdigest()}
    providers = {"status": "CAPTURED_PREIMPORT", "distributions": {
        "transformers": {"name": "transformers", "version": "5.12.1",
            "files": {"transformers/tokenization_utils_tokenizers.py": code}},
        "tokenizers": {"name": "tokenizers", "version": "0.22.2", "files": {}},
        "jinja2": {"name": "Jinja2", "version": "3.1.6", "files": {}},
    }}
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = Auto
    # Actual provider source tokenization_utils_tokenizers.py:1409-1410.
    tf.PreTrainedTokenizerFast = tf.TokenizersBackend = TokenizersBackend
    backend_module = types.ModuleType(_BACKEND_MODULE)
    backend_module.__file__ = str(provider_code)
    backend_module.PreTrainedTokenizerFast = backend_module.TokenizersBackend = TokenizersBackend
    tk = types.ModuleType("tokenizers")
    tk.Tokenizer = types.SimpleNamespace(from_file=lambda path: types.SimpleNamespace(
        pre_tokenizer=Sequence(), decoder=ByteLevel()))
    for name, value in (("transformers", tf), (_BACKEND_MODULE, backend_module), ("tokenizers", tk)):
        monkeypatch.setitem(sys.modules, name, value)

    directory = tmp_path / "tokenizer"
    directory.mkdir()
    assets = {}
    for name in ("config.json", "generation_config.json", "tokenizer_config.json",
                 "special_tokens_map.json", "tokenizer.json"):
        value = {"tokenizer_class": "PreTrainedTokenizerFast"} if name == "tokenizer_config.json" else {}
        raw = json.dumps(value).encode()
        (directory / name).write_bytes(raw)
        assets[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    trace = tmp_path / "pair-trace.jsonl"
    trace.write_text(json.dumps(workflows(75)[0]) + "\n")
    trace_bytes = trace.read_bytes()
    metadata = {
        "tokenizer_files": assets,
        "traces": {trace.name: {"bytes": len(trace_bytes),
            "sha256": hashlib.sha256(trace_bytes).hexdigest(), "calls": 75}},
        "driver_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "reviewed_server_sources": {_SERVER_LOADER_SOURCE: _SERVER_LOADER_SHA},
        "provider_identity_before_launch": "synthetic frozen provider identities",
        "provider_identity_expected": copy.deepcopy(providers),
        "chat_template": {"sha256": hashlib.sha256(template.encode()).hexdigest()},
    }
    metadata_path = tmp_path / "pair-metadata.json"
    metadata_raw = json.dumps(metadata).encode()
    metadata_path.write_bytes(metadata_raw)
    state.argv = ["--execute", "--trace", str(trace), "--tokenizer-dir", str(directory),
        "--metadata-file", str(metadata_path), "--metadata-sha256", hashlib.sha256(metadata_raw).hexdigest()]
    state.module, state.directory = module, directory
    state.providers, state.metadata, state.provider_code = providers, metadata, provider_code
    return state


def test_backend_pair_uses_exact_local_and_server_resolution_flow(monkeypatch, tmp_path):
    f = backend_pair_fixture(monkeypatch, tmp_path)
    local, server, evidence = f.module._load_tokenizer_pair(f.directory, f.providers, f.metadata)

    assert local is f.objects["local"] and server is f.objects["declared"]
    assert local is not server
    assert type(local) is type(server) is f.backend_class
    retry_kwargs = {**_SERVER_KWARGS, "use_fast": False}
    assert f.events == [
        ("local", str(f.directory), {"local_files_only": True}),
        ("initial", str(f.directory), _SERVER_KWARGS),
        ("retry", str(f.directory), retry_kwargs),
        ("declared", str(f.directory), retry_kwargs),
    ]
    assert f.module.loader_readiness(evidence)["ready"] is True
    assert evidence["actual_tokenizer_code"]["sha256"] == f.providers["distributions"]["transformers"]["files"][
        "transformers/tokenization_utils_tokenizers.py"]["sha256"]
    assert evidence["template_sha256"] == f.metadata["chat_template"]["sha256"]
    assert server.additional_stop_token_ids == {128009}
    assert not hasattr(local, "additional_stop_token_ids")


@pytest.mark.parametrize("error", [ValueError, TypeError, OSError, ImportError, RuntimeError])
def test_backend_retry_exceptions_stop_without_declared_step(monkeypatch, tmp_path, error):
    f = backend_pair_fixture(monkeypatch, tmp_path)
    f.retry_error = error("synthetic retry failed")

    with pytest.raises(RuntimeError):
        f.module._load_tokenizer_pair(f.directory, f.providers, f.metadata)

    assert [event[0] for event in f.events] == ["local", "initial", "retry"]
    assert all(not obj.encode_calls and not obj.template_calls for obj in f.objects.values())


@pytest.mark.parametrize("error", [OSError, ValueError, TypeError, ImportError])
def test_backend_declared_source_catch_list_keeps_retry_instance(monkeypatch, tmp_path, error):
    f = backend_pair_fixture(monkeypatch, tmp_path)
    f.declared_error = error("synthetic declared load failed")
    local, server, evidence = f.module._load_tokenizer_pair(f.directory, f.providers, f.metadata)

    assert local is f.objects["local"] and server is f.objects["retry"]
    assert local is not server and f.module.loader_readiness(evidence)["ready"] is True
    assert [event[0] for event in f.events] == ["local", "initial", "retry", "declared"]


def test_backend_declared_runtime_error_is_not_added_to_the_server_catch_list(monkeypatch, tmp_path):
    f = backend_pair_fixture(monkeypatch, tmp_path)
    f.declared_error = RuntimeError("synthetic declared runtime failure")
    with pytest.raises(RuntimeError, match="synthetic declared runtime failure"):
        f.module._load_tokenizer_pair(f.directory, f.providers, f.metadata)


@pytest.mark.parametrize("change", ["provider", "source", "template", "server_template", "code", "identity", "reuse", "component", "pattern"])
def test_backend_pair_unproven_identity_or_required_postload_fix_stays_unknown(monkeypatch, tmp_path, change):
    f = backend_pair_fixture(monkeypatch, tmp_path)
    if change == "provider":
        f.providers["distributions"]["transformers"]["version"] = "5.12.2"
    elif change == "source":
        f.metadata["reviewed_server_sources"][_SERVER_LOADER_SOURCE] = "0" * 64
    elif change == "template":
        f.metadata["chat_template"]["sha256"] = "0" * 64
    elif change == "server_template":
        f.objects["declared"].chat_template = "{{ changed }}"
    elif change == "code":
        f.provider_code.write_bytes(b"# changed provider bytes\n")
    elif change == "identity":
        # Same class name/module and isinstance relationship are insufficient.
        impostor = type("TokenizersBackend", (f.backend_class,), {"__module__": _BACKEND_MODULE})
        f.objects["declared"] = impostor()
    elif change == "reuse":
        f.objects["declared"] = f.objects["local"]
    elif change == "component":
        f.objects["declared"]._tokenizer.pre_tokenizer = type("Metaspace", (), {})()
    else:
        f.objects["declared"].special_tokens_pattern = "cls_sep"
    try:
        _, _, evidence = f.module._load_tokenizer_pair(f.directory, f.providers, f.metadata)
    except (ValueError, RuntimeError):
        pass
    else:
        result = f.module.loader_readiness(evidence)
        assert result["status"] == "TOKENIZER_LOADER_UNKNOWN" and result["ready"] is False


def test_explicit_pair_detects_id_mismatch_and_checks_all_75_unchanged_targets():
    module = load()
    original = workflows(75)
    before = copy.deepcopy(original)
    local, server = FakeTokenizer(), FakeTokenizer(server_ids=[11, 13])
    result = module.compare_calls(original, local, server_tokenizer=server)
    assert result["status"] == "TOKENIZER_CHECK_FAILED" and result["failed_calls"] == 75
    assert all(row["local_input_tokens"] == row["server_input_tokens"] == 2
               and not row["token_ids_equal"] and row["output_tokens"] == 8 for row in result["calls"])
    assert not local.encode_calls
    assert all(kwargs["tokenize"] is True for _, kwargs in local.template_calls)
    assert all(kwargs["tokenize"] is False for _, kwargs in server.template_calls)
    assert original == before
    for target, passed in ((8190, True), (8191, False)):
        calls = workflows(75, output_len=target)
        saved = copy.deepcopy(calls)
        checked = module.compare_calls(calls, FakeTokenizer(), server_tokenizer=FakeTokenizer())
        assert checked["checked_calls"] == 75
        assert (checked["status"] == "TOKENIZER_CHECK_PASSED") is passed
        assert all(row["output_tokens"] == target and row["total_tokens"] == target + 2 for row in checked["calls"])
        assert calls == saved


def test_default_executing_entry_uses_distinct_backend_pair_and_all_75_calls(monkeypatch, tmp_path):
    f = backend_pair_fixture(monkeypatch, tmp_path)
    output = io.StringIO()
    code = f.module.main(f.argv, metadata_reader=lambda: f.providers, output=output)

    result = json.loads(output.getvalue().splitlines()[-1])
    assert code == 0 and result["status"] == "TOKENIZER_CHECK_PASSED"
    assert result["checked_calls"] == 75 and result["loader_readiness"]["ready"] is True
    assert [event[0] for event in f.events] == ["local", "initial", "retry", "declared"]
    assert len(f.objects["local"].template_calls) == 75
    assert len(f.objects["declared"].template_calls) == 75
    assert not f.objects["local"].encode_calls
    assert "PRIVATE_PROMPT" not in output.getvalue() and "PRIVATE_RENDERED" not in output.getvalue()
