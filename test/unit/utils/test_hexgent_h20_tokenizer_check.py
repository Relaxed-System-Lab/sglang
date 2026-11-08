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
