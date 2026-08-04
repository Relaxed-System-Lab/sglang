from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


JsonDict = Dict[str, Any]


def build_phase_profiles(
    *,
    profile_dir: Path,
    model: str,
    worker_class: str,
    sources: Iterable[Tuple[str, Path]],
    output_tag: str,
) -> Dict[str, JsonDict]:
    all_prefill: List[JsonDict] = []
    all_decode: List[JsonDict] = []
    summary: Dict[str, JsonDict] = {}
    profile_dir.mkdir(parents=True, exist_ok=True)

    for name, path in sources:
        rows = _read_jsonl(path)
        rows = [_normalize_for_profile(row) for row in rows]
        good = [row for row in rows if _valid(row)]
        cleaned_path = profile_dir / f"raw_{name}_{output_tag}.clean.jsonl"
        prefill_path = profile_dir / f"prefill_{name}_{output_tag}.jsonl"
        decode_path = profile_dir / f"decode_{name}_{output_tag}.jsonl"
        _write_jsonl(cleaned_path, good)

        prefill = [_prefill_row(row, model=model, worker_class=worker_class) for row in good]
        decode = [_decode_row(row, model=model, worker_class=worker_class) for row in good]
        _write_jsonl(prefill_path, prefill)
        _write_jsonl(decode_path, decode)
        all_prefill.extend(prefill)
        all_decode.extend(decode)

        summary[name] = {
            "raw_rows": len(rows),
            "clean_rows": len(good),
            "dropped_rows": len(rows) - len(good),
            "workflows": len({row.get("workflow_id") for row in good}),
            "prefill_path": str(prefill_path),
            "decode_path": str(decode_path),
            "cleaned_path": str(cleaned_path),
        }

    _write_jsonl(profile_dir / f"prefill_all_{output_tag}.jsonl", all_prefill)
    _write_jsonl(profile_dir / f"decode_all_{output_tag}.jsonl", all_decode)
    (profile_dir / f"profile_summary_{output_tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build HexAgenT prefill/decode phase profiles from raw trace JSONL.")
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--worker-class", required=True)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Trace source in name=/path/to/raw.jsonl format. May be repeated.",
    )
    args = parser.parse_args(argv)
    sources = [_parse_source(source) for source in args.source]
    summary = build_phase_profiles(
        profile_dir=Path(args.profile_dir),
        model=args.model,
        worker_class=args.worker_class,
        sources=sources,
        output_tag=args.output_tag,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        "all_prefill",
        sum(int(item["clean_rows"]) for item in summary.values()),
        "all_decode",
        sum(int(item["clean_rows"]) for item in summary.values()),
    )
    return 0


def _parse_source(source: str) -> Tuple[str, Path]:
    if "=" not in source:
        raise SystemExit(f"--source must be name=path, got: {source}")
    name, path = source.split("=", 1)
    if not name or not path:
        raise SystemExit(f"--source must be name=path, got: {source}")
    return name, Path(path)


def _read_jsonl(path: Path) -> List[JsonDict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: List[JsonDict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _valid(row: JsonDict) -> bool:
    if row.get("error"):
        return False
    ttft = row.get("ttft_ms")
    total = row.get("total_latency_ms")
    if ttft is None or total is None:
        return False
    if float(total) < float(ttft):
        return False
    if row.get("input_tokens") is None or row.get("output_tokens") is None:
        return False
    return True


def _normalize_for_profile(row: JsonDict) -> JsonDict:
    row = dict(row)
    if (
        row.get("ttft_ms") is None
        and row.get("total_latency_ms") is not None
        and int(row.get("output_tokens") or 0) <= 1
        and not row.get("error")
    ):
        row["ttft_ms"] = row.get("total_latency_ms")
    return row


def _base(row: JsonDict, *, model: str, worker_class: str) -> JsonDict:
    input_len = int(row.get("input_tokens") or 0)
    output_len = int(row.get("output_tokens") or 0)
    return {
        "workflow_id": row.get("workflow_id"),
        "call_id": row.get("call_id"),
        "dataset": row.get("dataset"),
        "model": model,
        "worker_class": worker_class,
        "input_len": input_len,
        "output_len": output_len,
        "trial": 0,
        "ttft_ms": float(row.get("ttft_ms") or 0.0),
        "total_latency_ms": float(row.get("total_latency_ms") or 0.0),
        "output_tokens": output_len,
        "error": None,
    }


def _prefill_row(row: JsonDict, *, model: str, worker_class: str) -> JsonDict:
    out = _base(row, model=model, worker_class=worker_class)
    out["phase"] = "prefill"
    out["prefill_time_ms"] = out["ttft_ms"]
    out["timing_source"] = "proxy_stream_ttft_ms"
    return out


def _decode_row(row: JsonDict, *, model: str, worker_class: str) -> JsonDict:
    out = _base(row, model=model, worker_class=worker_class)
    decode_ms = max(0.0, out["total_latency_ms"] - out["ttft_ms"])
    decode_tokens = max(0, int(out["output_tokens"]) - 1)
    per_token = decode_ms / decode_tokens if decode_tokens > 0 else 0.0
    out["phase"] = "decode"
    out["decode_time_ms"] = decode_ms
    out["decode_time_per_token_ms"] = per_token
    out["decode_p50_ms"] = per_token
    out["decode_p95_ms"] = per_token
    out["timing_source"] = "wall_clock_total_latency_minus_ttft_ms"
    out["stream_timing_granularity"] = "request_total"
    return out


if __name__ == "__main__":
    raise SystemExit(main())
