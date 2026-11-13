"""Exact-token observations from the worker prefix-cache API.

These observations are planning hints. Runtime admission performs its own match
and locking. No storage query, remote history credit or KV reservation is made.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional

MAX_PREFIX_PROBES = 64
MAX_PREFIX_PROBE_TOKENS = 65_536
MAX_PREFIX_PROBE_TOTAL_TOKENS = 262_144


def validate_prefix_probes(probes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Bound HTTP query work and retain exact tokens, namespace and runtime cap."""
    if not isinstance(probes, list) or len(probes) > MAX_PREFIX_PROBES:
        raise ValueError("prefix_probes must be a list of at most 64 probes")
    total_tokens = 0
    result = []
    for probe in probes:
        if not isinstance(probe, dict):
            raise ValueError("Each prefix probe must be an object")
        token_ids = probe.get("input_ids")
        if not isinstance(token_ids, list) or len(token_ids) > MAX_PREFIX_PROBE_TOKENS:
            raise ValueError("Each prefix probe requires at most 65536 input_ids")
        total_tokens += len(token_ids)
        if total_tokens > MAX_PREFIX_PROBE_TOTAL_TOKENS:
            raise ValueError("prefix_probes exceed the total input token limit")
        if any(type(token) is not int for token in token_ids):
            raise ValueError("input_ids must contain integers")
        cap = probe.get("max_prefix_tokens")
        if type(cap) is not int or not 0 <= cap <= max(0, len(token_ids) - 1):
            raise ValueError("max_prefix_tokens must respect the runtime input-minus-one cap")
        extra_key = probe.get("extra_key")
        if extra_key is not None and not isinstance(extra_key, str):
            raise ValueError("extra_key must be the effective cache namespace or null")
        probe_id = probe.get("probe_id")
        if not isinstance(probe_id, str) or not probe_id:
            raise ValueError("Each prefix probe requires a probe_id")
        result.append(
            dict(probe_id=probe_id, input_ids=token_ids,
                 extra_key=extra_key, max_prefix_tokens=cap)
        )
    return result


def prefix_probe_digest(probe: Dict[str, Any]) -> str:
    """Bind a result to the full submitted input and its effective match key."""
    payload = [probe["input_ids"], probe.get("extra_key"), probe["max_prefix_tokens"]]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def probe_prefix_cache(
    probes: List[Dict[str, Any]], *, cache_enabled: Optional[bool],
    match_prefix: Optional[Callable], page_size: int, now: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Convert actual backend match results without granting storage credit.

    The callback accepts (input_ids, effective_extra_key, max_prefix_tokens) and
    returns the backend MatchResult. Page alignment is performed by that backend.
    """
    observed_at = time.time() if now is None else now
    observations = []
    for probe in probes:
        status = "unknown"
        device_tokens = host_tokens = None
        source = "prefix_cache_unavailable"
        if cache_enabled is False:
            status, device_tokens, host_tokens, source = "disabled", 0, 0, "cache_disabled"
        elif cache_enabled is True and match_prefix is not None:
            result = match_prefix(
                probe["input_ids"], probe["extra_key"], probe["max_prefix_tokens"]
            )
            device_tokens = len(result.device_indices)
            host_tokens = int(result.host_hit_length)
            status, source = "observed", "tree_cache.match_prefix"
        observations.append(dict(
            probe_id=probe["probe_id"], probe_digest=prefix_probe_digest(probe),
            input_tokens=len(probe["input_ids"]), max_prefix_tokens=probe["max_prefix_tokens"],
            status=status, device_prefix_tokens=device_tokens, host_prefix_tokens=host_tokens,
            storage_prefix_tokens=None, source=source, observed_at=observed_at,
            page_size=page_size,
        ))
    return observations


def _request_probe_id(request) -> str:
    return f"{request.hexgent.workflow_id}/{request.hexgent.call_id}"


def observed_prefix_tokens(request) -> List[Optional[int]]:
    """Return exact observed P-device prefixes; unknown replicas grant no credit.

    A URL may serve several DP replicas. Until a replica is bound, its safe
    planning credit is the minimum observation across all reported replicas.
    Disabled caches are known zero even when no exact input IDs were supplied.
    These are backend facts; the caller must separately ensure its service model
    uses the same input-token domain before granting timing credit.
    """
    probe = request.prefix_cache_probe
    probe_id = _request_probe_id(request)
    digest = None
    cap = None
    if isinstance(probe, dict) and probe.get("probe_id") == probe_id:
        token_ids = probe.get("input_ids")
        cap = probe.get("max_prefix_tokens")
        if (isinstance(token_ids, list) and type(cap) is int
                and 0 <= cap <= max(0, len(token_ids) - 1)):
            digest = prefix_probe_digest(probe)

    def device_prefix(snapshot):
        if snapshot.get("mode") not in {None, "prefill", "unified"}:
            return None
        if (snapshot.get("prefix_cache") or {}).get("enabled") is False:
            return 0
        if digest is None:
            return None
        for row in snapshot.get("prefix_observations") or []:
            if row.get("probe_id") != probe_id or row.get("probe_digest") != digest:
                continue
            if row.get("status") == "disabled" and row.get("source") == "cache_disabled":
                return 0
            if (row.get("status") != "observed"
                    or row.get("source") != "tree_cache.match_prefix"):
                continue
            tokens = row.get("device_prefix_tokens")
            if type(tokens) is int and 0 <= tokens <= cap:
                return tokens
        return None

    credits = []
    payloads = request.state.worker_snapshots or {}
    for worker in request.prefill_workers:
        payload = payloads.get(worker.url) or payloads.get(worker.id)
        if not isinstance(payload, dict):
            credits.append(None)
            continue
        if "snapshots" in payload:
            snapshots = payload["snapshots"]
        elif isinstance(payload.get("snapshot"), dict):
            snapshots = [payload["snapshot"]]
        else:
            snapshots = [payload]
        if not snapshots or payload.get("dp_rank_count", len(snapshots)) != len(snapshots):
            credits.append(None)
            continue
        values = [device_prefix(snapshot) for snapshot in snapshots]
        credits.append(None if any(value is None for value in values) else min(values))
    return credits


async def prepare_prefix_cache_probe(request, *, client=None, timeout_s: float = 1.0):
    """Bind a plain-text, unsalted, no-LoRA request to actual backend token IDs.

    Chat uses the backend's original message/tool/template renderer. Native
    /generate text uses its default special-token behavior; explicit input_ids
    bypass rendering as in the runtime. Completion-template and multimodal paths
    remain unknown. Workers in this profile must share the same tokenizer and
    run without LoRA model aliases. No scientific token/service labels change.
    """
    from dataclasses import replace

    unknown = replace(request, prefix_cache_probe=None)
    body = request.request_body
    if not isinstance(body, dict) or not request.prefill_workers:
        return unknown
    namespace_fields = ("cache_salt", "extra_key", "lora_path", "lora_id", "adapter_id")
    multimodal_fields = ("image_data", "audio_data", "video_data", "input_embeds",
                         "input_embeddings", "positional_embed_overrides", "modalities")
    if any(body.get(key) is not None for key in namespace_fields + multimodal_fields):
        return unknown
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            content = message.get("content")
            if isinstance(content, list) and any(
                not isinstance(part, dict) or part.get("type") != "text"
                or not isinstance(part.get("text"), str) for part in content
            ):
                return unknown

    route = request.route.rstrip("/")
    token_ids = body.get("input_ids")
    if route == "/v1/chat/completions":
        if not isinstance(messages, list):
            return unknown
        tokenize_body = dict(body)
    elif route == "/generate":
        if token_ids is None and not isinstance(body.get("text"), str):
            return unknown
        tokenize_body = dict(prompt=body.get("text"), add_special_tokens=True)
        model = body.get("model") or request.model
        if model is not None:
            tokenize_body["model"] = model
    else:
        return unknown

    if token_ids is None:
        url = request.prefill_workers[0].url.rstrip("/") + "/v1/tokenize"
        try:
            if client is None:
                import httpx
                async with httpx.AsyncClient(timeout=timeout_s) as tokenizer_client:
                    response = await tokenizer_client.post(
                        url, json=tokenize_body, timeout=timeout_s)
            else:
                response = await client.post(url, json=tokenize_body, timeout=timeout_s)
            response.raise_for_status()
            token_ids = response.json()["tokens"]
        except Exception:
            # Unavailable or rejected tokenization supplies no cache evidence.
            return unknown
    if not isinstance(token_ids, list):
        return unknown
    cap = max(0, len(token_ids) - 1)
    # The actual chat adapter fixes logprob_start_len=-1. Native /generate
    # exposes the cap used by Req._compute_max_prefix_len instead.
    if route == "/generate" and body.get("return_logprob"):
        start = body.get("logprob_start_len", -1)
        if type(start) is int and start >= 0:
            cap = min(cap, start)
    probe = dict(probe_id=_request_probe_id(request), input_ids=token_ids,
                 extra_key=None, max_prefix_tokens=cap)
    try:
        probe = validate_prefix_probes([probe])[0]
    except ValueError:
        return unknown
    return replace(request, prefix_cache_probe=probe)
