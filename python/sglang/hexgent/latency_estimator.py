"""Latency estimators for HexAgenT planner inputs.

The planner only needs per-worker stage costs.  This module keeps the existing
linear worker-label model as the compatibility path and adds a small roofline
estimator for requests that carry model and hardware parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sglang.hexgent.schemas import PlannerScheduleRequest, WorkerDescriptor


def _latency_ms(base_ms: float, coeff_ms: float, tokens: int) -> int:
    return max(1, int(round(float(base_ms) + float(coeff_ms) * max(1, tokens))))


def _optional_float(payload: Optional[Dict[str, Any]], *keys: str) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _optional_int(payload: Optional[Dict[str, Any]], *keys: str) -> Optional[int]:
    value = _optional_float(payload, *keys)
    if value is None:
        return None
    return int(value)


@dataclass(frozen=True)
class LatencyEstimate:
    prefill_ms_by_instance: List[int]
    decode_ms_by_instance: List[int]
    transfer_ms_by_decode_instance: List[int]


class LatencyEstimator(ABC):
    @abstractmethod
    def estimate(
        self,
        request: PlannerScheduleRequest,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> LatencyEstimate:
        raise NotImplementedError


class LinearWorkerLatencyEstimator(LatencyEstimator):
    """Compatibility estimator based on worker latency labels."""

    def estimate(
        self,
        request: PlannerScheduleRequest,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> LatencyEstimate:
        return LatencyEstimate(
            prefill_ms_by_instance=[
                self.prefill_ms(worker, input_tokens)
                for worker in request.prefill_workers
            ],
            decode_ms_by_instance=[
                self.decode_ms(worker, output_tokens) for worker in request.decode_workers
            ],
            transfer_ms_by_decode_instance=[
                self.transfer_ms(worker, input_tokens)
                for worker in request.decode_workers
            ],
        )

    @staticmethod
    def prefill_ms(worker: WorkerDescriptor, input_tokens: int) -> int:
        return _latency_ms(
            worker.prefill_base_ms,
            worker.prefill_ms_per_input_token,
            input_tokens,
        )

    @staticmethod
    def decode_ms(worker: WorkerDescriptor, output_tokens: int) -> int:
        return _latency_ms(
            worker.decode_base_ms,
            worker.decode_ms_per_output_token,
            output_tokens,
        )

    @staticmethod
    def transfer_ms(worker: WorkerDescriptor, input_tokens: int) -> int:
        if worker.transfer_ms_per_token <= 0:
            return 0
        return _latency_ms(0.0, worker.transfer_ms_per_token, input_tokens)


@dataclass(frozen=True)
class _ModelParams:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int
    tp_size: int
    w_bit: int
    a_bit: int
    kv_bit: int

    @property
    def head_dim(self) -> float:
        return self.hidden_size / max(1, self.num_attention_heads)

    @property
    def w_byte(self) -> float:
        return self.w_bit / 8.0

    @property
    def a_byte(self) -> float:
        return self.a_bit / 8.0

    @property
    def kv_byte(self) -> float:
        return self.kv_bit / 8.0


class RooflineLatencyEstimator(LatencyEstimator):
    """Approximate LLM latency with the same roofline principle as the simulator.

    The simulator computes each operator's achievable performance from peak OPS
    and memory bandwidth, then sums inference times.  This implementation keeps
    the same bottleneck model while aggregating common transformer operations so
    it remains cheap and dependency-free inside the online planner.
    """

    def estimate(
        self,
        request: PlannerScheduleRequest,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> LatencyEstimate:
        linear = LinearWorkerLatencyEstimator()
        return LatencyEstimate(
            prefill_ms_by_instance=[
                self.prefill_ms(
                    worker,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    request=request,
                    fallback=linear.prefill_ms(worker, input_tokens),
                )
                for worker in request.prefill_workers
            ],
            decode_ms_by_instance=[
                self.decode_ms(
                    worker,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    batch_size=1
                    + max(
                        0,
                        int(
                            request.state.decode_running_batch_sizes[index]
                            if index < len(request.state.decode_running_batch_sizes)
                            else 0
                        ),
                    ),
                    request=request,
                    fallback=linear.decode_ms(worker, output_tokens),
                )
                for index, worker in enumerate(request.decode_workers)
            ],
            transfer_ms_by_decode_instance=[
                self.transfer_ms(
                    worker,
                    input_tokens=input_tokens,
                    request=request,
                    fallback=linear.transfer_ms(worker, input_tokens),
                )
                for worker in request.decode_workers
            ],
        )

    def prefill_ms(
        self,
        worker: WorkerDescriptor,
        *,
        input_tokens: int,
        output_tokens: int = 1,
        request: Optional[PlannerScheduleRequest] = None,
        fallback: Optional[int] = None,
    ) -> int:
        del output_tokens
        model = _model_params(request, worker) if request else _worker_model_params(worker)
        hardware = (
            _hardware_params(request, worker) if request else _worker_hardware_params(worker)
        )
        if model is None or hardware is None:
            return fallback if fallback is not None else LinearWorkerLatencyEstimator.prefill_ms(worker, input_tokens)
        try:
            return self.estimate_prefill_ms(
                model=model,
                hardware=hardware,
                input_tokens=input_tokens,
            )
        except (TypeError, ValueError, ZeroDivisionError):
            return fallback if fallback is not None else LinearWorkerLatencyEstimator.prefill_ms(worker, input_tokens)

    def prefill_batch_ms(
        self,
        worker: WorkerDescriptor,
        *,
        input_tokens: List[int],
        request: Optional[PlannerScheduleRequest] = None,
    ) -> int:
        """Estimate one heterogeneous prefill forward batch.

        The batch shares model-weight traffic but each request contributes its
        own activation, KV, attention, and compute work. This lies between the
        prior ``max(single_request)`` approximation and a blindly serial sum.
        """

        token_counts = [max(1, int(tokens)) for tokens in input_tokens]
        if not token_counts:
            return 0
        model = _model_params(request, worker) if request else _worker_model_params(worker)
        hardware = (
            _hardware_params(request, worker) if request else _worker_hardware_params(worker)
        )
        if model is None or hardware is None:
            return sum(
                LinearWorkerLatencyEstimator.prefill_ms(worker, tokens)
                for tokens in token_counts
            )
        try:
            return self.estimate_prefill_batch_ms(
                model=model,
                hardware=hardware,
                input_tokens=token_counts,
            )
        except (TypeError, ValueError, ZeroDivisionError):
            return sum(
                LinearWorkerLatencyEstimator.prefill_ms(worker, tokens)
                for tokens in token_counts
            )

    def decode_ms(
        self,
        worker: WorkerDescriptor,
        *,
        input_tokens: int,
        output_tokens: int,
        batch_size: int = 1,
        request: Optional[PlannerScheduleRequest] = None,
        fallback: Optional[int] = None,
    ) -> int:
        model = _model_params(request, worker) if request else _worker_model_params(worker)
        hardware = (
            _hardware_params(request, worker) if request else _worker_hardware_params(worker)
        )
        if model is None or hardware is None:
            return fallback if fallback is not None else LinearWorkerLatencyEstimator.decode_ms(worker, output_tokens)
        try:
            return self.estimate_decode_total_ms(
                model=model,
                hardware=hardware,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                batch_size=batch_size,
            )
        except (TypeError, ValueError, ZeroDivisionError):
            return fallback if fallback is not None else LinearWorkerLatencyEstimator.decode_ms(worker, output_tokens)

    def decode_iteration_ms(
        self,
        worker: WorkerDescriptor,
        *,
        sequence_length: int,
        batch_size: int = 1,
        request: Optional[PlannerScheduleRequest] = None,
    ) -> int:
        model = _model_params(request, worker) if request else _worker_model_params(worker)
        hardware = (
            _hardware_params(request, worker) if request else _worker_hardware_params(worker)
        )
        if model is None or hardware is None:
            return LinearWorkerLatencyEstimator.decode_ms(worker, batch_size)
        ops, weight_bytes, dynamic_memory_bytes = self._stage_work_components(
            model=model,
            sequence_tokens=max(1, sequence_length),
            stage="decode",
        )
        batch_size = max(1, int(batch_size))
        ops *= batch_size
        memory_bytes = weight_bytes + dynamic_memory_bytes * batch_size
        try:
            return max(
                1,
                int(round(self._roofline_time_ms(ops, memory_bytes, model, hardware))),
            )
        except (TypeError, ValueError, ZeroDivisionError):
            return LinearWorkerLatencyEstimator.decode_ms(worker, batch_size)

    def transfer_ms(
        self,
        worker: WorkerDescriptor,
        *,
        input_tokens: int,
        request: Optional[PlannerScheduleRequest] = None,
        fallback: int = 0,
    ) -> int:
        model = _model_params(request, worker) if request else _worker_model_params(worker)
        hardware = (
            _hardware_params(request, worker) if request else _worker_hardware_params(worker)
        )
        if model is None or hardware is None:
            return fallback
        try:
            estimated = self.estimate_transfer_ms(
                model=model,
                hardware=hardware,
                input_tokens=input_tokens,
            )
        except (TypeError, ValueError, ZeroDivisionError):
            estimated = None
        return fallback if estimated is None else estimated

    def estimate_prefill_ms(
        self,
        *,
        model: _ModelParams,
        hardware: Dict[str, Any],
        input_tokens: int,
    ) -> int:
        ops, _, memory_bytes = self._stage_work_components(
            model=model,
            sequence_tokens=max(1, input_tokens),
            stage="prefill",
        )
        return max(1, int(round(self._roofline_time_ms(ops, memory_bytes, model, hardware))))

    def estimate_prefill_batch_ms(
        self,
        *,
        model: _ModelParams,
        hardware: Dict[str, Any],
        input_tokens: List[int],
    ) -> int:
        """Estimate a packed prefill batch with variable prompt lengths."""

        if not input_tokens:
            return 0
        ops = 0.0
        weight_bytes = 0.0
        dynamic_memory_bytes = 0.0
        for tokens in input_tokens:
            item_ops, item_weight, item_dynamic = self._stage_work_components(
                model=model,
                sequence_tokens=max(1, int(tokens)),
                stage="prefill",
            )
            ops += item_ops
            # A GEMM batch reuses the same model weights for all sequences.
            weight_bytes = max(weight_bytes, item_weight)
            dynamic_memory_bytes += item_dynamic
        return max(
            1,
            int(
                round(
                    self._roofline_time_ms(
                        ops,
                        weight_bytes + dynamic_memory_bytes,
                        model,
                        hardware,
                    )
                )
            ),
        )

    def estimate_decode_total_ms(
        self,
        *,
        model: _ModelParams,
        hardware: Dict[str, Any],
        input_tokens: int,
        output_tokens: int,
        batch_size: int = 1,
    ) -> int:
        total_ms = 0
        batch_size = max(1, int(batch_size))
        for generated_tokens in range(max(1, int(output_tokens))):
            ops, weight_bytes, dynamic_memory_bytes = self._stage_work_components(
                model=model,
                sequence_tokens=max(1, int(input_tokens)) + generated_tokens,
                stage="decode",
            )
            # Every active sequence participates in the same decode iteration.
            # Weight traffic is paid once by the batch, while activations and KV
            # traffic scale with the number of active sequences. Multiplying the
            # complete memory term by batch size treats a continuous batch as
            # independent requests and overestimates decode work.
            ops *= batch_size
            memory_bytes = weight_bytes + dynamic_memory_bytes * batch_size
            total_ms += max(
                1, int(round(self._roofline_time_ms(ops, memory_bytes, model, hardware)))
            )
        return max(1, total_ms)

    def estimate_transfer_ms(
        self,
        *,
        model: _ModelParams,
        hardware: Dict[str, Any],
        input_tokens: int,
    ) -> Optional[int]:
        effective_hardware = _effective_hardware(hardware)
        bandwidth = _optional_float(
            effective_hardware,
            "transfer_bandwidth",
            "transfer_bandwidth_bytes_per_s",
            "net_bandwidth",
            "network_bandwidth",
            "network_bandwidth_bytes_per_s",
        )
        if bandwidth is None:
            return None
        kv_bytes = (
            max(1, input_tokens)
            * model.num_hidden_layers
            * model.head_dim
            * model.num_key_value_heads
            * model.kv_byte
            * 2.0
        )
        if kv_bytes <= 0:
            return 0
        # P-D KV transfer has a non-negligible control-plane and setup cost.
        # Keep it separate from the bulk-link bandwidth so a same-host NVLink
        # microbenchmark is not mistaken for end-to-end request latency.
        startup_ms = _optional_float(
            effective_hardware,
            "transfer_startup_ms",
            "transfer_fixed_overhead_ms",
        ) or 0.0
        return max(1, int(round(startup_ms + kv_bytes / bandwidth * 1000.0)))

    def _stage_work(
        self,
        *,
        model: _ModelParams,
        sequence_tokens: int,
        stage: str,
    ) -> tuple[float, float]:
        ops, weight_bytes, dynamic_memory_bytes = self._stage_work_components(
            model=model,
            sequence_tokens=sequence_tokens,
            stage=stage,
        )
        return ops, weight_bytes + dynamic_memory_bytes

    def _stage_work_components(
        self,
        *,
        model: _ModelParams,
        sequence_tokens: int,
        stage: str,
    ) -> tuple[float, float, float]:
        linear_ops, linear_weight, linear_dynamic = self._linear_work_components(
            model, sequence_tokens, stage
        )
        attention_ops, attention_memory = self._attention_work(
            model, sequence_tokens, stage
        )
        elementwise_ops = sequence_tokens * model.hidden_size * 20.0
        elementwise_memory = sequence_tokens * model.hidden_size * model.a_byte * 8.0
        layer_ops = linear_ops + attention_ops + elementwise_ops
        layer_weight = linear_weight
        layer_dynamic_memory = linear_dynamic + attention_memory + elementwise_memory
        ops = layer_ops * model.num_hidden_layers
        weight_bytes = layer_weight * model.num_hidden_layers
        dynamic_memory_bytes = layer_dynamic_memory * model.num_hidden_layers
        if model.vocab_size > 0:
            ops += model.hidden_size * model.vocab_size
            weight_bytes += model.hidden_size * model.vocab_size * model.w_byte
            dynamic_memory_bytes += (model.hidden_size + model.vocab_size) * model.a_byte
        return ops, weight_bytes, dynamic_memory_bytes

    def _linear_work(
        self,
        model: _ModelParams,
        sequence_tokens: int,
        stage: str,
    ) -> tuple[float, float]:
        ops, weight_bytes, dynamic_memory_bytes = self._linear_work_components(
            model, sequence_tokens, stage
        )
        return ops, weight_bytes + dynamic_memory_bytes

    def _linear_work_components(
        self,
        model: _ModelParams,
        sequence_tokens: int,
        stage: str,
    ) -> tuple[float, float, float]:
        hidden = float(model.hidden_size)
        intermediate = float(model.intermediate_size)
        tp = float(max(1, model.tp_size))
        kv_hidden = hidden * float(model.num_key_value_heads) / max(
            1.0, float(model.num_attention_heads)
        )
        shapes = [
            (hidden, hidden / tp, False),
            (hidden, kv_hidden / tp, True),
            (hidden, kv_hidden / tp, True),
            (hidden / tp, hidden, False),
            (hidden, intermediate / tp, False),
            (hidden, intermediate / tp, False),
            (intermediate / tp, hidden, False),
        ]
        token_factor = float(sequence_tokens if stage == "prefill" else 1)
        ops = 0.0
        weight_bytes = 0.0
        dynamic_memory_bytes = 0.0
        for input_dim, output_dim, stores_kv in shapes:
            ops += input_dim * output_dim * token_factor * 2.0
            weight_bytes += input_dim * output_dim * model.w_byte
            dynamic_memory_bytes += input_dim * token_factor * model.a_byte
            if stores_kv:
                dynamic_memory_bytes += output_dim * token_factor * model.kv_byte
            else:
                dynamic_memory_bytes += output_dim * token_factor * model.a_byte
        return ops, weight_bytes, dynamic_memory_bytes

    def _attention_work(
        self,
        model: _ModelParams,
        sequence_tokens: int,
        stage: str,
    ) -> tuple[float, float]:
        seq = float(sequence_tokens)
        head_dim = model.head_dim
        heads = float(model.num_attention_heads)
        kv_heads = float(model.num_key_value_heads)
        if stage == "decode":
            ops = seq * head_dim * heads * 4.0 + heads * seq * 5.0
            memory = (
                head_dim * heads * model.a_byte
                + head_dim * heads * model.a_byte
                + seq * head_dim * kv_heads * model.kv_byte * 2.0
            )
            return ops, memory

        ops = seq * seq * head_dim * heads * 4.0 + heads * seq * seq * 5.0
        memory = (
            seq * head_dim * heads * model.a_byte
            + seq * head_dim * heads * model.a_byte
            + seq * head_dim * kv_heads * model.kv_byte * 2.0
        )
        return ops, memory

    def _roofline_time_ms(
        self,
        ops: float,
        memory_bytes: float,
        model: _ModelParams,
        hardware: Dict[str, Any],
    ) -> float:
        bandwidth = _hardware_bandwidth(hardware)
        peak_ops = _hardware_peak_ops(hardware, model)
        if bandwidth is None or peak_ops is None:
            raise ValueError("Roofline hardware params require bandwidth and peak OPS")
        compute_s = max(0.0, ops) / peak_ops if peak_ops > 0 else 0.0
        memory_s = max(0.0, memory_bytes) / bandwidth if bandwidth > 0 else 0.0
        return max(compute_s, memory_s) * 1000.0


class HybridLatencyEstimator(LatencyEstimator):
    """Use roofline where enough config exists, otherwise preserve linear labels."""

    def __init__(
        self,
        *,
        roofline: Optional[RooflineLatencyEstimator] = None,
        linear: Optional[LinearWorkerLatencyEstimator] = None,
    ) -> None:
        self.roofline = roofline or RooflineLatencyEstimator()
        self.linear = linear or LinearWorkerLatencyEstimator()
        self._prefill_cache: Dict[Tuple[Any, ...], Tuple[int, ...]] = {}
        self._decode_cache: Dict[Tuple[Any, ...], Tuple[int, ...]] = {}
        self._transfer_cache: Dict[Tuple[Any, ...], Tuple[int, ...]] = {}

    @staticmethod
    def _cache_value(value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (str(key), HybridLatencyEstimator._cache_value(item))
                    for key, item in value.items()
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(HybridLatencyEstimator._cache_value(item) for item in value)
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return repr(value)

    @classmethod
    def _worker_cache_key(cls, worker: WorkerDescriptor) -> Tuple[Any, ...]:
        return (
            worker.id,
            worker.url,
            worker.hardware,
            int(worker.tp_size),
            int(worker.weight_bits),
            int(worker.activation_bits),
            int(worker.kv_bits),
            float(worker.prefill_base_ms),
            float(worker.prefill_ms_per_input_token),
            float(worker.decode_base_ms),
            float(worker.decode_ms_per_output_token),
            float(worker.transfer_ms_per_token),
            cls._cache_value(worker.model_config),
            cls._cache_value(worker.hardware_params),
            cls._cache_value(worker.roofline),
        )

    @classmethod
    def _request_cache_key(cls, request: PlannerScheduleRequest) -> Tuple[Any, ...]:
        meta = request.hexgent
        return (
            cls._cache_value(meta.model_config),
            cls._cache_value(meta.hardware_params),
            cls._cache_value(meta.hardware_params_by_name),
            cls._cache_value(meta.roofline),
        )

    @staticmethod
    def _cache_get_or_compute(
        cache: Dict[Tuple[Any, ...], Tuple[int, ...]],
        key: Tuple[Any, ...],
        compute,
    ) -> List[int]:
        cached = cache.get(key)
        if cached is not None:
            return list(cached)
        result = tuple(int(value) for value in compute())
        if len(cache) >= 4096:
            cache.pop(next(iter(cache)))
        cache[key] = result
        return list(result)

    def estimate(
        self,
        request: PlannerScheduleRequest,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> LatencyEstimate:
        meta = request.hexgent
        request_key = self._request_cache_key(request)
        prefill_workers = tuple(
            self._worker_cache_key(worker) for worker in request.prefill_workers
        )
        decode_workers = tuple(
            self._worker_cache_key(worker) for worker in request.decode_workers
        )
        prefill = self._cache_get_or_compute(
            self._prefill_cache,
            (
                input_tokens,
                meta.estimated_prefill_time,
                request_key,
                prefill_workers,
            ),
            lambda: [
                self._prefill_ms(request, worker, input_tokens)
                for worker in request.prefill_workers
            ],
        )
        decode_batch_sizes = tuple(
            1
            + max(
                0,
                int(
                    request.state.decode_running_batch_sizes[index]
                    if index < len(request.state.decode_running_batch_sizes)
                    else 0
                ),
            )
            for index, _worker in enumerate(request.decode_workers)
        )
        decode = self._cache_get_or_compute(
            self._decode_cache,
            (
                input_tokens,
                output_tokens,
                meta.estimated_decode_time,
                request_key,
                decode_workers,
                decode_batch_sizes,
            ),
            lambda: [
                self._decode_ms(
                    request,
                    worker,
                    input_tokens,
                    output_tokens,
                    decode_batch_sizes[index],
                )
                for index, worker in enumerate(request.decode_workers)
            ],
        )
        transfer = self._cache_get_or_compute(
            self._transfer_cache,
            (
                input_tokens,
                meta.estimated_transfer_time,
                request_key,
                decode_workers,
                os.getenv("HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S"),
                os.getenv("HEXGENT_TRANSFER_STARTUP_MS"),
            ),
            lambda: [
                self._transfer_ms(request, worker, input_tokens)
                for worker in request.decode_workers
            ],
        )
        return LatencyEstimate(prefill, decode, transfer)

    def _prefill_ms(
        self,
        request: PlannerScheduleRequest,
        worker: WorkerDescriptor,
        input_tokens: int,
    ) -> int:
        profiled = _profile_ms(request.hexgent.estimated_prefill_time)
        if profiled is not None:
            return profiled
        model = _model_params(request, worker)
        hardware = _hardware_params(request, worker)
        if model is None or hardware is None:
            return self.linear.prefill_ms(worker, input_tokens)
        try:
            return self.roofline.estimate_prefill_ms(
                model=model,
                hardware=hardware,
                input_tokens=input_tokens,
            )
        except (TypeError, ValueError, ZeroDivisionError):
            return self.linear.prefill_ms(worker, input_tokens)

    def _decode_ms(
        self,
        request: PlannerScheduleRequest,
        worker: WorkerDescriptor,
        input_tokens: int,
        output_tokens: int,
        batch_size: int,
    ) -> int:
        profiled = _profile_ms(request.hexgent.estimated_decode_time)
        model = _model_params(request, worker)
        hardware = _hardware_params(request, worker)
        if profiled is not None and batch_size <= 1:
            return profiled
        if model is None or hardware is None:
            if profiled is not None:
                return max(1, int(round(profiled * max(1, batch_size))))
            return self.linear.decode_ms(worker, output_tokens)
        try:
            roofline_single = self.roofline.estimate_decode_total_ms(
                model=model,
                hardware=hardware,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                batch_size=1,
            )
            roofline_batch = self.roofline.estimate_decode_total_ms(
                model=model,
                hardware=hardware,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                batch_size=batch_size,
            )
            if profiled is not None:
                scale = max(1.0, roofline_batch / max(1, roofline_single))
                return max(1, int(round(profiled * scale)))
            return roofline_batch
        except (TypeError, ValueError, ZeroDivisionError):
            if profiled is not None:
                return max(1, int(round(profiled * max(1, batch_size))))
            return self.linear.decode_ms(worker, output_tokens)

    def _transfer_ms(
        self,
        request: PlannerScheduleRequest,
        worker: WorkerDescriptor,
        input_tokens: int,
    ) -> int:
        profiled = _profile_ms(request.hexgent.estimated_transfer_time)
        if profiled is not None:
            return profiled
        model = _model_params(request, worker)
        hardware = _hardware_params(request, worker)
        if model is not None and hardware is not None:
            try:
                estimated = self.roofline.estimate_transfer_ms(
                    model=model,
                    hardware=hardware,
                    input_tokens=input_tokens,
                )
            except (TypeError, ValueError, ZeroDivisionError):
                estimated = None
            if estimated is not None:
                return estimated
        return self.linear.transfer_ms(worker, input_tokens)


def _profile_ms(value: Any) -> Optional[int]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return max(1, int(round(parsed)))


def _effective_hardware(hardware: Dict[str, Any]) -> Dict[str, Any]:
    """Apply optional measured end-to-end P-D transfer calibration.

    The trace's nominal bandwidth describes the intended fabric.  A Pod may
    instead use Mooncake TCP fallback, so the measured value must be able to
    override only the transfer link without changing HBM roofline parameters.
    """

    result = dict(hardware)
    bandwidth_override = os.getenv("HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S")
    if bandwidth_override:
        try:
            bandwidth = float(bandwidth_override)
        except (TypeError, ValueError):
            bandwidth = 0.0
        if bandwidth > 0:
            result["transfer_bandwidth_bytes_per_s"] = bandwidth

    startup_override = os.getenv("HEXGENT_TRANSFER_STARTUP_MS")
    if startup_override:
        try:
            startup_ms = float(startup_override)
        except (TypeError, ValueError):
            startup_ms = -1.0
        if startup_ms >= 0:
            result["transfer_startup_ms"] = startup_ms
    return result


def _model_params(
    request: Optional[PlannerScheduleRequest],
    worker: WorkerDescriptor,
) -> Optional[_ModelParams]:
    payload = _model_payload(request, worker) if request else _worker_model_payload(worker)
    return _model_params_from_payload(payload, worker)


def _worker_model_params(worker: WorkerDescriptor) -> Optional[_ModelParams]:
    return _model_params_from_payload(_worker_model_payload(worker), worker)


def _model_params_from_payload(
    payload: Optional[Dict[str, Any]],
    worker: WorkerDescriptor,
) -> Optional[_ModelParams]:
    if payload is None:
        return None
    hidden_size = _optional_int(payload, "hidden_size", "n_embd")
    num_attention_heads = _optional_int(
        payload, "num_attention_heads", "n_head", "num_heads"
    )
    num_hidden_layers = _optional_int(
        payload, "num_hidden_layers", "n_layer", "num_layers"
    )
    intermediate_size = _optional_int(
        payload, "intermediate_size", "ffn_hidden_size", "mlp_hidden_size"
    )
    if (
        hidden_size is None
        or num_attention_heads is None
        or num_hidden_layers is None
        or intermediate_size is None
    ):
        return None
    num_key_value_heads = (
        _optional_int(payload, "num_key_value_heads", "num_kv_heads")
        or num_attention_heads
    )
    a_bit = _optional_int(payload, "a_bit", "activation_bits", "act_bits") or 16
    kv_bit = _optional_int(payload, "kv_bit", "kv_bits") or a_bit
    return _ModelParams(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        vocab_size=_optional_int(payload, "vocab_size") or 0,
        tp_size=_worker_tp_size(worker, payload),
        w_bit=_optional_int(payload, "w_bit", "weight_bits") or 16,
        a_bit=a_bit,
        kv_bit=kv_bit,
    )


def _model_payload(
    request: PlannerScheduleRequest,
    worker: WorkerDescriptor,
) -> Optional[Dict[str, Any]]:
    worker_payload = _worker_model_payload(worker)
    if worker_payload is not None:
        return worker_payload
    meta = request.hexgent
    if isinstance(meta.model_config, dict):
        return meta.model_config
    if isinstance(meta.roofline, dict):
        model = meta.roofline.get("model_config") or meta.roofline.get("model")
        if isinstance(model, dict):
            return model
        if "hidden_size" in meta.roofline:
            return meta.roofline
    return None


def _worker_model_payload(worker: WorkerDescriptor) -> Optional[Dict[str, Any]]:
    if isinstance(worker.model_config, dict):
        return worker.model_config
    if isinstance(worker.roofline, dict):
        model = worker.roofline.get("model_config") or worker.roofline.get("model")
        if isinstance(model, dict):
            return model
    return None


def _worker_tp_size(worker: WorkerDescriptor, payload: Dict[str, Any]) -> int:
    try:
        worker_tp_size = int(getattr(worker, "tp_size", 1) or 1)
    except (TypeError, ValueError):
        worker_tp_size = 1
    if worker_tp_size > 1:
        return worker_tp_size
    return max(1, _optional_int(payload, "tp_size", "tensor_parallel_size") or 1)


def _hardware_params(
    request: Optional[PlannerScheduleRequest],
    worker: WorkerDescriptor,
) -> Optional[Dict[str, Any]]:
    worker_hardware = _worker_hardware_params(worker)
    if worker_hardware is not None or request is None:
        return worker_hardware
    meta = request.hexgent
    if isinstance(meta.hardware_params_by_name, dict):
        for key in (worker.hardware, worker.id, worker.url):
            if key is None:
                continue
            value = meta.hardware_params_by_name.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(meta.roofline, dict):
        hardware_by_name = (
            meta.roofline.get("hardware_params_by_name")
            or meta.roofline.get("hardware_by_name")
            or meta.roofline.get("hardware")
        )
        if isinstance(hardware_by_name, dict):
            for key in (worker.hardware, worker.id, worker.url):
                if key is None:
                    continue
                value = hardware_by_name.get(key)
                if isinstance(value, dict):
                    return value
        hardware_params = meta.roofline.get("hardware_params")
        if isinstance(hardware_params, dict):
            return hardware_params
    if isinstance(meta.hardware_params, dict):
        return meta.hardware_params
    return None


def _worker_hardware_params(worker: WorkerDescriptor) -> Optional[Dict[str, Any]]:
    if isinstance(worker.hardware_params, dict):
        return worker.hardware_params
    if isinstance(worker.roofline, dict):
        hardware = worker.roofline.get("hardware_params") or worker.roofline.get(
            "hardware"
        )
        if isinstance(hardware, dict):
            return hardware
    # The Rust router always carries the worker's hardware label, but does not
    # need to duplicate static HBM/compute constants in every planner request.
    from sglang.hexgent.hardware import resolve_hardware_profile

    return resolve_hardware_profile(worker.hardware)


def _hardware_bandwidth(hardware: Dict[str, Any]) -> Optional[float]:
    return _optional_float(
        hardware,
        "bandwidth",
        "memory_bandwidth",
        "bandwidth_bytes_per_s",
        "memory_bandwidth_bytes_per_s",
        "hbm_bandwidth_bytes_per_s",
    )


def _hardware_peak_ops(
    hardware: Dict[str, Any],
    model: _ModelParams,
) -> Optional[float]:
    if model.w_bit <= 8 and model.a_bit <= 8 and model.kv_bit <= 8:
        int8_ops = _optional_float(
            hardware,
            "INT8",
            "int8_ops",
            "int8_flops",
            "peak_int8_ops",
        )
        if int8_ops is not None:
            return int8_ops
    return _optional_float(
        hardware,
        "FP16",
        "fp16_ops",
        "fp16_flops",
        "peak_flops",
        "max_OPS",
        "max_ops",
    )
