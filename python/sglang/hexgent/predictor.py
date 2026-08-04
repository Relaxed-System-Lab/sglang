"""Output-length prediction hooks for HexAgenT.

The production implementation can load a BERT/proxy model, while tests and
minimal deployments use a deterministic fallback. Request-supplied predictions
always take precedence because traces or upstream clients may already provide a
better estimate.
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence


class OutputLengthPredictor(ABC):
    @abstractmethod
    def predict(
        self,
        text: str,
        *,
        predicted_output_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> int:
        raise NotImplementedError


@dataclass
class DeterministicOutputLengthPredictor(OutputLengthPredictor):
    """A deterministic fallback that needs no model weights.

    This is intentionally conservative: it never returns zero and it scales
    slowly with prompt size when neither the client nor sampling params provide
    a stronger bound.
    """

    default_tokens: int = 128
    min_tokens: int = 1
    max_fallback_tokens: int = 1024

    def predict(
        self,
        text: str,
        *,
        predicted_output_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> int:
        if predicted_output_tokens is not None and int(predicted_output_tokens) > 0:
            return int(predicted_output_tokens)
        if max_tokens is not None and int(max_tokens) > 0:
            return int(max_tokens)

        tokenish_words = max(1, len(str(text).split()))
        estimated = int(round(self.default_tokens * math.sqrt(tokenish_words / 64.0)))
        return max(self.min_tokens, min(self.max_fallback_tokens, estimated))


def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in items] or None


class BertProxyOutputLengthPredictor(OutputLengthPredictor):
    """Optional BERT/proxy predictor wrapper.

    The proxy-model reference implementation fine-tunes BERT-style sequence
    classifiers for regression, binary, multiclass, and ordinal output-length
    prediction. This wrapper keeps serving integration generic: mount any
    compatible Hugging Face sequence-classification checkpoint and configure how
    logits map to token counts.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        task_type: Optional[str] = None,
        class_values: Optional[Sequence[int]] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        architecture: Optional[str] = None,
        weights_path: Optional[str] = None,
        hidden_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        model_index: int = 0,
        model_count: int = 0,
    ) -> None:
        model_path = model_path or os.environ.get("HEXGENT_PROXY_MODEL_PATH")
        if not model_path:
            raise ValueError("A proxy model path is required")
        self.task_type = (
            task_type
            or os.environ.get("HEXGENT_PROXY_TASK_TYPE")
            or os.environ.get("HEXGENT_PROXY_MODEL_TYPE")
            or "regression"
        ).strip().lower()
        if self.task_type == "classification":
            self.task_type = "multiclass"
        if self.task_type not in {"regression", "binary", "multiclass", "ordinal"}:
            raise ValueError(f"Unsupported proxy task type: {self.task_type}")
        self.class_values = list(
            class_values
            or _parse_int_list(os.environ.get("HEXGENT_PROXY_CLASS_VALUES"))
            or _parse_int_list(os.environ.get("HEXGENT_PROXY_OUTPUT_BUCKETS"))
            or []
        )
        self.device = device or os.environ.get("HEXGENT_PROXY_DEVICE")
        self.max_length = int(max_length)
        self.architecture = (
            architecture
            or os.environ.get("HEXGENT_PROXY_ARCHITECTURE")
            or "hf_sequence_classification"
        ).strip().lower()
        self.weights_path = weights_path or os.environ.get("HEXGENT_PROXY_WEIGHTS_PATH")
        self.hidden_dim = int(
            hidden_dim or os.environ.get("HEXGENT_PROXY_HIDDEN_DIM", "128")
        )
        self.num_classes = int(
            num_classes
            or os.environ.get(
                "HEXGENT_PROXY_NUM_CLASSES",
                str(len(self.class_values) or (3 if self.task_type == "binary" else 5)),
            )
        )
        self.model_index = int(model_index)
        self.model_count = int(
            model_count or os.environ.get("HEXGENT_PROXY_MODEL_COUNT", "0")
        )

        try:
            from transformers import (
                AutoConfig,
                AutoModelForSequenceClassification,
                AutoTokenizer,
                BertModel,
            )
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("transformers and torch are required") from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.architecture == "proxy_repo":
            if not self.weights_path:
                raise ValueError(
                    "HEXGENT_PROXY_WEIGHTS_PATH is required for proxy_repo architecture"
                )
            self._model = self._build_proxy_repo_model(
                torch=torch,
                bert_model_cls=BertModel,
                config=AutoConfig.from_pretrained(model_path),
                model_path=model_path,
            )
            state_dict = torch.load(
                self.weights_path,
                map_location=self.device or "cpu",
            )
            self._model.load_state_dict(state_dict)
        elif self.architecture == "hf_sequence_classification":
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported proxy architecture: {self.architecture}")
        if self.device and hasattr(self._model, "to"):
            self._model.to(self.device)
        self._model.eval()

    def _build_proxy_repo_model(self, *, torch, bert_model_cls, config, model_path):
        outer = self

        class ProxyRepoHead(torch.nn.Module):  # pragma: no cover - exercised with real deps
            def __init__(self):
                super().__init__()
                self.bert = bert_model_cls.from_pretrained(model_path)
                self.cls = torch.nn.Linear(config.hidden_size, outer.hidden_dim)
                self.relu = torch.nn.ReLU()
                fc1_input_dim = outer.hidden_dim + max(0, outer.model_count)
                self.fc1 = torch.nn.Linear(fc1_input_dim, outer.hidden_dim)
                output_dim = 1 if outer.task_type in {"regression", "ordinal"} else outer.num_classes
                self.fc2 = torch.nn.Linear(outer.hidden_dim, output_dim)
                self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

            def forward(self, input_ids, attention_mask, model_name=None):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.last_hidden_state[:, 0, :]
                output = self.relu(self.cls(pooled))
                if outer.model_count > 0:
                    if model_name is None:
                        model_name = torch.zeros(
                            (output.shape[0], outer.model_count),
                            dtype=output.dtype,
                            device=output.device,
                        )
                        model_name[:, min(outer.model_index, outer.model_count - 1)] = 1
                    output = torch.cat((output, model_name), dim=-1)
                output = self.relu(self.fc1(output))
                output = self.fc2(output)
                if outer.task_type in {"binary", "multiclass"}:
                    output = self.logsoftmax(output)
                return output.squeeze(-1)

        return ProxyRepoHead()

    def _class_value(self, index: int) -> int:
        if self.class_values:
            index = min(max(0, index), len(self.class_values) - 1)
            return max(1, int(self.class_values[index]))
        return max(1, int(index) + 1)

    def _logits_to_list(self, logits) -> List[float]:
        if hasattr(logits, "detach"):
            logits = logits.detach()
        if hasattr(logits, "cpu"):
            logits = logits.cpu()
        if hasattr(logits, "reshape"):
            logits = logits.reshape(-1)
        if hasattr(logits, "tolist"):
            values = logits.tolist()
        else:
            values = list(logits)
        if values and isinstance(values[0], list):
            values = [item for row in values for item in row]
        return [float(value) for value in values]

    def _predict_from_logits(self, values: Sequence[float]) -> int:
        if not values:
            return 1
        if self.task_type == "regression":
            return max(1, int(round(float(values[0]))))
        if self.task_type == "binary":
            chosen = 1 if float(values[-1]) >= float(values[0]) else 0
            return self._class_value(chosen)
        if self.task_type == "multiclass":
            chosen = max(range(len(values)), key=lambda idx: float(values[idx]))
            return self._class_value(chosen)

        # Ordinal classifiers are commonly trained with BCE-with-logits heads:
        # the number of positive thresholds crossed is the predicted bucket.
        threshold = float(os.environ.get("HEXGENT_PROXY_ORDINAL_THRESHOLD", "0"))
        chosen = sum(1 for value in values if float(value) > threshold)
        return self._class_value(chosen)

    def predict(
        self,
        text: str,
        *,
        predicted_output_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> int:
        if predicted_output_tokens is not None and int(predicted_output_tokens) > 0:
            return int(predicted_output_tokens)
        if max_tokens is not None and int(max_tokens) > 0:
            return int(max_tokens)

        inputs = self._tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        if self.device:
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        with self._torch.no_grad():
            output = self._model(**inputs)
        logits = output.logits if hasattr(output, "logits") else output
        return self._predict_from_logits(self._logits_to_list(logits))


def build_output_length_predictor() -> OutputLengthPredictor:
    model_path = os.environ.get("HEXGENT_PROXY_MODEL_PATH")
    if model_path:
        try:
            return BertProxyOutputLengthPredictor(model_path)
        except Exception:
            # The planner must remain usable even when the optional proxy model
            # is not mounted in the router container.
            pass
    return DeterministicOutputLengthPredictor()
