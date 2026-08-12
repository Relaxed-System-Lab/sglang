"""HexAgenT experiment tooling and planner integration for SGLang."""

__all__ = [
    "AgenticPDExactScheduler",
    "BertProxyOutputLengthPredictor",
    "DeterministicOutputLengthPredictor",
    "HexgentPlanner",
    "OutputLengthPredictor",
]


def __getattr__(name):
    if name == "AgenticPDExactScheduler":
        from sglang.hexgent.scheduler import AgenticPDExactScheduler

        return AgenticPDExactScheduler
    if name == "HexgentPlanner":
        from sglang.hexgent.planner import HexgentPlanner

        return HexgentPlanner
    if name in {
        "BertProxyOutputLengthPredictor",
        "DeterministicOutputLengthPredictor",
        "OutputLengthPredictor",
    }:
        from sglang.hexgent import predictor

        return getattr(predictor, name)
    raise AttributeError(name)
