"""Checks for report weighting and the public configuration contract."""
from pathlib import Path

import pytest

from cda_unet.config import load_config, validate_config
from cda_unet.metrics import score_binary, summarize
from cda_unet.schema import get_panel


@pytest.mark.parametrize("modality", ["mass", "flow"])
def test_standard_configs(modality):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / f"{modality}.yaml")
    assert config["resolution"] == 101 and config["rank"] == 32
    assert config["training"]["epochs"] == 1000
    assert config["training"]["parent_policy"] == "consensus"
    assert config["evaluation"]["parent_policy"] == "self"
    config["rank"] = 16
    with pytest.raises(ValueError, match="rank"):
        validate_config(config)


def test_metrics_confusion_counts_and_undefined_cases():
    result = score_binary([1, 1, 0, 0], [1, 0, 1, 0])
    assert all(result[key] == 1 for key in ("tp", "tn", "fp", "fn"))
    assert all(result[key] == 0.5 for key in ("accuracy", "recall", "precision", "f1"))
    empty_positive = score_binary([0, 0], [0, 0])
    assert empty_positive["accuracy"] == 1
    assert empty_positive["f1"] == empty_positive["recall"] == empty_positive["precision"] == 0
    with pytest.raises(ValueError):
        score_binary([1], [0, 1])


def test_overall_weights_samples_and_gates_equally():
    panel = get_panel("flow")
    records = []
    for gate in panel.gates:
        records.append({"sample_id": "small", "gate": gate.name, **score_binary([1], [1])})
        records.append({"sample_id": "large", "gate": gate.name, **score_binary([1] * 99, [0] * 99)})
    result = summarize(records, panel)
    assert result["n_samples"] == 2
    assert result["overall"]["f1"] == 0.5  # Not event-count-weighted F1 = 0.0198.
    with pytest.raises(ValueError):
        summarize(records + [records[0]], panel)
    with pytest.raises(ValueError):
        summarize(records[:-1], panel)
