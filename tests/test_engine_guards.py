"""Protocol checks that require no optimization or pretrained weights."""

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from cda_unet.data import Sample
from cda_unet.engine import _load_parents, _predict, aggregate_reports
from cda_unet.schema import get_panel


def _report(fold):
    panel = get_panel("mass")
    sample_id = f"synthetic_{fold}"
    return {
        "schema_version": 1, "modality": "mass", "parent_policy": "self",
        "cache_signature": "synthetic-cache", "config": {"training": {"folds": 2}},
        "epoch": 1000, "fold": fold, "test_ids": [sample_id],
        "records": [{"sample_id": sample_id, "gate": gate.name, "gate_id": gate_id,
                     "tp": 4, "tn": 4, "fp": 1, "fn": 1,
                     "accuracy": 0.8, "recall": 0.8, "precision": 0.8, "f1": 0.8}
                    for gate_id, gate in enumerate(panel.gates)],
    }


def _report_paths(tmp_path, reports):
    paths = []
    for index, report in enumerate(reports):
        path = tmp_path / f"report_{index}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    return paths


@pytest.mark.parametrize("field,value,message", [
    ("epoch", 900, "epochs"),
    ("fold", 0, "duplicate fold"),
    ("fold", -1, "out-of-range fold"),
    ("fold", 2, "out-of-range fold"),
    ("fold", 1.0, "out-of-range fold"),
])
def test_aggregation_refuses_mixed_budgets_or_invalid_folds(tmp_path, field, value, message):
    reports = [_report(0), _report(1)]
    reports[1][field] = value
    paths = _report_paths(tmp_path, reports)
    with pytest.raises(ValueError, match=message):
        aggregate_reports(paths, tmp_path / "aggregate")


def test_exact_fold_coverage_and_four_decimal_csv_export(tmp_path):
    paths = _report_paths(tmp_path, [_report(0), _report(1)])
    partial = aggregate_reports(paths[:1], tmp_path / "partial")
    assert partial["complete_cross_validation"] is False
    complete = aggregate_reports(paths, tmp_path / "complete")
    assert complete["complete_cross_validation"] is True
    assert complete["summary"]["n_samples"] == 2
    assert "0.8000" in (tmp_path / "complete" / "per_gate.csv").read_text()
    assert "0.8000" in (tmp_path / "complete" / "per_sample.csv").read_text()
    assert "python" in complete["software_versions"]


@pytest.fixture
def external_parents(tmp_path):
    panel = get_panel("mass")
    channels = np.random.default_rng(3).uniform(size=(12, len(panel.channels)))
    source = tmp_path / "synthetic.csv"
    pd.DataFrame(channels, columns=panel.channels).to_csv(source, index=False)
    predictions = np.zeros((12, len(panel.gates)), dtype=np.uint8)
    predictions[:6, 0] = 1
    predictions[:3, 1] = 1
    values = {"predictions": predictions, "gate_names": np.array([gate.name for gate in panel.gates]),
              "source_sha256": np.array(hashlib.sha256(source.read_bytes()).hexdigest())}
    return source, panel, values, Sample("synthetic", channels)


@pytest.mark.parametrize("fault", ["source", "order", "shape", "binary"])
def test_supplied_parents_require_exact_event_identity_and_gate_schema(tmp_path, external_parents, fault):
    source, panel, values, _ = external_parents
    valid_path = tmp_path / "valid.npz"
    np.savez_compressed(valid_path, **values)
    np.testing.assert_array_equal(_load_parents(valid_path, source, panel, 12), values["predictions"])
    bad = copy.deepcopy(values)
    if fault == "source":
        bad["source_sha256"] = np.array("wrong-hash")
    elif fault == "order":
        bad["gate_names"] = bad["gate_names"][::-1]
    elif fault == "shape":
        bad["predictions"] = bad["predictions"][:-1]
    elif fault == "binary":
        bad["predictions"][0, 0] = 2
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **bad)
    with pytest.raises(ValueError):
        _load_parents(path, source, panel, 12)


class _ConstantLogits(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.called_gates = []

    def forward(self, images, gate_id):
        self.called_gates.append(gate_id)
        return torch.full_like(images, self.value)


def test_inference_respects_external_parent_rows_and_skips_empty_branches(external_parents):
    _, panel, values, sample = external_parents
    config = {"resolution": 101, "evaluation": {"threshold": 0.0}}
    model = _ConstantLogits(1.0)
    result = _predict(model, sample, panel, config, torch.device("cpu"), values["predictions"])
    assert result[:, 0].all()
    by_name = {gate.name: index for index, gate in enumerate(panel.gates)}
    for gate_id, gate in enumerate(panel.gates):
        if gate.parent is not None:
            expected = values["predictions"][:, by_name[gate.parent]]
            np.testing.assert_array_equal(result[:, gate_id], expected)
    assert model.called_gates == [0, 1, 2, 3]
    empty_model = _ConstantLogits(-1.0)
    self_result = _predict(empty_model, sample, panel, config, torch.device("cpu"))
    assert not self_result.any()
    assert empty_model.called_gates == [0]
