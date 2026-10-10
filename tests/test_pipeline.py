"""Small CPU integration tests; fixtures contain invented measurements only."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cda_unet.schema import get_panel


def make_samples(directory: Path, panel_name: str, count: int = 4) -> dict[str, int]:
    """Use every deepest label so all branches occur in every synthetic sample."""
    directory.mkdir(parents=True, exist_ok=True)
    panel = get_panel(panel_name)
    event_counts = {}
    for subject in range(count):
        rng = np.random.default_rng(104 + subject)
        labels = np.tile(panel.consensus_labels, 6)
        values = rng.uniform(0, 10, size=(len(labels), len(panel.channels)))
        for label_index, label in enumerate(panel.consensus_labels):
            values[labels == label] += label_index
        frame = pd.DataFrame(values, columns=panel.channels)
        frame[panel.label_column] = labels
        subject_id = f"synthetic_{subject:02d}"
        frame.to_csv(directory / f"{subject_id}.csv", index=False)
        event_counts[subject_id] = len(frame)
    return event_counts


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module", params=["mass", "flow"])
def workflow(request, tmp_path_factory):
    # Import here so collection also provides a useful missing-engine error.
    from cda_unet.engine import evaluate_checkpoint, prepare_data, train_fold

    directory = tmp_path_factory.mktemp(f"pipeline_{request.param}")
    source = directory / "samples"
    cache = directory / "cache"
    counts = make_samples(source, request.param)
    config = {
        "schema_version": 1,
        "modality": request.param,
        "resolution": 101,
        "rank": 32,
        "conditioning": "xy_gate",
        "normalization": "input_group_encoder_gate_decoder",
        "features": [4, 8, 16, 32],
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "seed": 17,
            "folds": 2,
            "fold_strategy": "balanced",
            "shuffle": True,
            "parent_policy": "consensus",
            "checkpoint_every": 100,
            "amp": False,
        },
        "evaluation": {"parent_policy": "self", "threshold": 0.0},
    }
    prepare_data(config, source, cache)
    manifest = json.loads((cache / "manifest.json").read_text())
    checkpoints = []
    reports = []
    first_state = None
    for fold in range(2):
        checkpoint = train_fold(config, cache, directory / "runs", fold, device="cpu")
        if fold == 0:
            first_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        checkpoint = train_fold(config, cache, directory / "runs", fold, device="cpu", resume=checkpoint, epochs=2)
        checkpoints.append(Path(checkpoint))
        destination = directory / f"evaluation_{fold}"
        evaluate_checkpoint(checkpoint, source, destination, device="cpu")
        reports.append(destination / "report.json")
    return {
        "directory": directory, "source": source, "cache": cache,
        "counts": counts, "config": config, "manifest": manifest,
        "checkpoints": checkpoints, "reports": reports,
        "first_state": first_state,
    }


def test_training_splits_are_sample_disjoint_and_resume_updates_state(workflow):
    all_ids = set(workflow["counts"])
    heldout = set()
    for fold in workflow["manifest"]["folds"]:
        training, testing = set(fold["train_ids"]), set(fold["test_ids"])
        assert training and testing
        assert training.isdisjoint(testing)
        assert training | testing == all_ids
        assert heldout.isdisjoint(testing)
        heldout.update(testing)
    assert heldout == all_ids
    first = workflow["first_state"]
    final = torch.load(workflow["checkpoints"][0], map_location="cpu", weights_only=True)
    assert first["epoch"] == 1 and final["epoch"] == 2
    assert first["train_ids"] == final["train_ids"]
    assert first["test_ids"] == final["test_ids"]
    assert len(final["history"]) == 2
    assert final["optimizer_state"]["state"]
    for parameter_id, first_optimizer in first["optimizer_state"]["state"].items():
        final_optimizer = final["optimizer_state"]["state"][parameter_id]
        assert final_optimizer["step"] == 2 * first_optimizer["step"]
    assert any(not torch.equal(first["model_state"][key], value)
               for key, value in final["model_state"].items() if value.is_floating_point())


def test_evaluation_uses_only_heldout_samples_and_all_cell_denominators(workflow):
    panel = get_panel(workflow["config"]["modality"])
    for fold, report_path in enumerate(workflow["reports"]):
        report = json.loads(report_path.read_text())
        assert {row["sample_id"] for row in report["records"]} == set(workflow["manifest"]["folds"][fold]["test_ids"])
        assert len(report["records"]) == len(panel.gates) * 2
        for row in report["records"]:
            assert sum(row[key] for key in ("tp", "tn", "fp", "fn")) == workflow["counts"][row["sample_id"]]
            for key in ("accuracy", "recall", "precision", "f1"):
                assert 0 <= row[key] <= 1


def test_prediction_ignores_annotations_and_preserves_parent_containment(workflow):
    from cda_unet.engine import predict_file

    panel = get_panel(workflow["config"]["modality"])
    sample_id = workflow["manifest"]["folds"][0]["test_ids"][0]
    original = workflow["source"] / f"{sample_id}.csv"
    frame = pd.read_csv(original)
    frame[panel.label_column] = "NOT A VALID CONSENSUS LABEL"
    for gate in panel.gates:
        frame[gate.name] = "NOT A BINARY LABEL"
    alternate_dir = workflow["directory"] / "corrupted_annotations"
    alternate_dir.mkdir(exist_ok=True)
    corrupted = alternate_dir / original.name
    frame.to_csv(corrupted, index=False)
    expected = workflow["directory"] / "prediction_original.npz"
    actual = workflow["directory"] / "prediction_corrupted.npz"
    predict_file(workflow["checkpoints"][0], original, expected, device="cpu")
    predict_file(workflow["checkpoints"][0], corrupted, actual, device="cpu")
    with np.load(expected, allow_pickle=False) as archive:
        expected_predictions = archive["predictions"].copy()
    with np.load(actual, allow_pickle=False) as archive:
        predictions = archive["predictions"].copy()
        assert list(archive["gate_names"]) == [gate.name for gate in panel.gates]
    np.testing.assert_array_equal(predictions, expected_predictions)
    assert predictions.shape == (workflow["counts"][sample_id], len(panel.gates))
    assert np.isin(predictions, [0, 1]).all()
    by_name = {gate.name: index for index, gate in enumerate(panel.gates)}
    for gate_id, gate in enumerate(panel.gates):
        if gate.parent is not None:
            assert np.all(predictions[:, gate_id] <= predictions[:, by_name[gate.parent]])


def test_aggregation_rejects_duplicate_heldout_records(workflow):
    from cda_unet.engine import aggregate_reports

    output = workflow["directory"] / "aggregate"
    aggregate_reports(workflow["reports"], output)
    report = json.loads((output / "report.json").read_text())
    assert report["complete_cross_validation"]
    assert report["summary"]["n_samples"] == len(workflow["counts"])
    with pytest.raises(ValueError, match="[Dd]uplicate|[Oo]verlap"):
        aggregate_reports(workflow["reports"] + [workflow["reports"][0]], workflow["directory"] / "bad_aggregate")


def test_training_rejects_modified_prepared_input(workflow):
    from cda_unet.engine import train_fold

    entry = workflow["manifest"]["samples"][0]
    cache_file = workflow["cache"] / entry["cache_file"]
    original = cache_file.read_bytes()
    try:
        cache_file.write_bytes(original + b"unexpected modification")
        with pytest.raises(ValueError, match="[Cc]ache|[Hh]ash|[Ss]ignature|[Mm]ismatch|modified"):
            train_fold(workflow["config"], workflow["cache"], workflow["directory"] / "stale_run", 0, device="cpu")
    finally:
        cache_file.write_bytes(original)


def test_evaluation_rejects_changed_heldout_source(workflow):
    from cda_unet.engine import evaluate_checkpoint

    sample_id = workflow["manifest"]["folds"][0]["test_ids"][0]
    source = workflow["source"] / f"{sample_id}.csv"
    original = source.read_bytes()
    try:
        source.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="missing or changed"):
            evaluate_checkpoint(workflow["checkpoints"][0], workflow["source"], workflow["directory"] / "changed_source_evaluation", device="cpu")
    finally:
        source.write_bytes(original)


def test_resume_matches_uninterrupted_cpu_training(workflow):
    from cda_unet.engine import train_fold

    checkpoint = train_fold(workflow["config"], workflow["cache"], workflow["directory"] / "uninterrupted", 0, device="cpu", epochs=2)
    uninterrupted = torch.load(checkpoint, map_location="cpu", weights_only=True)
    resumed = torch.load(workflow["checkpoints"][0], map_location="cpu", weights_only=True)
    assert uninterrupted["history"] == resumed["history"]
    for name, value in uninterrupted["model_state"].items():
        torch.testing.assert_close(value, resumed["model_state"][name], rtol=0, atol=0)
    for parameter_id, state in uninterrupted["optimizer_state"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, resumed["optimizer_state"]["state"][parameter_id][name], rtol=0, atol=0)


@pytest.mark.parametrize("change, message", [("epoch", "epochs"), ("fold", "duplicate fold"), ("invalid_fold", "out-of-range fold")])
def test_aggregation_rejects_incompatible_fold_reports(workflow, change, message):
    from cda_unet.engine import aggregate_reports

    report = json.loads(workflow["reports"][1].read_text())
    if change == "epoch":
        report["epoch"] -= 1
    elif change == "fold":
        report["fold"] = 0
    else:
        report["fold"] = 2
    modified = workflow["directory"] / f"report_modified_{change}.json"
    modified.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=message):
        aggregate_reports([workflow["reports"][0], modified], workflow["directory"] / f"invalid_aggregate_{change}")
