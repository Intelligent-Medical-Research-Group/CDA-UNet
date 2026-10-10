"""Regression checks for event identity, parent isolation, and raster semantics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cda_unet.data import Sample, backproject, discover_samples, fill_hull, load_sample, rasterize
from cda_unet.schema import get_panel


def sample_with_axes(panel_name="mass", x=(0, 1, 2), y=(0, 1, 2)):
    panel = get_panel(panel_name)
    channels = np.zeros((len(x), len(panel.channels)), dtype=np.float64)
    gate = panel.gates[0]
    channels[:, panel.channels.index(gate.x)] = x
    channels[:, panel.channels.index(gate.y)] = y
    labels = np.ones((len(x), len(panel.gates)), dtype=np.uint8)
    return panel, Sample("synthetic", channels, labels)


def write_csv(path: Path, panel_name, labels=None):
    panel = get_panel(panel_name)
    n = len(labels) if labels is not None else 3
    table = pd.DataFrame(np.arange(n * len(panel.channels), dtype=np.float64).reshape(n, -1), columns=panel.channels)
    if labels is not None:
        table[panel.label_column] = labels
    table.to_csv(path, index=False)
    return panel, table


@pytest.mark.parametrize("panel_name", ["mass", "flow"])
def test_schema_is_topological_and_ancestor_labels_expand(panel_name):
    panel = get_panel(panel_name)
    seen = {}
    for gate in panel.gates:
        assert gate.parent is None or gate.parent in seen
        assert gate.x in panel.channels and gate.y in panel.channels
        assert set(gate.positive_labels) <= set(panel.consensus_labels)
        if gate.parent is not None:
            assert set(gate.positive_labels) <= set(seen[gate.parent].positive_labels)
        seen[gate.name] = gate


def test_csv_deepest_consensus_expands_ancestors(tmp_path):
    path = tmp_path / "sample.csv"
    panel, _ = write_csv(path, "flow", ["CD4naive", "CD8", "Unfiltered"])
    sample = load_sample(path, panel, require_labels=True)
    assert sample.channels.dtype == np.float64
    np.testing.assert_array_equal(sample.labels, [[1, 1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 1, 0, 0], [0] * 7])


def test_unlabeled_inference_ignores_malformed_annotations(tmp_path):
    path = tmp_path / "sample.csv"
    panel, table = write_csv(path, "flow", ["INVALID", None, "Unfiltered"])
    table["CD3"] = ["invalid", 7, -3]
    table.to_csv(path, index=False)
    sample = load_sample(path, panel)
    assert sample.labels is None
    assert rasterize(sample, panel, 0, None).target is None
    with pytest.raises(ValueError, match="Unknown consensus"):
        load_sample(path, panel, require_labels=True)


def test_required_labels_and_measurement_validation(tmp_path):
    path = tmp_path / "sample.csv"
    panel, table = write_csv(path, "mass")
    assert load_sample(path, panel).labels is None
    with pytest.raises(ValueError, match="Missing required columns"):
        load_sample(path, panel, require_labels=True)
    table.loc[0, panel.channels[0]] = np.inf
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Non-finite"):
        load_sample(path, panel)


@pytest.mark.parametrize("binary, message", [([1, 2, 0], "Non-binary"), ([0, 0, 0], "Consensus mismatch")])
def test_optional_flow_binary_columns_are_audited(tmp_path, binary, message):
    path = tmp_path / "sample.csv"
    panel, table = write_csv(path, "flow", ["CD4naive", "CD8", "Unfiltered"])
    table["CD3"] = binary
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_sample(path, panel, require_labels=True)


def test_numpy_round_to_even_and_axis_inversion():
    panel, sample = sample_with_axes(x=(0, 0.005, 0.015, 0.025, 1), y=(0, 0.005, 0.015, 0.025, 1))
    raster = rasterize(sample, panel, 0, None, include_target=True)
    np.testing.assert_array_equal(raster.rows, [100, 100, 98, 98, 0])
    np.testing.assert_array_equal(raster.cols, [0, 0, 2, 2, 100])
    assert raster.image.shape == (1, 101, 101)
    assert raster.image.dtype == np.float32
    assert raster.target.dtype == np.uint8
    assert raster.image[0, 100, 0] == 1
    assert raster.image[0, 0, 100] == 0.5


def test_constant_axes_and_empty_parent():
    panel, sample = sample_with_axes(x=(3, 3, 3), y=(7, 7, 7))
    raster = rasterize(sample, panel, 0, None)
    np.testing.assert_array_equal(raster.rows, [100, 100, 100])
    np.testing.assert_array_equal(raster.cols, [0, 0, 0])
    assert raster.image.sum() == 1
    empty = rasterize(sample, panel, 1, np.zeros(3, dtype=bool), include_target=True)
    assert empty.event_indices.size == 0
    assert empty.image.sum() == empty.target.sum() == 0
    np.testing.assert_array_equal(backproject(np.ones((101, 101)), empty, 3), [0, 0, 0])


def test_parent_is_explicit_and_backprojection_preserves_event_identity():
    panel, sample = sample_with_axes(x=(0, 1, 2), y=(0, 1, 2))
    with pytest.raises(ValueError, match="explicit parent_selection"):
        rasterize(sample, panel, 1, None, include_target=True)
    parent = np.array([0, 1, 1], dtype=np.uint8)
    raster = rasterize(sample, panel, 0, parent)
    np.testing.assert_array_equal(raster.rows, [100, 0])
    np.testing.assert_array_equal(raster.cols, [0, 100])
    np.testing.assert_array_equal(raster.event_indices, [1, 2])
    np.testing.assert_array_equal(backproject(np.ones((101, 101)), raster, 3), [0, 1, 1])
    with pytest.raises(ValueError, match="boolean or binary"):
        rasterize(sample, panel, 0, [0, 2, 1])
    with pytest.raises(ValueError, match="shape"):
        rasterize(sample, panel, 0, [True])


def test_target_requires_labels_and_cannot_include_nonparent_events():
    panel, sample = sample_with_axes(x=(0, 1, 2), y=(0, 1, 2))
    sample.labels[:, 0] = [1, 0, 0]
    raster = rasterize(sample, panel, 0, np.array([0, 1, 1]), include_target=True)
    assert not raster.target.any()
    sample.labels = None
    with pytest.raises(ValueError, match="requires consensus labels"):
        rasterize(sample, panel, 0, None, include_target=True)


def test_flow_scatter_clamps_only_coordinate_copy():
    panel, sample = sample_with_axes("flow", x=(-10, -1, 1), y=(-5, 0, 2))
    original = sample.channels.copy()
    raster = rasterize(sample, panel, 0, None)
    np.testing.assert_array_equal(raster.rows, [100, 100, 0])
    np.testing.assert_array_equal(raster.cols, [0, 0, 100])
    np.testing.assert_array_equal(sample.channels, original)
    sample.channels[:, panel.channels.index("CD3Q605")] = [-10, -1, 1]
    sample.channels[:, panel.channels.index("LIVEDEAD")] = [-5, 0, 2]
    fluorescence = rasterize(sample, panel, 2, np.ones(3, dtype=bool))
    np.testing.assert_array_equal(fluorescence.rows, [100, 18, 0])
    np.testing.assert_array_equal(fluorescence.cols, [0, 71, 100])


def test_small_and_collinear_hulls_keep_only_occupied_pixels():
    for points in [[], [(0, 0)], [(0, 0), (5, 0), (0, 5)], [(0, 0), (1, 1), (2, 2), (4, 4)]]:
        mask = np.zeros((6, 6), dtype=np.uint8)
        for row, col in points:
            mask[row, col] = 1
        np.testing.assert_array_equal(fill_hull(mask), mask)
    square = np.zeros((6, 6), dtype=np.uint8)
    square[0, 0] = square[0, 5] = square[5, 0] = square[5, 5] = 1
    np.testing.assert_array_equal(fill_hull(square), np.ones((6, 6), dtype=np.uint8))


def test_backprojection_requires_binary_2d_mask():
    panel, sample = sample_with_axes()
    raster = rasterize(sample, panel, 0, None)
    with pytest.raises(ValueError, match="binary mask"):
        backproject(np.full((101, 101), 0.5), raster, 3)
    with pytest.raises(ValueError, match="2-D mask"):
        backproject(np.ones((1, 101, 101)), raster, 3)


def test_discovery_requires_unique_ids_and_handles_uppercase_csv(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "b.CSV").touch()
    (tmp_path / "nested" / "a.csv").touch()
    (tmp_path / "ignored.txt").touch()
    assert {p.stem for p in discover_samples(tmp_path)} == {"a", "b"}
    (tmp_path / "nested" / "b.csv").touch()
    with pytest.raises(ValueError, match="Duplicate sample ID"):
        discover_samples(tmp_path)
