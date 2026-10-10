"""CSV loading, density-map construction, and cell-wise backprojection.

Only measurement columns are touched in label-free mode. Non-root rasterization
always requires an explicit parent selection, so ground-truth ancestors cannot
silently enter deployment inference.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from .schema import Panel


@dataclass
class Sample:
    sample_id: str
    channels: np.ndarray
    labels: np.ndarray | None = None


@dataclass
class Raster:
    image: np.ndarray
    target: np.ndarray | None
    event_indices: np.ndarray
    rows: np.ndarray
    cols: np.ndarray


def discover_samples(input_dir: str | Path) -> list[Path]:
    """Find CSV samples recursively; every CSV must be a measurement file.

    Use a dedicated sample directory, without metadata CSVs. Unique basenames
    are required because output files and split manifests use the basename as
    the sample identifier. Files are returned in a deterministic path order.
    """
    directory = Path(input_dir)
    if not directory.is_dir():
        raise ValueError(f"Sample directory does not exist: {directory}")
    paths = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".csv")
    if not paths:
        raise ValueError(f"No CSV samples found in {directory}")
    seen: dict[str, Path] = {}
    for path in paths:
        if path.stem in seen:
            raise ValueError(f"Duplicate sample ID {path.stem!r}: {seen[path.stem]} and {path}")
        seen[path.stem] = path
    return paths


def load_sample(path: str | Path, panel: Panel, require_labels: bool = False) -> Sample:
    """Read a released CSV without altering event order or raw measurements.

    ``require_labels=False`` deliberately ignores all annotation columns,
    including malformed labels. Training/evaluation must explicitly request
    labels. Supervision is expanded from the deepest ``GaterConsensus`` label;
    any flow binary gate columns present are checked against that expansion.
    """
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a sample CSV, got {path}")
    header = pd.read_csv(path, nrows=0).columns
    required = list(panel.channels) + ([panel.label_column] if require_labels else [])
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    audit_names = [gate.name for gate in panel.gates if panel.name == "flow" and gate.name in header] if require_labels else []
    table = pd.read_csv(path, usecols=list(dict.fromkeys(required + audit_names)))
    if table.empty:
        raise ValueError(f"Empty sample: {path}")
    try:
        channels = table[list(panel.channels)].to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric measurement in {path}") from exc
    if not np.isfinite(channels).all():
        raise ValueError(f"Non-finite measurement in {path}")
    labels = None
    if require_labels:
        codes = pd.Index(panel.consensus_labels).get_indexer(table[panel.label_column])
        if np.any(codes < 0):
            unknown = table.loc[codes < 0, panel.label_column].value_counts(dropna=False).to_dict()
            raise ValueError(f"Unknown consensus labels in {path}: {unknown}")
        lookup = np.array([[label in gate.positive_labels for gate in panel.gates]
                           for label in panel.consensus_labels], dtype=np.uint8)
        labels = lookup[codes]
        for gate_id, gate in enumerate(panel.gates):
            if gate.name not in audit_names:
                continue
            supplied = table[gate.name].to_numpy()
            if not np.isin(supplied, [0, 1]).all():
                raise ValueError(f"Non-binary gate label in {path}, gate {gate.name}")
            if not np.array_equal(supplied, labels[:, gate_id]):
                raise ValueError(f"Consensus mismatch in {path}, gate {gate.name}")
    return Sample(path.stem, channels, labels)


def fill_hull(mask: np.ndarray) -> np.ndarray:
    """Fill the occupied-pixel convex hull using the original OpenCV recipe.

    With at most three occupied pixels or a collinear set, retain only occupied
    pixels. This also handles empty targets and rare degenerate parent subsets.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    points = np.argwhere(mask).astype(np.int32)
    if len(points) <= 3:
        return mask.copy()
    try:
        hull = ConvexHull(points)
    except QhullError:
        return mask.copy()
    vertices = points[hull.vertices][:, ::-1].copy()
    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(filled, pts=[vertices], color=1)
    return filled


def rasterize(
    sample: Sample,
    panel: Panel,
    gate_id: int,
    parent_selection: np.ndarray | None,
    include_target: bool = False,
    resolution: int = 101,
) -> Raster:
    """Construct one density image in the selected parent population.

    Coordinates use per-sample/per-parent min-max normalization to [0, R-1],
    NumPy round-to-even, and an inverted first coordinate (the image row).
    Counts are divided by their peak. For flow data only, scatter coordinates
    are clamped below at zero on a copy; fluorescence and raw arrays are kept.
    A target hull includes only positive events inside the supplied parent.
    """
    if not isinstance(gate_id, (int, np.integer)) or not 0 <= gate_id < len(panel.gates):
        raise ValueError(f"Invalid gate index: {gate_id}")
    if not isinstance(resolution, (int, np.integer)) or isinstance(resolution, bool) or resolution < 2:
        raise ValueError("Resolution must be an integer of at least 2")
    gate = panel.gates[gate_id]
    channels = np.asarray(sample.channels)
    if channels.ndim != 2 or channels.shape[1] != len(panel.channels):
        raise ValueError(f"Measurements must have shape (events, {len(panel.channels)})")
    if not np.isfinite(channels).all():
        raise ValueError("Non-finite sample measurements")
    n_events = len(channels)
    if parent_selection is None:
        if gate.parent is not None:
            raise ValueError(f"Gate {gate.name} requires explicit parent_selection")
        indices = np.arange(n_events, dtype=np.int64)
    else:
        parent_selection = np.asarray(parent_selection)
        if parent_selection.shape != (n_events,):
            raise ValueError(f"Parent selection must have shape {(n_events,)}")
        if not np.isin(parent_selection, [0, 1]).all():
            raise ValueError("Parent selection must be boolean or binary")
        indices = np.flatnonzero(parent_selection)
    if include_target:
        if sample.labels is None:
            raise ValueError("Target construction requires consensus labels")
        labels = np.asarray(sample.labels)
        if labels.shape != (n_events, len(panel.gates)) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Labels must be a binary (events, gates) array")
    image = np.zeros((1, resolution, resolution), dtype=np.float32)
    target = np.zeros((1, resolution, resolution), dtype=np.uint8) if include_target else None
    if not len(indices):
        return Raster(image, target, indices, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    axes = [panel.channels.index(gate.x), panel.channels.index(gate.y)]
    xy = channels[np.ix_(indices, axes)].astype(np.float64, copy=True)
    for axis, name in enumerate((gate.x, gate.y)):
        if name in panel.scatter_channels:
            np.maximum(xy[:, axis], 0, out=xy[:, axis])
    lower = xy.min(axis=0)
    width = xy.max(axis=0) - lower
    normalized = np.divide(xy - lower, width, out=np.zeros_like(xy), where=width != 0) * (resolution - 1)
    pixels = np.rint(normalized).astype(np.int64)
    pixels[:, 0] = resolution - 1 - pixels[:, 0]
    rows, cols = pixels[:, 0], pixels[:, 1]
    counts = np.bincount(rows * resolution + cols, minlength=resolution * resolution).reshape(resolution, resolution)
    image[0] = (counts / counts.max()).astype(np.float32)
    if include_target:
        positive = labels[indices, gate_id].astype(bool)
        target[0, rows[positive], cols[positive]] = 1
        target[0] = fill_hull(target[0])
    return Raster(image, target, indices, rows, cols)


def backproject(mask: np.ndarray, raster: Raster, n_events: int) -> np.ndarray:
    """Map a binary 2-D mask to original event order; exclude non-parent cells."""
    mask = np.asarray(mask)
    if mask.shape != raster.image.shape[-2:]:
        raise ValueError(f"Expected a 2-D mask of shape {raster.image.shape[-2:]}, got {mask.shape}")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Backprojection requires a binary mask; threshold probabilities first")
    if not isinstance(n_events, (int, np.integer)) or n_events < 0:
        raise ValueError("n_events must be a non-negative integer")
    if len(raster.event_indices) and (raster.event_indices.min() < 0 or raster.event_indices.max() >= n_events):
        raise ValueError("Raster event indices are outside n_events")
    predictions = np.zeros(n_events, dtype=np.uint8)
    predictions[raster.event_indices] = mask[raster.rows, raster.cols]
    return predictions
