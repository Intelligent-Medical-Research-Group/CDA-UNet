"""Reproducible subject-level preparation, training and cell-wise inference.

Only compact density images and binary raster targets enter the training
cache. The default release trains with consensus parents and evaluates a
model's own predicted hierarchy. Supplied parent arrays are an explicit,
separately recorded evaluation/training option.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from .data import backproject, discover_samples, load_sample, rasterize
from .config import validate_config
from .model import CDAUNet
from .schema import get_panel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _software_versions() -> dict:
    return {"python": platform.python_version(), "pytorch": str(torch.__version__),
            "numpy": str(np.__version__), "cuda_runtime": torch.version.cuda}


def _write_metric_tables(output_dir: Path, records: list[dict], summary: dict) -> None:
    metrics = ("accuracy", "recall", "precision", "f1")
    tables = (
        ("per_sample.csv", ("sample_id", "gate", "gate_id", "tp", "tn", "fp", "fn", *metrics), records),
        ("per_gate.csv", ("gate", "n_samples", *metrics), summary["per_gate"] + [
            {"gate": "Overall", "n_samples": summary["n_samples"], **summary["overall"]}]),
    )
    for filename, fields, rows in tables:
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: f"{float(row[key]):.4f}" if key in metrics else row.get(key, "") for key in fields})


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".json-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _save_torch(path: Path, value) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(value, stream)
    os.replace(temporary, path)


def _data_options(config: dict) -> dict:
    train = config["training"]
    return {
        "schema_version": config["schema_version"], "modality": config["modality"],
        "resolution": config["resolution"], "parent_policy": train["parent_policy"],
        "seed": train["seed"], "folds": train["folds"], "fold_strategy": train["fold_strategy"],
    }


def _panel_contract(panel) -> dict:
    return {"channels": list(panel.channels), "gates": [
        {"name": gate.name, "parent": gate.parent, "x": gate.x, "y": gate.y,
         "positive_labels": list(gate.positive_labels)} for gate in panel.gates]}


def _load_parents(path: Path, sample_path: Path, panel, event_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        required = {"predictions", "gate_names", "source_sha256"}
        if not required.issubset(data.files):
            raise ValueError(f"Parent file {path} requires {sorted(required)}")
        names = data["gate_names"].tolist()
        if names != [gate.name for gate in panel.gates]:
            raise ValueError(f"Parent gate order does not match the panel: {path}")
        if data["source_sha256"].shape != () or str(data["source_sha256"].item()) != _sha256(sample_path):
            raise ValueError(f"Parent source SHA256 does not match the exact event file: {path}")
        predictions = data["predictions"]
        if predictions.shape != (event_count, len(panel.gates)) or not np.isin(predictions, [0, 1]).all():
            raise ValueError(f"Parent predictions must be binary (events, gates): {path}")
        return predictions.astype(np.uint8, copy=True)


def _cache_identity(manifest: dict) -> str:
    return _json_fingerprint({key: manifest[key] for key in
                              ("data_options", "panel", "samples", "folds")})


def _check_manifest(manifest: dict, config: dict, cache_dir: Path, verify_files: bool = True) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported cache schema version")
    if manifest["data_options"] != _data_options(config):
        raise ValueError("Cache preprocessing or fold options differ from the requested configuration")
    if manifest["panel"] != _panel_contract(get_panel(config["modality"])):
        raise ValueError("Cache panel schema does not match this software version")
    if manifest["data_signature"] != _cache_identity(manifest):
        raise ValueError("Cache manifest content does not match its recorded signature")
    if verify_files:
        for record in manifest["samples"]:
            path = cache_dir / record["cache_file"]
            if not path.is_file() or _sha256(path) != record["cache_sha256"]:
                raise ValueError(f"Missing or modified training cache: {path}")


def prepare_data(config: dict, data_dir: Path, cache_dir: Path, parents_dir: Path | None = None) -> dict:
    """Create or verify an immutable image cache and subject-level split manifest."""
    config = validate_config(config)
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    parents_dir = None if parents_dir is None else Path(parents_dir)
    panel = get_panel(config["modality"])
    parent_policy = config["training"]["parent_policy"]
    if parent_policy not in ("consensus", "provided"):
        raise ValueError("Training parent_policy must be consensus or provided")
    if (parents_dir is not None) != (parent_policy == "provided"):
        raise ValueError("Provided training parents require both parent_policy=provided and parents_dir")
    files = sorted(discover_samples(data_dir), key=lambda path: path.stem)
    fold_count = int(config["training"]["folds"])
    if not 2 <= fold_count <= len(files):
        raise ValueError("Cross-validation needs at least two nonempty folds and one subject per fold")
    permutation = np.random.default_rng(config["training"]["seed"]).permutation(len(files))
    strategy = config["training"]["fold_strategy"]
    if strategy == "balanced":
        subsets = np.array_split(permutation, fold_count)
    else:
        chunk_size = int(np.ceil(len(files) / fold_count))
        subsets = [permutation[start:start + chunk_size] for start in range(0, len(files), chunk_size)]
        if len(subsets) != fold_count:
            raise ValueError("Requested chunk strategy cannot produce the requested number of nonempty folds")
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        _check_manifest(manifest, config, cache_dir)
        recorded = {row["sample_id"]: row for row in manifest["samples"]}
        if set(recorded) != {path.stem for path in files}:
            raise ValueError("The existing cache contains a different set of subjects; use a new cache directory")
        for path in files:
            row = recorded[path.stem]
            if _sha256(path) != row["source_sha256"]:
                raise ValueError(f"Source CSV changed after cache preparation: {path}")
            if parents_dir is not None and _sha256(parents_dir / f"{path.stem}.npz") != row["parent_sha256"]:
                raise ValueError(f"Provided parent file changed for {path.stem}")
        return manifest
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise FileExistsError("Cache directory is nonempty without a valid manifest; use a new directory")
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "samples").mkdir()
    records = []
    gate_index = {gate.name: index for index, gate in enumerate(panel.gates)}
    for index, path in enumerate(files):
        sample = load_sample(path, panel, require_labels=True)
        parents_path = None if parents_dir is None else parents_dir / f"{sample.sample_id}.npz"
        parents = sample.labels if parents_path is None else _load_parents(parents_path, path, panel, len(sample.channels))
        images, targets = [], []
        for gate_id, gate in enumerate(panel.gates):
            selection = None if gate.parent is None else parents[:, gate_index[gate.parent]]
            raster = rasterize(sample, panel, gate_id, selection, include_target=True, resolution=config["resolution"])
            images.append(raster.image)
            targets.append(raster.target)
        cache_file = f"samples/sample_{index:05d}.npz"
        np.savez_compressed(cache_dir / cache_file, images=np.stack(images), targets=np.stack(targets))
        record = {"sample_id": sample.sample_id, "relative_path": path.relative_to(data_dir).as_posix(),
                  "source_sha256": _sha256(path), "event_count": len(sample.channels),
                  "cache_file": cache_file, "cache_sha256": _sha256(cache_dir / cache_file)}
        if parents_path is not None:
            record["parent_sha256"] = _sha256(parents_path)
        records.append(record)
    all_ids = [record["sample_id"] for record in records]
    folds = []
    for fold, subset in enumerate(subsets):
        test_ids = [all_ids[int(index)] for index in subset]
        test_set = set(test_ids)
        folds.append({"fold": fold, "train_ids": [name for name in all_ids if name not in test_set], "test_ids": test_ids})
    manifest = {"schema_version": 1, "config": copy.deepcopy(config), "data_options": _data_options(config),
                "panel": _panel_contract(panel), "gate_names": [gate.name for gate in panel.gates],
                "samples": records, "folds": folds}
    manifest["data_signature"] = _cache_identity(manifest)
    _write_json(manifest_path, manifest)
    return manifest


def _device(name: str) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("A CUDA device was requested but CUDA is unavailable")
    return device


def _read_checkpoint(path: Path) -> dict:
    # Checkpoints consist only of tensors and Python primitives. Never fall
    # back to unrestricted pickle loading for externally obtained files.
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"schema_version", "config", "model_state", "epoch", "fold", "train_ids", "test_ids", "samples"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Not a CDA-UNet release checkpoint")
    if checkpoint["schema_version"] != 1:
        raise ValueError("Unsupported checkpoint schema version")
    checkpoint["config"] = validate_config(checkpoint["config"])
    if type(checkpoint["epoch"]) is not int or checkpoint["epoch"] < 1:
        raise ValueError("Checkpoint epoch must be a positive integer")
    if type(checkpoint["fold"]) is not int or not 0 <= checkpoint["fold"] < checkpoint["config"]["training"]["folds"]:
        raise ValueError("Checkpoint fold is outside the configured fold range")
    for field in ("train_ids", "test_ids"):
        ids = checkpoint[field]
        if not isinstance(ids, list) or not ids or not all(isinstance(item, str) for item in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Checkpoint {field} must contain unique nonempty subject IDs")
    if set(checkpoint["train_ids"]) & set(checkpoint["test_ids"]):
        raise ValueError("Checkpoint train/test subjects overlap")
    return checkpoint


def train_fold(config: dict, cache_dir: Path, output_dir: Path, fold: int, device: str = "auto",
               resume: Path | None = None, epochs: int | None = None) -> Path:
    """Train one fold without reading held-out images or selecting on test scores.

    ``epochs`` is the total target epoch, not the number of additional epochs.
    It is useful for smoke checks and interrupted runs; the canonical release
    configuration records 1000 epochs. Resume requires an identical cache and
    training configuration and restores optimizer and random-number state.
    """
    config = validate_config(config)
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    _check_manifest(manifest, config, cache_dir)
    if type(fold) is not int or not 0 <= fold < len(manifest["folds"]):
        raise ValueError("Requested fold is outside the cache split manifest")
    split = manifest["folds"][fold]
    train_ids, test_ids = split["train_ids"], split["test_ids"]
    if not train_ids or not test_ids or set(train_ids) & set(test_ids):
        raise ValueError("Fold must have disjoint, nonempty train and test subjects")
    training = config["training"]
    target_epochs = training["epochs"] if epochs is None else epochs
    if type(target_epochs) is not int or target_epochs < 1:
        raise ValueError("epochs must be a positive integer")
    run_dir = output_dir / f"fold_{fold:02d}"
    if run_dir.exists() and any(run_dir.iterdir()) and resume is None:
        raise FileExistsError(f"Run already exists: {run_dir}; specify an explicit resume checkpoint")
    run_dir.mkdir(parents=True, exist_ok=True)
    selected_device = _device(device)
    seed = int(training["seed"]) + fold
    np.random.seed(seed)
    torch.manual_seed(seed)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    panel = get_panel(config["modality"])
    model = CDAUNet(panel, rank=config["rank"], features=config["features"]).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    start_epoch, history = 0, []
    if resume is not None:
        checkpoint = _read_checkpoint(Path(resume))
        if checkpoint.get("cache_signature") != manifest["data_signature"] or checkpoint["fold"] != fold:
            raise ValueError("Resume checkpoint belongs to a different cache or fold")
        if checkpoint["config"] != config or checkpoint["train_ids"] != train_ids or checkpoint["test_ids"] != test_ids:
            raise ValueError("Resume checkpoint configuration or subject split differs")
        if "optimizer_state" not in checkpoint:
            raise ValueError("Resume requires optimizer state")
        if (run_dir / "last.pt").exists():
            latest = _read_checkpoint(run_dir / "last.pt")
            if latest["epoch"] > checkpoint["epoch"]:
                raise ValueError("An earlier checkpoint cannot overwrite a later run; choose a new output directory")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch, history = int(checkpoint["epoch"]), list(checkpoint["history"])
        torch.set_rng_state(checkpoint["rng_state"])
        if selected_device.type == "cuda" and checkpoint.get("cuda_rng_states"):
            saved = checkpoint["cuda_rng_states"]
            if len(saved) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(saved)
        if target_epochs <= start_epoch:
            raise ValueError("Resume target epochs must exceed the checkpoint's completed epoch")
    lookup = {record["sample_id"]: record for record in manifest["samples"]}
    # The held-out sample caches are never loaded into the optimizer pipeline.
    images, targets = [], []
    for sample_id in train_ids:
        with np.load(cache_dir / lookup[sample_id]["cache_file"], allow_pickle=False) as data:
            images.append(torch.from_numpy(data["images"].copy()))
            targets.append(torch.from_numpy(data["targets"].astype(np.float32)))
    image_tensor, target_tensor = torch.stack(images), torch.stack(targets)
    amp = bool(training["amp"] and selected_device.type == "cuda" and torch.cuda.is_bf16_supported())
    batch_size = int(training["batch_size"])
    interval = int(training["checkpoint_every"])
    if batch_size < 1 or interval < 1:
        raise ValueError("batch_size and checkpoint_every must be positive")
    _write_json(run_dir / "config.json", config)
    model.train()
    last_path = run_dir / "last.pt"
    for epoch in range(start_epoch + 1, target_epochs + 1):
        rng = np.random.default_rng(seed + 1_000_003 * epoch)
        batches = []
        for gate_id in range(len(panel.gates)):
            order = rng.permutation(len(train_ids)) if training["shuffle"] else np.arange(len(train_ids))
            batches.extend((gate_id, order[start:start + batch_size]) for start in range(0, len(order), batch_size))
        if training["shuffle"]:
            rng.shuffle(batches)
        loss_sum = 0.0
        for gate_id, indices in batches:
            batch_images = image_tensor[indices, gate_id].to(selected_device)
            batch_targets = target_tensor[indices, gate_id].to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=selected_device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(batch_images, gate_id)
                loss = F.binary_cross_entropy_with_logits(logits, batch_targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}, gate {gate_id}")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
        row = {"epoch": epoch, "loss": loss_sum / len(batches), "batches": len(batches)}
        history.append(row)
        print(f"fold={fold} epoch={epoch}/{target_epochs} loss={row['loss']:.6f}", flush=True)
        if epoch % interval == 0 or epoch == target_epochs:
            checkpoint = {"schema_version": 1, "config": copy.deepcopy(config), "epoch": epoch,
                          "fold": fold, "train_ids": train_ids, "test_ids": test_ids,
                          "samples": manifest["samples"], "panel": manifest["panel"],
                          "cache_signature": manifest["data_signature"], "history": history,
                          "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                          "rng_state": torch.get_rng_state(),
                          "cuda_rng_states": torch.cuda.get_rng_state_all() if selected_device.type == "cuda" else [],
                          "precision": "bfloat16" if amp else "float32", "software_versions": _software_versions()}
            _save_torch(last_path, checkpoint)
            if epoch % interval == 0:
                _save_torch(run_dir / f"epoch_{epoch:04d}.pt", checkpoint)
            _write_json(run_dir / "history.json", history)
    return last_path


def _inference_model(checkpoint: dict, device: str):
    config = checkpoint["config"]
    panel = get_panel(config["modality"])
    if checkpoint.get("panel") != _panel_contract(panel):
        raise ValueError("Checkpoint panel schema differs from the installed schema")
    selected_device = _device(device)
    model = CDAUNet(panel, rank=config["rank"], features=config["features"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(selected_device).eval()
    return model, panel, selected_device


def _predict(model, sample, panel, config, selected_device, supplied_parents=None) -> np.ndarray:
    predictions = np.zeros((len(sample.channels), len(panel.gates)), dtype=np.uint8)
    names = {gate.name: index for index, gate in enumerate(panel.gates)}
    with torch.inference_mode():
        for gate_id, gate in enumerate(panel.gates):
            parents = predictions if supplied_parents is None else supplied_parents
            selection = None if gate.parent is None else parents[:, names[gate.parent]]
            raster = rasterize(sample, panel, gate_id, selection, resolution=config["resolution"])
            if not len(raster.event_indices):
                continue
            images = torch.from_numpy(raster.image[None]).to(selected_device)
            logits = model(images, gate_id)[0, 0].cpu().float().numpy()
            mask = logits > float(config["evaluation"]["threshold"])
            predictions[:, gate_id] = backproject(mask, raster, len(sample.channels))
    return predictions


def _save_predictions(path: Path, predictions, sample_id: str, source_path: Path, panel) -> None:
    if path.suffix.lower() != ".npz":
        raise ValueError("Prediction output must have an .npz extension")
    if path.exists():
        raise FileExistsError(f"Prediction output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, predictions=predictions, gate_names=np.array([gate.name for gate in panel.gates]),
                        sample_id=np.array(sample_id), source_sha256=np.array(_sha256(source_path)))


def _evaluation_policy(config: dict, supplied: bool) -> str:
    policy = config["evaluation"]["parent_policy"]
    if policy not in ("self", "provided"):
        raise ValueError("Evaluation parent_policy must be self or provided")
    if supplied:
        return "provided"
    if policy == "provided":
        raise ValueError("Provided-parent evaluation requires parent prediction files")
    return "self"


def evaluate_checkpoint(checkpoint: Path, data_dir: Path, output_dir: Path, device: str = "auto",
                        parents_dir: Path | None = None) -> dict:
    """Evaluate only the held-out subjects recorded in one checkpoint."""
    from .metrics import score_binary, summarize

    checkpoint_path, output_dir = Path(checkpoint), Path(output_dir)
    saved = _read_checkpoint(checkpoint_path)
    parent_policy = _evaluation_policy(saved["config"], parents_dir is not None)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output directory is nonempty: {output_dir}")
    files = {path.stem: path for path in discover_samples(data_dir)}
    sources = {row["sample_id"]: row for row in saved["samples"]}
    for sample_id in saved["test_ids"]:
        if sample_id not in files or _sha256(files[sample_id]) != sources[sample_id]["source_sha256"]:
            raise ValueError(f"Held-out source file is missing or changed: {sample_id}")
    model, panel, selected_device = _inference_model(saved, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_id in saved["test_ids"]:
        path = files[sample_id]
        sample = load_sample(path, panel, require_labels=True)
        supplied = None if parents_dir is None else _load_parents(Path(parents_dir) / f"{sample_id}.npz", path, panel, len(sample.channels))
        predictions = _predict(model, sample, panel, saved["config"], selected_device, supplied)
        _save_predictions(output_dir / "predictions" / f"{sample_id}.npz", predictions, sample_id, path, panel)
        for gate_id, gate in enumerate(panel.gates):
            records.append({"sample_id": sample_id, "gate": gate.name, "gate_id": gate_id,
                            **score_binary(sample.labels[:, gate_id], predictions[:, gate_id])})
    report = {"schema_version": 1, "modality": panel.name, "config": saved["config"],
              "checkpoint": checkpoint_path.name, "checkpoint_sha256": _sha256(checkpoint_path),
              "epoch": saved["epoch"], "fold": saved["fold"], "test_ids": saved["test_ids"],
              "parent_policy": parent_policy, "cache_signature": saved["cache_signature"],
              "records": records, "summary": summarize(records, panel),
              "software_versions": _software_versions(),
              "training_software_versions": saved.get("software_versions", {})}
    _write_metric_tables(output_dir, records, report["summary"])
    _write_json(output_dir / "report.json", report)
    return report


def predict_file(checkpoint: Path, input_path: Path, output_path: Path, device: str = "auto",
                 parents_file: Path | None = None) -> dict:
    """Predict labels without reading any consensus or expert annotation column."""
    saved = _read_checkpoint(Path(checkpoint))
    parent_policy = _evaluation_policy(saved["config"], parents_file is not None)
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Prediction output already exists: {output_path}")
    model, panel, selected_device = _inference_model(saved, device)
    sample = load_sample(input_path, panel, require_labels=False)
    supplied = None if parents_file is None else _load_parents(Path(parents_file), input_path, panel, len(sample.channels))
    predictions = _predict(model, sample, panel, saved["config"], selected_device, supplied)
    _save_predictions(output_path, predictions, sample.sample_id, input_path, panel)
    return {"sample_id": sample.sample_id, "events": len(sample.channels), "parent_policy": parent_policy,
            "gate_names": [gate.name for gate in panel.gates], "positive_counts": predictions.sum(axis=0).tolist(),
            "output": str(output_path)}


def aggregate_reports(paths: list[Path], output_dir: Path) -> dict:
    """Aggregate non-overlapping out-of-fold subject records with equal weights."""
    from .metrics import summarize

    if not paths:
        raise ValueError("At least one evaluation report is required")
    reports = [json.loads(Path(path).read_text()) for path in paths]
    first = reports[0]
    required = ("schema_version", "modality", "parent_policy", "cache_signature", "config", "epoch")
    seen, seen_folds, records = set(), set(), []
    for report in reports:
        if any(report.get(key) != first.get(key) for key in required):
            raise ValueError("Cannot aggregate reports with different panels, protocols, caches, configurations or epochs")
        fold = report["fold"]
        if type(fold) is not int or not 0 <= fold < first["config"]["training"]["folds"]:
            raise ValueError("Evaluation report contains an out-of-range fold index")
        if fold in seen_folds:
            raise ValueError("Evaluation reports contain duplicate fold indices")
        seen_folds.add(fold)
        ids = report["test_ids"]
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("Evaluation reports contain overlapping test subjects")
        if {row["sample_id"] for row in report["records"]} != set(ids):
            raise ValueError("Report records do not match its declared test subjects")
        seen.update(ids)
        records.extend(report["records"])
    panel = get_panel(first["modality"])
    summary = summarize(records, panel)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Aggregate output directory is nonempty: {output_dir}")
    result = {key: first[key] for key in required}
    result.update({"reports": [str(Path(path)) for path in paths], "test_ids": sorted(seen),
                   "folds": [report["fold"] for report in reports], "epochs": [report["epoch"] for report in reports],
                   "records": records, "summary": summary,
                   "complete_cross_validation": seen_folds == set(range(first["config"]["training"]["folds"])),
                   "software_versions": _software_versions()})
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metric_tables(output_dir, records, summary)
    _write_json(output_dir / "report.json", result)
    return result
