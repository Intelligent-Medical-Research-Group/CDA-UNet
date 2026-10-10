"""Read and validate the small, explicit public experiment configuration."""
from pathlib import Path
import copy
import yaml
from .schema import get_panel


def validate_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    if cfg.get("schema_version") != 1:
        raise ValueError("Configuration schema_version must be 1")
    get_panel(cfg.get("modality"))
    if cfg.get("resolution") != 101 or cfg.get("rank") != 32:
        raise ValueError("The public reference configuration uses resolution 101 and rank 32")
    if cfg.get("conditioning") != "xy_gate":
        raise ValueError("conditioning must be xy_gate (ordered channel IDs and gate ID)")
    if cfg.get("normalization") != "input_group_encoder_gate_decoder":
        raise ValueError("Use input_group_encoder_gate_decoder normalization")
    features = cfg.get("features", [64, 128, 256, 512])
    if len(features) != 4 or any(type(x) is not int or x < 1 for x in features):
        raise ValueError("features must be four positive integer channel widths")
    if any(b != 2*a for a,b in zip(features, features[1:])):
        raise ValueError("Successive channel widths must double")
    cfg["features"] = features
    train = cfg.get("training", {})
    for key in ("epochs", "batch_size", "folds", "checkpoint_every"):
        if type(train.get(key)) is not int or train[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if train["folds"] < 2:
        raise ValueError("Subject cross-validation requires at least two folds")
    if type(train.get("seed")) is not int or train["seed"] < 0:
        raise ValueError("training.seed must be a nonnegative integer")
    train["learning_rate"] = float(train.get("learning_rate", 0))
    if not 0 < train["learning_rate"] < 1:
        raise ValueError("training.learning_rate must be between 0 and 1")
    if train.get("fold_strategy") not in ("balanced", "chunks"):
        raise ValueError("fold_strategy must be balanced or chunks")
    if train.get("parent_policy") not in ("consensus", "provided"):
        raise ValueError("Training parents must be consensus or explicitly provided")
    if type(train.get("shuffle")) is not bool or type(train.get("amp")) is not bool:
        raise ValueError("training.shuffle and training.amp must be YAML booleans")
    evaluation = cfg.get("evaluation", {})
    if evaluation.get("parent_policy") not in ("self", "provided"):
        raise ValueError("Evaluation parents must be self or explicitly provided")
    evaluation["threshold"] = float(evaluation.get("threshold", 0))
    if evaluation["threshold"] != 0:
        raise ValueError("The reference uses logit > 0, equivalent to probability > 0.5")
    return cfg


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return validate_config(data)
