"""Whole-event binary metrics followed by subject- and gate-balanced means."""
from collections import defaultdict
import numpy as np

METRICS = ("accuracy", "recall", "precision", "f1")


def score_binary(ytrue, ypred) -> dict:
    truth, prediction = np.asarray(ytrue), np.asarray(ypred)
    if truth.ndim != 1 or truth.shape != prediction.shape or not truth.size:
        raise ValueError("Scoring expects equal, nonempty event vectors")
    if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Scoring expects binary event labels")
    truth, prediction = truth.astype(bool), prediction.astype(bool)
    tp = int(np.count_nonzero(truth & prediction))
    tn = int(np.count_nonzero(~truth & ~prediction))
    fp = int(np.count_nonzero(~truth & prediction))
    fn = int(np.count_nonzero(truth & ~prediction))
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=(tp+tn)/truth.size,
                recall=tp/(tp+fn) if tp+fn else 0.0,
                precision=tp/(tp+fp) if tp+fp else 0.0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)


def summarize(records: list[dict], panel) -> dict:
    """Do not pool events or reconstruct F1 from averaged recall/precision."""
    if not records:
        raise ValueError("Cannot summarize empty evaluation records")
    gate_names = [gate.name for gate in panel.gates]
    groups, per_subject, seen = defaultdict(list), defaultdict(set), set()
    for row in records:
        key = (str(row["sample_id"]), str(row["gate"]))
        if key in seen:
            raise ValueError(f"Duplicate subject/gate in evaluation: {key}")
        if key[1] not in gate_names:
            raise ValueError(f"Unknown gate: {key[1]}")
        for metric in METRICS:
            value = float(row[metric])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {metric} for {key}")
        seen.add(key); groups[key[1]].append(row); per_subject[key[0]].add(key[1])
    for sample, names in per_subject.items():
        if names != set(gate_names):
            raise ValueError(f"Incomplete gate coverage for {sample}")
    per_gate = []
    for gate in gate_names:
        rows = groups[gate]
        per_gate.append({"gate": gate, "n_samples": len(rows),
                         **{m: float(np.mean([float(r[m]) for r in rows])) for m in METRICS}})
    overall = {m: float(np.mean([row[m] for row in per_gate])) for m in METRICS}
    return {"n_samples": len(per_subject), "n_gates": len(gate_names),
            "per_gate": per_gate, "overall": overall}
