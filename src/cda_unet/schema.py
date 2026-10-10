"""Channel vocabularies and ancestor-expanded labels for Pennsieve 408 v2.

Gate order is topological and is part of the model's checkpoint contract.
The consensus annotation stores the deepest attained gate, not independent
binary labels. Each gate therefore includes the labels of its descendants.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Gate:
    name: str
    display: str
    parent: str | None
    x: str
    y: str
    positive_labels: tuple[str, ...]


@dataclass(frozen=True)
class Panel:
    name: str
    channels: tuple[str, ...]
    gates: tuple[Gate, ...]
    label_column: str
    scatter_channels: tuple[str, ...]
    consensus_labels: tuple[str, ...]


_MASS_LABELS = (
    "Unfiltered", "Single cell gate 1", "Single cell gate 2", "Granulocyte",
    "Mononuclear", "Neutrophil", "Eosinophil", "T cell", "B cell", "Non T Non B cell",
)

MASS = Panel(
    name="mass",
    channels=("Event_length", "CD45", "CD19", "CD16", "CD294", "CD3", "CD66b", "DNA1", "DNA2"),
    gates=(
        Gate("Single_cell_gate_1", "Single cell gate 1", None, "DNA1", "Event_length", _MASS_LABELS[1:]),
        Gate("Single_cell_gate_2", "Single cell gate 2", "Single_cell_gate_1", "DNA2", "CD45", _MASS_LABELS[2:]),
        Gate("Granulocyte", "Granulocyte gate", "Single_cell_gate_2", "CD66b", "CD45", ("Granulocyte", "Neutrophil", "Eosinophil")),
        Gate("Mononuclear", "Mononuclear gate", "Single_cell_gate_2", "CD66b", "CD45", ("Mononuclear", "T cell", "B cell", "Non T Non B cell")),
        Gate("Neutrophil", "Neutrophil gate", "Granulocyte", "CD16", "CD294", ("Neutrophil",)),
        Gate("Eosinophil", "Eosinophil gate", "Granulocyte", "CD16", "CD294", ("Eosinophil",)),
        Gate("T_cell", "T cell gate", "Mononuclear", "CD3", "CD19", ("T cell",)),
        Gate("B_cell", "B cell gate", "Mononuclear", "CD3", "CD19", ("B cell",)),
        Gate("No_T_Non_B_cell", "Non-T-non-B cell gate", "Mononuclear", "CD3", "CD19", ("Non T Non B cell",)),
    ),
    label_column="GaterConsensus",
    scatter_channels=(),
    consensus_labels=_MASS_LABELS,
)

FLOW = Panel(
    name="flow",
    channels=("FSC-A", "FSC-W", "SSC-A", "SSC-W", "LIVEDEAD", "CD3Q605", "CD45RAQ655", "CD8Q705", "CD4PETR", "CCR7PECY7"),
    gates=(
        Gate("Lymphocyte", "Lymphocyte gate", None, "FSC-A", "SSC-A", ("Lymphocyte", "Singlet", "CD3", "CD4", "CD8", "CD4naive", "CD8naive")),
        Gate("Singlet", "Single cell gate", "Lymphocyte", "FSC-W", "SSC-W", ("Singlet", "CD3", "CD4", "CD8", "CD4naive", "CD8naive")),
        Gate("CD3", "CD3 gate", "Singlet", "CD3Q605", "LIVEDEAD", ("CD3", "CD4", "CD8", "CD4naive", "CD8naive")),
        Gate("CD4", "CD4 gate", "CD3", "CD4PETR", "CD8Q705", ("CD4", "CD4naive")),
        Gate("CD8", "CD8 gate", "CD3", "CD4PETR", "CD8Q705", ("CD8", "CD8naive")),
        Gate("CD4naive", "CD4 naive gate", "CD4", "CD45RAQ655", "CCR7PECY7", ("CD4naive",)),
        Gate("CD8naive", "CD8 naive gate", "CD8", "CD45RAQ655", "CCR7PECY7", ("CD8naive",)),
    ),
    label_column="GaterConsensus",
    scatter_channels=("FSC-A", "FSC-W", "SSC-A", "SSC-W"),
    consensus_labels=("Unfiltered", "Lymphocyte", "Singlet", "CD3", "CD4", "CD8", "CD4naive", "CD8naive"),
)


def get_panel(name: str) -> Panel:
    """Return the exact mass-nine-gate or flow-seven-gate schema."""
    if name == "mass":
        return MASS
    if name == "flow":
        return FLOW
    raise ValueError(f"Unknown panel {name!r}; expected 'mass' or 'flow'")
