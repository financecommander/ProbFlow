"""Core types for ProbFlow probabilistic models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Variable:
    """A discrete random variable with a finite set of states."""

    name: str
    states: List[str]

    @property
    def num_states(self) -> int:
        return len(self.states)


@dataclass
class Factor:
    """A factor (potential function) over a set of variables."""

    variables: List[Variable]
    values: np.ndarray  # shape: (|var1|, |var2|, ...)

    def __post_init__(self) -> None:
        expected = tuple(v.num_states for v in self.variables)
        if self.values.shape != expected:
            raise ValueError(
                f"Factor shape {self.values.shape} does not match "
                f"variable cardinalities {expected}"
            )


@dataclass
class Node:
    """A node in a factor graph / Bayesian network."""

    variable: Variable
    prior: Optional[np.ndarray] = None  # shape: (num_states,)
    children: List[Node] = field(default_factory=list)
    parents: List[Node] = field(default_factory=list)
    cpt: Optional[np.ndarray] = None  # conditional probability table

    @property
    def name(self) -> str:
        return self.variable.name

    def add_child(self, child: Node, cpt: np.ndarray) -> None:
        """Add a child node with its conditional probability table."""
        self.children.append(child)
        child.parents.append(self)
        child.cpt = cpt
