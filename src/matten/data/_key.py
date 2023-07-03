"""
Put all keys in a file as attributes, to make functions annotated by them jittable.

Based on nequip.
"""
from typing import Dict, Final

import torch

# type of atomic data
Type = Dict[str, torch.Tensor]

# positions of nodes in 3D space
POSITIONS: Final[str] = "pos"
# WEIGHTS_KEY: Final[str] = "weights"

# (possibly equivariant) attributes on node; fixed
NODE_ATTRS: Final[str] = "node_attrs"

# features on node, e.g. embedding of atomic species; learnable
NODE_FEATURES: Final[str] = "node_features"

EDGE_INDEX: Final[str] = "edge_index"
EDGE_CELL_SHIFT: Final[str] = "edge_cell_shift"
EDGE_VECTORS: Final[str] = "edge_vectors"
EDGE_LENGTH: Final[str] = "edge_lengths"

# spherical part of edge vector (i.e. expansion of the unit displacement vector
# on spherical harmonics); fixed
EDGE_ATTRS: Final[str] = "edge_attrs"

# radial part of the edge vector (i.e. distance between atoms), fixed
EDGE_EMBEDDING: Final[str] = "edge_embedding"

# message from neighboring nodes to a central node
EDGE_MESSAGE: Final[str] = "edge_message"

CELL: Final[str] = "cell"
# PBC: Final[str] = "pbc"

NUM_NEIGH: Final[str] = "num_neigh"

ATOMIC_NUMBERS: Final[str] = "atomic_numbers"
SPECIES_INDEX: Final[str] = "species_index"
PER_ATOM_ENERGY: Final[str] = "atomic_energy"
TOTAL_ENERGY: Final[str] = "total_energy"
# FORCE: Final[str] = "forces"

BATCH: Final[str] = "batch"
