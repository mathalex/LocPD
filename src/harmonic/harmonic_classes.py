import numpy as np

from itertools import combinations

from dataclasses import dataclass
from dataclasses import astuple, asdict

from typing import Tuple, Dict, List

@dataclass
class Simplex:
    vertices: Tuple[int]
    index: int = None
    time: float = None

    def __repr__(self):
        return "({})".format(", ".join(map(str, self.vertices)))

    @property
    def dim(self):
        return len(self.vertices) - 1

@dataclass
class Chain:
    elements: List[Simplex]
    coefficients: List[float]

@dataclass
class PersistenceRepresentative:
    birth_index: int
    death_index: int
    birth_time: float
    death_time: float
    birth_simplex: Simplex
    death_simplex: Simplex
    birth_cycle: Chain
    death_cycle: Chain

@dataclass
class HarmonicPersistenceRepresentative(PersistenceRepresentative):
    birth_harmonic_cycle: Chain
    death_harmonic_cycle: Chain

@dataclass
class PersistenceDiagram:
    elements: List[PersistenceRepresentative]

    def as_numpy(self):
        pass

@dataclass
class HarmonicPersistenceDiagram(PersistenceDiagram):
    elements: List[HarmonicPersistenceRepresentative]

@dataclass
class FilteredComplex:
    seq: List[Simplex]
    boundary_matrix: np.ndarray
    reduced_boundary_matrix: np.ndarray

    # init -> list of simplices to boundary matrix
    # reduce -> boundary matrix to reduced boundary matrix
    # boundary21, boundary10 (up to index/time, default=None means all matrix)
    # laplacian1 (up to index/time, default=None means all matrix)


class VietorisRipsFiltration():
    
    def __init__(self, X, distance_matrix=False):
        def pairwise_distances(X):
            return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)

        if (distance_matrix):
            self.X = X
        else:
            self.X = pairwise_distances(X)

        self.n_vertices = X.shape[0]

    def apply(self):
        # return FiltredComplex


