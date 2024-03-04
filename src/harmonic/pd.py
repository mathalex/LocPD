# harmonic.harmonic fully contains this

from sys import orig_argv
import numpy as np

from itertools import combinations, groupby
from bisect import bisect
from operator import attrgetter

from dataclasses import dataclass
from dataclasses import astuple, asdict

from typing import Tuple, Dict, List

@dataclass
class Simplex:
    vertices: Tuple[int]
    index: int = None
    time: float = None
    weight: float = None

    def __repr__(self):
        return "({})".format(", ".join(map(str, self.vertices)))

    @property
    def dim(self):
        return len(self.vertices) - 1

    @property
    def boundary(self):
        if self.dim==0:
            faces = []
        else:
            faces = [Simplex(item) for item in combinations(self.vertices, self.dim)][::-1]
        return faces

@dataclass
class PersistenceRepresentative:
    birth_simplex: Simplex
    death_simplex: Simplex

@dataclass
class PersistenceDiagram:
    elements: List[PersistenceRepresentative]

    def num_representatives(self, dim=0):
        n_representatives = {0: 0, 1: 0}

        for representative in self.elements:
            representative_dim = representative.birth_simplex.dim
            n_representatives[representative_dim] = n_representatives[representative_dim] + 1

        return n_representatives[dim]

    def representatives_graded(self, k=0):

        representatives_graded = {}

        representatives = sorted(self.elements, key=lambda element: (element.birth_simplex.dim)) # , element.birth_simplex.index, element.death_simplex.index

        for k_repr, k_representatives in groupby(representatives, key=lambda representative: representative.birth_simplex.dim):
            k_representatives = list(k_representatives)
            representatives_graded[k_repr] = k_representatives

        return representatives_graded[k]

    def as_numpy(self, index=False):
        pd = np.zeros((len(self.elements), 3))
        
        sorted_elements = sorted(self.elements, key=lambda element: (element.birth_simplex.dim, element.birth_simplex.index, element.death_simplex.index))

        for i, element in enumerate(sorted_elements):
            if index==False:
                pd[i,:] = np.array([element.birth_simplex.dim, element.birth_simplex.time, element.death_simplex.time])
            else:
                pd[i,:] = np.array([element.birth_simplex.dim, element.birth_simplex.index, element.death_simplex.index])

        return pd.astype(int)

class FilteredComplex:

    def __init__(self, filtration: List[Simplex], oriented=False):
        self.filtration = filtration
        self.oriented = oriented
        self.boundary_matrix = None
        self.reduced_boundary_matrix = None
        self.persistence_diagram = None

        self.simplex_to_index = {}
        for simplex in self.filtration:
            self.simplex_to_index[simplex.vertices] = simplex.index

        n_simplices = len(self.filtration)
        self.boundary_matrix = np.zeros((n_simplices, n_simplices), dtype=int)

        # building boundary matrix
        for simplex in self.filtration:
            for q, face in enumerate(simplex.boundary):
                i, j = self.simplex_to_index[face.vertices], simplex.index
                self.boundary_matrix[i,j] = 1

    def get_reduced_boundary_matrix(self):
        
        def matrix_reduction(matrix: np.ndarray) -> np.ndarray:
            
            def low(column: np.ndarray) -> int:
                if np.any(column!=0):
                    return np.flatnonzero(column)[-1] 
                return -1

            def reduceable(matrix, j, lows, pivots):
                is_reduceable = False
                if lows[j]!=-1 and pivots[lows[j]]!=-1:
                    is_reduceable = pivots[lows[j]]<j
                return is_reduceable

            # set lows and pivots
            lows = [low(column) for column in matrix.T]
            
            pivots = np.ones(matrix.shape[0]).astype(int) * -1
            for i in range(matrix.shape[0]):
                for j in range(i+1, matrix.shape[0]):
                    if (matrix[i,j]!=0 and lows[j]==i):
                        pivots[i] = j
                        break

            pivots = list(pivots)

            for i in range(0, matrix.shape[1]):
                while reduceable(matrix, i, lows, pivots):
                    j = pivots[lows[i]]
                    matrix[:,i] = (matrix[:,j] + matrix[:,i]) % 2
                    lows[i] = low(matrix[:,i]) # update lows
                
                if lows[i]!=-1:
                    pivots[lows[i]] = i; # update pivots
                    
            return matrix

        if (self.reduced_boundary_matrix==None): # cached
            self.reduced_boundary_matrix = matrix_reduction(self.boundary_matrix)
            self.persistence_diagram = self.get_persistence_diagram()

        return self.reduced_boundary_matrix

    def view_boundary_matrix(self, index=None, order=1):
        
        self.simplices_at_index = {}
        self.simplices_index_idx = {}

        filtration_index = sorted(self.filtration[:index+1], key=lambda simplex: (len(simplex.vertices), simplex.index))
        for k, k_simplices in groupby(filtration_index, key=lambda simplex: len(simplex.vertices)):
            k_simplices = list(k_simplices)
            self.simplices_at_index[k-1] = k_simplices
            self.simplices_index_idx[k-1] = [simplex.index for simplex in k_simplices]

        if order==1:
            B = self.oriented_boundary_matrix[self.simplices_index_idx[0],:][:,self.simplices_index_idx[1]]
        elif order==2:
            B = self.oriented_boundary_matrix[self.simplices_index_idx[1],:][:,self.simplices_index_idx[2]]

        return B

    def get_persistence_diagram(self):
        def low(column):
            column = (column!=0).astype(int)
            argwhere = np.argwhere(column)
            if argwhere.shape[0]==0:
                lowest = -1
            else:
                lowest = argwhere[-1,0]
            return lowest

        persistence_representatives = []
        for j in range(len(self.filtration)):
            i_low = low(self.reduced_boundary_matrix[:,j])
            if i_low!=-1:
                birth_simplex, death_simplex = self.filtration[i_low], self.filtration[j]
                if (death_simplex.index - birth_simplex.index) > 1:
                    persistence_representative = PersistenceRepresentative(birth_simplex, death_simplex)
                    persistence_representatives.append(persistence_representative)

        return PersistenceDiagram(persistence_representatives)

    @property
    def harmonic_persistence_diagram(self):
        pass

class IndexFiltration:
    
    def __init__(self, cmplx):
        self.cmplx = cmplx

    def __call__(self, identity=False):
        
        if identity==False:
            filtered_cmplx = sorted(self.cmplx, key=lambda simplex: (simplex.index, simplex.vertices))
        else: # if identity - set index and time as they passed
            filtered_cmplx = self.cmplx
            for i, simplex in enumerate(filtered_cmplx):
                simplex.index = i

        for simplex in filtered_cmplx:
            simplex.time = simplex.index

        return FilteredComplex(filtered_cmplx)

class VietorisRipsFiltration:
    
    def __init__(self, X, distance_matrix=False):
        def pairwise_distances(X):
            return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)

        if (distance_matrix):
            self.X = X
        else:
            self.X = pairwise_distances(X)

        self.n_vertices = X.shape[0]

    def __call__(self):
        def f(simplex):
            if simplex.dim==0:
                f = 0
            elif simplex.dim==1:
                i, j = simplex.vertices
                f = self.X[i,j]
            else:
                i, j, k = simplex.vertices
                f = max([self.X[i,j], self.X[i,k], self.X[j,k]])
            return f

        # TODO: refactor
        vertices = [Simplex(item) for item in combinations(range(self.n_vertices), 1)]
        edges = [Simplex(item) for item in combinations(range(self.n_vertices), 2)]
        triangles = [Simplex(item) for item in combinations(range(self.n_vertices), 3)]
        cmplx = [item for lst in [vertices, edges, triangles] for item in lst]

        for simplex in cmplx:
            simplex.time = f(simplex)

        filtered_cmplx = sorted(cmplx, key=lambda simplex: (simplex.time, simplex.dim, simplex.vertices))

        for i, simplex in enumerate(filtered_cmplx):
            simplex.index = i
            
        return FilteredComplex(filtered_cmplx)
