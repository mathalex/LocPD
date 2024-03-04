import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

from harmonic.harmonic import VietorisRipsFiltration
from harmonic.drawing import plot, plot2

from bisect import bisect
from operator import attrgetter
from itertools import groupby

def harmonic_cycle(m, eigenvector, simplices, aggregation=None):
    H = np.zeros((m, m))

    idx_i, idx_j = [], []
    for simplex in simplices:
        idx_i.append(simplex.vertices[0])
        idx_j.append(simplex.vertices[1])

    print("!!!")
    print(idx_i)

    H[idx_i,idx_j] = np.abs(eigenvector)
    H = H + H.T # symmetrize  # changed

    if aggregation=="sum":
        H = H.sum(axis=0)
    elif aggregation=="mean":
        H = H.mean(axis=0)
    elif aggregation=="max":
        H = H.max(axis=0)

    H = H / H.sum() # normalization

    return H

# set random state
rs = np.random.RandomState(42)

# generate data
n_points = 20
noise = rs.normal(1, 0.3) * 0.3
X, _ = make_circles(n_samples=(n_points, 0), noise=noise, random_state=42)

# compute VR filtration of X
vr_filtration = VietorisRipsFiltration(X)
filtered_complex = vr_filtration()

# reduce the boundary matrix
filtered_complex.get_reduced_boundary_matrix()

# get persistence diagram
pd = filtered_complex.persistence_diagram
pdn = pd.as_numpy(index=True)

print("PERSISTENCE DIAGRAM")
print(pdn)

# plot
#plot2(X, edges, eigenvectors_list, harmonic_cycles, indices)

edges = []
eigenvectors_list = []
harmonic_cycles = []
indices = []

for i, eq in enumerate(pd.elements):
    if eq.birth_simplex.dim==1:
        birth_index = eq.birth_simplex.index
        eigenvalues, eigenvectors = filtered_complex.spectra(birth_index)
        simplices1 = filtered_complex.simplices_at_index[1]
        hc = harmonic_cycle(n_points, eigenvectors[:,i], simplices1, aggregation="max")

        edges.append(simplices1)
        eigenvectors_list.append(eigenvectors[:,i])
        harmonic_cycles.append(hc)
        indices.append(birth_index)

        print("BIRTH INDEX", birth_index)
        print("EIGENVALUES", eigenvalues)
        print("EDGES", simplices1)
        print("HARMONIC CYCLE", hc)

