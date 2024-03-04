import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

from harmonic.harmonic import VietorisRipsFiltration
from harmonic.drawing import plot

from bisect import bisect
from operator import attrgetter
from itertools import groupby

def harmonic_cycle(m, eigenvector, simplices, aggregation=None):
    H = np.zeros((m, m))

    idx_i, idx_j = [], []
    for simplex in simplices:
        idx_i.append(simplex.vertices[0])
        idx_j.append(simplex.vertices[1])

    H[idx_i,idx_j] = np.abs(eigenvector)
    H = H + H.T # symmetrize  # was H = H + H

    if aggregation=="sum":
        H = H.sum(axis=0)
    elif aggregation=="mean":
        H = H.mean(axis=0)
    elif aggregation=="max":
        H = H.max(axis=0)

    H = H / H.sum() # normalization

    print(H.shape)
    return H

# generate data
n_points = 20
noise = np.random.normal(1, 0.15) * 0.15
X, _ = make_circles(n_samples=(n_points, 0), noise=noise, random_state=42)

# compute VR filtration of X
vr_filtration = VietorisRipsFiltration(X)
filtered_complex = vr_filtration()

# reduce the boundary matrix
filtered_complex.get_reduced_boundary_matrix()

# get spectra
index = 100
eigenvalues, eigenvectors = filtered_complex.spectra(index)
print(eigenvectors)

# get harmonic cycle
h = harmonic_cycle(n_points, eigenvectors[:,0], filtered_complex.simplices_at_index[1], aggregation="max")
print("HARMONIC CYCLE")
print(h)

#plot(X, filtered_complex, eigenvalues, eigenvectors, h, index)