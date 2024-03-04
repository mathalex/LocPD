from math import hypot
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

    H[idx_i,idx_j] = np.abs(eigenvector)
    H = H + H # symmetrize

    if aggregation=="sum":
        H = H.sum(axis=0)
    elif aggregation=="max":
        H = H.max(axis=0)

    H = H / H.sum() # normalization

    return H

# set random state
rs = np.random.RandomState(42)

# generate data
n_points = 25
dispersion = 0.15
noise = rs.normal(1, dispersion) * dispersion
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
Hxs = []
Hys = []
Hzs = []
harmonic_cycles = []
indices = []

threshold = 1e-6

# for each H_1 homology class
for i, eq in enumerate(pd.elements):
    if eq.birth_simplex.dim==1:

        # birth and death indices
        birth_index = eq.birth_simplex.index
        death_index = eq.death_simplex.index

        # eigenvectors, eigenvalues and edges at birth-1, birth, death-1, and death
        eigenvalues_v, eigenvectors_v = filtered_complex.spectra(birth_index-1)
        eigenvalues_x, eigenvectors_x = filtered_complex.spectra(birth_index)
        eigenvalues_y, eigenvectors_y = filtered_complex.spectra(death_index-1)
        eigenvalues_z, eigenvectors_z = filtered_complex.spectra(death_index)

        # eigenvectors corresponding to near-zero eigenvalues
        eigenvectors_v = eigenvectors_v[:,np.isclose(eigenvalues_v, np.zeros_like(eigenvalues_v), threshold)]
        eigenvectors_x = eigenvectors_x[:,np.isclose(eigenvalues_x, np.zeros_like(eigenvalues_x), threshold)]
        eigenvectors_y = eigenvectors_y[:,np.isclose(eigenvalues_y, np.zeros_like(eigenvalues_y), threshold)]
        eigenvectors_z = eigenvectors_z[:,np.isclose(eigenvalues_z, np.zeros_like(eigenvalues_z), threshold)]

        print(i+1, "BIRTH, DEATH", birth_index, death_index)

        print("V SHAPE", eigenvectors_v.shape)
        print("X SHAPE", eigenvectors_x.shape)
        print("Y SHAPE", eigenvectors_y.shape)
        print("Z SHAPE", eigenvectors_z.shape)

        # optimal cycle
        if eigenvectors_v.shape[1]!=0:
            lambdas = np.zeros(eigenvectors_x.shape[1])
            for j in range(eigenvectors_x.shape[1]):
                Hvx_j = np.concatenate((eigenvectors_v, eigenvectors_x[:-1,j][:,np.newaxis]), axis=1)
                print("(V | x_j)", Hvx_j.shape)
                _, singular_values, _ = np.linalg.svd(Hvx_j)
                lambdas[j] = np.min(singular_values)
            idx = np.argmax(lambdas)
            optimal_cycle = eigenvectors_x[:,idx]
        else:
            optimal_cycle = eigenvectors_x[:,0]

        # cycle of optimal volume
        if eigenvectors_z.shape[1]!=0:
            lambdas = np.zeros(eigenvectors_y.shape[1])
            for j in range(eigenvectors_y.shape[1]):
                Hzy_j = np.concatenate((eigenvectors_z, eigenvectors_y[:,j][:,np.newaxis]), axis=1)
                print("(Z | y_j)", Hzy_j.shape)
                _, singular_values, _ = np.linalg.svd(Hzy_j)
                lambdas[j] = np.min(singular_values)
            idx = np.argmax(lambdas)
            optimal_volume_cycle = eigenvectors_y[:,idx]
        else:
            optimal_volume_cycle = eigenvectors_y[:,0]

        print("OPTIMAL CYCLE")
        print(optimal_cycle)
        print("OPTIMAL VOLUME CYCLE")
        print(optimal_volume_cycle)
