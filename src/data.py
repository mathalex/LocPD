# draws point cloud

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from harmonic.harmonic import VietorisRipsFiltration

#np.random.seed(42)

label = 1

r = 1.2

U1 = np.random.uniform(size=15)
U2 = np.random.uniform(size=15)

if (label==0):
    disk = plt.Circle((-0.75, 0), 1, color="r", linewidth=0, alpha=0.05, fill=True)
    sphere = plt.Circle((0.75, 0), 1, color="r", linewidth=2, alpha=0.05, fill=False)

    Phi = r * np.sqrt(U2) * np.cos(2 * np.pi * U1) - 0.85
    Psi = r * np.sqrt(U2) * np.sin(2 * np.pi * U1)
    D = np.concatenate((Phi[:,np.newaxis], Psi[:,np.newaxis]), axis=1)

    bias = np.concatenate((np.ones((12, 1)) * 0.75, np.zeros((12, 1))), axis=1)
    noise = np.random.normal(1, 0.1) * 0.15
    S, _ = make_circles(n_samples=(12, 0), noise=noise, random_state=42)
    S = S + bias

    X = np.concatenate((D, S), axis=0)
    D_idx, S_idx = np.array(range(0, 13)), np.array(range(13, 25))

else:
    disk = plt.Circle((0.75, 0), 1, color="r", linewidth=0, alpha=0.05, fill=True)
    sphere = plt.Circle((-0.75, 0), 1, color="r", linewidth=2, alpha=0.05, fill=False)

    Phi = r * np.sqrt(U2) * np.cos(2 * np.pi * U1) + 0.85
    Psi = r * np.sqrt(U2) * np.sin(2 * np.pi * U1)
    D = np.concatenate((Phi[:,np.newaxis], Psi[:,np.newaxis]), axis=1)

    bias = np.concatenate((np.ones((12, 1)) * 0.75, np.zeros((12, 1))), axis=1)
    noise = np.random.normal(1, 0.1) * 0.15
    S, _ = make_circles(n_samples=(12, 0), noise=noise, random_state=42)
    S = S - bias    

    X = np.concatenate((S, D), axis=0)
    S_idx, D_idx = np.array(range(0, 12)), np.array(range(12, 25))

# compute VR filtration of X
vr_filtration = VietorisRipsFiltration(X)
filtered_complex = vr_filtration()

# reduce the boundary matrix
filtered_complex.get_reduced_boundary_matrix()

# get persistence diagram
pdn = filtered_complex.persistence_diagram.as_numpy(index=True)
print(pdn)

fig, ax = plt.subplots(figsize=(7,4))
ax.set_xlim(-1.85, 1.85)
ax.set_ylim(-1.1, 1.1)
# ax.add_artist(disk)
# ax.add_artist(sphere)
ax.scatter(X[D_idx,0], X[D_idx,1], c="k", s=10)
ax.scatter(X[S_idx,0], X[S_idx,1], c="r", s=10)
plt.show()