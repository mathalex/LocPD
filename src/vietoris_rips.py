import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

from harmonic.harmonic import Simplex, FilteredComplex, VietorisRipsFiltration

def harmonic_cycle(m, eigenvector, simplices, aggregation=None):
    H = np.zeros((m, m)) # 190

    idx_i, idx_j = [], []
    for simplex in simplices:
        idx_i.append(simplex.vertices[0])
        idx_j.append(simplex.vertices[1])

    H[idx_i,idx_j] = np.abs(eigenvector)
    H = H + H.T # symmetrize

    if aggregation=="sum":
        H = H.sum(axis=0)
    elif aggregation=="mean":
        H = H.mean(axis=0)
    elif aggregation=="max":
        H = H.max(axis=0)

    H = H / H.sum() # normalization

    return H

np.random.seed(42)
X = np.random.normal(size=(10, 2))
noise = np.random.normal(1, 0.3) * 0.3
X, _ = make_circles(n_samples=(20, 0), noise=noise, random_state=42)

vr_filtration = VietorisRipsFiltration(X)

filtered_complex = vr_filtration()

for simplex in filtered_complex.filtration:
    print(simplex.vertices, simplex.index, simplex.time)

#print(filtered_complex.filtration[10].time)

filtered_complex.get_reduced_boundary_matrix()

#print()
persistence_diagram = filtered_complex.persistence_diagram
#print(persistence_diagram)
print(persistence_diagram.as_numpy(index=True))

pd1_idx = persistence_diagram.as_numpy(index=True)[:,0]==1
pd1 = persistence_diagram.as_numpy(index=True)[pd1_idx]
print(pd1)

index1 = pd1[0,1]
index2 = pd1[0,2]-1

print("INDICES", index1, index2)

print("NOISE: ", noise)

#print(filtered_complex.view_boundary_matrix(index=index, order=1))
#print(filtered_complex.view_boundary_matrix(index=index, order=2))

eigenvalues1, eigenvectors1 = filtered_complex.spectra(index=index1)
edges1 = filtered_complex.simplices_index_idx[1]

h1 = harmonic_cycle(20, eigenvectors1[:,0], filtered_complex.simplices_at_index[1], aggregation="max")

eigenvalues2, eigenvectors2 = filtered_complex.spectra(index=index2)
edges2 = filtered_complex.simplices_index_idx[1]

h2 = harmonic_cycle(20, eigenvectors2[:,0], filtered_complex.simplices_at_index[1], aggregation="max")

#print(eigenvectors.shape, len(filtered_complex.simplices_index_idx[1]))

#print(filtered_complex.simplices_index_idx)

#for simplex_idx in filtered_complex.simplices_index_idx[1]:
#    print(filtered_complex.filtration[simplex_idx])

print(filtered_complex.simplices_index_idx[1])
print(filtered_complex.simplices_at_index[1])

print(h1)
print(h2)
print(h1 @ h2)
print(np.corrcoef(h1, h2)[0,1])

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10.75,7))

for i in range(2):
    for j in range(3):
        
        if (i==0):
            ax[i,j].scatter(X[:,0], X[:,1], c=h1, cmap="Reds", s=15)
        else:
            ax[i,j].scatter(X[:,0], X[:,1], c=h2, cmap="Reds", s=15)

        if i==0:
            alphas = np.abs(eigenvectors1[:,j]) ** 2 / np.max(np.abs(eigenvectors1[:,j]) ** 2)
            edges = edges1
            eigenvalues = eigenvalues1
        else:
            alphas = np.abs(eigenvectors2[:,j]) ** 2 / np.max(np.abs(eigenvectors2[:,j]) ** 2)
            edges = edges2
            eigenvalues = eigenvalues2

        for k, simplex_idx in enumerate(edges):
            edge = filtered_complex.filtration[simplex_idx]
            x_values = [X[edge.vertices[0],0], X[edge.vertices[1],0]]
            y_values = [X[edge.vertices[0],1], X[edge.vertices[1],1]]
            ax[i,j].set_title("$\lambda_{}$={:.4f}".format(j, eigenvalues[j]))
            ax[i,j].plot(x_values, y_values, "k-", alpha=0.15, linewidth=0.5)
            ax[i,j].plot(x_values, y_values, "r-", alpha=alphas[k])

plt.suptitle("Birth: {:.0f}, death: {:.0f}".format(index1, index2))
plt.tight_layout()
plt.show()