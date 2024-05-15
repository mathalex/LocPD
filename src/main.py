import numpy as np

from harmonic.drawing import *
from harmonic.harmonic import *

def gen(X):
    N = len(X)
    #print("Total pts:", len(X))

    # compute VR filtration of X
    vr_filtration = VietorisRipsFiltration(X)
    filtered_complex = vr_filtration()

    #print("@@@")

    # reduce the boundary matrix
    filtered_complex.get_reduced_boundary_matrix()

    #print("!!!")

    # print diagram
    # print(filtered_complex.persistence_diagram.as_numpy(index=True))

    threshold = 1e-6
    #print("LEN", len(filtered_complex.persistence_diagram.elements))
    ##print(len(filtered_complex.persistence_diagram.elements))
    ph_lens = []
    for i, eq in enumerate(filtered_complex.persistence_diagram.elements):
        ph_lens.append((eq.death_simplex.index - eq.birth_simplex.index, i, eq.death_simplex.index))
    #print(sorted(ph_lens, reverse=True))
    #print(sorted(ph_lens, reverse=True))
    ok_ids = list(map(lambda x: x[1], sorted(ph_lens, reverse=True)[:min(5, len(ph_lens))]))
    #print(ok_ids)

    # for each H_1 homology class
    a, b = [], []
    for i, eq in enumerate(filtered_complex.persistence_diagram.elements):
        if i not in ok_ids:
            continue
        if eq.birth_simplex.dim==1:

            # birth and death indices
            birth_index = eq.birth_simplex.index
            death_index = eq.death_simplex.index

            # eigenvectors, eigenvalues and edges at birth-1, birth, death-1, and death
            eigenvalues_v, eigenvectors_v = filtered_complex.spectra(birth_index-1)
            eigenvalues_x, eigenvectors_x = filtered_complex.spectra(birth_index)
            eigenvalues_y, eigenvectors_y = filtered_complex.spectra(death_index-1)
            eigenvalues_z, eigenvectors_z = filtered_complex.spectra(death_index)
            #print(eigenvectors_v.shape, eigenvectors_x.shape, eigenvectors_y.shape, eigenvectors_z.shape)

            # eigenvectors corresponding to near-zero eigenvalues
            eigenvectors_v = eigenvectors_v[:,np.isclose(eigenvalues_v, np.zeros_like(eigenvalues_v), threshold)]
            eigenvectors_x = eigenvectors_x[:,np.isclose(eigenvalues_x, np.zeros_like(eigenvalues_x), threshold)]
            eigenvectors_y = eigenvectors_y[:,np.isclose(eigenvalues_y, np.zeros_like(eigenvalues_y), threshold)]
            eigenvectors_z = eigenvectors_z[:,np.isclose(eigenvalues_z, np.zeros_like(eigenvalues_z), threshold)]
            #print(eigenvectors_v.shape, eigenvectors_x.shape, eigenvectors_y.shape, eigenvectors_z.shape)

            # print(i+1, "BIRTH, DEATH", birth_index, death_index)

            # print("V SHAPE", eigenvectors_v.shape)
            # print("X SHAPE", eigenvectors_x.shape)
            # print("Y SHAPE", eigenvectors_y.shape)
            # print("Z SHAPE", eigenvectors_z.shape)

            # optimal cycle
            if eigenvectors_v.shape[1]!=0:
                lambdas = np.zeros(eigenvectors_x.shape[1])
                for j in range(eigenvectors_x.shape[1]):
                    Hvx_j = np.concatenate((eigenvectors_v, eigenvectors_x[:-1,j][:,np.newaxis]), axis=1)
                    # print("(V | x_j)", Hvx_j.shape)
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
                    # print("(Z | y_j)", Hzy_j.shape)
                    _, singular_values, _ = np.linalg.svd(Hzy_j)
                    lambdas[j] = np.min(singular_values)
                idx = np.argmax(lambdas)
                optimal_volume_cycle = eigenvectors_y[:,idx]
            else:
                optimal_volume_cycle = eigenvectors_y[:,0]

            # print("OPTIMAL CYCLE", i)
            # print(optimal_cycle)
            # print("OPTIMAL VOLUME CYCLE", i)
            # print(optimal_volume_cycle)

            # update persistence diagram
            filtered_complex.persistence_diagram.elements[i].birth_harmonic_cycle = optimal_cycle
            filtered_complex.persistence_diagram.elements[i].death_harmonic_cycle = optimal_volume_cycle
            a.append([birth_index, death_index])
            b.append(harmonic_cycle0(N, optimal_cycle, filtered_complex.simplices_at_index[1][:len(optimal_cycle)], aggregation="max"))
            #print(len(optimal_cycle))
    return np.array(a), np.array(b)

print(gen(np.array([[1, 1], [1, 2]])))
