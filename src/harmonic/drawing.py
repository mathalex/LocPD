import matplotlib.pyplot as plt

import numpy as np

def stack(X, idx):
    ret = np.empty((0, 2))
    for _id in idx:
        ret = np.vstack((ret, X[_id,:]))
    return ret

def harmonic_cycle0(m, eigenvector, simplices, aggregation=None):
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

def plot(X, filtered_complex, eigenvalues, eigenvectors, harmonic_cycles, indices):
    def stack(idx):
        ret = np.empty((0, 2))
        for _id in idx:
            ret = np.vstack((ret, X[_id,:]))
        return ret

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(14.25,4))

    for j in range(len(indices)):

        h = harmonic_cycles[j]
        index = indices[j]
        alphas = np.abs(eigenvectors[:,j]) ** 1.75 / np.max(np.abs(eigenvectors[:,j]) ** 1.75)

        # plot edges
        edges = filtered_complex.simplices_at_index[1]
        for k, edge in enumerate(edges):
            x_values = [X[edge.vertices[0],0], X[edge.vertices[1],0]]
            y_values = [X[edge.vertices[0],1], X[edge.vertices[1],1]]
            ax[j].set_title("$\lambda_{}$={:.4f}".format(j, eigenvalues[j]))
            ax[j].plot(x_values, y_values, "k-", alpha=0.15, linewidth=0.5)
            ax[j].plot(x_values, y_values, "r-", alpha=alphas[k])

        # plot triangles
        # triangles = filtered_complex.simplices_at_index[2]
        # for triangle in triangles:
        #     t = plt.Polygon(stack(triangle.vertices), color="red", alpha=0.025)
        #     ax[j].add_patch(t)

        if (j==0):
            ax[j].scatter(X[:,0], X[:,1], c=h, cmap="Reds", s=22.5)
        else:
            ax[j].scatter(X[:,0], X[:,1], c="k", alpha=0.5, s=10)

    plt.suptitle("Index: {:.0f}".format(index))
    plt.tight_layout()
    plt.show()


def plot2(X, edges_list, eigenvectors, harmonic_cycles, indices):
    def stack(idx):
        ret = np.empty((0, 2))
        for _id in idx:
            ret = np.vstack((ret, X[_id,:]))
        return ret

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(14.25,4))

    for j, _ in enumerate(indices):

        edges = edges_list[j]
        eigenvector = np.abs(eigenvectors[j])
        h = harmonic_cycles[j]
        index = indices[j]

        alphas = eigenvector ** 1.75 / np.max(eigenvector ** 1.75)

        # plot edges
        for k, edge in enumerate(edges):
            x_values = [X[edge.vertices[0],0], X[edge.vertices[1],0]]
            y_values = [X[edge.vertices[0],1], X[edge.vertices[1],1]]
            ax[j].set_title("Index={}".format(index)) 
            ax[j].plot(x_values, y_values, "k-", alpha=0.15, linewidth=0.5)
            ax[j].plot(x_values, y_values, "r-", alpha=alphas[k])

        # plot triangles
        # triangles = filtered_complex.simplices_at_index[2]
        # for triangle in triangles:
        #     t = plt.Polygon(stack(triangle.vertices), color="red", alpha=0.025)
        #     ax[j].add_patch(t)

        if (j==0):
            ax[j].scatter(X[:,0], X[:,1], c=h, cmap="Reds", s=22.5)
        else:
            ax[j].scatter(X[:,0], X[:,1], c="k", alpha=0.5, s=10)

    plt.suptitle("Index: {:.0f}".format(index))
    plt.tight_layout()
    plt.show()

def plot_harmonic(X, filtered_complex):

    # get 1-dimensional homology classes
    representaives1 = filtered_complex.persistence_diagram.representatives_graded(k=1)
    n_representatives1 = len(representaives1)

    width = n_representatives1 * 4.25 + (n_representatives1-1) * 0.5

    fig, ax = plt.subplots(nrows=2, ncols=n_representatives1, figsize=(width,8.965))

    birth_row, death_row = 0, 1

    # plot points
    for row in range(2):
        for col in range(0, n_representatives1):
            ax[row,col].scatter(X[:,0], X[:,1], c="k", alpha=0.2, s=3)

    # representative columns
    for col in range(0, n_representatives1):

        repr_idx = col
        birth_index = representaives1[repr_idx].birth_simplex.index
        death_index = representaives1[repr_idx].death_simplex.index

        ax[birth_row,col].set_title("Optimal cycle at {}, [{}, {})".format(birth_index, birth_index, death_index))
        ax[death_row,col].set_title("Optimal volume cycle at {}, [{}, {})".format(death_index-1, birth_index, death_index))

        vertices_birth = filtered_complex.filtration_at_index(birth_index, k=0)
        edges_birth = filtered_complex.filtration_at_index(birth_index, k=1)
        alphas_birth = np.abs(representaives1[repr_idx].birth_harmonic_cycle)

        # aggregation
        m = len(vertices_birth)
        A = np.zeros((m, m))

        for vertex in vertices_birth:
            idx = vertex.vertices[0]
            x, y = X[idx,0], X[idx,1]
            ax[birth_row,col].annotate(idx, (x+0.02, y+0.01))

        for k, edge in enumerate(edges_birth):
            i, j = edge.vertices[0], edge.vertices[1]
            A[i,j] = alphas_birth[k]
            print("BIRTH col={}, edge[{},{}]={}".format(col, i, j, alphas_birth[k]))

        # repr0_birth = A.sum(axis=0) # aggregate

        # mean aggregation
        sum = np.sum(A, axis=0)
        nz = np.count_nonzero(A, axis=0)
        repr0_birth = np.nan_to_num(sum / nz)
        repr0_birth = repr0_birth / repr0_birth.sum(axis=0) # normalize

        # print("VERTICES BIRTH", len(vertices_birth), vertices_birth)
        # print("EDGES / ALPHAS LEN", len(edges_birth), len(alphas_birth))
        # print("repr0_birth", repr0_birth)

        for k, edge in enumerate(edges_birth):
            x_values = [X[edge.vertices[0],0], X[edge.vertices[1],0]]
            y_values = [X[edge.vertices[0],1], X[edge.vertices[1],1]]
            ax[birth_row,col].plot(x_values, y_values, "k-", alpha=0.25, linewidth=0.33)
            ax[birth_row,col].plot(x_values, y_values, "r-", alpha=alphas_birth[k], linewidth=3)

        scaling_factor = 250
        ax[birth_row,col].scatter(X[:,0], X[:,1], c="k", alpha=1, s=scaling_factor*repr0_birth)

        vertices_death = filtered_complex.filtration_at_index(death_index-1, k=0)
        edges_death = filtered_complex.filtration_at_index(death_index-1, k=1)
        alphas_death = np.abs(representaives1[repr_idx].death_harmonic_cycle)

        # aggregation
        m = len(vertices_birth)
        A = np.zeros((m, m))

        for k, edge in enumerate(edges_death):
            i, j = edge.vertices[0], edge.vertices[1]
            A[i,j] = alphas_death[k]

        # mean aggregation
        sum = np.sum(A, axis=0)
        nz = np.count_nonzero(A, axis=0)
        repr0_death = np.nan_to_num(sum / nz)
        repr0_death = repr0_death / repr0_death.sum(axis=0) # normalize

        for k, edge in enumerate(edges_death):
            x_values = [X[edge.vertices[0],0], X[edge.vertices[1],0]]
            y_values = [X[edge.vertices[0],1], X[edge.vertices[1],1]]
            ax[death_row,col].plot(x_values, y_values, "k-", alpha=0.25, linewidth=0.33)
            ax[death_row,col].plot(x_values, y_values, "r-", alpha=alphas_death[k], linewidth=3)

        scaling_factor = 250
        ax[death_row,col].scatter(X[:,0], X[:,1], c="k", alpha=0.8, s=scaling_factor*repr0_death)

        # print("# EDGES BIRTH/DEATH", len(edges_birth), len(edges_death))
        # print(representaives1[repr_idx])

        # plot triangles
        triangles_birth = filtered_complex.filtration_at_index(birth_index, k=2)
        for triangle in triangles_birth:
            t = plt.Polygon(stack(X, triangle.vertices), color="red", alpha=0.03)
            ax[birth_row,col].add_patch(t)

        triangles_death = filtered_complex.filtration_at_index(death_index-1, k=2)
        for triangle in triangles_death:
            t = plt.Polygon(stack(X, triangle.vertices), color="red", alpha=0.03)
            ax[death_row,col].add_patch(t)

    # plt.suptitle("Index: {:.0f}".format(index))
    plt.tight_layout()
    plt.show()