import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

from harmonic.harmonic import VietorisRipsFiltration
from harmonic.drawing import plot, plot_harmonic

# set random state
rs = np.random.RandomState(42)

# generate data
# n_points = 30
# noise = rs.normal(1, 0.2) * 0.2
# X, _ = make_circles(n_samples=(n_points, 0), noise=noise, random_state=42)

def check_collision(current_circle, circles):
    """
        checks whether current_circle not intersects with circles
        returns: False if there is a collision, True otherwise
    """
    x, y, radius = current_circle
    if x-radius < 0 or x+radius > 1 or y-radius < 0 or y+radius > 1:
        return False
    return all(np.sqrt((x - xc)**2 + (y - yc)**2) >= (radius + rc) for xc, yc, rc in circles)

def in_circle(point, circle):
        """
            checks whether point is inside the circle
        """
        x, y = point
        xc, yc, r = circle
        return (x - xc) ** 2 + (y - yc) ** 2 < r ** 2


def generate_circles(num_circles,
                     r_boundaries,
                     x_boundaries,
                     y_boundaries,
                     num_attempts=1000,
                     seed=None):
    """
        Generates non-intersecting circles inside [0,1]x[0,1].
        num_circles - number of circles
        r_boundaries - circles radius boundaries
        x/y_boundaries - circles placement boundaries

        returns: list of non-intersecting circles
    """
    np.random.seed(seed)
    circles = []
    for _ in range(num_circles):
        for _ in range(num_attempts):
            radius = np.random.uniform(*(r_boundaries))
            x = np.random.uniform(x_boundaries[0] + radius, x_boundaries[1] - radius)
            y = np.random.uniform(y_boundaries[0] + radius, y_boundaries[1] - radius)
            new_circle = (x, y, radius)
            if check_collision((x, y, radius), circles):
                circles.append(new_circle)
                break
    return circles


def generate_example(num_points,
                     num_circles,
                     r_boundaries,
                     x_boundaries=(0, 1),
                     y_boundaries=(0, 1),
                     seed=None):
    """
    """
    np.random.seed(seed)
    points = np.random.rand(num_points, 2)
    circles = generate_circles(num_circles, r_boundaries, x_boundaries, y_boundaries, seed=seed)
    filtered_points = np.array([point for point in points if not any(in_circle(point, circle) for circle in circles)])
    return filtered_points, circles

X, _ = generate_example(num_circles=2, num_points=40, r_boundaries=(0.1, 0.15), x_boundaries=(0.5, 0.99))
print(X)

# compute VR filtration of X
vr_filtration = VietorisRipsFiltration(X)
filtered_complex = vr_filtration()

print("@@@")

# reduce the boundary matrix
filtered_complex.get_reduced_boundary_matrix()

print("!!!")

# print diagram
# print(filtered_complex.persistence_diagram.as_numpy(index=True))

threshold = 1e-6
print(len(filtered_complex.persistence_diagram.elements))

# for each H_1 homology class
for i, eq in enumerate(filtered_complex.persistence_diagram.elements):
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

# plot harmonic diagram
plot_harmonic(X, filtered_complex)
