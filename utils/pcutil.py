import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

# Don't delete this line, even if PyCharm says it's an unused import.
# It is required for projection='3d' in add_subplot()
from mpl_toolkits.mplot3d import Axes3D
import torch

def rand_rotation_matrix(deflection=1.0, seed=None):
    """Creates a random rotation matrix.

    Args:
        deflection: the magnitude of the rotation. For 0, no rotation; for 1,
                    completely random rotation. Small deflection => small
                    perturbation.

    DOI: http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
         http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    """
    if seed is not None:
        np.random.seed(seed)

    theta, phi, z = np.random.uniform(size=(3,))

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (np.sin(phi) * r,
         np.cos(phi) * r,
         np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def add_gaussian_noise_to_pcloud(pcloud, mu=0, sigma=1):
    gnoise = np.random.normal(mu, sigma, pcloud.shape[0])
    gnoise = np.tile(gnoise, (3, 1)).T
    pcloud += gnoise
    return pcloud


def add_rotation_to_pcloud(pcloud):
    r_rotation = rand_rotation_matrix()

    if len(pcloud.shape) == 2:
        return pcloud.dot(r_rotation)
    else:
        return np.asarray([e.dot(r_rotation) for e in pcloud])


def apply_augmentations(batch, conf):
    if conf.gauss_augment is not None or conf.z_rotate:
        batch = batch.copy()

    if conf.gauss_augment is not None:
        mu = conf.gauss_augment['mu']
        sigma = conf.gauss_augment['sigma']
        batch += np.random.normal(mu, sigma, batch.shape)

    if conf.z_rotate:
        r_rotation = rand_rotation_matrix()
        r_rotation[0, 2] = 0
        r_rotation[2, 0] = 0
        r_rotation[1, 2] = 0
        r_rotation[2, 1] = 0
        r_rotation[2, 2] = 1
        batch = batch.dot(r_rotation)
    return batch


def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with
    resolution^3 cells, that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside
    the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def plot_3d_point_cloud(x, y, z, show=True, show_axis=True, in_u_sphere=False,
                        marker='.', s=8, alpha=.8, figsize=(5, 5), elev=10,
                        azim=240, axis=None, title=None, *args, **kwargs):
    plt.switch_backend('agg')
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    if type(x). __module__ != np. __name__:
        x = x.to('cpu')
    else:
        x = torch.from_numpy(x.astype(np.float32)).clone().to('cpu').detach().numpy().copy()
    if type(y). __module__ != np. __name__:
        y = y.to('cpu')
    else:
        y = torch.from_numpy(y.astype(np.float32)).clone().to('cpu').detach().numpy().copy()
    if type(z). __module__ != np. __name__:
        z = z.to('cpu')
    else:
        z = torch.from_numpy(z.astype(np.float32)).clone().to('cpu').detach().numpy().copy()

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        # Multiply with 0.7 to squeeze free-space.
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig


def transform_point_clouds(X, only_z_rotation=False, deflection=1.0):
    r_rotation = rand_rotation_matrix(deflection)
    if only_z_rotation:
        r_rotation[0, 2] = 0
        r_rotation[2, 0] = 0
        r_rotation[1, 2] = 0
        r_rotation[2, 1] = 0
        r_rotation[2, 2] = 1
    X = X.dot(r_rotation).astype(np.float32)
    return X
