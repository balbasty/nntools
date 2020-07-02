import numpy as np
import itertools
from .utils import sub2ind


def identity_grid(shape, dtype=None):
    """Generate a dense identity grid

    Parameters
    ----------
    shape : iterable of length D
        Shape of the dense grid.
    dtype : type, default=mat.dtype
        Output data type.

    Returns
    -------
    grid : np.ndarray of shape (*shape, D)
        Dense identity grid.

    """

    grid = np.stack(np.meshgrid(*(np.arange(s, dtype=dtype) for s in shape),
                                indexing='ij', copy=False), axis=-1)
    return grid


def affine_grid(mat, shape, dtype=None):
    """Generate a dense affine grid.

    Parameters
    ----------
    mat : array_like of shape (D, D+1) or (D+1, D+1)
        Affine matrix.
        - mat[:D, :D] contains the rotation part of the affine transform
        - mat[D, :D] contains the translation part of the affine transform
    shape : iterable of length D
        Shape of the dense grid.
    dtype : type, default=mat.dtype
        Output data type.

    Returns
    -------
    grid : np.ndarray of shape (*shape, D)
        Dense affine grid.

    """
    mat = np.asarray(mat, dtype=dtype)
    dim = mat.shape[1] - 1
    assert(len(shape) == dim)
    if dtype is None:
        dtype = mat.dtype

    # Generate identity grid
    grid = identity_grid(shape, dtype)

    # Compose with affine
    rotation = mat[:dim, :dim]
    translation = mat[:dim, dim].reshape((1,)*dim + (dim,))
    grid = np.dot(grid, rotation.transpose())
    grid += translation

    return grid


def sample_grid(x, grid, order=1, bound='reflect'):
    """Sample a volume at specified coordinates.

    Parameters
    ----------
    x : array_like of shape (*input_spatial, *features)
        Input volume
    grid : array_like of shape (*output_spatial, dim)
        Grid of coordinates
    order : int, default=1
        Order of the B-spline coefficients that encode ``x``.
        Often called 'interpolation order'.
    bound : {'wrap', 'nearest', 'mirror', 'reflect'} or scalar,
            default=0
        Boundary conditions when sampling out-of-bounds.

    Returns
    -------
    y : np.ndarray of shape (*output_spatial, *features)


    """
    if isinstance(bound, str):
        if bound == 'wrap':
            bound = bound_wrap
        elif bound == 'nearest':
            bound = bound_nearest
        elif bound == 'mirror':
            bound = bound_mirror
        elif bound == 'reflect':
            bound = bound_reflect
        else:
            raise ValueError('Unknown boundary condition {}'.format(bound))
    # else: constant value

    if order in (0, 'nearest'):
        return sample_grid_nearest(x, grid, bound)
    elif order in (1, 'linear'):
        return sample_grid_linear(x, grid, bound)
    else:
        raise ValueError('Interpolation order {} not implemented'
                         .format(order))


def sample_grid_nearest(x, grid, bound):
    dim = grid.shape[-1]
    grid = np.round(grid).astype(np.int64)
    grid = np.stack(tuple(bound(grid[..., d], x.shape[d])
                    for d in range(dim)), axis=dim)
    grid = sub2ind([grid[..., d] for d in range(dim)], x.shape[:dim])
    flattened_shape = (np.prod(x.shape[:dim]),) + x.shape[dim:]
    x = x.reshape(flattened_shape)
    x = x[grid, ...]
    return x
    # TODO: constant value mode -> create a mask of out-of-bound voxels


def sample_grid_linear(x, grid, bound):
    dim = grid.shape[-1]

    # Weights
    grid_corner0 = np.floor(grid)
    weights = grid_corner0 - grid
    weights += 1

    # Corners
    grid_corner0 = grid_corner0.astype(np.int64)
    corners = list(itertools.product([False, True], repeat=dim))
    grid_corners = [grid_corner0]
    for corner in corners[1:]:
        grid_corner1 = grid_corner0.copy()
        grid_corner1[..., corner] += 1
        grid_corners.append(grid_corner1)
    del grid_corner0, grid_corner1

    # Convert indices using bound function
    grid_corners = [np.stack(tuple(bound(g[..., d], x.shape[d])
                                   for d in range(dim)), axis=dim)
                    for g in grid_corners]
    # TODO: constant value mode -> create a mask of out-of-bound voxels

    # Linear indices
    grid_corners = [sub2ind([g[..., d] for d in range(dim)], x.shape[:dim])
                    for g in grid_corners]
    flattened_shape = (-1,) + x.shape[dim:]
    x = x.reshape(flattened_shape)

    # Interpolate
    x0 = np.zeros_like(x, shape=grid.shape[:dim] + x.shape[dim:])
    for corner, grid in zip(corners, grid_corners):
        w = weights.copy()
        w[..., corner] *= -1
        w[..., corner] += 1
        w = np.prod(w, axis=-1)
        x0 += w * x[grid, ...]

    return x0


def reliability_grid(grid):
    """Compute a reliability map.

    Reliability maps quantify "how much" an outptu voxel is interpolated.
    If an output voxel is perfectly aligned with an input voxel, its
    reliability is 1; if it is exactly in-between 2 voxels in a dimension
    and aligned in all other dimensions, its reliability is 0.5; if it is
    in-between 4 nodes, its reliability is 0.25; etc.

    Parameters
    ----------
    grid : array_like of shape (*spatial, D)

    Returns
    -------
    weight : array_like of shape (*spatial)

    """
    grid = np.asarray(grid)
    weight = grid - np.floor(grid)
    weight[weight < 0.5] = 1-weight[weight < 0.5]
    weight = weight.prod(axis=-1)
    return weight


def bound_wrap(i, n, inplace=True):
    i = np.asarray(i)
    return np.mod(i, n, out=i if inplace else None)


def bound_nearest(i, n, inplace=True):
    i = np.asarray(i)
    return np.clip(i, min=0, max=n-1, out=i if inplace else None)


def bound_reflect(i, n, inplace=True):
    i = np.asarray(i)
    if not inplace:
        i = i.copy()
    n2 = n*2
    pre = (i < 0)
    # i[pre] = n2 - 1 - ((-i[pre]-1) % n2)
    i[pre] *= -1
    i[pre] -= 1
    i[pre] %= n2
    i[pre] *= -1
    i[pre] += n2 - 1
    # i[~pre] = (i[~pre] % n2)
    i[~pre] %= n2
    post = (i >= n)
    # i[post] = n2 - i[post] - 1
    i[post] *= -1
    i[post] += n2 - 1
    return i


def bound_mirror(i, n, inplace=True):
    i = np.asarray(i)
    if n == 1:
        if inplace:
            i[...] = 0
            return i
        else:
            return np.zeros_like(i)
    else:
        if not inplace:
            i = i.copy()
        n2 = (n-1)*2
        pre = (i < 0)
        # i[pre] = -i[pre]
        i[pre] *= -1
        # i = i % n2
        i %= n2
        post = (i >= n)
        # i[post] = n2 - i[post]
        i[post] *= -1
        i[post] += n2
        return i



