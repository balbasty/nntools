import re
import numpy as np
import itertools
from .linalg import lmdiv, rmdiv, meanm, dexpm
from .utils import sub2ind, majority
from scipy.linalg import logm


def affine_layout(layout, dtype=np.float64):
    """Compute the origin affine matrix for different voxel layouts.

    Resources
    ---------
    .. https://nipy.org/nibabel/image_orientation.html
    .. https://nipy.org/nibabel/neuro_radio_conventions.html

    Parameters
    ----------
    layout : str
        Voxel layout are described by permutation of up to three letters:
            * 'R' for *left to Right* or
              'L' for *right to Left*
            * 'A' for *posterior to Anterior* or
              'P' for *anterior to Posterior*
            * 'S' for *inferior to Superior* or
              'I' for *superior to Inferior*
        The most common layout is 'RAS', which maps to the "world"
        orientation 'RAS+' with an identity matrix.
        If the first voxel dimension browsed the brain from right to
        left, the layout would be 'LAS'.
        Note that this layout is only approximate; in practice, angled
        field-of-views would fall between these layouts.

        The number of letters defines the dimension of the matrix

    dtype : str or type
        Data type of the matrix

    Returns
    -------
    mat : (D+1, D+1) np.ndarray
        Corresponding affine matrix.
    """

    # Author
    # ------
    # .. Yael Balbastre <yael.balbastre@gmail.com>

    layout = layout.upper()
    dim = len(layout)

    # STEP 1: Find flips (L, P, I) and substitute them
    flip = [False, ] * dim
    if layout.find('L') > 0:
        flip[0] = True
        layout = layout.replace('L', 'R')
    if dim > 0 and layout.find('P') > 0:
        flip[1] = True
        layout = layout.replace('P', 'A')
    if dim > 1 and layout.find('I') > 0:
        flip[2] = True
        layout = layout.replace('I', 'S')

    # STEP 2: Find permutations
    perm = [layout.find('R')]
    if dim > 0:
        perm.append(layout.find('A'))
        if dim > 1:
            perm.append(layout.find('S'))

    # STEP 3: Create matrix
    mat = np.ones(dim+1, dtype=dtype)
    mat[flip + [False]] *= -1
    mat = np.diag(mat)
    mat = mat[:, perm]

    return mat


def affine_find_layout(mat):
    """Find voxel layout associated with an affine matrix.

    Parameters
    ----------
    mat : (D+1, D+1) array_like
        Affine matrix

    Returns
    -------
    layout : str
        Voxel layout (see ``affine_layout``)

    """

    # Author
    # ------
    # .. Yael Balbastre <yael.balbastre@gmail.com>

    # Extract linear component + remove voxel scaling
    mat = np.asarray(mat).astype(np.float64)
    dim = mat.shape[-1] - 1
    mat = mat[:dim, :dim]
    vs = (mat ** 2).sum(1)
    mat = rmdiv(mat, np.diag(vs))
    I = np.eye(dim, dtype=np.float64)

    min_sos = np.inf
    min_layout = None

    def check_space(space):
        layout = affine_layout(space)[:dim, :dim]
        sos = ((rmdiv(mat, layout) - I) ** 2).sum()
        if sos < min_sos:
            return space
        else:
            return min_layout

    if dim == 3:
        for D1 in ('R', 'L'):
            for D2 in ('A', 'P'):
                for D3 in ('S', 'I'):
                    spaces = itertools.permutations([D1, D2, D3])
                    spaces = list(''.join(space) for space in spaces)
                    for space in spaces:
                        min_layout = check_space(space)
    elif dim == 2:
        for D1 in ('R', 'L'):
            for D2 in ('A', 'P'):
                spaces = itertools.permutations([D1, D2])
                spaces = list(''.join(space) for space in spaces)
                for space in spaces:
                    min_layout = check_space(space)
    elif dim == 1:
        for D1 in ('R', 'L'):
            min_layout = check_space(D1)

    return min_layout


def affine_parameters(mat, basis=None, layout='RAS'):
    """Compute the parameters of an affine matrix in a basis of the algebra.

    Parameters
    ----------
    mat : (D+1, D+1) array_like
        Affine matrix
    basis : (F, D+1, D+1) array_like, default=affine basis
        Basis of the Lie algebra.
        It must be orthonormal with respect to the matrix dot product
        trace(A.t() @ B)
    layout : str or (D+1, D+1) array_like, default='RAS'
        "Point" at which to take the matrix exponential.
        If a string, builds the corresponding matrix using ``affine_layout``

    Returns
    -------
    prm : (F,) np.ndarray
        Parameters

    """

    # Author
    # ------
    # .. Yael Balbastre <yael.balbastre@gmail.com>

    mat = np.asarray(mat)
    in_dtype = mat.dtype
    dim = mat.shape[-1] - 1
    mat = mat.astype(np.float64)
    if basis is None:
        basis = affine_basis('Aff', dim)
    nb_basis = basis.shape[0]

    # Project to tangent space
    if layout == 'RAS':
        mat = logm(mat)
    else:
        if isinstance(layout, str):
            layout = affine_layout(layout)
        mat = rmdiv(mat, layout)
        mat = logm(mat)

    # Project to orthonormal basis in the tangent space
    prm = np.zeros(nb_basis, dtype=in_dtype)
    for n_basis in range(nb_basis):
        prm[n_basis] = np.trace(np.matmul(mat, basis[n_basis, ...].transpose()))

    return prm


def affine_subbasis(mode, dim=3, dtype='float64'):
    """Return a basis for the algebra of some (Lie) groups of matrices.

    The basis is returned in homogeneous coordinates, even if
    the group required does not require translations. To extract the linear
    part of the basis: lin = basis[:-1, :-1].

    Parameters
    ----------
    mode : {'T', 'R', 'Z', 'S', 'ISO'}
        Group that should be encoded by the basis set:
            * 'T'   : Translations
            * 'R'   : Rotations
            * 'Z'   : zooms
            * 'S'   : shears
            * 'ISO' : isotropic zoom
    dim : {1, 2, 3}, default=3
        Dimension
    dtype : str or type, default='float64'
        Data type of the returned array

    Returns
    -------
    basis : np.ndarray of shape (F, dim+1, dim+1)
        Basis set, where ``F`` is the number of basis functions.

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python code

    if dim not in (1, 2, 3):
        raise ValueError('dim must be one of 1, 2, 3')
    if mode not in ('T', 'R', 'Z', 'S', 'ISO'):
        raise ValueError('mode {} not implemented'.format(mode))

    if mode == 'T':
        basis = np.zeros((dim, dim+1, dim+1), dtype=dtype)
        for i in range(dim):
            basis[i, i, -1] = 1
    elif mode == 'Z':
        basis = np.zeros((dim, dim+1, dim+1), dtype=dtype)
        for i in range(dim):
            basis[i, i, i] = 1
    elif mode == 'ISO':
        basis = np.zeros((1, dim+1, dim+1), dtype=dtype)
        for i in range(dim):
            basis[0, i, i] = 1
    elif mode == 'R':
        basis = np.zeros((dim*(dim-1)//2, dim+1, dim+1), dtype=dtype)
        k = 0
        for i in range(dim):
            for j in range(i+1, dim):
                basis[k, i, j] = 1/np.sqrt(2)
                basis[k, j, i] = -1/np.sqrt(2)
                k += 1
    elif mode == 'S':
        basis = np.zeros((dim*(dim-1)//2, dim+1, dim+1), dtype=dtype)
        k = 0
        for i in range(dim):
            for j in range(i+1, dim):
                basis[k, i, j] = 1/np.sqrt(2)
                basis[k, j, i] = 1/np.sqrt(2)
                k += 1
    return basis


def affine_basis(group='SE', dim=3, dtype='float64'):
    """Generate basis set for the algebra of some (Lie) group of matrices.

    The basis is returned in homogeneous coordinates, even if
    the group required does not require translations. To extract the linear
    part of the basis: lin = basis[:-1, :-1].

    Parameters
    ----------
    group : {'T', 'SO', 'SE', 'SL', 'Aff'}, default='SE'
        Group that should be encoded by the basis set:
            * 'T'   : Translations
            * 'SO'  : Special Orthogonal (rotations)
            * 'SE'  : Special Euclidean (translations + rotations)
            * 'SL'  : Special Linear (rotations + zooms + shears)
            * 'Aff' : Affine (translations + rotations + zooms + shears)
    dim : {1, 2, 3}, default=3
        Dimension
    dtype : str or type, default='float64'
        Data type of the returned array

    Returns
    -------
    basis : (F, dim+1, dim+1) ndarray
        Basis set, where ``F`` is the number of basis functions.

    """
    # TODO:
    # - other groups?

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python code

    if dim not in (1, 2, 3):
        raise ValueError('dim must be one of 1, 2, 3')
    if group not in ('T', 'SO', 'SE', 'SL', 'Aff'):
        raise ValueError('group must be one of T, SO, SE, SL, Aff')

    if group == 'T':
        return affine_subbasis('T', dim, dtype=dtype)
    elif group == 'SO':
        return affine_subbasis('R', dim, dtype=dtype)
    elif group == 'SE':
        return np.concatenate((affine_subbasis('T', dim, dtype=dtype),
                               affine_subbasis('R', dim, dtype=dtype)))
    elif group == 'SL':
        return np.concatenate((affine_subbasis('R', dim, dtype=dtype),
                               affine_subbasis('Z', dim, dtype=dtype),
                               affine_subbasis('S', dim, dtype=dtype)))
    elif group == 'Aff':
        return np.concatenate((affine_subbasis('T', dim, dtype=dtype),
                               affine_subbasis('R', dim, dtype=dtype),
                               affine_subbasis('Z', dim, dtype=dtype),
                               affine_subbasis('S', dim, dtype=dtype)))


def change_layout(mat, shape, layout='RAS'):
    """Reorient an affine matrix / a volume to match a target layout.

    Parameters
    ----------
    mat : (D+1+, D+1) array_like
        Orientation matrix
    shape : (D,) array_like or (shape*, features*) array_like
        Shape or Volume
    layout : str or (D+1+, D+1) array_like
        Name of a layout or corresponding matrix

    Returns
    -------
    mat : (D+1, D+1) np.ndarray
        Reoriented orientation matrix
    shape : (D,) np.ndarray or (permuted_shape*, features*) np.ndarray
        Reoriented shape or  volume

    """

    mat = np.asarray(mat)
    dim = mat.shape[-1] - 1
    shape = np.asarray(shape)
    array = None
    if len(shape.shape) > 1:
        array = shape
        shape = array.shape[:dim]

    # Find combination of 90 degree rotations and flips that brings
    # all the matrices closest to the target layout.
    # In practice, combinations are implemented as permutations
    # (= 90º rotation + flip) and flips.
    perms = list(itertools.permutations(range(dim)))
    flips = list(itertools.product([True, False], repeat=dim))
    if isinstance(layout, str):
        layout = affine_layout(layout)

    # Remove scale and translation
    R0 = mat[:dim, :dim]
    vs = (R0**2).sum(axis=1)
    R0 = rmdiv(R0, np.diag(vs))
    min_sos = np.inf
    min_R = np.eye(dim)
    min_perm = list(range(dim))
    I = layout[:dim, :dim]

    for perm in perms:
        # Build permutation matrix
        P = np.zeros(dim*dim)
        P[sub2ind([perm, range(dim)], (dim, dim))] = 1
        P = P.reshape((dim, dim))
        for flip in flips:
            # Build flip matrix
            F = np.diag([2*f-1 for f in flip])

            # Combine and compare
            R = np.matmul(F, P)
            sos = ((rmdiv(R0, R) - I) ** 2).sum()
            if sos < min_sos:
                min_sos = sos
                min_R = R
                min_perm = perm

    # Flips also include a translation; they are defined by the
    # affine mapping:
    # . 0 -> d-1
    # . d-1 -> 0
    transformed_corner = np.matmul(min_R, shape)
    iR = np.linalg.inv(min_R)
    T = (iR.sum(0)-1)/2 * (transformed_corner+1)
    min_R = np.concatenate((iR, T[:, None]), axis=1)
    pad = np.array([[0]*dim + [1]], dtype=min_R.dtype)
    min_R = np.concatenate((min_R, pad), axis=0)
    mat = np.matmul(mat, min_R)

    if array:
        array = array.transpose(min_perm)
        return mat, array
    else:
        shape = shape[list(min_perm)]
        return mat, shape


def mean_affine(mats, shapes):
    """Compute a mean orientation matrix.

    Parameters
    ----------
    mats : (N, D+1, D+1) array_like or list[(D+1, D+1) array_like]
        Input orientation matrices
    shapes : (N, D) array_like or list[(D,) array like]
        Input shape

    Returns
    -------
    mat : (D+1, D+1) np.ndarray
        Mean orientation matrix, with an RAS layout

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Mikael Brudfors <brudfors@gmail.com> : Python port
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python port
    #
    # License
    # -------
    # The original Matlab code is (C) 2019-2020 WCHN / John Ashburner
    # and was distributed as part of [SPM](https://www.fil.ion.ucl.ac.uk/spm)
    # under the GNU General Public Licence (version >= 2).

    if not isinstance(mats, np.ndarray):
        mats = list(mats)
        mats = np.stack(mats)
    mats = mats.copy()
    dim = mats.shape[-1] - 1

    if shapes is not None:
        if not isinstance(shapes, np.ndarray):
            shapes = list(shapes)
            shapes = np.stack(shapes)

    # STEP 1: Reorient to RAS layout
    # ------
    # Computing an exponential mean only works if all matrices are
    # "close". In particular, if the voxel layout associated with these
    # matrices is different (e.g., RAS vs LAS vs RSA), the exponential
    # mean will fail. The first step is therefore to reorient all
    # matrices so that they map to a common voxel layout.
    # We choose RAS as the common layout, as it makes further steps
    # easier ans matches the world space orientation.
    RAS = np.eye(dim+1, dtype=np.float64)
    for mat, shape in zip(mats, shapes):
        mat[:, :], shape[:] = change_layout(mat, shape, RAS)

    # STEP 2: Compute exponential barycentre
    # ------
    mat = meanm(mats)

    # STEP 3: Remove spurious shears
    # ------
    # We want the matrix to be "rigid" = the combination of a
    # rotation+translation (T*R) in world space and of a "voxel size"
    # scaling (Z), i.e., M = T*R*Z.
    # We look for the matrix that can be encoded without shear bases
    # that is the closest to the original matrix (in terms of the
    # Frobenius norm of the residual matrix)
    basis_shears = affine_subbasis('S', dim)
    prm_shears = affine_parameters(mat, basis_shears)
    if (prm_shears ** 2).sum() > 1e-8:
        basis_rotation = affine_subbasis('R', dim)
        basis_zoom = affine_subbasis('Z', dim)
        nb_prm_rotation = basis_rotation.shape[0]
        basis = np.concatenate((basis_rotation, basis_zoom), axis=0)
        prm = affine_parameters(mat, basis)
        for n_iter in range(10000):
            R, dR = dexpm(prm[:nb_prm_rotation], basis_rotation)
            Z, dZ = dexpm(prm[nb_prm_rotation:], basis_zoom)
            M = R*Z
            dM = np.concatenate((np.matmul(dR, Z), np.matmul(R, dZ)))
            diff = M - mat
            gradient = np.tensordot(dM, diff.transpose, axis=2)
            hessian = np.tensordot(dM, dM.transpose, axis=2)
            prm = prm - lmdiv(hessian, gradient)
            if (gradient ** 2).sum() < 1e-8:
                print('Early stopping at iteration {}'.format(n_iter))
                break
        mat[:dim, :dim] = M[:dim, :dim]

    return mat

