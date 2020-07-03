import re
import numpy as np
import itertools
from warnings import warn
from .linalg import lmdiv, rmdiv, mm, meanm, dexpm
from .utils import sub2ind, majority
from scipy.linalg import logm, expm
from copy import deepcopy


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
        The most common layout is 'RAS', which maps to the 'world'
        orientation 'RAS+' with an identity matrix.
        If the first voxel dimension browsed the brain from right to
        left, the layout would be 'LAS'.
        Note that this layout is only approximate; in practice, angled
        field-of-views would fall sort of in-between these layouts.

        The number of letters defines the dimension of the matrix
        ('R' -> 1D, 'RA' -> 2D, 'RAS' : 3D).

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
    mat = mat[:, perm + [3]]

    return mat


def affine_find_layout(mat):
    """Find the voxel layout associated with an affine matrix.

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
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original idea
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python code

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


def _format_basis(basis, dim=None):
    """Transform an Outter/Inner Lie basis into a list of arrays."""

    basis0 = basis
    basis = deepcopy(basis0)

    # Guess dimension
    if dim is None:
        if isinstance(basis, np.ndarray):
            dim = basis.shape[-1] - 1
        else:
            for outer_basis in basis:
                if isinstance(outer_basis, np.ndarray):
                    dim = outer_basis.shape[0] - 1
                    break
                elif not isinstance(outer_basis, str):
                    for inner_basis in outer_basis:
                        if not isinstance(inner_basis, str):
                            inner_basis = np.asarray(inner_basis)
                            dim = inner_basis.shape[0] - 1
                            break
    if dim is None:
        # Guess failed
        dim = 3

    # Helper to convert named bases to matrices
    def name_to_basis(name):
        if outer_basis in affine_basis_choices:
            return affine_basis(name, dim)
        elif outer_basis in affine_subbasis_choices:
            return affine_subbasis(name, dim)
        else:
            raise ValueError('Unknown basis name {}.'
                             .format(name))

    # Convert 'named' bases to matrix bases
    if not isinstance(basis, np.ndarray):
        basis = list(basis)
        for n_outer, outer_basis in enumerate(basis):
            if isinstance(outer_basis, str):
                basis[n_outer] = name_to_basis(outer_basis)
            elif not isinstance(outer_basis, np.ndarray):
                outer_basis = list(outer_basis)
                for n_inner, inner_basis in enumerate(outer_basis):
                    if isinstance(inner_basis, str):
                        outer_basis[n_inner] = name_to_basis(inner_basis)
                    else:
                        outer_basis[n_inner] = np.asarray(inner_basis)
                outer_basis = np.concatenate(outer_basis)
                basis[n_outer] = outer_basis

    return basis, dim


def affine_matrix(prm, basis, dim=None, layout='RAS', prod=True):
    r"""Reconstruct an affine matrix from its Lie parameters.

    Affine matrices are encoded as product of sub-matrices, where
    each sub-matrix is encoded in a Lie algebra. Finally, the right
    most matrix is a 'layout' matrix (see affine_layout).
    ..math: M   = exp(A_1) \times ... \times exp(A_n) \times L
    ..math: A_i = \sum_k = p_{ik} B_{ik}

    An SPM-like construction (as in ``spm_matrix``) would be:
    >>> M = affine_matrix(prm, ['T', 'R[0]', 'R[1]', 'R[2]', 'Z', 'S'])
    Rotations need to be split by axis because they do not commute.

    Parameters
    ----------
    prm : vector_like or vector_like[vector_like]
        Parameters in the Lie algebra(s).

    basis : vector_like[basis_like]
        The outer level corresponds to matrices in the product (*i.e.*,
        exponentiated matrices), while the inner level corresponds to
        Lie algebras.

    dim : int, default=guess or 3
        If not provided, the function tries to guess it from the shape
        of the basis matrices. If the dimension cannot be guessed
        (because all bases are named bases), the default is 3.

    layout : str or matrix_like, default='RAS'
        A layout matrix.

    Returns
    -------
    mat : (D+1, D+1) np.ndarray
        Reconstructed affine matrix

    """

    # Author
    # ------
    # .. Yael Balbastre <yael.balbastre@gmail.com>

    # Make sure basis is a vector_like of (F, D+1, D+1) ndarray
    basis, dim = _format_basis(basis, dim)

    # Check length
    nb_basis = np.sum([len(b) for b in basis])
    prm = np.asarray(prm).flatten()
    in_dtype = prm.dtype
    if len(prm) != nb_basis:
        raise ValueError('Number of parameters and number of bases '
                         'do not match. Got {} and {}'
                         .format(len(prm), nb_basis))

    # Helper to reconstruct a log-matrix
    def recon(p, B):
        p = np.asarray(p, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        return expm((B*p[:, None, None]).sum(axis=0))

    # Reconstruct each sub matrix
    n_prm = 0
    mats = []
    for a_basis in basis:
        nb_prm = a_basis.shape[0]
        a_prm = prm[n_prm:(n_prm+nb_prm)]
        mats.append(recon(a_prm, a_basis))
        n_prm += nb_prm

    # Add layout matrix
    if layout != 'RAS':
        if isinstance(layout, str):
            layout = affine_layout(layout)
        mats.append(layout)

    # Matrix product
    if prod:
        return mm(np.stack(mats)).astype(in_dtype)
    else:
        return mats


def _affine_parameters_single_basis(mat, basis, layout='RAS'):

    # Project to tangent space
    if not isinstance(layout, str) or layout != 'RAS':
        if isinstance(layout, str):
            layout = affine_layout(layout)
        mat = rmdiv(mat, layout)
    mat = logm(mat)

    # Project to orthonormal basis in the tangent space
    prm = np.zeros(basis.shape[0], dtype=np.float64)
    for n_basis in range(basis.shape[0]):
        prm[n_basis] = np.trace(np.matmul(mat, basis[n_basis, ...].transpose()))

    return prm


def affine_parameters(mat, basis, layout='RAS', max_iter=10000, tol=1e-16,
                      max_line_search=6):
    """Compute the parameters of an affine matrix in a basis of the algebra.

    This function finds the matrix closest to ``mat`` (in the least squares
    sense) that can be encoded in the specified basis.

    Parameters
    ----------
    mat : (D+1, D+1) array_like
        Affine matrix

    basis : vector_like[basis_like]
        Basis of the Lie algebra(s).

    layout : str or (D+1, D+1) array_like, default='RAS'
        "Point" at which to take the matrix exponential
        (see affine_layout)

    max_iter : int, default=10000
        Maximum number of Gauss-Newton iterations in the least-squares fit.

    tol : float, default = 1e-8
        Tolerance criterion for convergence.
        It is based on the squared norm of the GN step divided by the
        squared norm of the input matrix.

    max_line_search: int, default=6
        Maximum number of line search steps.
        If zero: no line-search is performed.

    Returns
    -------
    prm : ndarray
        Parameters in the specified basis

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original GN fit in Matlab
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python code

    # Format mat
    mat = np.asarray(mat)
    in_dtype = mat.dtype
    dim = mat.shape[-1] - 1
    mat = mat.astype(np.float64)

    # Format basis
    basis, _ = _format_basis(basis, dim)
    nb_basis = np.sum([len(b) for b in basis])

    # Create layout matrix
    if isinstance(layout, str):
        layout = affine_layout(layout)

    def gauss_newton():
        # Predefine these values in case max_iter == 0
        n_iter = -1
        # Gauss-Newton optimisation
        prm = np.zeros(nb_basis, dtype=np.float64)
        M = affine_matrix(prm, basis)
        sos = ((M - mat) ** 2).sum()
        norm = (mat ** 2).sum()
        crit = np.inf
        for n_iter in range(max_iter):

            # Compute derivative of each submatrix with respect to its basis
            Ms = []
            dMs = []
            n_basis = 0
            for a_basis in basis:
                nb_a_basis = a_basis.shape[0]
                a_prm = prm[n_basis:(n_basis+nb_a_basis)]
                M, dM = dexpm(a_prm, a_basis)
                Ms.append(M)
                dMs.append(dM)
                n_basis += nb_a_basis
            M = np.stack(Ms)

            # Compute derivative of the full matrix with respect to each basis
            for n_mat, dM in enumerate(dMs):
                if n_mat > 0:
                    pre = mm(M[:n_mat, ...], axis=0)
                    dM = mm(pre, dM)
                if n_mat < M.shape[0]-1:
                    post = mm(M[(n_mat+1):, ...], axis=0)
                    dM = mm(dM, post)
                dMs[n_mat] = dM
            dM = np.concatenate(dMs)
            M = mm(M, axis=0)

            # Multiply with layout
            M = mm(M, layout)
            dM = mm(dM, layout)

            # Compute gradient/Hessian of the loss (squared residuals)
            diff = M - mat
            diff = diff.flatten()
            dM = dM.reshape((nb_basis, -1))
            gradient = mm(dM, diff)
            hessian = mm(dM, dM.transpose())
            delta_prm = lmdiv(hessian, gradient)
            # prm -= delta_prm

            crit = (delta_prm ** 2).sum() / norm
            if crit < tol:
                break

            if max_line_search == 0:
                # We trust the Gauss-Newton step
                prm -= delta_prm
            else:
                # Line Search
                sos0 = sos
                prm0 = prm
                M0 = M
                armijo = 1
                success = False
                for _ in range(max_line_search):
                    prm = prm0 - armijo * delta_prm
                    M = affine_matrix(prm, basis)
                    sos = ((M - mat) ** 2).sum()
                    if sos < sos0:
                        success = True
                        break
                    else:
                        armijo /= 2
                if not success:
                    prm = prm0
                    M = M0
                    break

        if crit >= tol:
            warn('Gauss-Newton optimisation did not converge: '
                 'n_iter = {}, sos = {}.'.format(n_iter + 1, crit),
                 RuntimeWarning)

        return prm, M

    prm, M = gauss_newton()

    # TODO: should I stack parameters per basis?
    return prm.astype(in_dtype), M.astype(in_dtype)


affine_subbasis_choices = ('T', 'R', 'Z', 'S', 'ISO')


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
    if mode not in affine_subbasis_choices:
        raise ValueError('mode must be one of {}.'
                         .format(affine_subbasis_choices))

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


affine_basis_choices =  ('T', 'SO', 'SE', 'SL', 'Aff')


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
    if group not in affine_basis_choices:
        raise ValueError('group must be one of {}.'
                         .format(affine_basis_choices))

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

    # Convert to (N,, D+1, D+1) ndarray + copy
    # We copy because we're going to write inplace.
    shapes = np.asarray(shapes)
    mats = np.array(mats, copy=True)
    dim = mats.shape[-1] - 1

    # STEP 1: Reorient to RAS layout
    # ------
    # Computing an exponential mean only works if all matrices are
    # "close". In particular, if the voxel layout associated with these
    # matrices is different (e.g., RAS vs LAS vs RSA), the exponential
    # mean will fail. The first step is therefore to reorient all
    # matrices so that they map to a common voxel layout.
    # We choose RAS as the common layout, as it makes further steps
    # easier and matches the world space orientation.
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
    # We look for the matrix that can be encoded without shears
    # that is the closest to the original matrix (in terms of the
    # Frobenius norm of the residual matrix)
    _, M = affine_parameters(mat, ['R', 'Z'])
    mat[:dim, :dim] = M[:dim, :dim]

    return mat

