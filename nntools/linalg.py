from warnings import warn
import numpy as np
from scipy.linalg import logm, expm


def lmdiv(A, B, rcond=None):
    r"""Left matrix division A\B.

    Parameters
    ----------
    A : (M, [N]) array_like
    B : (M, [K]) array_like

    Returns
    -------
    X : (N, [K]) np.ndarray

    """
    A = np.asarray(A)
    B = np.asarray(B)
    if len(A.shape) == 1:
        A = A[..., None]
    X = np.linalg.lstsq(A, B, rcond=rcond)[0]
    return X


def rmdiv(A, B, rcond=None):
    r"""Right matrix division A/B.

    Parameters
    ----------
    A : (M, [N]) array_like
    B : (K, [N]) array_like

    Returns
    -------
    X : (M, K) np.ndarray

    """
    A = np.asarray(A)
    B = np.asarray(B)
    if len(A.shape) == 1:
        A = A[..., None]
    if len(B.shape) == 1:
        B = B[..., None]
    return np.linalg.lstsq(B.transpose(), A.transpose(), rcond=rcond)[0].transpose()


def matmul(x1, x2=None, axis=None, **kwargs):
    """Matrix multiplication (with extended capabilities)

    Parameters
    ----------
    x1 : array_like
        First matrix
    x2 : array_like, optional
        Second matrix
    axis : int, optional
        Axis along which to extract matrices. Default axis is 0
    **kwargs
        Other keyword only arguments. See numpy.matmul

    Returns
    -------
    x : ndarray
        * If x2 is None: extract matrices from x1 along an axis and
          multiply them together.
        * Else: classic matrix multiplication x1 @ x2. See numpy.matmul

    """
    if x2 is None:
        # Product across a dimension of x1
        x1 = np.asarray(x1)
        if axis is None:
            axis = 0
        x = np.take(x1, 0, axis=axis)
        for n_mat in range(1, x1.shape[axis]):
            x = np.matmul(x, np.take(x1, n_mat, axis=axis))
        return x
    else:
        # Product of two matricws
        if axis is not None:
            raise ValueError('Cannot use ``b`` and ``axis`` together.')
        return np.matmul(x1, x2, **kwargs)


def mm(*args, **kwargs):
    """Alias for matmul"""
    return matmul(*args, **kwargs)


def meanm(mats, max_iter=1024, tol=1e-20):
    """Compute the exponential barycentre of a set of matrices.

    Parameters
    ----------
    mats : (N, M, M) array_like
        Set of square invertible matrices
    max_iter : int, default=1024
        Maximum number of iterations
    tol : float, default=1E-20
        Tolerance for early stopping.
        The tolerance criterion is the sum-of-squares of the residuals
        in log-space, _i.e._, :math:`||\sum_n \log_M(A_n) / N||^2`

    Returns
    -------
    mean_mat : (M, M) np.ndarray
        Mean matrix.

    References
    ----------
    .. [1]  Xavier Pennec, Vincent Arsigny.
        "Exponential Barycenters of the Canonical Cartan Connection and
        Invariant Means on Lie Groups."
        Matrix Information Geometry, Springer, pp.123-168, 2012.
        ⟨10.1007/978-3-642-30232-9_7⟩. ⟨hal-00699361⟩
        https://hal.inria.fr/hal-00699361

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Mikael Brudfors <brudfors@gmail.com> : Python port
    #
    # License
    # -------
    # The original Matlab code is (C) 2012-2019 WCHN / John Ashburner
    # and was distributed as part of [SPM](https://www.fil.ion.ucl.ac.uk/spm)
    # under the GNU General Public Licence (version >= 2).

    mats = np.asarray(mats)
    dim = mats.shape[2]-1
    in_dtype = mats.dtype
    acc_dtype = np.float64

    mean_mat = np.eye(dim+1, dtype=acc_dtype)
    zero_mat = np.zeros((dim+1, dim+1), dtype=acc_dtype)
    for n_iter in range(max_iter):
        mean_log_mat = zero_mat.copy()
        for mat in mats:
            mean_log_mat += logm(lmdiv(mean_mat, mat.astype(acc_dtype)))
        mean_log_mat /= mats.shape[0]
        mean_mat = np.matmul(mean_mat, expm(mean_log_mat))
        if (mean_log_mat ** 2).sum() < tol:
            break
    return mean_mat.astype(in_dtype)


def dexpm(X, basis, max_order=10000, tol=1e-32):
    """Derivative of the matrix exponential.

    This function evaluates the matrix exponential and its derivative
    using a Taylor approximation. A faster integration technique, based
    e.g. on scaling and squaring, could have been used instead.

    Parameters
    ----------
    X : {(F,), (D, D)} array_like
        If vector_like: parameters of the log-matrix in the basis set
        If matrix_like: log-matrix
    basis : (F, D, D) array_like
        Basis set
    max_order : int, default=10000
        Order of the Taylor expansion
    tol : float, default=1e-32
        Tolerance for early stopping
        The criterion is based on the Frobenius norm of the last term of
        the Taylor series.

    Returns
    -------
    eX : (D, D) np.ndarray
        Matrix exponential
    dX : (F, D, D) np.ndarray
        Derivative of the matrix exponential with respect to the
        parameters in the basis set

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Mikael Brudfors <brudfors@gmail.com> : Python port
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python port
    #
    # License
    # -------
    # The original Matlab code is (C) 2012-2019 WCHN / John Ashburner
    # and was distributed as part of [SPM](https://www.fil.ion.ucl.ac.uk/spm)
    # under the GNU General Public Licence (version >= 2).

    X = np.asarray(X)
    in_dtype = X.dtype
    X = X.astype(np.float64)
    if basis is None:
        return None
    basis = np.asarray(basis, dtype=np.float64)
    nb_basis = basis.shape[0]

    if len(X.shape) == 1:
        # Assume that input contains parameters in the algebra
        # -> reconstruct matrix
        X = (basis * X[:, None, None]).sum(axis=0)

    # Aliases
    I = np.eye(X.shape[0], dtype=np.float64)
    E = I + X                           # expm(X)
    dE = basis.copy()                   # dexpm(X)
    En = X.copy()                       # n-th Taylor coefficient of expm
    dEn = basis.copy()                  # n-th Taylor coefficient of dexpm
    for n_order in range(2, max_order+1):
        for n_basis in range(nb_basis):
            dEn[n_basis, ...] = (np.matmul(dEn[n_basis, ...], X) +
                                 np.matmul(En, basis[n_basis, ...]))/n_order
        dE += dEn
        En = np.matmul(En, X)/n_order
        E += En
        if (En ** 2).sum() < En.size * tol:
            break
    if (En ** 2).sum() >= En.size * tol:
        warn('expm did not converge.', RuntimeWarning)

    E = E.astype(in_dtype)
    dE = dE.astype(in_dtype)
    return E, dE


def check_commutative(mats):
    """ Check if a list of matrices commmute.

    Parameters
    ----------
    mats - (N, K, K) array_like or iterable[(K, K) array_like]

    Returns
    -------
    is_commutative : bool
        True if all pairs of matrices commute
    table : (N, N) np.ndarray
        Table of commutativity

    """
    mats = np.stack((mat for mat in mats))
    nb_matrices = len(mats)
    check = np.eye(nb_matrices, dtype=bool)
    for i in range(nb_matrices):
        for j in range(i, nb_matrices):
            check[i, j] = np.allclose(mats[i, ...] @ mats[j, ...],
                                      mats[j, ...] @ mats[i, ...])
            check[j, i] = check[i, j]
    return check.all(), check


def check_orthonormal(mats):
    """ Check if a list of matrices forms an orthonormal basis.

    The basis is with respect to the matrix dot product:
        <A, B> = trace(A.t() @ B)

    Parameters
    ----------
    mats - (N, K, K) array_like or iterable[(K, K) array_like]

    Returns
    -------
    is_orthonormal : bool
        True if the basis is orthonormal
    table : (N, N) np.ndarray
        Dot product table

    """
    mats = np.stack((mat for mat in mats))
    nb_matrices = len(mats)
    dot = np.eye(nb_matrices, dtype=np.float64)
    for i in range(nb_matrices):
        for j in range(i, nb_matrices):
            dot[i, j] = np.trace(mats[i, ...] @ mats[j, ...].transpose())
            dot[j, i] = dot[i, j]
    eye = np.eye(nb_matrices, dtype=np.float64)
    return np.allclose(dot, eye), dot
