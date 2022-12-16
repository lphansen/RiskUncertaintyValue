"""
Functions to facilitate the computation in expansion solvers. 

"""
import numpy as np
from scipy import linalg
from numba import njit


@njit
def mat(h, shape):
    r"""
    For a vector (column or row) vec of length mn, mat(h, (m, n)) 
    produces an (m, n) matrix created by ‘columnizing’ the vector:

    .. math::
        H_{ij} = h_{(j-1)m+i}

    Parameters
    ----------
    h : (mn, 1) ndarray
    shape : tuple of ints
         Shape of H.

    Returns
    -------
    H : (m, n) ndarray

    References
    ---------
    Borovicka, Hansen (2014). See http://larspeterhansen.org/wp-content/uploads/2016/10/Examining-Macroeconomic-Models-through-the-Lens-of-Asset-Pricing.pdf

    """
    H = h.reshape((shape[1], shape[0])).T
    return H


@njit
def vec(H):
    r"""
    For an (m, n) matrix H , vec(H) produces a column vector
    of length mn created by stacking the columns of H:

    .. math::
        [vec(H)]_{(j-1)m+i} = H_{ij}

    Parameters
    ----------
    H : (m, n) ndarray

    Returns
    -------
    h : (n*m, 1) ndarray

    References
    ---------
    Borovicka, Hansen (2014). See http://larspeterhansen.org/wp-content/uploads/2016/10/Examining-Macroeconomic-Models-through-the-Lens-of-Asset-Pricing.pdf

    """
    H_T = H.T.copy()
    h = H_T.reshape(-1, 1)
    return h


@njit
def sym(M):
    r"""
    Computes :math:`\frac{1}{2} (M + M^T)`.

    Parameters
    ----------
    M : (m, m) ndarray

    Returns
    -------
    sym_M : (m, m) ndarray

    References
    ---------
    Borovicka, Hansen (2014). See http://larspeterhansen.org/wp-content/uploads/2016/10/Examining-Macroeconomic-Models-through-the-Lens-of-Asset-Pricing.pdf

    """
    sym_M = (M + M.T) / 2
    return sym_M


@njit
def cal_E_ww(E_w, Cov_w):
    """
    Computes expectation of :math:`W \otimes W`, where W follows multivariate normal distribution.

    Parameters
    ----------
    E_w : (m, 1) ndarray
         Expectation of W.
    Cov_w : (m, m) ndarray
         Covariance matrix of W.

    Returns
    -------
    E_ww: (m*m, 1) ndarray
         Expectaton of :math:`W \otimes W`.

    """
    m = E_w.shape[0]
    E_ww = np.zeros((m**2, 1))
    for i in range(m):
        for j in range(m):
            E_ww[i*m+j] = E_w[i, 0]*E_w[j, 0]+Cov_w[i,j]
    return E_ww


@njit
def solve_matrix_equation(A, B, C, D):
    r"""
    Solves for:

    .. math::
        A\psi + B\psiC + D = 0

    The solution to the equation is:

    .. math::
        \psi = \text{mat}\{-[I\otimes A + C^\prime\otimes B\]^{-1}\text{vec}(D)}_{n,m}

    Parameters
    ----------
    A : (n, n) ndarray
    B : (n, n) ndarray
    C : (m, m) ndarray
    D : (n, m) ndarray

    Returns
    -------
    res : (n, m) ndarray

    References
    ----------
    Borovicka and Hansen (2014).
    https://www.borovicka.org/research.html
    """
    n = B.shape[1]
    m = C.shape[0]
    LHS = np.kron(np.eye(m), A) + np.kron(C.T, B)
    RHS = - vec(D)
    vec_res = np.linalg.solve(LHS, RHS)
    res = mat(vec_res, (n, m))
    return res


def gschur(A, B, tol=1e-9):
    """
    Performs generalized schur decomposition (QZ decomposition) with reordering.
    Pushes explosive eigenvalues (i.e., > 1) to the right bottom.

    Parameters
    ----------
    A: (m, m) ndarray to decompose
    B: (m, m) ndarray to decompose
    tol: a tolerance level added to the threshold 1, allowing for numerical disturbance

    Returns
    -------
    AA : (m, m) ndarray
        Generalized Schur form of A.
    BB : (m, m) ndarray
        Generalized Schur form of B.
    a : (m,) ndarray
        alpha = alphar + alphai * 1j.
    b : (m,) ndarray
        See reference.
    Q : (m, m) ndarray
        The left Schur vectors.
    Z : (m, m) ndarray
        The right Schur vectors.

    References
    ---------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.ordqz.html

    """
    def sort_req(alpha, beta):
        return np.abs(alpha) <= (1+tol)*np.abs(beta)

    BB, AA, a, b, Q, Z = linalg.ordqz(B, A, sort=sort_req)

    return AA, BB, a, b, Q, Z
