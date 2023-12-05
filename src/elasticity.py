"""
Tools to compute exposure/price elasticities.

Reference: Borovicka, Hansen (2014). See http://larspeterhansen.org/.

"""
import numpy as np
from scipy.stats import norm
from lin_quad import LinQuadVar
from lin_quad_util import log_E_exp, next_period, kron_prod, M_mapping
from utilities import mat, vec, sym
from numba import njit, prange

def exposure_elasticity(log_M_growth, X1_tp1, X2_tp1, T=400, shock=0, percentile=0.5):
    r"""
    Computes exposure elasticity for M.

    Parameters
    ----------
    log_M_growth : LinQuadVar
        Log growth of multiplicative functional M.
        e.g. log consumption growth, :math:`\log \frac{C_{t+1}}{C_t}`
    X1_tp1 : LinQuadVar
        Stores the coefficients of laws of motion for X1.
    X2_tp1 : LinQuadVar
        Stores the coefficients of laws of motion for X2.        
    T : int
        Time horizon.
    shock : int
        Position of the initial shock, starting from 0.
    percentile : float
        Specifies the percentile of the elasticities.

    Returns
    -------
    elasticities : (T, n_Y) ndarray
        Exposure elasticities.

    Reference
    ---------
    Borovicka, Hansen (2014). See http://larspeterhansen.org/.

    """
    n_Y, n_X, n_W = log_M_growth.shape
    if n_Y != 1:
        raise ValueError("The dimension of input should be 1.")

    α = np.zeros(n_W)
    α[shock] = 1    
    p = norm.ppf(percentile)

    Σ_tilde_t, μ_t0, μ_t1 = _elasticity_coeff(log_M_growth, X1_tp1, X2_tp1, T)

    kron_product = np.kron(X1_tp1['x'], X1_tp1['x'])
    x_mean = np.linalg.solve(np.eye(n_X)-X1_tp1['x'],X1_tp1['c'])
    x_cov = mat(np.linalg.solve(np.eye(n_X**2)-kron_product,
                                vec(X1_tp1['w']@X1_tp1['w'].T)), (n_X, n_X))

    elasticities = _exposure_elasticity_loop(T, n_Y, α, Σ_tilde_t, μ_t0,
                                             μ_t1, percentile, x_mean, x_cov, p)

    return elasticities


def price_elasticity(log_G_growth, log_S_growth, X1_tp1, X2_tp1, T=400, shock=0, percentile=0.5):
    r"""
    Computes price elasticity.

    Parameters
    ----------
    log_G_growth : LinQuadVar
        Log growth of multiplicative functional G.
        e.g. log consumption growth, :math:`\log \frac{C_{t+1}}{C_t}`
    log_S_growth : LinQuadVar
        Log growth of multiplicative functional S.
        e.g. log stochastic discount factor, :math:`\log \frac{S_{t+1}}{S_t}`
    X1_tp1 : LinQuadVar
        Stores the coefficients of laws of motion for X1.
    X2_tp2 : LinQuadVar or None
        Stores the coefficients of laws of motion for X2.        
    T : int
        Time horizon.
    shock : int
        Position of the initial shock, starting from 0.
    percentile : float
        Specifies the percentile of the elasticities.

    Returns
    -------
    elasticities : (T, dim) ndarray
        Price elasticities.

    Reference
    ---------
    Borovicka, Hansen (2014). See http://larspeterhansen.org/.

    """
    if log_G_growth.shape != log_S_growth.shape:
        raise ValueError("The dimensions of G and S do not match.")
    else:
        n_Y, n_X, n_W = log_G_growth.shape
        if n_Y != 1:
            raise ValueError("The dimension of inputs should be (1, n_X, n_W)")
    α = np.zeros(n_W)
    α[shock] = 1          

    p = norm.ppf(percentile)

    Σ_tilde_expo_t, μ_expo_t0, μ_expo_t1 \
        = _elasticity_coeff(log_G_growth, X1_tp1, X2_tp1, T)
    Σ_tilde_value_t, μ_value_t0, μ_value_t1\
        = _elasticity_coeff(log_G_growth+log_S_growth, X1_tp1, X2_tp1, T)

    kron_product = np.kron(X1_tp1['x'], X1_tp1['x'])
    x_mean = np.linalg.solve(np.eye(n_X)-X1_tp1['x'],X1_tp1['c'])
    x_cov = mat(np.linalg.solve(np.eye(n_X**2)-kron_product,
                                vec(X1_tp1['w']@X1_tp1['w'].T)), (n_X, n_X))
    
    elasticities = _price_elasticity_loop(T, n_Y, α, Σ_tilde_expo_t, Σ_tilde_value_t, 
                           μ_expo_t0, μ_value_t0, μ_expo_t1, μ_value_t1,
                           percentile, x_mean, x_cov, p)

    return elasticities


@njit(parallel=True)
def _exposure_elasticity_loop(T, n_Y, α, Σ_tilde_t, μ_t0, μ_t1, percentile, x_mean, x_cov, p):
    elasticities = np.zeros((T, n_Y))
    if percentile == 0.5:
        for t in prange(T):
            elasticity = (α@Σ_tilde_t[t]@μ_t0[t])[0] +(α@Σ_tilde_t[t]@μ_t1[t]@x_mean)[0]
            elasticities[t] = elasticity
    else:
        for t in prange(T):
            elasticity = (α@Σ_tilde_t[t]@μ_t0[t])[0] +(α@Σ_tilde_t[t]@μ_t1[t]@x_mean)[0]
            A = α@Σ_tilde_t[t]@μ_t1[t]
            elasticity = _compute_percentile(A, elasticity, x_cov, p)
            elasticities[t] = elasticity
    return elasticities


@njit(parallel=True)
def _price_elasticity_loop(T, n_Y, α, Σ_tilde_expo_t, Σ_tilde_value_t, 
                           μ_expo_t0, μ_value_t0, μ_expo_t1, μ_value_t1,
                           percentile, x_mean, x_cov, p):
    elasticities = np.zeros((T, n_Y))
    if percentile == 0.5:
        for t in prange(T):
            elasticity = (α @ (Σ_tilde_expo_t[t] @ μ_expo_t0[t] \
                               - Σ_tilde_value_t[t] @ μ_value_t0[t]))[0]\
                          +(α@(Σ_tilde_expo_t[t]@μ_expo_t1[t]@x_mean\
                              - Σ_tilde_value_t[t] @ μ_value_t1[t]@x_mean))[0]
            elasticities[t] = elasticity        
    else:
        for t in prange(T):
            elasticity = (α @ (Σ_tilde_expo_t[t] @ μ_expo_t0[t]\
                               - Σ_tilde_value_t[t] @ μ_value_t0[t]))[0]\
                            +(α@(Σ_tilde_expo_t[t]@μ_expo_t1[t]@x_mean\
                              - Σ_tilde_value_t[t] @ μ_value_t1[t]@x_mean))[0]
            A = α @ (Σ_tilde_expo_t[t]@μ_expo_t1[t]\
                     - Σ_tilde_value_t[t]@μ_value_t1[t])
            elasticity = _compute_percentile(A, elasticity, x_cov, p)
            elasticities[t] = elasticity
    return elasticities


@njit
def _compute_percentile(A, Ax_mean, x_cov, p):
    """
    Compute percentile of the scalar Ax, where A is vector coefficient and
    x follows multivariate normal distribution.
    
    Parameters
    ----------
    A : (N, ) ndarray
        Coefficient of Ax.
    Ax_mean : float
        Mean of Ax.
    x_cov : (N, N) ndarray
        Covariance matrix of x.
    p : float
        Percentile of a standard normal distribution.

    Returns
    -------
    res : float
        Percentile of Ax.

    """
    Ax_var = A@x_cov@A.T
    Ax_std = np.sqrt(Ax_var)
    res = Ax_mean + Ax_std * p
    return res
    
    
def _Φ_star(log_M_growth, X1_tp1, X2_tp1, T):
    r"""
    Computes :math:`\Phi^*_{0,t-1}`, :math:`\Phi^*_{1,t-1}`, 
        :math:`\Phi^*_{2,t-1}`, :math:`\Phi^*_{3,t-1}`.

    Parameters
    ----------
    log_G_growth : LinQuadVar
        Log growth of multiplicative functional M.
        e.g. log consumption growth, :math:`\log \frac{C_{t+1}}{C_t}`
    X1_tp1 : LinQuadVar
        Stores the coefficients of laws of motion for X1.
    X2_tp2 : LinQuadVar or None
        Stores the coefficients of laws of motion for X2.
    T : int
        Time horizon.

    Returns
    -------
    Φ_star_1tm1_all : (T, 1, n_X) ndarray
    Φ_star_2tm1_all : (T, 1, n_X) ndarray
    Φ_star_3tm1_all : (T, 1, n_X**2) ndarray

    Reference
    ---------
    Borovicka, Hansen (2014). See http://larspeterhansen.org/.

    """
    _, n_X, _ = X1_tp1.shape
    
    Φ_star_1tm1_all = np.zeros((T, 1, n_X))
    Φ_star_2tm1_all = np.zeros((T, 1, n_X))
    Φ_star_3tm1_all = np.zeros((T, 1, n_X**2))
    log_M_growth_distort = log_E_exp(log_M_growth)
    X1X1 = kron_prod(X1_tp1, X1_tp1)

    for i in range(1, T):
        Φ_star_1tm1_all[i] = log_M_growth_distort['x']
        Φ_star_2tm1_all[i] = log_M_growth_distort['x2']
        Φ_star_3tm1_all[i] = log_M_growth_distort['xx']
        temp = next_period(log_M_growth_distort, X1_tp1, X2_tp1, X1X1)
        log_M_growth_distort = log_E_exp(log_M_growth + temp)

    return Φ_star_1tm1_all, Φ_star_2tm1_all, Φ_star_3tm1_all


def _elasticity_coeff(log_M_growth, X1_tp1, X2_tp1, T):
    r"""
    Computes :math:`\mu_{t,0}`, :math:`\mu_{t,1}`, :math:`\tilde{\Sigma}_t`

    Parameters
    ----------
    log_M_growth : LinQuadVar
        Log difference of multiplicative functional.
        e.g. log consumption growth, :math:`\log \frac{C_{t+1}}{C_t}`
    X1_tp1 : LinQuadVar
        Stores the coefficients of laws of motion for X1.
    X2_tp2 : LinQuadVar or None
        Stores the coefficients of laws of motion for X2.        
    T : int
        Time horizon.

    Returns
    -------
    Σ_tilde_t_all : (T, n_W, n_W) ndarray
    μ_t0_all : (T, n_W, 1) ndarray
    μ_t1_all : (T, n_W, n_X) ndarray

    Reference
    ---------
    Borovicka, Hansen (2014). See http://larspeterhansen.org/.

    """
    _, n_X, n_W = log_M_growth.shape
    
    Φ_star_1tm1_all, Φ_star_2tm1_all, Φ_star_3tm1_all = _Φ_star(log_M_growth, X1_tp1, X2_tp1, T)
    Ψ_0 = log_M_growth['w']
    Ψ_1 = log_M_growth['xw']
    Ψ_2 = log_M_growth['ww']
    Λ_10 = X1_tp1['w']
    if log_M_growth.second_order:
        Λ_20 = X2_tp1['w']
        Λ_21 = X2_tp1['xw']
        Λ_22 = X2_tp1['ww']
    else:
        Λ_20 = np.zeros((n_X,n_W))
        Λ_21 = np.zeros((n_X,n_X*n_W))
        Λ_22 = np.zeros((n_X,n_W**2))
    Θ_10 = X1_tp1['c']
    Θ_11 = X1_tp1['x']
    
    Σ_tilde_t_all, μ_t0_all, μ_t1_all \
        = _elasticity_coeff_inner_loop(Φ_star_1tm1_all, Φ_star_2tm1_all, Φ_star_3tm1_all, 
                                       Ψ_0, Ψ_1, Ψ_2, Λ_10, Λ_20, Λ_21, Λ_22,
                                       Θ_10, Θ_11, n_X, n_W, T)   
    
    return Σ_tilde_t_all, μ_t0_all, μ_t1_all


@njit
def _elasticity_coeff_inner_loop(Φ_star_1tm1_all, Φ_star_2tm1_all, Φ_star_3tm1_all, 
                                 Ψ_0, Ψ_1, Ψ_2, Λ_10, Λ_20, Λ_21, Λ_22,
                                 Θ_10, Θ_11, n_X, n_W, T):
    Σ_tilde_t_all = np.zeros((T, n_W, n_W))
    μ_t0_all = np.zeros((T, n_W, 1))
    μ_t1_all = np.zeros((T, n_W, n_X))    

    kron_Λ_10_Λ_10 = np.kron(Λ_10,Λ_10)
    kron_Θ_10_Λ_10_sum = np.kron(Θ_10,Λ_10) + np.kron(Λ_10,Θ_10)

    temp = np.kron(Λ_10, Θ_11[:, 0:1].copy())
    for j in range(1, n_X):
        temp = np.hstack((temp, np.kron(Λ_10, Θ_11[:, j:j+1].copy())))

    kron_Θ_11_Λ_10_term = np.kron(Θ_11, Λ_10) + temp

    for t in prange(T):
        Φ_star_1tm1 = Φ_star_1tm1_all[t]
        Φ_star_2tm1 = Φ_star_2tm1_all[t]
        Φ_star_3tm1 = Φ_star_3tm1_all[t]

        Σ_tilde_t_inv = np.eye(n_W)\
                        - 2 * sym(mat(Ψ_2 + Φ_star_2tm1@Λ_22
                                      + Φ_star_3tm1@kron_Λ_10_Λ_10,
                                      (n_W, n_W)))

        μ_t0 = (Ψ_0 + Φ_star_1tm1@Λ_10 + Φ_star_2tm1@Λ_20
                + Φ_star_3tm1 @ kron_Θ_10_Λ_10_sum).T

        μ_t1 = mat(Ψ_1 + Φ_star_2tm1 @ Λ_21
                   + Φ_star_3tm1 @ kron_Θ_11_Λ_10_term,(n_W, n_X))
        Σ_tilde_t_all[t] = np.linalg.inv(Σ_tilde_t_inv)
        μ_t0_all[t] = μ_t0
        μ_t1_all[t] = μ_t1
    
    return Σ_tilde_t_all, μ_t0_all, μ_t1_all

def exposure_elasticity_type2(log_M_growth, X1_tp1, X2_tp1, T=400, shock=0, percentile=0.5):
    r"""
    Computes exposure elasticity for M.

    Parameters
    ----------
    log_M_growth : LinQuadVar
        Log growth of multiplicative functional M.
        e.g. log consumption growth, :math:`\log \frac{C_{t+1}}{C_t}`
    X1_tp1 : LinQuadVar
        Stores the coefficients of laws of motion for X1.
    X2_tp1 : LinQuadVar
        Stores the coefficients of laws of motion for X2.        
    T : int
        Time horizon.
    shock : int
        Position of the initial shock, starting from 0.
    percentile : float
        Specifies the percentile of the elasticities.

    Returns
    -------
    elasticities : (T, n_Y) ndarray
        Exposure elasticities.

    """
    n_Y, n_X, n_W = log_M_growth.shape
    if n_Y != 1:
        raise ValueError("The dimension of input should be 1.")

    α = np.zeros(n_W)
    α[shock] = 1    
    p = norm.ppf(percentile)

    Σ_tilde_t, μ_t0, μ_t1 = _elasticity_coeff(log_M_growth, X1_tp1, X2_tp1, T)

    kron_product = np.kron(X1_tp1['x'], X1_tp1['x'])
    x_mean = np.linalg.solve(np.eye(n_X)-X1_tp1['x'],X1_tp1['c'])
    x_cov = mat(np.linalg.solve(np.eye(n_X**2)-kron_product, vec(X1_tp1['w']@X1_tp1['w'].T)), (n_X, n_X))

    ψ_x = X1_tp1['x']
    ψ_w = X1_tp1['w']
    ψ_q = X1_tp1['c']

    elasticities = _exposure_elasticity_loop2(ψ_x, ψ_w, ψ_q, T, n_Y, α, Σ_tilde_t, μ_t0, μ_t1, percentile, x_mean, x_cov, p)

    return elasticities[1:]

@njit
def _exposure_elasticity_loop2(ψ_x, ψ_w, ψ_q, T, n_Y, α, Σ_tilde_t, μ_t0, μ_t1, percentile, x_mean, x_cov, p):

    mu_tilde_0 = np.zeros(μ_t0.shape)
    mu_tilde_1 = np.zeros(μ_t1.shape)
    for t in prange(1, T):
        mu_tilde_0[t] = Σ_tilde_t[t]@μ_t0[t]
        mu_tilde_1[t] = Σ_tilde_t[t]@μ_t1[t]

    mu_hat_0 = np.zeros(μ_t0.shape)
    mu_hat_1 = np.zeros(μ_t1.shape)
    mu_hat_0[1] = mu_tilde_0[1].copy()
    mu_hat_1[1] = mu_tilde_1[1].copy()

    for t in prange(1, T-1):
        mu_hat_0[t+1] = mu_hat_0[t]+mu_hat_1[t]@(ψ_q+ψ_w@mu_tilde_0[t+1])
        mu_hat_1[t+1] = mu_hat_1[t]@ψ_x+mu_hat_1[t]@ψ_w@mu_tilde_1[t+1]

    elasticities = np.zeros((T, n_Y))
    if percentile == 0.5:
        for t in prange(1,T):
            elasticities[t] = (α@mu_hat_0[t])[0] + (α@mu_hat_1[t])[0]
    else:
        for t in prange(1,T):
            elasticity = compute_percentile_T(α@mu_hat_1[t], x_cov, p)
            elasticities[t] = (α@mu_hat_0[t])[0] + elasticity

    return elasticities

@njit
def compute_percentile_T(A, x_cov, p):
    r"""
    Compute percentile of the scalar Ax, where A is vector coefficient and x follows multivariate normal distribution.
    
    Parameters
    ----------
    A : (N, ) ndarray
        Coefficient of Ax.
    Ax_mean : float
        Mean of Ax.
    x_cov : (N, N) ndarray
        Covariance matrix of x.
    p : float
        Percentile of a standard normal distribution.

    Returns
    -------
    res : float
        Percentile of Ax.

    """
    Ax_var = A@x_cov@A.T
    Ax_std = np.sqrt(Ax_var)
    res = Ax_std * p
    return res



def exposure_elasticity_type2_alpha_extend(gc_tp1_growth, X1_tp1, X2_tp1, T, α0, α1, x0=None, percentile=0.5, gpu= False, MCsize = 10**3):
    r"""
    Computes exposure elasticity for M.

    Parameters
    ----------
    gc_tp1_growth : LinQuadVar
        Log growth of multiplicative functional M.
        e.g. log consumption growth, :math:`\log \frac{C_{t+1}}{C_t}`
    X1_tp1 : LinQuadVar
        Stores the coefficients of laws of motion for X1.
    X2_tp1 : LinQuadVar
        Stores the coefficients of laws of motion for X2.        
    T : int
        Time horizon.
    shock : int
        Position of the initial shock, starting from 0.
    percentile : float
        Specifies the percentile of the elasticities.

    Returns
    -------
    elasticities : (T, n_Y) ndarray
        Exposure elasticities.

    """
    n_Y, n_X, n_W = gc_tp1_growth.shape
    if n_Y != 1:
        raise ValueError("The dimension of input should be 1.")

    Σ_tilde_t, μ_t0, μ_t1 = _elasticity_coeff(gc_tp1_growth, X1_tp1, X2_tp1, T)

    kron_product = np.kron(X1_tp1['x'], X1_tp1['x'])
    x_mean = np.linalg.solve(np.eye(n_X)-X1_tp1['x'],X1_tp1['c'])
    x_cov = mat(np.linalg.solve(np.eye(n_X**2)-kron_product, vec(X1_tp1['w']@X1_tp1['w'].T)), (n_X, n_X))
    if x0 == None:
        if gpu:
            X0 = cp.random.multivariate_normal(x_mean.flatten(), x_cov, size=MCsize).T
        else:
            X0 = np.random.multivariate_normal(x_mean.flatten(), x_cov, size=MCsize).T
    else:
        X0 = x0

    ψ_x = X1_tp1['x']
    ψ_w = X1_tp1['w']
    ψ_q = X1_tp1['c']

    elasticities, mu_hat_0, mu_hat_1, mu_hat_2 = _exposure_elasticity_loop2_alpha_extend(ψ_x, ψ_w, ψ_q, T, n_Y, α0, α1, Σ_tilde_t, μ_t0, μ_t1, percentile, X0, gpu)

    return elasticities[1:]

def _exposure_elasticity_loop2_alpha_extend(ψ_x, ψ_w, ψ_q, T, n_Y, α0, α1, Σ_tilde_t, μ_t0, μ_t1, percentile, X0, gpu):

    mu_tilde_0 = np.zeros(μ_t0.shape)
    mu_tilde_1 = np.zeros(μ_t1.shape)
    for t in prange(1, T):
        mu_tilde_0[t] = Σ_tilde_t[t]@μ_t0[t]
        mu_tilde_1[t] = Σ_tilde_t[t]@μ_t1[t]

    mu_hat_0 = np.zeros([T,n_Y,n_Y])
    mu_hat_1 = np.zeros([T,n_Y,μ_t1.shape[2]])
    mu_hat_2 = np.zeros([T,μ_t1.shape[2],μ_t1.shape[2]])
    mu_hat_0[1] = α0.T@mu_tilde_0[1]
    mu_hat_1[1] = α0.T@mu_tilde_1[1]+mu_tilde_0[1].T@α1
    mu_hat_2[1] = 0.5*(α1.T@mu_tilde_1[1]+mu_tilde_1[1].T@α1)

    for t in prange(1, T-1):
        mu_hat_0[t+1] = mu_hat_0[t]+mu_hat_1[t]@(ψ_q+ψ_w@mu_tilde_0[t+1]) + \
                        (ψ_q+ψ_w@mu_tilde_0[t+1]).T@mu_hat_2[t]@(ψ_q+ψ_w@mu_tilde_0[t+1]) + \
                        np.trace(ψ_w.T@mu_hat_2[t]@ψ_w@Σ_tilde_t[t+1])
        mu_hat_1[t+1] = mu_hat_1[t]@ψ_x+mu_hat_1[t]@ψ_w@mu_tilde_1[t+1] + \
                        2*(ψ_q+ψ_w@mu_tilde_0[t+1]).T@mu_hat_2[t]@(ψ_x+ψ_w@mu_tilde_1[t+1])
        mu_hat_2[t+1] = (ψ_x+ψ_w@mu_tilde_1[t+1]).T@mu_hat_2[t]@(ψ_x+ψ_w@mu_tilde_1[t+1])

    elasticities = np.zeros((T, n_Y))
    for t in prange(1,T):
        if gpu:
            cp0 = cp.array(mu_hat_0[t])
            cp1 = cp.array(mu_hat_1[t])
            cp2 = cp.array(mu_hat_2[t])
            elasticity = cp0 + cp.matmul(cp1,X0) + cp.sum(np.multiply(cp2.T@X0,X0),axis=0)
            #+ cp.diagonal(cp.matmul(cp.matmul(X0.T,cp2),X0))
            elasticities[t] = cp.quantile(elasticity, percentile).get()
        else:
            # prodtemp = np.zeros([1,X0.shape[1]])
            # for i in range(X0.shape[1]):               
            #     prodtemp[:,i] = (X0[:,[i]].T@mu_hat_2[t]@X0[:,[i]])[0,0]
            # print(mu_hat_2[t].T.shape)
            # print(np.multiply(mu_hat_2[t].T@X0,X0).shape)
            # print(X0.shape)
            elasticity = mu_hat_0[t] + mu_hat_1[t]@X0 + np.sum(np.multiply(mu_hat_2[t].T@X0,X0),axis=0)
            #+ np.diagonal(X0.T@mu_hat_2[t]@X0)
                
            elasticities[t] = np.quantile(elasticity, percentile)
    return elasticities, mu_hat_0, mu_hat_1, mu_hat_2

def price_elasticity_type2_alpha_extend(log_G_growth, log_S_growth, X1_tp1, X2_tp1, T, α0, α1, x0=None, percentile=0.5, gpu= False, MCsize = 10**3):
    r"""
    Computes price elasticity.

    Parameters
    ----------
    log_G_growth : LinQuadVar
        Log growth of multiplicative functional G.
        e.g. log consumption growth, :math:`\log \frac{C_{t+1}}{C_t}`
    log_S_growth : LinQuadVar
        Log growth of multiplicative functional S.
        e.g. log stochastic discount factor, :math:`\log \frac{S_{t+1}}{S_t}`
    X1_tp1 : LinQuadVar
        Stores the coefficients of laws of motion for X1.
    X2_tp2 : LinQuadVar or None
        Stores the coefficients of laws of motion for X2.        
    T : int
        Time horizon.
    shock : int
        Position of the initial shock, starting from 0.
    percentile : float
        Specifies the percentile of the elasticities.

    Returns
    -------
    elasticities : (T, dim) ndarray
        Price elasticities.

    Reference
    ---------
    Borovicka, Hansen (2014). See http://larspeterhansen.org/.

    """
    if log_G_growth.shape != log_S_growth.shape:
        raise ValueError("The dimensions of G and S do not match.")
    else:
        n_Y, n_X, n_W = log_G_growth.shape
        if n_Y != 1:
            raise ValueError("The dimension of inputs should be (1, n_X, n_W)")
    kron_product = np.kron(X1_tp1['x'], X1_tp1['x'])
    x_mean = np.linalg.solve(np.eye(n_X)-X1_tp1['x'],X1_tp1['c'])
    x_cov = mat(np.linalg.solve(np.eye(n_X**2)-kron_product, vec(X1_tp1['w']@X1_tp1['w'].T)), (n_X, n_X))
    if x0 == None:
        if gpu:
            X0 = cp.random.multivariate_normal(x_mean.flatten(), x_cov, size=MCsize).T
        else:
            X0 = np.random.multivariate_normal(x_mean.flatten(), x_cov, size=MCsize).T
    else:
        X0 = x0

    ψ_x = X1_tp1['x']
    ψ_w = X1_tp1['w']
    ψ_q = X1_tp1['c']

    Σ_tilde_expo_t, μ_expo_t0, μ_expo_t1 \
        = _elasticity_coeff(log_G_growth, X1_tp1, X2_tp1, T)
    Σ_tilde_value_t, μ_value_t0, μ_value_t1\
        = _elasticity_coeff(log_G_growth+log_S_growth, X1_tp1, X2_tp1, T)
    
    elasticities, mu_hat_expo_0, mu_hat_expo_1, mu_hat_expo_2, mu_hat_value_0, mu_hat_value_1, mu_hat_value_2 = _price_elasticity_loop2_alpha_extend(ψ_x, ψ_w, ψ_q, T, n_Y, α0, α1, Σ_tilde_expo_t, Σ_tilde_value_t, 
                           μ_expo_t0, μ_value_t0, μ_expo_t1, μ_value_t1, percentile, X0, gpu)

    return elasticities

def _price_elasticity_loop2_alpha_extend(ψ_x, ψ_w, ψ_q, T, n_Y, α0, α1, Σ_tilde_expo_t, Σ_tilde_value_t, μ_expo_t0, μ_value_t0, μ_expo_t1, μ_value_t1, percentile, X0, gpu):

    mu_tilde_expo_0 = np.zeros(μ_expo_t0.shape)
    mu_tilde_expo_1 = np.zeros(μ_expo_t1.shape)
    mu_tilde_value_0 = np.zeros(μ_value_t0.shape)
    mu_tilde_value_1 = np.zeros(μ_value_t1.shape)
    for t in prange(1, T):
        mu_tilde_expo_0[t] = Σ_tilde_expo_t[t]@μ_expo_t0[t]
        mu_tilde_expo_1[t] = Σ_tilde_expo_t[t]@μ_expo_t1[t]
        mu_tilde_value_0[t] = Σ_tilde_value_t[t]@μ_value_t0[t]
        mu_tilde_value_1[t] = Σ_tilde_value_t[t]@μ_value_t1[t]

    mu_hat_expo_0 = np.zeros([T,n_Y,n_Y])
    mu_hat_expo_1 = np.zeros([T,n_Y,μ_expo_t1.shape[2]])
    mu_hat_expo_2 = np.zeros([T,μ_expo_t1.shape[2],μ_expo_t1.shape[2]])
    mu_hat_value_0 = np.zeros([T,n_Y,n_Y])
    mu_hat_value_1 = np.zeros([T,n_Y,μ_expo_t1.shape[2]])
    mu_hat_value_2 = np.zeros([T,μ_value_t1.shape[2],μ_value_t1.shape[2]])

    mu_hat_expo_0[1] = α0.T@mu_tilde_expo_0[1]
    mu_hat_expo_1[1] = α0.T@mu_tilde_expo_1[1]+mu_tilde_expo_0[1].T@α1
    mu_hat_expo_2[1] = 0.5*(α1.T@mu_tilde_expo_1[1]+mu_tilde_expo_1[1].T@α1)
    mu_hat_value_0[1] = α0.T@mu_tilde_value_0[1]
    mu_hat_value_1[1] = α0.T@mu_tilde_value_1[1]+mu_tilde_value_0[1].T@α1
    mu_hat_value_2[1] = 0.5*(α1.T@mu_tilde_value_1[1]+mu_tilde_value_1[1].T@α1)

    for t in prange(1, T-1):
        mu_hat_expo_0[t+1] = mu_hat_expo_0[t]+mu_hat_expo_1[t]@(ψ_q+ψ_w@mu_tilde_expo_0[t+1]) + \
                            (ψ_q+ψ_w@mu_tilde_expo_0[t+1]).T@mu_hat_expo_2[t]@(ψ_q+ψ_w@mu_tilde_expo_0[t+1]) + \
                            np.trace(ψ_w.T@mu_hat_expo_2[t]@ψ_w@Σ_tilde_expo_t[t+1])
        mu_hat_expo_1[t+1] = mu_hat_expo_1[t]@ψ_x+mu_hat_expo_1[t]@ψ_w@mu_tilde_expo_1[t+1] + \
                        2*(ψ_q+ψ_w@mu_tilde_expo_0[t+1]).T@mu_hat_expo_2[t]@(ψ_x+ψ_w@mu_tilde_expo_1[t+1])
        mu_hat_expo_2[t+1] = (ψ_x+ψ_w@mu_tilde_expo_1[t+1]).T@mu_hat_expo_2[t]@(ψ_x+ψ_w@mu_tilde_expo_1[t+1])

        mu_hat_value_0[t+1] = mu_hat_value_0[t]+mu_hat_value_1[t]@(ψ_q+ψ_w@mu_tilde_value_0[t+1]) + \
                            (ψ_q+ψ_w@mu_tilde_value_0[t+1]).T@mu_hat_value_2[t]@(ψ_q+ψ_w@mu_tilde_value_0[t+1]) + \
                            np.trace(ψ_w.T@mu_hat_value_2[t]@ψ_w@Σ_tilde_value_t[t+1])
        mu_hat_value_1[t+1] = mu_hat_value_1[t]@ψ_x+mu_hat_value_1[t]@ψ_w@mu_tilde_value_1[t+1] + \
                        2*(ψ_q+ψ_w@mu_tilde_value_0[t+1]).T@mu_hat_value_2[t]@(ψ_x+ψ_w@mu_tilde_value_1[t+1])
        mu_hat_value_2[t+1] = (ψ_x+ψ_w@mu_tilde_value_1[t+1]).T@mu_hat_value_2[t]@(ψ_x+ψ_w@mu_tilde_value_1[t+1])

    elasticities = np.zeros((T, n_Y))
    for t in prange(1,T):
        if gpu:
            cp_expo_0 = cp.array(mu_hat_expo_0[t])
            cp_expo_1 = cp.array(mu_hat_expo_1[t])
            cp_expo_2 = cp.array(mu_hat_expo_2[t])

            cp_value_0 = cp.array(mu_hat_value_0[t])
            cp_value_1 = cp.array(mu_hat_value_1[t])
            cp_value_2 = cp.array(mu_hat_value_2[t])

            elasticity = cp0 + cp.matmul(cp1,X0) + cp.sum(np.multiply(cp2.T@X0,X0),axis=0)

            elasticity = cp_expo_0 + cp.matmul(cp_expo_1,X0) + cp.sum(np.multiply(cp_expo_2.T@X0,X0),axis=0) -\
                        cp_value_0 - cp.matmul(cp_value_1,X0) - cp.sum(np.multiply(cp_value_2.T@X0,X0),axis=0) 
            elasticities[t] = cp.quantile(elasticity, percentile).get()
        else:
            elasticity = mu_hat_expo_0[t] + mu_hat_expo_1[t]@X0 + np.sum(np.multiply(mu_hat_expo_2[t].T@X0,X0),axis=0) -\
                        mu_hat_value_0[t] - mu_hat_expo_1[t]@X0 - np.sum(np.multiply(mu_hat_value_2[t].T@X0,X0),axis=0)
            elasticities[t] = np.quantile(elasticity, percentile)
            
    return elasticities[1:], mu_hat_expo_0, mu_hat_expo_1, mu_hat_expo_2, mu_hat_value_0, mu_hat_value_1, mu_hat_value_2
