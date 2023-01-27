"""
Tools for operations on LinQuadVar.

"""
import numpy as np
from scipy.stats import norm
from scipy import optimize
from utilities import vec, mat, sym, cal_E_ww
from lin_quad import LinQuadVar
from numba import njit
from numba import prange
import scipy as sp
import seaborn as sns
import time

def previous_period(Y, X1_tp1, X2_tp1=None):
    """
    Gets representation for Y_{t+1} when Y only contains time t+1 state variables.

    Parameters
    ----------
    Y : LinQuadVar
        Stores the coefficient of Y.
    X1_tp1 : LinQuadVar
        Stores the coefficients of laws of motion for X1.
    X2_tp2 : LinQuadVar or None
        Stores the coefficients of laws of motion for X2. Does not need
        to be specified when Y['x2'] is zero ndarray.

    Returns
    -------
    Y_next : LinQuadVar

    """
    if not Y.deterministic:
        raise ValueError("Y should only contain time t+1 state variables.")
    n_Y, n_X, n_W = Y.shape
    
    ψ_x = X1_tp1['x']
    ψ_w = X1_tp1['w']
    ψ_q = X1_tp1['c']
    
    if X2_tp1 is not None:
        ψ_xx = X2_tp1['xx']
        ψ_xw = X2_tp1['xw'] * 0.5
        ψ_xq = X2_tp1['x'] * 0.5
        ψ_ww = X2_tp1['ww']
        ψ_wq = X2_tp1['w'] * 0.5
        ψ_qq = X2_tp1['c']
    else:
        ψ_xx = np.zeros((n_X, n_X**2))
        ψ_xw = np.zeros((n_X, n_X*n_W))
        ψ_xq = np.zeros((n_X, n_X))
        ψ_ww = np.zeros((n_X, n_W**2))
        ψ_wq = np.zeros((n_X, n_W))
        ψ_qq = np.zeros((n_X, 1))
    
    Y_previous = LinQuadVar({'x2': Y['x2']@ψ_x}, (n_Y, n_X, n_W))
    Y_previous += LinQuadVar({'xx': Y['x2']@ψ_xx + Y['xx']@np.kron(ψ_x,ψ_x)}, (n_Y, n_X, n_W))
    Y_previous += LinQuadVar({'xw': 2*Y['x2']@ψ_xw + Y['xx']@np.kron(ψ_x,ψ_w) + \
                vec(mat(Y['xx']@np.kron(ψ_w,ψ_x),(n_X,n_W)).T).T}, (n_Y, n_X, n_W))
    Y_previous += LinQuadVar({'ww': Y['x2']@ψ_ww + Y['xx']@np.kron(ψ_w,ψ_w)}, (n_Y, n_X, n_W))
    Y_previous += LinQuadVar({'x':  2*Y['x2']@ψ_xq + Y['xx']@(np.kron(ψ_x,ψ_q)+np.kron(ψ_q,ψ_x))\
                + Y['x']@ψ_x}, (n_Y, n_X, n_W))
    Y_previous += LinQuadVar({'w': 2*Y['x2']@ψ_wq + Y['xx']@(np.kron(ψ_w,ψ_q)+np.kron(ψ_q,ψ_w))\
                + Y['x']@ψ_w}, (n_Y, n_X, n_W))
    Y_previous += LinQuadVar({'c': Y['x2']@ψ_qq + Y['xx']@np.kron(ψ_q,ψ_q) + Y['x']@ψ_q + Y['c']}, (n_Y, n_X, n_W))

    return Y_previous

def next_period(Y, X1_tp1, X2_tp1=None, X1X1=None):
    """
    Gets representation for Y_{t+1} when Y only contains time t variables.

    Parameters
    ----------
    Y : LinQuadVar
        Stores the coefficient of Y.
    X1_tp1 : LinQuadVar
        Stores the coefficients of laws of motion for X1.
    X2_tp2 : LinQuadVar or None
        Stores the coefficients of laws of motion for X2. Does not need
        to be specified when Y['x2'] is zero ndarray.
    X1X1 : LinQuadVar or None
        Stores the coefficients of :math:`X_{1,t+1}\otimes X_{1,t+1}`.
        If None, the function will recalculate it.

    Returns
    -------
    Y_next : LinQuadVar

    """
    if not Y.deterministic:
        raise ValueError("Y should only contain time t variables.")
    n_Y, n_X, n_W = Y.shape
    Y_next = LinQuadVar({'c': Y['c']}, (n_Y, n_X, n_W))\
            + matmul(Y['x'], X1_tp1)
    # if np.any(Y['xx'] != 0)
    if Y['xx'].any():
        if X1X1 is None:
            X1X1 = kron_prod(X1_tp1, X1_tp1)
        Y_next = Y_next + matmul(Y['xx'], X1X1)
    if Y['x2'].any():
        Y_next = Y_next + matmul(Y['x2'], X2_tp1)

    return Y_next


def kron_prod(Y1, Y2):
    """
    Computes the Kronecker product of Y1 and Y2, where Y1 and Y2
    do not have second-order terms.
    
    Parameters
    ----------
    Y1 : LinQuadVar
        Y1.second_order should be False.
    Y2 : LinQuadVar
        Y2.second_order should be False.

    Returns
    -------
    Y_kron : LinQuadVar
        Stores coefficients for the Kronecker product of Y1 and Y2.

    """
    if Y1.second_order or Y2.second_order:
        raise ValueError('Y1.second_order and Y2.second_order should be False.')
    n_Y1, n_X, n_W = Y1.shape
    n_Y2, _, _ = Y2.shape
    kron_prod = {}
    terms = ['x', 'w', 'c']
    for key_left in terms:
        for key_right in terms:
            # if np.any(Y1[key_left] != 0) and np.any(Y2[key_right] != 0)
            if Y1[key_left].any() and Y2[key_right].any():
                kron_prod[key_left+key_right] = np.kron(Y1[key_left], Y2[key_right])
            else:
                _, m1 = Y1[key_left].shape
                _, m2 = Y2[key_right].shape
                kron_prod[key_left+key_right] = np.zeros((n_Y1*n_Y2, m1*m2))
    # Combine terms
    xx = kron_prod['xx']
    wx = kron_prod['wx']
    wx_reshape = np.vstack([vec(mat(wx[row:row+1, :].T, (n_X, n_W)).T).T for row in range(wx.shape[0])])
    xw = kron_prod['xw'] + wx_reshape
    ww = kron_prod['ww']
    x = kron_prod['xc'] + kron_prod['cx']
    w = kron_prod['wc'] + kron_prod['cw']
    c = kron_prod['cc']
    
    Y_kron = LinQuadVar({'xx': xx, 'xw': xw, 'ww': ww, 'x': x, 'w': w, 'c': c},
                        (n_Y1*n_Y2, n_X, n_W))
    
    return Y_kron

def lq_sum(lq_list):
    """
    Compute the sum of a list of LinQuadVar.
    
    Parameters
    ----------
    lq_list: a list of LinQuadVar

    Returns
    ----------
    lq_sum : LinQuadVar
        sum of a list of LinQuadVar.

    """
    lq_sum = LinQuadVar({'c':np.zeros([lq_list[0].shape[0],1])},lq_list[0].shape)
    for i in range(len(lq_list)):
        lq_sum += lq_list[i]

    return lq_sum

def matmul(matrix, Y):
    """
    Computes matrix@Y[key] for each key in Y.
    
    Parameters
    ----------
    matrix : (n1, n2) ndarray
    Y : (n2, n_X, n_W) LinQuadVar
    
    Returns
    Y_new : (n1, n_X, n_W) LinQuadVar

    """
    Y_new_coeffs = {}
    n_Y, n_X, n_W = Y.shape
    for key in Y.coeffs:
        Y_new_coeffs[key] = matrix @ Y.coeffs[key]
    Y_new = LinQuadVar(Y_new_coeffs, (matrix.shape[0], n_X, n_W), False)
    return Y_new


def concat(Y_list):
    """
    Concatenates a list of LinQuadVar.

    Parameters
    ----------
    Y_list : list of (n_Yi, n_X, n_W) LinQuadVar

    Returns
    -------
    Y_cat : (n_Y1+n_Y2..., n_X, n_W) LinQuadVar
    
    See Also
    --------
    LinQuadVar.split : Splits the N-dimensional Y into N 1-dimensional Ys.

    """
    terms = []
    for Y in Y_list:
        terms = set(terms) | set(Y.coeffs.keys())
    Y_cat = {}
    for key in terms:
        Y_coeff_list = [Y[key] for Y in Y_list]
        Y_cat[key] = np.concatenate(Y_coeff_list, axis=0)
    temp = list(Y_cat.keys())[0]
    n_Y_cat = Y_cat[temp].shape[0]
    n_X = Y_list[0].shape[1]
    n_W = Y_list[0].shape[2]
    Y_cat = LinQuadVar(Y_cat, (n_Y_cat, n_X, n_W), False)

    return Y_cat
    
def E(Y, E_w, E_ww=None):
    r"""
    Computes :math:`E[Y_{t+1} \mid \mathfrak{F}_t]`,
    The expecation calculated in this function does not have the state dependent terms 

    Parameters
    ----------
    Y : LinQuadVar
        The LinQuadVar to be taken expectation
    E_w : (n_W, 1) ndarray
        Expectation of the shock vector.
    E_ww : (n_W, n_W) ndarray
        Expectation of the kronecker product of shock vectors.
        Used when the Y has non-zero coefficient on 'ww' term.

    Returns
    -------
    E_Y : LinQuadVar
        Expectation of Y
    """
    n_Y, n_X, n_W = Y.shape
    if Y.deterministic:
        return LinQuadVar(Y.coeffs, Y.shape)
    else:
        E_Y = {}
        E_Y['x2'] = Y['x2']
        E_Y['xx'] = Y['xx']
        temp = np.vstack([E_w.T@mat(Y['xw'][row: row+1, :], (n_W, n_X))
                          for row in range(n_Y)])
        E_Y['x'] = temp + Y['x']
        E_Y['c'] = Y['c'] + Y['w'] @ E_w
        if Y['ww'].any():
            # E_ww = cal_E_ww(E_w, Cov_w)
            E_Y['c'] += Y['ww'] @ E_ww
        E_Y = LinQuadVar(E_Y, Y.shape, False)
        return E_Y
        
def N_tilde_measure(log_N, var_shape):
    """
    Computes the distored distribution of shocks implied by the change of measure N_tilde

    Parameters
    ----------
    log_N : LinQuadVar
        Log N tilde
    var_shape : tuple of ints
        (1, n_X, n_W)

    Returns
    ----------
    change_of_measure : dict
        A dictionary containing the distored distribution of shocks implied by the change of measure N_tilde
        Λ : ndarray
            transformed coefficients on ww term for log N tilde
        H_0 : ndarray
            coefficients on w term for log N tilde
        H_1 : ndarray
            transformed coefficients on xw term for log N tilde
        Λ_tilde_inv : ndarray
            distorted covariance matrix
        Γ : ndarray
            matrix square root of the distorted covariance matrix
        H_tilde_0 : ndarray
            distorted mean coefficients on constant terms
        H_tilde_1 : ndarray
            distorted mean coefficients on x terms
        H_tilde_1_aug : ndarray
            distorted mean coefficients on x terms augmented by zero matrices
    """
    n_Y, n_X, n_W = var_shape

    Ψ_0 = log_N['w']
    Ψ_1 = log_N['xw']
    Ψ_2 = log_N['ww']
    
    Λ = -sym(mat(2*Ψ_2,(n_W,n_W)))
    H_0 = Ψ_0.T
    H_1 = mat(Ψ_1, (n_W, n_X))
    Λ_tilde = np.eye(n_W) + Λ
    Λ_tilde_inv = np.linalg.inv(Λ_tilde) 
    H_tilde_0 = Λ_tilde_inv@H_0
    H_tilde_1 = Λ_tilde_inv@H_1
    Γ = sp.linalg.sqrtm(Λ_tilde_inv)
    H_tilde_1_aug = np.block([[np.zeros([n_W,n_Y]),H_tilde_1]])
    change_of_measure = {
                        'Λ':Λ, 
                        'H_0':H_0, 
                        'H_1':H_1, 
                        'Λ_tilde':Λ_tilde, 
                        'Λ_tilde_inv':Λ_tilde_inv,
                        'H_tilde_0':H_tilde_0,
                        'H_tilde_1':H_tilde_1,
                        'Γ':Γ,
                        'H_tilde_1_aug':H_tilde_1_aug}
    
    return change_of_measure

def E_N_tp1(Y, change_of_measure):
    """
    Computes the expectation implied by log N tilde.
    The expecation calculated in this function has the state dependent terms 

    Parameters
    ----------
    Y : LinQuadVar
        The LinQuadVar to be taken expectation
    change_of_measure : dict
        A dictionary containing the distored distribution of shocks implied by the change of measure N_tilde
        returned by the funciton N_tilde_measure
    
    Returns
    ----------
    E_Y : LinQuadVar
        Expectation of Y
    """
    n_Y, n_X, n_W = Y.shape
    E_Y = {}
    E_Y['x2'] = Y['x2']
    E_Y['xx'] = Y['xx'] + Y['ww'] @ np.kron(change_of_measure['H_tilde_1'],change_of_measure['H_tilde_1']) \
                + (Y['xw'].reshape([n_X,n_W])@change_of_measure['H_tilde_1']).T.reshape([n_Y,n_X**2])
    E_Y['x'] = Y['x'] + Y['w'] @ change_of_measure['H_tilde_1'] \
                + Y['ww'] @ (np.kron(change_of_measure['H_tilde_0'],change_of_measure['H_tilde_1']) + np.kron(change_of_measure['H_tilde_1'],change_of_measure['H_tilde_0']))\
                + (Y['xw'].reshape([n_X,n_W])@change_of_measure['H_tilde_0']).T.reshape([n_Y,n_X])
    E_Y['c'] = Y['c'] + Y['w'] @ change_of_measure['H_tilde_0'] + Y['ww'] @ (np.kron(change_of_measure['H_tilde_0'],change_of_measure['H_tilde_0'])+vec(change_of_measure['Λ_tilde_inv']))
    E_Y = LinQuadVar(E_Y, Y.shape, False)
    return E_Y

def kron_comm(AB, nX, nW):
    if not np.any(AB):
        return AB
    kcAB = np.zeros(AB.shape)
    for i in prange(AB.shape[0]):
        kcAB[i] = vec(mat(AB[i:i+1, :].T, (nX, nW)).T).T
    return kcAB


def log_E_exp(Y):
    r"""
    Computes :math:`\log E[\exp(Y_{t+1}) \mid \mathfrak{F}_t]`,
    assuming shocks follow iid normal distribution.

    Parameters
    ----------
    Y : LinQuadVar

    Returns
    -------
    Y_sol : LinQuadVar

    References
    ---------
    Borovicka, Hansen (2014). See http://larspeterhansen.org/.

    """
    n_Y, n_X, n_W = Y.shape
    if n_Y != 1:
        raise ValueError('Y should be scalar-valued.')
    if Y.deterministic:
        return LinQuadVar(Y.coeffs, Y.shape)
    else:
        x2, xx, x, c = _log_E_exp_jit(Y['x2'], Y['x'], Y['w'],
                                      Y['c'], Y['xx'], Y['xw'],
                                      Y['ww'], n_X, n_W)
        Y_sol = LinQuadVar({'x2': x2, 'xx': xx, 'x': x, 'c':c}, Y.shape, False)
        return Y_sol


def simulate(Y, X1_tp1, X2_tp1, Ws):
    """
    Simulate a time path for `Y` given shocks `Ws`.

    Parameters
    ----------
    Y : LinQuadVar
        Variable to be simulated.
    X1_tp1 : LinQuadVar
        Stores the coefficients of laws of motion for X1.
    X2_tp2 : LinQuadVar or None
        Stores the coefficients of laws of motion for X2. Does not need to be
        specified when Y only has first-order terms.
    Ws : (T, n_W) ndarray
        n_W dimensional shocks for T periods to be fed into the system.

    Returns
    -------
    sim_result : (T, n_Y) ndarray
        Simulated Ys.

    """
    n_Y, n_X, n_W = Y.shape
    T = Ws.shape[0]
    Ws = Ws.reshape(T, n_W, 1)
    x1 = np.zeros((T, n_X, 1))
    x2 = np.zeros((T, n_X, 1))

    for i in range(1, T):
        x1[i] = X1_tp1(x1[i-1], np.zeros((n_X, 1)), Ws[i])

    if Y.second_order:
        for i in range(1, T):
            x2[i] = X2_tp1(x1[i-1], x2[i-1], Ws[i])
    sim_result = np.vstack([Y(x1[i], x2[i], Ws[i]).ravel() for i in range(T)])

    return sim_result


@njit
def _log_E_exp_jit(x2, x, w, c, xx, xw, ww, n_X, n_W):
    Σ = np.eye(n_W) - sym(mat(2 * ww, (n_W, n_W)))
    Σ_xw_solved = np.linalg.solve(Σ, mat(xw, (n_W, n_X)))
    new_x2 = x2
    new_xx = xx + 0.5 * vec(mat(xw, (n_W, n_X)).T
                                      @ Σ_xw_solved).T
    new_x = x + w @ Σ_xw_solved
    new_c = c - 1. / 2 * np.log(np.linalg.det(Σ))\
        + 1. / 2 * w @ np.linalg.solve(Σ, w.T)

    return new_x2, new_xx, new_x, new_c

def distance(Y1, Y2, keys_to_compare = None):
    dist = 0.
    if keys_to_compare is None:
        keys_to_compare = set(Y1.coeffs.keys() | Y2.coeffs.keys())
    for key in keys_to_compare:
        temp = np.max(np.abs(Y1[key] - Y2[key]))
        if temp > dist:
            dist = temp
    return dist

def M_mapping(M, f, X1_tp1, X2_tp1, second_order = True):
    r'''
    Computes coefficients of a LinQuadVar after one iteration of M mapping

    Parameters
    ----------
    log_M_growth : LinQuadVar
        Log difference of multiplicative functional.
        e.g. log consumption growth, :math:`\log \frac{C_{t+1}}{C_t}`
    f : LinQuadVar
        The function M Mapping operate on. 
        e.g. A function that is identically one, log_f = LinQuadVar({'c': np.zeros((1,1))}, log_M_growth.shape)
    X1_tp1 : LinQuadVar 
        Stores the coefficients of laws of motion for X1.
    X2_tp2 : LinQuadVar or None
        Stores the coefficients of laws of motion for X2.  
    second_order: boolean
        Whether the second order expansion of the state evoluton equation has been input
        
    Returns
    -------
    LinQuadVar, stores the coefficients of the new LinQuadVar after one iteration of M Mapping
    '''
    if second_order:
        return log_E_exp(M + next_period(f, X1_tp1, X2_tp1))
    else:
        if X2_tp1 != None:
            print('The second order expansion for law of motion is not used in the first order expansion.')
        return log_E_exp(M + next_period(f, X1_tp1))
    

def Q_mapping(M, f, X1_tp1, X2_tp1, tol = 1e-10, max_iter = 20000, second_order = True):
    r'''
    Computes limiting coefficients of a LinQuadVar by recursively applying the M mapping operator till convergence, returns the eigenvalue and eigenvector.

    Parameters
    ----------
    log_M_growth : LinQuadVar
        Log difference of multiplicative functional.
        e.g. log consumption growth, :math:`\log \frac{C_{t+1}}{C_t}`
    f : LinQuadVar
        The function M Mapping operate on. 
        e.g. A function that is identically one, log_f = LinQuadVar({'c': np.zeros((1,1))}, log_M_growth.shape)
    X1_tp1 : LinQuadVar 
        Stores the coefficients of laws of motion for X1.
    X2_tp2 : LinQuadVar or None
        Stores the coefficients of laws of motion for X2.  
    tol: float
        tolerance for convergence
    max_iter: int
        maximum iteration
    second_order: boolean
        Whether the second order expansion of the state evoluton equation has been input

    Returns
    -------
    Qf_components_log : List of LinQuadVar
        stores the coefficients of the LinQuadVar in each iteration of M Mapping
    f: LinQuadVar
        The function M Mapping operate on. 
        e.g. A function that is identically one, log_f = LinQuadVar({'c': np.zeros((1,1))}, log_M_growth.shape)
    η: float
        The eigenvalue
    η_series: list of float
        The convergence path of the eigenvalue 
    '''
    η_series = []
    Qf_components_log = []
#    Qf_evaluate = 0.
    for i in range(max_iter):
        Qf_components_log.append(f)
#        Qf_evaluate += np.exp(f(*x))
        if second_order:
            f_next = M_mapping(M, f, X1_tp1, X2_tp1, second_order = second_order)
        else:
            if X2_tp1 != None:
                print('The second order expansion for law of motion is not used in the first order expansion.')
            f_next = M_mapping(M, f, X1_tp1, second_order = second_order)
        η = (f_next['c'] - f['c']).item()
        η_series.append(η)
        
        if distance(f, f_next, ['x', 'xx', 'x2']) < tol:
            break
        f = f_next
    
    print('Convergence periods:',i)
    if i >= max_iter-1:
        print("Warning: Q iteration may not have converged.")
    Qf_components_log.append(f_next)
#    Qf_evaluate += np.exp(f_next(*x))/(1-np.exp(η))
        
    return Qf_components_log, f, η, η_series

def Q_mapping_no_cons(M, f, X1_tp1, X2_tp1, tol = 1e-10, max_iter = 20000, second_order = True):
    r'''
    Computes limiting coefficients of a LinQuadVar by recursively applying the M mapping operator till convergence, returns the eigenvalue and eigenvector.

    Parameters
    ----------
    log_M_growth : LinQuadVar
        Log difference of multiplicative functional.
        e.g. log consumption growth, :math:`\log \frac{C_{t+1}}{C_t}`
    f : LinQuadVar
        The function M Mapping operate on. 
        e.g. A function that is identically one, log_f = LinQuadVar({'c': np.zeros((1,1))}, log_M_growth.shape)
    X1_tp1 : LinQuadVar 
        Stores the coefficients of laws of motion for X1.
    X2_tp2 : LinQuadVar or None
        Stores the coefficients of laws of motion for X2.  
    tol: float
        tolerance for convergence
    max_iter: int
        maximum iteration
    second_order: boolean
        Whether the second order expansion of the state evoluton equation has been input

    Returns
    -------
    Qf_components_log : List of LinQuadVar
        stores the coefficients of the LinQuadVar in each iteration of M Mapping
    f: LinQuadVar
        The function M Mapping operate on. 
        e.g. A function that is identically one, log_f = LinQuadVar({'c': np.zeros((1,1))}, log_M_growth.shape)
    η: float
        The eigenvalue
    η_series: list of float
        The convergence path of the eigenvalue 
    '''
    η_series = []
    Qf_components_log = []
#    Qf_evaluate = 0.
    for i in range(max_iter):
        Qf_components_log.append(f)
#        Qf_evaluate += np.exp(f(*x))
        if second_order:
            f_next = M_mapping(M, f, X1_tp1, X2_tp1, second_order = second_order)
        else:
            if X2_tp1 != None:
                print('The second order expansion for law of motion is not used in the first order expansion.')
            f_next = M_mapping(M, f, X1_tp1, second_order = second_order)
        η = (f_next['c'] - f['c']).item()
        η_series.append(η)
        
        if distance(f, f_next, ['x', 'xx', 'x2']) < tol:
            break
        f_next = LinQuadVar({'x':f_next['x'],\
                            'w':f_next['w'],\
                            'xx':f_next['xx'],\
                            'xw':f_next['xw'],\
                            'ww':f_next['ww'],\
                            'x2':f_next['x2']}, shape = f_next.shape)
        f = f_next
    
    print('Convergence periods:',i)
    if i >= max_iter-1:
        print("Warning: Q iteration may not have converged.")
    Qf_components_log.append(f_next)
#    Qf_evaluate += np.exp(f_next(*x))/(1-np.exp(η))
        
    return Qf_components_log, f, η, η_series

def Q_mapping_eval(Qf_components_log, η, x):
    """
    Evaluate all the Qf_components_log given x recurisvely
    """
    Qf_evaluate = 0.

    for i in range(len(Qf_components_log)-1):
        Qf_evaluate += np.exp(Qf_components_log[i](*x))

    Qf_evaluate += np.exp(Qf_components_log[-1](*x))/(1-np.exp(η))

    return Qf_evaluate.item()

def Q_mapping_eval_all(Qf_components_log, η, X_series):
    """
    Evaluate all the Qf_components_log given x collectively
    """
    main_periods = len(Qf_components_log)-1
    Qf_evaluate_period = np.zeros([main_periods+1, X_series[0].shape[1]])
    for i in range(main_periods):
        Qf_evaluate_period[i,:] = np.exp(Qf_components_log[i](*X_series))

    Qf_evaluate_period[-1,:] = np.exp(Qf_components_log[-1](*X_series))/(1-np.exp(η))
    
    Qf_evaluate = Qf_evaluate_period.sum(axis=0)
    return Qf_evaluate

def eval_main(Qf_components_log, x):
    return np.exp(Qf_components_log(*x))

def E_exp_W(Y, x):
    _, n_X, n_W = Y.shape
    x1, x2 = x
    Y_x2, Y_x, Y_w, Y_c, Y_xx, Y_xw, Y_ww = Y['x2'], Y['x'], Y['w'], Y['c'], Y['xx'], Y['xw'], Y['ww']

    return _E_exp_W_jit(Y_x2, Y_x, Y_w, Y_c, Y_xx, Y_xw, Y_ww, n_X, n_W, x1, x2)

@njit
def _E_exp_W_jit(Y_x2, Y_x, Y_w, Y_c, Y_xx, Y_xw, Y_ww, n_X, n_W, x1, x2):

    A = x1.T @ mat(Y_xw, (n_W, n_X)).T + Y_w
    B = Y_ww
    C = np.exp(Y_x2@x2 + Y_xx@np.kron(x1, x1) + Y_x@x1 + Y_c)
    temp = np.eye(n_W) - sym(mat(2*B, (n_W, n_W)))
    temp_inv = np.linalg.inv(temp)
    
    term_0 = C
    term_1 = np.linalg.det(temp)**(-1./2)
    term_2 = np.exp(1./2 * A@temp_inv@A.T)
    term_3 = temp_inv@A.T

    return term_0[0,0]*term_1*term_2[0,0]*term_3