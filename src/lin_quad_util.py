"""
Tools for operations on LinQuadVar.

"""
import numpy as np
from scipy.stats import norm
from scipy import optimize
from utilities import vec, mat, sym, cal_E_ww
from lin_quad import LinQuadVar
from numba import njit
from copy import deepcopy
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

    lq_sum = LinQuadVar({'c':np.zeros([1,1])},lq_list[0].shape)
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
    

# def E(Y, E_w, Cov_w=None):
def E(Y, E_w, E_ww=None):
    r"""
    Computes :math:`E[Y_{t+1} \mid \mathfrak{F}_t]`,
    Parameters
    ----------
    Y : LinQuadVar
    E_w : (n_W, 1) ndarray
    Cov_w : (n_W, n_W) ndarray
        Used when the Y has non-zero coefficient on 'ww' term.
    Returns
    -------
    E_Y : LinQuadVar
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
    if second_order:
        return log_E_exp(M + next_period(f, X1_tp1, X2_tp1))
    else:
        if X2_tp1 != None:
            print('The second order expansion for law of motion is not used in the first order expansion.')
        return log_E_exp(M + next_period(f, X1_tp1))
    

def Q_mapping(M, f, X1_tp1, X2_tp1, tol = 1e-10, max_iter = 20000, second_order = True):
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
    
def Q_mapping_eval(Qf_components_log, η, x):
    Qf_evaluate = 0.

    for i in range(len(Qf_components_log)-1):
        Qf_evaluate += np.exp(Qf_components_log[i](*x))

    Qf_evaluate += np.exp(Qf_components_log[-1](*x))/(1-np.exp(η))

    return Qf_evaluate.item()

def Q_mapping_eval_all(Qf_components_log, η, X_series):

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

def Q_mapping_eval_tp11(Qf_components_log, η, x):
    Qf_evaluate = 0.

    for i in range(11,len(Qf_components_log)-1):
        Qf_evaluate += np.exp(Qf_components_log[i](*x))

    Qf_evaluate += np.exp(Qf_components_log[-1](*x))/(1-np.exp(η))

    return Qf_evaluate.item()

def Q_mapping_11(M, f, X1_tp1, X2_tp1, tol = 1e-10, max_iter = 10000, second_order = True):
    Qf_components_log = []
#    Qf_evaluate = 0.
    for i in range(11):
        Qf_components_log.append(f)
#        Qf_evaluate += np.exp(f(*x))
        if second_order:
            f_next = M_mapping(M, f, X1_tp1, X2_tp1)
        else:
            if X2_tp1 != None:
                print('The second order expansion for law of motion is not used in the first order expansion.')
            f_next = M_mapping(M, f, X1_tp1, second_order = second_order)
        η = (f_next['c'] - f['c']).item()
        
    Qf_components_log.append(f_next)
    
    return Qf_components_log, f, η

def Q_mapping_eval_11(Qf_components_log, η, x):
    Qf_evaluate = 0.

    for i in range(len(Qf_components_log)):
        Qf_evaluate += np.exp(Qf_components_log[i](*x))

    return Qf_evaluate.item()

def calc_PD(Qf_components_log, η, x):
    numerator = Q_mapping_eval_tp11(Qf_components_log, η, x)
    Qf_components_log_D_11, _, η_D_third_11= Q_mapping_11(log_D_growth, log_f, modelSol.Z1_tp1, modelSol.Z2_tp1)
    denominator = Q_mapping_eval_11(Qf_components_log_D_11, η_D_third_11, (np.zeros([n_Z,1]),np.zeros([n_Z,1])))    
    return numerator/denominator

def Q_der_17(Qf_components_log, Qf_evaluate, η, log_M, x, X1_tp1, X2_tp1, α):
    
    numerator = 0.
    for i in range(len(Qf_components_log) - 1):
        component = previous_period(Qf_components_log[i], X1_tp1, X2_tp1)
        numerator += E_exp_W(log_M + component, x)
    numerator += E_exp_W(log_M + previous_period(Qf_components_log[-1], \
                                                 X1_tp1, X2_tp1) ,x)/(1-np.exp(η))
    return (α.T@numerator / Qf_evaluate).item()

def pd_ratio_adjust(model_Sol, gc0_tp1, gd0_tp1, gc1_tp1, gd1_tp1, vmc_t_order, rmc_t_order, pd_t_order, args = (), tol = 1e-7):

    γ, β, ρ, α, ϕ_e, σ_squared, ν_1, σ_w, μ, μ_d, ϕ, ϕ_d, ϕ_c, π = args

    η_m = (np.log(β) - ρ*gc0_tp1['c']+gd0_tp1['c']).item()
    dM1_tp1 = ((ρ-1)*(previous_period(model_Sol['X1_t'][vmc_t_order],model_Sol.Z1_tp1,model_Sol.Z2_tp1)-model_Sol['X1_t'][rmc_t_order])-gc1_tp1)+gd1_tp1

    n_Y, n_Z, n_W = model_Sol.var_shape 

    def return_order1_t(order1_t_coeffs):
        return LinQuadVar({'c': np.array([[order1_t_coeffs[0]]]), 'x':np.array([order1_t_coeffs[1:]])},(1, n_Z, n_W))

    def E_N1_tp1(Y):
        E_Y = {}
        temp_x = Y['w'] @ model_Sol['μ_1']
        E_Y['x'] = Y['x'] + temp_x 
        E_Y['c'] = Y['c'] + Y['w'] @ model_Sol['μ_0']
        E_Y = LinQuadVar(E_Y, (1, n_Z, n_W), False)
        return E_Y

    def solve_pd1_t_first(order1_init_coeffs):
        pd1_t = return_order1_t(order1_init_coeffs)
        LHS = E_N1_tp1(dM1_tp1 + np.exp(η_m) *previous_period(pd1_t, model_Sol.Z1_tp1, model_Sol.Z2_tp1))
        LHS_list = [LHS['c'].item()]+ [i for i in LHS['x'][0]] 
        return list(np.array(LHS_list) - np.array(order1_init_coeffs))
    
    pd1_t_sol = optimize.root(solve_pd1_t_first, x0 = [0]*(1 + n_Z),tol = tol)['x']
    pd1_t = return_order1_t(pd1_t_sol)
    print(optimize.root(solve_pd1_t_first, x0 = [0]*(1 +n_Z),tol = tol)['message'])

    N = model_Sol.X1_t['x'][:n_Y]
    N[pd_t_order,:] = pd1_t['x'][0]
    C = model_Sol.X1_t['c'][:n_Y]
    C[pd_t_order,:] = pd1_t['c'][0]

    X0_t = model_Sol.X0_t

    Y1_t = LinQuadVar({'x': N, 'c': C}, (n_Y, n_Z, n_W), False)
    X1_t = concat([Y1_t, model_Sol.Z1_t])

    G = model_Sol['G_M1_0']
    Y2_t = LinQuadVar({'x2': N,
                    'xx': G[:, 1+n_Z:1+n_Z+n_Z**2],
                    'x': G[:, 1:1+n_Z],
                    'c': G[:, :1]}, (n_Y, n_Z, n_W), False)
    X2_t = concat([Y2_t, model_Sol.Z2_t])
    X_t = X0_t + X1_t + X2_t*0.5

    Z1Z1 = kron_prod(model_Sol.Z1_tp1, model_Sol.Z1_tp1)

    X_tp1 = next_period(X_t, model_Sol.Z1_tp1, model_Sol.Z2_tp1, Z1Z1)
    X1_tp1 = next_period(X1_t, model_Sol.Z1_tp1)
    X2_tp1 = next_period(X2_t, model_Sol.Z1_tp1, model_Sol.Z2_tp1, Z1Z1)

    model_Sol_adj = deepcopy(model_Sol)
    model_Sol_adj['X_t'] = X_t
    model_Sol_adj['X1_t'] = X1_t
    model_Sol_adj['X2_t'] = X2_t
    model_Sol_adj['X_tp1'] = X_tp1
    model_Sol_adj['X1_tp1'] = X1_tp1
    model_Sol_adj['X2_tp1'] = X2_tp1

    return model_Sol_adj