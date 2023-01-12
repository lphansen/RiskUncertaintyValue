import numpy as np
import autograd.numpy as anp
import scipy as sp
import seaborn as sns
from scipy import optimize

from lin_quad_util import E, cal_E_ww, matmul, concat, next_period, kron_prod, log_E_exp, lq_sum
from utilities import mat, vec, sym, gschur
from derivatives import compute_derivatives
from lin_quad import LinQuadVar

'''
This Python script provides functions to solve for the discrete-time 
dynamic macro-finance models under uncertainty, based on the 
perturbation method. These models feature EZ recursive preferences.

Developed and maintained by the MFR research team.
Codes updated on Jan. 11, 2023, 6:34 P.M. CT
Check and obtain the lastest version on 
https://github.com/lphansen/RiskUncertaintyValue 
'''

def uncertain_expansion(eq, ss, var_shape, args, gc, init_util = None, iter_tol = 1e-8, max_iter = 50):
    
    """
    This function solves a system with recursive utility via small-noise
        expansion, given a set of equilibrium conditions, steady states,
        consumption growth, and other model configurations. 

    The solver returns a class storing solved variables. In particular, 
    it stores the solved variables represented by a linear or
    linear-quadratic function of the first- and second-order derivatives
    of states. It also stores laws of motion for these state derivatives.

    Parameters
    ----------
    eq_cond : callable
        Returns [Q psi_1-psi_2, phi_var - phi], Q psi_1-psi_2 satisfy the 
        forward-looking equations E[N Q psi_1-psi_2]=0, and phi satisfy 
        the state equations phi_var - phi=0

        ``eq_cond(Var_t, Var_tp1, W_tp1, q, *args) -> (n_JX, ) ndarray``

        where Var_t and Var_tp1 are variables at time t and t+1 respectively,
        W_tp1 are time t+1 shocks, and q is perturbation parameter.
        Note that in Var_t and Var_tp1, state variables must follow endogenous 
        variables. The first one must be q_t or q_tp1. In the equilibrium 
        conditions, state evolution equations should follow forward-looking 
        equations.

        Under 'psi1' mode, returns psi_1, 

        ``eq_cond(Var_t, Var_tp1, W_tp1, q, *args) -> (n_J, ) ndarray``

        'psi1' mode specification is mandatory. Other modes are optional.
    ss : callable or (n_JX, ) ndarray
        Steady states or the function for calculating steady states.
        It follows the same order as Var_t and Var_tp1 in equilibrium 
        conditions.
    var_shape : tuple of ints
        (n_J, n_X, n_W). Number of jump variables, states and
        shocks respectively.
    args : tuple of floats/ints
        Model parameters, the first three elements are fixed recursive 
        utility parameters, γ, β, ρ
        The last item is reserved to `mode`, a string that specifies 
        what the `eq_cond` should return.
    gc_tp1_fun : callable
        Function to approximate the log growth of consumption.
    init_util: dict
        Initialization of $mu^0, Upsilon^2_0, Upsilon^2_1,$ and $Upsilon^2_2$. 
        Users may provide a dictionary that maps the keys `mu_0`, `Upsilon_0`, 
        `Upsilon_1`, `Upsilon_2` to the corresponding matrices for initialization. 
        If None, zero matrices are used.
    iter_tol : float
        The tolerance for iteration in the expansion.
    max_iter : int
        The maximum of iterations in the expansion.

    Returns
    -------
    res : ModelSolution
        The model solution represented as a ModelSolution object. Important
        attributes are: J_t the approximated jump variables as a linear or linear-
        quadratic function of state derivatives; X1_tp1 (and X2_tp1) the laws of
        motion for the first-order (and second-order) derivatives of states. 

    """

    n_J, n_X, n_W = var_shape
    n_JX = n_J + n_X
    γ = args[0]
    β = args[1]
    ρ = args[2]
    df, ss = take_derivatives(eq, ss, var_shape, True, args)
    H_fun_list = [(lambda y: (lambda JX_t, JX_tp1, W_tp1, q, *args: eq(JX_t, JX_tp1, W_tp1, q, *(*list(args)[:-1], 'psi1'))[y]))(i) for i in range(n_J)]

    X0_t = LinQuadVar({'c': ss[n_J:].reshape(-1, 1)}, (n_X, n_X, n_W), False)
    J0_t = LinQuadVar({'c': ss[:n_J].reshape(-1, 1)}, (n_J, n_X, n_W), False)
    JX0_t = concat([J0_t, X0_t])
    X1_t = LinQuadVar({'x': np.eye(n_X)}, (n_X, n_X, n_W), False)
    X2_t = LinQuadVar({'x2': np.eye(n_X)}, (n_X, n_X, n_W), False)
    
    H0_t = [approximate_fun(H_fun, np.concatenate([np.zeros(1), ss]), (1, n_X, n_W), None, None, None, None, args, second_order = False, zeroth_order= True) for H_fun in H_fun_list]
    if init_util == None:
        util_sol = {'μ_0': np.zeros([n_W,1]),
                    'Upsilon_2':np.zeros([n_W,n_W]),
                    'Upsilon_1':np.zeros([n_W,n_X]),
                    'Upsilon_0':np.zeros([n_W,1])}
    else:
        util_sol = init_util
                
    μ_0_series = []
    J1_t_series = []
    J2_t_series = []
    error_series = []

    i = 0
    error = 1

    while error > iter_tol and i < max_iter:
        
        J1_t, JX1_t, JX1_tp1, JX1_tp1_tilde, X1_tp1, X1_tp1_tilde, schur = first_order_expansion(df, util_sol, var_shape, H0_t, args)
        adj, lq1, lq2, lq3, lq4, H_approx = compute_adj(H_fun_list, ss, var_shape, util_sol, JX1_t, X1_tp1, H0_t, args)
        J2_t, JX2_t, JX2_tp1, JX2_tp1_tilde, X2_tp1, X2_tp1_tilde, schur = second_order_expansion(df, util_sol, X1_tp1, X1_tp1_tilde, JX1_t, JX1_tp1, JX1_tp1_tilde, adj, var_shape)
        util_sol = solve_utility(γ, β, ρ, ss, var_shape, args, X1_tp1, X2_tp1, JX1_t, JX2_t, gc)
        μ_0_series.append(util_sol['μ_0'])
        J1_t_series.append(J1_t(*(np.ones([n_X,1]),np.ones([n_X,1]),np.ones([n_X,1]))))
        J2_t_series.append(J2_t(*(np.ones([n_X,1]),np.ones([n_X,1]),np.ones([n_X,1]))))

        if i > 0:
            error_1 = np.max(np.abs(μ_0_series[i] - μ_0_series[i-1]))
            error_2 = np.max(np.abs(J1_t_series[i] - J1_t_series[i-1])) 
            error_3 = np.max(np.abs(J2_t_series[i] - J2_t_series[i-1])) 
            error = np.max([error_1, error_2, error_3])
            print('Iteration {}: error = {:.9g}'.format(i, error))
            error_series.append(error)
        
        i+=1

    JX_t = JX0_t + JX1_t + 0.5*JX2_t
    JX_tp1 = next_period(JX_t, X1_tp1, X2_tp1)
    JX_tp1_tilde = next_period(JX_t, X1_tp1_tilde, X2_tp1_tilde)

    return ModelSolution({
                        'X0_t':X0_t, 'X1_t':X1_t, 'X2_t':X2_t, 'X1_tp1': X1_tp1, 'X2_tp1': X2_tp1,
                        'X1_tp1_tilde': X1_tp1_tilde, 'X2_tp1_tilde': X2_tp1_tilde,
                        'J0_t':J0_t, 'J1_t':J1_t, 'J2_t':J2_t,
                        'JX0_t':JX0_t, 'JX1_t':JX1_t, 'JX2_t':JX2_t, 'JX1_tp1': JX1_tp1, 'JX2_tp1': JX2_tp1, 'JX_t': JX_t, 'JX_tp1': JX_tp1,
                        'JX1_tp1_tilde': JX1_tp1_tilde, 'JX2_tp1_tilde': JX2_tp1_tilde, 'JX_tp1_tilde': JX_tp1_tilde,
                        'util_sol':util_sol, 'log_N0': util_sol['log_N0'], 'log_N_tilde':util_sol['log_N_tilde'],
                        'vmr1_tp1':util_sol['vmr1_tp1'], 'vmr2_tp1':util_sol['vmr2_tp1'],
                        'gc_tp1':util_sol['gc_tp1'],'gc0_tp1':util_sol['gc0_tp1'],'gc1_tp1':util_sol['gc1_tp1'],'gc2_tp1':util_sol['gc2_tp1'],
                        'var_shape': var_shape, 'args':args, 
                        'ss': ss, 'second_order': True})
    
def take_derivatives(f, ss, var_shape, second_order,
                      args):
    """
    Take first- or second-order derivatives.

    """
    n_J, n_X, n_W = var_shape
    n_JX = n_J + n_X
    W_0 = np.zeros(n_W)
    q_0 = 0.

    if callable(ss):
        ss = ss(*args)

    dfq = compute_derivatives(f=lambda Var_t, Var_tp1, W_tp1, q:
                             anp.atleast_1d(f(Var_t, Var_tp1, W_tp1, q, *args)),
                             X=[ss, ss, W_0, q_0],
                             second_order=second_order)
                 
    df = {'xt':dfq['xt'][:,1:],\
         'xtp1':dfq['xtp1'][:,1:],\
         'wtp1':dfq['wtp1'],\
         'q':dfq['q'],\
         'xtxt':dfq['xtxt'][:,n_JX+1:][:,np.tile(np.concatenate([np.array([False]),np.repeat(True, n_JX)]),n_JX)],\
         'xtxtp1':dfq['xtxtp1'][:,n_JX+1:][:,np.tile(np.concatenate([np.array([False]),np.repeat(True, n_JX)]),n_JX)],\
         'xtwtp1':dfq['xtwtp1'][:,n_W:],\
         'xtq':dfq['xtq'][:,1:],\
         'xtp1xtp1':dfq['xtp1xtp1'][:,n_JX+1:][:,np.tile(np.concatenate([np.array([False]),np.repeat(True, n_JX)]),n_JX)],\
         'xtp1wtp1':dfq['xtp1wtp1'][:,n_W:],\
         'xtp1q':dfq['xtp1q'][:,1:],\
         'wtp1wtp1':dfq['wtp1wtp1'],\
         'wtp1q':dfq['wtp1q'],\
         'qq':dfq['qq']}                         
    ss = ss[1:]            
    return df, ss

def first_order_expansion(df, util_sol, var_shape, H0, args):
    """
    Implements first-order expansion.

    """
    n_J, n_X, n_W = var_shape
    n_JX = n_J + n_X
    HQ_loc_list = [i for i in range(n_J)]
    
    γ = args[0]
    ρ = args[2]

    schur = schur_decomposition(-df['xtp1'], df['xt'], (n_J, n_X, n_W))
    μ_0 = util_sol['μ_0']
    H_1 = np.zeros([n_JX,1])
    for i in range(len(H0)):
        H_1[HQ_loc_list[i]] = μ_0.T@μ_0*(ρ-1)/2/(1-γ)*H0[i]
    
    f_1_xtp1 = df['xtp1'][:n_J]
    f_1_wtp1 = df['wtp1'][:n_J]
    f_2_xtp1 = df['xtp1'][n_J:]
    f_2_xt = df['xt'][n_J:]
    f_2_wtp1 = df['wtp1'][n_J:]

    RHS = - (df['wtp1']@μ_0+H_1)
    LHS = df['xtp1'] + df['xt']
    D = np.linalg.solve(LHS, RHS)
    C = D[:n_J] - schur['N']@D[n_J:]

    ψ_tilde_x = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_xt@schur['N_block'])
    ψ_tilde_w = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_wtp1)
    ψ_tilde_q = D[n_J:] - ψ_tilde_x@D[n_J:]

    RHS_org = - (np.block([[(f_1_xtp1@schur['N_block']@ψ_tilde_w+ f_1_wtp1)@μ_0], [np.zeros((n_X, 1))]])+H_1)
    LHS_org = df['xtp1'] + df['xt']
    D_org = np.linalg.solve(LHS_org, RHS_org)
    ψ_q = D_org[n_J:] - ψ_tilde_x@D_org[n_J:]

    X1_tp1 = LinQuadVar({'x': ψ_tilde_x, 'w': ψ_tilde_w, 'c': ψ_q}, (n_X, n_X, n_W), False)
    X1_tp1_tilde = LinQuadVar({'x': ψ_tilde_x, 'w': ψ_tilde_w, 'c': ψ_tilde_q}, (n_X, n_X, n_W), False)
    J1_t = LinQuadVar({'x': schur['N'], 'c': C}, (n_J,n_X,n_W), False)
    X1_t = LinQuadVar({'x': np.eye(n_X)}, (n_X, n_X, n_W), False)
    JX1_t = concat([J1_t, X1_t])
    JX1_tp1 = next_period(JX1_t, X1_tp1)
    JX1_tp1_tilde = next_period(JX1_t, X1_tp1_tilde)
    
    return J1_t, JX1_t, JX1_tp1, JX1_tp1_tilde, X1_tp1, X1_tp1_tilde, schur

def compute_adj(H_fun_list, ss, var_shape, util_sol, JX1_t, X1_tp1, H0, args):
    """
    Computes additional recursive utility adjustment

    """
    n_J, n_X, n_W = var_shape
    n_JX = n_J + n_X
    HQ_loc_list = [i for i in range(n_J)]

    γ = args[0]
    ρ = args[2]

    H_approx = [approximate_fun(H_fun, np.concatenate([np.zeros(1), ss]), (1, n_X, n_W), concat([LinQuadVar({'c':np.zeros([1,1])}, shape=(1,n_X,n_W)),JX1_t]), None, X1_tp1, None, args, second_order = False) for H_fun in H_fun_list]
    μ_0 = util_sol['μ_0']
    Upsilon_2 = util_sol['Upsilon_2']
    Upsilon_1 = util_sol['Upsilon_1']
    Upsilon_0 = util_sol['Upsilon_0']
    Theta_0 = [app[2]['c']+app[2]['w']@μ_0 for app in H_approx]
    Theta_1 = [app[2]['x'] for app in H_approx]
    Theta_2 = [app[2]['w'] for app in H_approx]

    lq1 = [LinQuadVar({'x': (1-γ) * (Theta_2[i]) @ Upsilon_1,\
                       'c': (1-γ) * (Theta_2[i]) @ Upsilon_0}, (1, n_X, n_W)) for i in range(len(HQ_loc_list))] 
    lq2 = [LinQuadVar({'x': (ρ-1) * μ_0.T @ Upsilon_1 * H0[i],\
                        'c': (ρ-1) * μ_0.T @ Upsilon_0 * H0[i]},(1, n_X, n_W)) for i in range(len(HQ_loc_list))]
    lq3 = [LinQuadVar({'x': 2 * (ρ-1)/(1-γ)*(0.5 * (μ_0.T @ μ_0) @ Theta_1[i]),\
                        'c': 2 * (ρ-1)/(1-γ)*(Theta_2[i] @ μ_0 + 0.5 * (μ_0.T@μ_0) @ Theta_0[i])},(1, n_X, n_W)) for i in range(len(HQ_loc_list))]
    lq4 = [LinQuadVar({'c': ((ρ-1)/(1-γ))**2 * (μ_0.T@μ_0 + 0.25 * (μ_0.T@μ_0)**2) * H0[i]},(1, n_X, n_W)) for i in range(len(HQ_loc_list))]
    
    adj = [lq_sum([lq1[i], lq2[i], lq3[i], lq4[i]]) for i in range(len(HQ_loc_list))]
    adj_aug = {'x':np.zeros([n_JX,n_X]),'c':np.zeros([n_JX,1])}

    for i in range(len(HQ_loc_list)):
        adj_aug['x'][HQ_loc_list[i]] = adj[i]['x']
        adj_aug['c'][HQ_loc_list[i]] = adj[i]['c']

    return adj_aug, lq1, lq2, lq3, lq4, H_approx

def second_order_expansion(df, util_sol, X1_tp1, X1_tp1_tilde, JX1_t, JX1_tp1, JX1_tp1_tilde, adj, var_shape):
    """
    Implements second-order expansion.

    """
    n_J, n_X, n_W = var_shape

    schur = schur_decomposition(-df['xtp1'], df['xt'], (n_J, n_X, n_W))
    μ_0 = util_sol['μ_0']

    def guess_verify(Wtp1, M_E_w, X1_tp1, JX1_tp1):
        
        D2 = combine_second_order_terms(df, JX1_t, JX1_tp1, Wtp1)
        D2_coeff = np.block([[D2['c'], D2['x'], D2['w'], D2['xx'], D2['xw'], D2['ww']]])
        
        cov_w = np.eye(n_W)
        M_E_ww = cal_E_ww(M_E_w, cov_w)
        E_D2 = E(D2, M_E_w, M_E_ww)
        E_D2_coeff = np.block([[E_D2['c']+adj['c'], E_D2['x']+adj['x'], E_D2['xx']]])
        X1X1 = kron_prod(X1_tp1, X1_tp1)
        M_mat = form_M0(M_E_w, M_E_ww, X1_tp1, X1X1)
        LHS = np.eye(n_J*M_mat.shape[0]) - np.kron(M_mat.T, np.linalg.solve(schur['Λ22'], schur['Λp22']))
        RHS = vec(-np.linalg.solve(schur['Λ22'], (schur['Q'].T@E_D2_coeff)[-n_J:]))
        D_tilde_vec = np.linalg.solve(LHS, RHS)
        D_tilde = mat(D_tilde_vec, (n_J, 1+n_X+n_X**2))
        G = np.linalg.solve(schur['Z21'], D_tilde)
        Y2_coeff = -df['xtp1'][n_J:]@schur['N_block']
        C_hat = np.linalg.solve(Y2_coeff, D2_coeff[n_J:])
        C_hat_coeff = np.split(C_hat, np.cumsum([1, n_X, n_W, n_X**2, n_X*n_W]),axis=1)
        G_block = np.block([[G], [np.zeros((n_X, 1+n_X+n_X**2))]])
        Gp_hat = np.linalg.solve(Y2_coeff, df['xtp1'][n_J:]@G_block)
        G_hat = np.linalg.solve(Y2_coeff, df['xt'][n_J:]@G_block)
        c_1, x_1, w_1, xx_1, xw_1, ww_1 = C_hat_coeff
        c_2, x_2, xx_2 = np.split(G_hat, np.cumsum([1, n_X]), axis=1)
        var = LinQuadVar({'c': Gp_hat[:, :1], 'x': Gp_hat[:, 1:1+n_X], 'xx': Gp_hat[:, 1+n_X:1+n_X+n_X**2]}, (n_X, n_X, n_W), False)
        var_next = next_period(var, X1_tp1, None, X1X1)
        ψ_x2 = X1_tp1['x']
        ψ_tilde_xx = var_next['xx'] + xx_1 + xx_2
        ψ_tilde_xw = var_next['xw'] + xw_1
        ψ_tilde_xq = var_next['x'] + x_1 + x_2
        ψ_tilde_ww = var_next['ww'] + ww_1
        ψ_tilde_wq = var_next['w'] + w_1
        ψ_tilde_qq = var_next['c'] + c_1 + c_2

        X2_tp1 = LinQuadVar({'x2': ψ_x2, 'xx': ψ_tilde_xx, 'xw': ψ_tilde_xw, 'ww': ψ_tilde_ww, 'x': ψ_tilde_xq, 'w': ψ_tilde_wq, 'c': ψ_tilde_qq}, (n_X, n_X, n_W),False)

        return X2_tp1, G

    Wtp1 = LinQuadVar({'w': np.eye(n_W),'c':μ_0}, (n_W, n_X, n_W), False)
    M_E_w = np.zeros([n_W,1])
    X2_tp1_tilde, G = guess_verify(Wtp1, M_E_w, X1_tp1_tilde, JX1_tp1_tilde)
    
    Wtp1 = LinQuadVar({'w': np.eye(n_W)}, (n_W, n_X, n_W), False)
    M_E_w = μ_0
    X2_tp1, _ = guess_verify(Wtp1, M_E_w, X1_tp1, JX1_tp1)

    J2_t = LinQuadVar({'x2': schur['N'], 'xx': G[:, 1+n_X:1+n_X+(n_X)**2], 'x': G[:, 1:1+n_X], 'c': G[:, :1]}, (n_J, n_X, n_W), False)
    X2_t = LinQuadVar({'x2': np.eye(n_X)}, (n_X, n_X, n_W), False)
    JX2_t = concat([J2_t, X2_t])
    JX2_tp1 = next_period(JX2_t, X1_tp1, X2_tp1)
    JX2_tp1_tilde = next_period(JX2_t, X1_tp1_tilde, X2_tp1_tilde)

    return J2_t, JX2_t, JX2_tp1, JX2_tp1_tilde, X2_tp1, X2_tp1_tilde, schur
    
def schur_decomposition(df_tp1, df_t, var_shape):
    
    n_J, n_X, n_W = var_shape
    Λp, Λ, a, b, Q, Z = gschur(df_tp1, df_t)
    Λp22 = Λp[-n_J:, -n_J:]
    Λ22 = Λ[-n_J:, -n_J:]
    Z21 = Z.T[-n_J:, :n_J]
    Z22 = Z.T[-n_J:, n_J:]
    N = -np.linalg.solve(Z21, Z22)
    N_block = np.block([[N], [np.eye(n_X)]])
    schur_decomposition = {'N':N,'N_block':N_block,'Λp':Λp,'Λ':Λ,'Q':Q,'Z':Z,'Λp22':Λp22,'Λ22':Λ22,'Z21':Z21,'Z22':Z22}
    
    return schur_decomposition

def combine_second_order_terms(df, JX1_t, JX1_tp1, Wtp1):

    _, n_X, n_W = JX1_tp1.shape

    xtxt = kron_prod(JX1_t, JX1_t)
    xtxtp1 = kron_prod(JX1_t, JX1_tp1)
    xtwtp1 = kron_prod(JX1_t, Wtp1)
    xtp1xtp1 = kron_prod(JX1_tp1, JX1_tp1)
    xtp1wtp1 = kron_prod(JX1_tp1, Wtp1)
    wtp1wtp1 = kron_prod(Wtp1, Wtp1)
    res = matmul(df['xtxt'], xtxt)\
        + matmul(2*df['xtxtp1'], xtxtp1)\
        + matmul(2*df['xtwtp1'], xtwtp1)\
        + matmul(2*df['xtq'], JX1_t)\
        + matmul(df['xtp1xtp1'], xtp1xtp1)\
        + matmul(2*df['xtp1wtp1'], xtp1wtp1)\
        + matmul(2*df['xtp1q'], JX1_tp1)\
        + matmul(df['wtp1wtp1'], wtp1wtp1)\
        + matmul(2*df['wtp1q'], Wtp1)\
        + LinQuadVar({'c': df['qq']}, (df['qq'].shape[0], n_X, n_W), False)

    return res

def form_M0(M0_E_w, M0_E_ww, X1_tp1, X1X1):

    _, n_X, n_W = X1_tp1.shape
    M0_mat_11 = np.eye(1)
    M0_mat_12 = np.zeros((1, n_X))
    M0_mat_13 = np.zeros((1, n_X**2))
    M0_mat_21 = X1_tp1['w']@M0_E_w + X1_tp1['c']
    M0_mat_22 = X1_tp1['x']
    M0_mat_23 = np.zeros((n_X, n_X**2))
    M0_mat_31 = X1X1['ww']@M0_E_ww + X1X1['w']@M0_E_w + X1X1['c']
    temp = np.vstack([M0_E_w.T@mat(X1X1['xw'][row: row+1, :], (n_W, n_X))
                      for row in range(X1X1.shape[0])])    
    M0_mat_32 = temp + X1X1['x']
    M0_mat_33 = X1X1['xx']
    M0_mat = np.block([[M0_mat_11, M0_mat_12, M0_mat_13],
                       [M0_mat_21, M0_mat_22, M0_mat_23],
                       [M0_mat_31, M0_mat_32, M0_mat_33]])
    return M0_mat

def approximate_fun(fun, ss, var_shape, JX1_t, JX2_t, X1_tp1, X2_tp1, args, second_order = True, zeroth_order = False):
    """
    Approximates function given state evolutions and jump varibles

    """
    _, n_X, n_W = var_shape
    
    W_0 = np.zeros(n_W)
    q_0 = 0.
    
    dfun = compute_derivatives(f=lambda JX_t, JX_tp1, W_tp1, q:
                                   anp.atleast_1d(fun(JX_t, JX_tp1, W_tp1, q, *args)),
                                   X=[ss, ss, W_0, q_0],
                                   second_order=True)
    
    fun_zero_order = fun(ss, ss, W_0, q_0, *args)
    if zeroth_order:
        return fun_zero_order
    else:
        JX1_tp1 = next_period(JX1_t, X1_tp1)
        fun_first_order = matmul(dfun['xtp1'], JX1_tp1)\
            + matmul(dfun['xt'], JX1_t)\
            + LinQuadVar({'w': dfun['wtp1'], 'c': dfun['q'].reshape(-1, 1)},
                        (1, n_X, n_W), False)
        fun_approx = fun_zero_order + fun_first_order

        if second_order:
            JX2_tp1 = next_period(JX2_t, X1_tp1, X2_tp1)
            Wtp1 = LinQuadVar({'w': np.eye(n_W)}, (n_W, n_X, n_W), False)

            temp1 = combine_second_order_terms(dfun, JX1_t, JX1_tp1, Wtp1)
            temp2 = matmul(dfun['xt'], JX2_t)\
                + matmul(dfun['xtp1'], JX2_tp1)
            fun_second_order = temp1 + temp2
            fun_approx = fun_approx + fun_second_order*0.5

            return fun_approx, fun_zero_order, fun_first_order, fun_second_order
        else:
            return fun_approx, fun_zero_order, fun_first_order

def solve_utility(γ, β, ρ, ss, var_shape, args, X1_tp1, X2_tp1, JX1_t, JX2_t, gc_tp1_fun, tol = 1e-10):
    """
    Solves continuation values and forms approximation of change of measure

    """
    _, n_X, n_W = var_shape

    gc_tp1_approx = approximate_fun(gc_tp1_fun, np.concatenate([np.zeros(1), ss]), (1, n_X, n_W), concat([LinQuadVar({'c':np.zeros([1,1])}, shape=(1,n_X,n_W)),JX1_t]), concat([LinQuadVar({'c':np.zeros([1,1])}, shape=(1,n_X,n_W)),JX2_t]), X1_tp1, X2_tp1, args)

    gc_tp1 = gc_tp1_approx[0]
    gc0_tp1 = gc_tp1_approx[1]
    gc1_tp1 = gc_tp1_approx[2]
    gc2_tp1 = gc_tp1_approx[3]

    def return_order1_t(order1_t_coeffs):
        return LinQuadVar({'c': np.array([[order1_t_coeffs[0]]]), 'x':np.array([order1_t_coeffs[1:(1+n_X)]])},(1, n_X, n_W))
    
    def return_order2_t(order2_t_coeffs):
            return LinQuadVar({'c': np.array([[order2_t_coeffs[0]]]), 'x':np.array([order2_t_coeffs[1:(1+n_X)]]),\
                            'x2':np.array([order2_t_coeffs[(1+n_X):(1+n_X+n_X)]]), 'xx':np.array([order2_t_coeffs[(1+n_X+n_X):(1+n_X+n_X+n_X*n_X)]])},(1, n_X, n_W))
    
    λ = β * np.exp((1-ρ) * gc0_tp1)
    
    def solve_vmc1_t(order1_init_coeffs):
        vmc1_t = return_order1_t(order1_init_coeffs)
        LHS = λ/(1-γ) *log_E_exp((1-γ)*(next_period(vmc1_t,X1_tp1)+gc1_tp1))
        LHS_list = [LHS['c'].item()]+ [i for i in LHS['x'][0]] 
        return list(np.array(LHS_list) - np.array(order1_init_coeffs))   
        
    vmc1_t_sol = optimize.root(solve_vmc1_t, x0 = [0]*(1 + n_X),tol = tol)
    if vmc1_t_sol['success'] ==False:
        print(vmc1_t_sol['message'])
    vmc1_t = return_order1_t(vmc1_t_sol['x'])
    rmc1_t = log_E_exp((1-γ)*(next_period(vmc1_t,X1_tp1)+gc1_tp1))/(1-γ)
    vmc1_tp1 = next_period(vmc1_t,X1_tp1)
    vmr1_tp1 = vmc1_tp1 + gc1_tp1 - rmc1_t
    log_N0 = ((1-γ)*(vmc1_tp1 + gc1_tp1)-log_E_exp((1-γ)*(vmc1_tp1 + gc1_tp1)))
    μ_0 = log_N0['w'].T
    Ew0 = μ_0
    Eww0 = cal_E_ww(Ew0,np.eye(Ew0.shape[0]))

    def solve_vmc2_t(order2_init_coeffs):
        vmc2_t = return_order2_t(order2_init_coeffs)
        LHS = λ *E(next_period(vmc2_t, X1_tp1, X2_tp1) + gc2_tp1, Ew0, Eww0) + (1-ρ)*(1-λ)/λ*kron_prod(vmc1_t,vmc1_t)
        LHS_list = [LHS['c'].item()]+ [i for i in LHS['x'][0]] + [i for i in LHS['x2'][0]] + [i for i in LHS['xx'][0]] 
        return list(np.array(LHS_list) - np.array(order2_init_coeffs))
    
    vmc2_t_sol = optimize.root(solve_vmc2_t, x0 = [0]*(1 + n_X+n_X+n_X*n_X),tol = tol)
    if vmc2_t_sol['success'] ==False:
        print(vmc2_t_sol['message'])
    vmc2_t = return_order2_t(vmc2_t_sol['x'])
    rmc2_t = E(next_period(vmc2_t, X1_tp1, X2_tp1) + gc2_tp1, Ew0,Eww0)
    vmc2_tp1 = next_period(vmc2_t, X1_tp1, X2_tp1)
    vmr2_tp1 = vmc2_tp1 + gc2_tp1 - rmc2_t
    log_N_tilde = (1-γ)*(vmc1_tp1+gc1_tp1 + 0.5*(vmc2_tp1+gc2_tp1))-log_E_exp((1-γ)*(vmc1_tp1+gc1_tp1 + 0.5*(vmc2_tp1+gc2_tp1)))
    Upsilon_2 = vmr2_tp1['ww'].reshape(n_W,n_W).T*2
    Upsilon_1 = vmr2_tp1['xw'].reshape((n_X,n_W)).T
    Upsilon_0 = vmr2_tp1['w'].T + (Upsilon_2@μ_0)

    util_sol = {'μ_0': μ_0, 
                'Upsilon_2': Upsilon_2, 
                'Upsilon_1': Upsilon_1, 
                'Upsilon_0': Upsilon_0, 
                'log_N0': log_N0,
                'log_N_tilde': log_N_tilde,
                'gc_tp1': gc_tp1,
                'gc0_tp1': gc0_tp1,
                'gc1_tp1': gc1_tp1,
                'gc2_tp1': gc2_tp1,
                'vmr1_tp1': vmr1_tp1, 
                'vmr2_tp1': vmr2_tp1}

    return util_sol


class ModelSolution(dict):
    """
    Represents the model solution.

    Attributes
    ----------

    'X0_t' : LinQuadVar
        State Variables zeroth order approximation, time t
    'X1_t' : LinQuadVar
        State Variables first order approximation, time t
    'X2_t' : LinQuadVar
        State Variables second order approximation, time t
    'X1_tp1': LinQuadVar
        State Variables first order approximation, time t+1, original measure
    'X2_tp1': LinQuadVar
        State Variables second order approximation, time t+1, original measure
    'X1_tp1_tilde': LinQuadVar
        State Variables first order approximation, time t+1, the distorted measure
    'X2_tp1_tilde': LinQuadVar
        State Variables second order approximation, time t+1, the distorted measure
    'J0_t': LinQuadVar
        Jump Variables zeroth order approximation, time t
    'J1_t': LinQuadVar
        Jump Variables first order approximation, time t
    'J2_t': LinQuadVar
        Jump Variables second order approximation, time t
    'JX0_t': LinQuadVar
        Vector of Jump and State Variables zeroth order approximation, time t
    'JX1_t': LinQuadVar
        Vector of Jump and State Variables first order approximation, time t
    'JX2_t': LinQuadVar
        Vector of Jump and State Variables second order approximation, time t
    'JX1_tp1': LinQuadVar
        Vector of Jump and State Variables first order approximation, time t+1, original measure
    'JX2_tp1': LinQuadVar
        Vector of Jump and State Variables second order approximation, time t+1, original measure
    'JX_t': LinQuadVar
        Vector of Jump and State Variables approximation, time t
    'JX_tp1': LinQuadVar
        Vector of Jump and State Variables approximation, time t+1, original measure
    'JX1_tp1_tilde': LinQuadVar
        Vector of Jump and State Variables first order approximation, time t+1, the distorted measure
    'JX2_tp1_tilde': LinQuadVar
        Vector of Jump and State Variables second order approximation, time t+1, the distorted measure
    'util_sol': dict
        Solutions of the continuation values
    'log_N0': LinQuadVar
        logN0_t+1 change of measure     
    'log_N_tilde': LinQuadVar
        logNtilde_t+1 change of measure  
    'vmr1_tp1': LinQuadVar
        First order approximation of log V_t+1 - log R_t = log V1_t+1 - log R1_t  
    'vmr2_tp1': LinQuadVar
        Second order approximation of log V_t+1 - log R_t = log V2_t+1 - log R2_t  
    'gc_tp1': LinQuadVar
        Approximation of consumption growth
    'gc0_tp1': LinQuadVar
        Zeroth order approximation of consumption growth
    'gc1_tp1': LinQuadVar
        First order approximation of consumption growth
    'gc2_tp1': LinQuadVar
        Second order approximation of consumption growth
    nit : int
        Number of iterations performed.
    second_order : bool
        If True, the solution is in second-order.
    var_shape : tuple of ints
        (n_J, n_X, n_W). Number of endogenous variables, states and shocks
        respectively.
    ss : (n_JX, ) ndarray
        Steady states.
    message : str
        Message from the solver.

    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())    