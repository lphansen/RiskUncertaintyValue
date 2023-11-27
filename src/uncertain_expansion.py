import numpy as np
import autograd.numpy as anp
import scipy as sp
import seaborn as sns
from scipy import optimize

from lin_quad_util import E, cal_E_ww, matmul, concat, next_period, kron_prod, log_E_exp, lq_sum, simulate
from utilities import mat, vec, gschur
from derivatives import compute_derivatives
from lin_quad import LinQuadVar
from elasticity import exposure_elasticity, price_elasticity

'''
This Python script provides functions to solve for the discrete-time 
dynamic macro-finance models under uncertainty, based on the small-
noise expansion method. These models feature EZ recursive preferences.

Developed and maintained by the MFR research team.
Codes updated on Jan. 26, 2023, 11:34 P.M. CT
Check and obtain the lastest version on 
https://github.com/lphansen/RiskUncertaintyValue 
'''

def uncertain_expansion(eq, ss, var_shape, args, gc, approach = '1', init_util = None, iter_tol = 1e-8, max_iter = 50):
    """
    This function solves a system with recursive utility via small-noise
    expansion, given a set of equilibrium conditions, steady states,
    log consumption growth, and other model configurations. 

    The solver returns a class storing solved variables. In particular, 
    it stores the solved variables represented by a linear or
    linear-quadratic function of the first- and second-order derivatives
    of states. It also stores laws of motion for these state derivatives.

    Parameters
    ----------
    eq_cond : callable
        Returns [Q psi_1-psi_2, phi], Q psi_1-psi_2 satisfy the 
        forward-looking equations E[N Q psi_1-psi_2]=0, and phi satisfy 
        the state equations phi=0

        ``eq_cond(Var_t, Var_tp1, W_tp1, q, mode, *args) -> (n_JX, ) ndarray``

        where Var_t and Var_tp1 are variables at time t and t+1 respectively,
        W_tp1 are time t+1 shocks, and q is perturbation parameter.
        Note that in Var_t and Var_tp1, state variables must follow jump 
        variables. The first one must be q_t or q_tp1. In the equilibrium 
        conditions, state evolution equations should follow forward-looking 
        equations.

        Under 'psi1' mode, returns psi_1, 

        ``eq_cond(Var_t, Var_tp1, W_tp1, q, mode, *args) -> (n_J, ) ndarray``

        'psi1' mode specification is necessary. Other modes are optional.
    ss : callable or (n_JX, ) ndarray
        Steady states or the function for calculating steady states.
        It follows the same order as Var_t and Var_tp1 in equilibrium 
        conditions.
    var_shape : tuple of ints
        (n_J, n_X, n_W). Number of jump variables, states and
        shocks respectively.
    args : tuple of floats/ndarray
        Model parameters, the first three elements are fixed recursive 
        utility parameters, γ, β, ρ
    gc : callable
        Function to approximate the log growth of consumption.
    approach : string
        '1': Solve the system using the approach 1 shown in 'Exploring 
            Recursive Utility' notes.
        '2': Solve the system using the approach 2 shown in 'Exploring 
            Recursive Utility' notes.
        Reference : https://larspeterhansen.org/class-notes/
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
        attributes are: JX_t the approximated variables as a linear or linear-
        quadratic function of state derivatives; X1_tp1 (and X2_tp1) the laws of
        motion for the first-order (and second-order) derivatives of states. 

    """
    n_J, n_X, n_W = var_shape
    n_JX = n_J + n_X
    γ = args[0]
    β = args[1]
    ρ = args[2]
    df, ss = take_derivatives(eq, ss, var_shape, True, args)
    H_fun_list = [(lambda y: (lambda JX_t, JX_tp1, W_tp1, q, *args: eq(JX_t, JX_tp1, W_tp1, q, 'psi1', *args)[y]))(i) for i in range(n_J)]

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

    util_sol['Σ_tilde'] = np.linalg.inv(np.eye(n_W)-0.5*util_sol['Upsilon_2']*(1-γ))
    util_sol['Γ_tilde'] = sp.linalg.sqrtm(util_sol['Σ_tilde'])
    util_sol['μ_tilde_t'] = LinQuadVar({'x':0.5*(1-γ)*util_sol['Σ_tilde']@util_sol['Upsilon_1'], 'c':(1-γ)*util_sol['Σ_tilde']@(1/(1-γ)*util_sol['μ_0']+0.5*(util_sol['Upsilon_0']-util_sol['Upsilon_2']@util_sol['μ_0']))},shape=(n_W,n_X,n_W))
            
    μ_0_series = []
    J1_t_series = []
    J2_t_series = []
    error_series = []

    i = 0
    error = 1

    while error > iter_tol and i < max_iter:
        if approach == '1':
            J1_t, JX1_t, JX1_tp1, JX1_tp1_tilde, X1_tp1, X1_tp1_tilde = first_order_expansion_approach_1(df, util_sol, var_shape, H0_t, args)
            adj = compute_adj_approach_1(H_fun_list, ss, var_shape, util_sol, JX1_t, X1_tp1, H0_t, args)
            J2_t, JX2_t, JX2_tp1, JX2_tp1_tilde, X2_tp1, X2_tp1_tilde = second_order_expansion_approach_1(df, util_sol, X1_tp1, X1_tp1_tilde, JX1_t, JX1_tp1, JX1_tp1_tilde, adj, var_shape)
        elif approach == '2':
            J1_t, JX1_t, JX1_tp1, JX1_tp1_tilde, X1_tp1, X1_tp1_tilde = first_order_expansion_approach_2(df, util_sol, var_shape, H0_t, args)
            adj = compute_adj_approach_2(H_fun_list, ss, var_shape, util_sol, JX1_t, X1_tp1, H0_t, args)
            J2_t, JX2_t, JX2_tp1, JX2_tp1_tilde, X2_tp1, X2_tp1_tilde = second_order_expansion_approach_2(df, util_sol, X1_tp1, X1_tp1_tilde, JX1_t, JX1_tp1, JX1_tp1_tilde, adj, var_shape)
        else: 
            raise ValueError('Please input approach 1 or 2 with string.')
        util_sol = solve_utility(ss, var_shape, args, X1_tp1, X2_tp1, JX1_t, JX2_t, gc)
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
                        'X0_t':X0_t, 
                        'X1_t':X1_t, 
                        'X2_t':X2_t, 
                        'X1_tp1': X1_tp1, 
                        'X2_tp1': X2_tp1,
                        'X1_tp1_tilde': X1_tp1_tilde, 
                        'X2_tp1_tilde': X2_tp1_tilde,
                        'J0_t':J0_t, 
                        'J1_t':J1_t, 
                        'J2_t':J2_t,
                        'JX0_t':JX0_t, 
                        'JX1_t':JX1_t, 
                        'JX2_t':JX2_t, 
                        'JX1_tp1': JX1_tp1, 
                        'JX2_tp1': JX2_tp1, 
                        'JX_t': JX_t, 
                        'JX_tp1': JX_tp1,
                        'JX1_tp1_tilde': JX1_tp1_tilde, 
                        'JX2_tp1_tilde': JX2_tp1_tilde, 
                        'JX_tp1_tilde': JX_tp1_tilde,
                        'util_sol':util_sol, 
                        'log_N0': util_sol['log_N0'], 
                        'log_N_tilde':util_sol['log_N_tilde'],
                        'vmr1_tp1':util_sol['vmr1_tp1'], 
                        'vmr2_tp1':util_sol['vmr2_tp1'],
                        'gc_tp1':util_sol['gc_tp1'],
                        'gc0_tp1':util_sol['gc0_tp1'],
                        'gc1_tp1':util_sol['gc1_tp1'],
                        'gc2_tp1':util_sol['gc2_tp1'],
                        'var_shape': var_shape, 
                        'args':args, 
                        'ss': ss, 
                        'second_order': True})
    
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
                             anp.atleast_1d(f(Var_t, Var_tp1, W_tp1, q, 'eq_cond', *args)),
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

def first_order_expansion_approach_1(df, util_sol, var_shape, H0, args):
    """
    Implements first-order expansion using approach 1.

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
    
    return J1_t, JX1_t, JX1_tp1, JX1_tp1_tilde, X1_tp1, X1_tp1_tilde

def compute_adj_approach_1(H_fun_list, ss, var_shape, util_sol, JX1_t, X1_tp1, H0, args):
    """
    Computes additional recursive utility adjustment using approach 1.

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

    return adj_aug

def second_order_expansion_approach_1(df, util_sol, X1_tp1, X1_tp1_tilde, JX1_t, JX1_tp1, JX1_tp1_tilde, adj, var_shape):
    """
    Implements second-order expansion using approach 1.

    """
    n_J, n_X, n_W = var_shape

    schur = schur_decomposition(-df['xtp1'], df['xt'], (n_J, n_X, n_W))
    μ_0 = util_sol['μ_0']

    Wtp1 = LinQuadVar({'w': np.eye(n_W),'c':μ_0}, (n_W, n_X, n_W), False)
    D2 = combine_second_order_terms(df, JX1_t, JX1_tp1_tilde, Wtp1)
    D2_coeff = np.block([[D2['c'], D2['x'], D2['w'], D2['xx'], D2['xw'], D2['ww']]])
    M_E_w = np.zeros([n_W,1])
    cov_w = np.eye(n_W)
    M_E_ww = cal_E_ww(M_E_w, cov_w)
    E_D2 = E(D2, M_E_w, M_E_ww)
    E_D2_coeff = np.block([[E_D2['c']+adj['c'], E_D2['x']+adj['x'], E_D2['xx']]])
    X1X1_tilde = kron_prod(X1_tp1_tilde, X1_tp1_tilde)
    M_mat = form_M0(M_E_w, M_E_ww, X1_tp1_tilde, X1X1_tilde)
    LHS = np.eye(n_J*M_mat.shape[0]) - np.kron(M_mat.T, np.linalg.solve(schur['Λ22'], schur['Λp22']))
    RHS = vec(-np.linalg.solve(schur['Λ22'], (schur['Q'].T@E_D2_coeff)[-n_J:]))
    D_tilde_vec = np.linalg.solve(LHS, RHS)
    D_tilde = mat(D_tilde_vec, (n_J, 1+n_X+n_X**2))
    G = np.linalg.solve(schur['Z21'], D_tilde)

    def solve_second_state(X1_tp1, JX1_tp1, Wtp1):

        D2 = combine_second_order_terms(df, JX1_t, JX1_tp1, Wtp1)
        D2_coeff = np.block([[D2['c'], D2['x'], D2['w'], D2['xx'], D2['xw'], D2['ww']]])

        X1X1 = kron_prod(X1_tp1, X1_tp1)
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

        return X2_tp1

    Wtp1 = LinQuadVar({'w': np.eye(n_W),'c':μ_0}, (n_W, n_X, n_W), False)
    X2_tp1_tilde = solve_second_state(X1_tp1_tilde, JX1_tp1_tilde, Wtp1)
    
    Wtp1 = LinQuadVar({'w': np.eye(n_W)}, (n_W, n_X, n_W), False)
    X2_tp1 = solve_second_state(X1_tp1, JX1_tp1, Wtp1)

    J2_t = LinQuadVar({'x2': schur['N'], 'xx': G[:, 1+n_X:1+n_X+(n_X)**2], 'x': G[:, 1:1+n_X], 'c': G[:, :1]}, (n_J, n_X, n_W), False)
    X2_t = LinQuadVar({'x2': np.eye(n_X)}, (n_X, n_X, n_W), False)
    JX2_t = concat([J2_t, X2_t])
    JX2_tp1 = next_period(JX2_t, X1_tp1, X2_tp1)
    JX2_tp1_tilde = next_period(JX2_t, X1_tp1_tilde, X2_tp1_tilde)

    return J2_t, JX2_t, JX2_tp1, JX2_tp1_tilde, X2_tp1, X2_tp1_tilde

def first_order_expansion_approach_2(df, util_sol, var_shape, H0, args):
    """
    Implements first-order expansion using approach 2.

    """
    n_J, n_X, n_W = var_shape
    n_JX = n_J + n_X
    HQ_loc_list = [i for i in range(n_J)]
    
    γ = args[0]
    ρ = args[2]
    μ_0 = util_sol['μ_0']
    Σ_tilde = util_sol['Σ_tilde']
    Γ_tilde = util_sol['Γ_tilde']
    μ_tilde_t = util_sol['μ_tilde_t']

    H_1_x = np.zeros([n_JX,n_X])
    H_1_c = np.zeros([n_JX,1])
    for i in range(len(H0)):
        H_1_x[HQ_loc_list[i]] = (ρ-1)/(1-γ)*μ_0.T@μ_tilde_t['x']*H0[i]
        H_1_c[HQ_loc_list[i]] = μ_0.T@μ_0*(ρ-1)/2/(1-γ)*H0[i]+(ρ-1)/(1-γ)*μ_0.T@(μ_tilde_t['c']-μ_0)*H0[i]
    df_adj = np.block([[np.zeros([n_JX,n_J]),H_1_x]])
    df_mix = np.block([[np.zeros([n_JX,n_J]),df['wtp1']@μ_tilde_t['x']]])

    schur = schur_decomposition(-df['xtp1'], df['xt']+df_adj+df_mix, (n_J, n_X, n_W))
    
    f_1_xtp1 = df['xtp1'][:n_J]
    f_1_wtp1 = df['wtp1'][:n_J]
    f_2_xtp1 = df['xtp1'][n_J:]
    f_2_xt_tilde = (df['xt']+df_adj+df_mix)[n_J:]
    f_2_wtp1 = df['wtp1'][n_J:]

    RHS = - (df['wtp1']@μ_tilde_t['c']+H_1_c)
    LHS = df['xtp1'] + df['xt'] + df_adj + df_mix
    D = np.linalg.solve(LHS, RHS)
    C = D[:n_J] - schur['N']@D[n_J:]

    ψ_tilde_x = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_xt_tilde@schur['N_block'])
    ψ_tilde_w = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_wtp1)@Γ_tilde
    ψ_tilde_q = D[n_J:] - ψ_tilde_x@D[n_J:]

    f_2_xt_org = df['xt'][n_J:]
    ψ_x = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_xt_org@schur['N_block'])
    ψ_w = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_wtp1)

    df_mix_org = df_mix.copy()
    df_mix_org[n_J:] = 0

    RHS_org = - (np.block([[(f_1_xtp1@schur['N_block']@ψ_w + f_1_wtp1)@μ_tilde_t['c']], [np.zeros((n_X, 1))]])+H_1_c)
    LHS_org = df['xtp1'] + df['xt'] + df_adj + df_mix_org
    D_org = np.linalg.solve(LHS_org, RHS_org)
    ψ_q = D_org[n_J:] - ψ_tilde_x@D_org[n_J:]

    X1_tp1 = LinQuadVar({'x': ψ_x, 'w': ψ_w, 'c': ψ_q}, (n_X, n_X, n_W), False)
    X1_tp1_tilde = LinQuadVar({'x': ψ_tilde_x, 'w': ψ_tilde_w, 'c': ψ_tilde_q}, (n_X, n_X, n_W), False)
    J1_t = LinQuadVar({'x': schur['N'], 'c': C}, (n_J,n_X,n_W), False)
    X1_t = LinQuadVar({'x': np.eye(n_X)}, (n_X, n_X, n_W), False)
    JX1_t = concat([J1_t, X1_t])
    JX1_tp1 = next_period(JX1_t, X1_tp1)
    JX1_tp1_tilde = next_period(JX1_t, X1_tp1_tilde)
    
    return J1_t, JX1_t, JX1_tp1, JX1_tp1_tilde, X1_tp1, X1_tp1_tilde

def compute_adj_approach_2(H_fun_list, ss, var_shape, util_sol, JX1_t, X1_tp1, H0, args):
    """
    Computes additional recursive utility adjustment using approach 2.

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
    Σ_tilde = util_sol['Σ_tilde']
    Γ_tilde = util_sol['Γ_tilde']
    μ_tilde_t = util_sol['μ_tilde_t']
    Theta_0 = [app[2]['c']+app[2]['w']@μ_0 for app in H_approx]
    Theta_1 = [app[2]['x'] for app in H_approx]
    Theta_2 = [app[2]['w'] for app in H_approx]

    lq1 = [LinQuadVar({'xx': np.kron(2*(ρ-1)/(1-γ)*μ_0.T@μ_tilde_t['x'], Theta_1[i]) + np.kron(2*(ρ-1)/(1-γ)*μ_0.T@μ_tilde_t['x'], Theta_2[i]@μ_tilde_t['x']),
                    'x': 2*(ρ-1)/(1-γ)*μ_0.T@μ_tilde_t['x']*(Theta_0[i]+Theta_2[i]@μ_tilde_t['c'] - Theta_2[i]@μ_0)+\
                            2*(ρ-1)/(1-γ)*(μ_0.T@μ_tilde_t['c'] - μ_0.T@μ_0 + 0.5*μ_0.T@μ_0) * (Theta_1[i] + Theta_2[i]@μ_tilde_t['x']),
                    'c': 2*(ρ-1)/(1-γ)*Theta_2[i]@Σ_tilde@μ_0 + 2*(ρ-1)/(1-γ)*(μ_0.T @ μ_tilde_t['c']-μ_0.T @ μ_0 + 0.5*μ_0.T @ μ_0)*(Theta_0[i]+Theta_2[i]@(μ_tilde_t['c'] - μ_0))
                    }, (1, n_X, n_W)) for i in range(len(HQ_loc_list))] 
    lq2 = [LinQuadVar({'xx': ((1-ρ)/(1-γ))**2*np.kron(μ_0.T@μ_tilde_t['x'],μ_0.T@μ_tilde_t['x'])*H0[i]+\
                            (ρ-1)/2*(μ_tilde_t['x'].T@Upsilon_2@μ_tilde_t['x']).reshape([1,n_X*n_X])*H0[i]+\
                            (ρ-1)*(μ_tilde_t['x'].T@Upsilon_1).reshape([1,n_X*n_X])*H0[i],
                    'x': ((1-ρ)/(1-γ))**2*(2*(μ_0.T@μ_tilde_t['c']-0.5*μ_0.T@μ_0)*μ_0.T@μ_tilde_t['x'])*H0[i]+\
                            (ρ-1)/2*2*((μ_tilde_t['c']-μ_0).T@Upsilon_2@μ_tilde_t['x'])*H0[i]+\
                            (ρ-1)*((μ_tilde_t['x'].T@Upsilon_0).T +(μ_tilde_t['c'].T - μ_0.T)@Upsilon_1)*H0[i],
                    'c': ((1-ρ)/(1-γ))**2*(μ_0.T@Σ_tilde@μ_0+(μ_0.T@μ_tilde_t['c']-0.5*μ_0.T@μ_0)**2)*H0[i]+\
                            (ρ-1)/2*(np.trace(Upsilon_2@Σ_tilde-Upsilon_2)+(μ_tilde_t['c']-μ_0).T@Upsilon_2@(μ_tilde_t['c']-μ_0))*H0[i]+\
                            (ρ-1)*(μ_tilde_t['c'] - μ_0).T@Upsilon_0*H0[i]
                    }, (1, n_X, n_W)) for i in range(len(HQ_loc_list))] 
    adj = [lq_sum([lq1[i], lq2[i]]) for i in range(len(HQ_loc_list))]
    adj_aug = {'xx':np.zeros([n_JX,n_X*n_X]),'x':np.zeros([n_JX,n_X]),'c':np.zeros([n_JX,1])}

    for i in range(len(HQ_loc_list)):
        adj_aug['xx'][HQ_loc_list[i]] = adj[i]['xx']
        adj_aug['x'][HQ_loc_list[i]] = adj[i]['x']
        adj_aug['c'][HQ_loc_list[i]] = adj[i]['c']
        
    return adj_aug

def second_order_expansion_approach_2(df, util_sol, X1_tp1, X1_tp1_tilde, JX1_t, JX1_tp1, JX1_tp1_tilde, adj, var_shape):
    """
    Implements second-order expansion using approach 2.

    """
    n_J, n_X, n_W = var_shape
    schur = schur_decomposition(-df['xtp1'], df['xt'], (n_J, n_X, n_W))

    μ_0 = util_sol['μ_0']
    Σ_tilde = util_sol['Σ_tilde']
    Γ_tilde = util_sol['Γ_tilde']
    μ_tilde_t = util_sol['μ_tilde_t']

    Wtp1 = LinQuadVar({'w': Γ_tilde, 'c':μ_tilde_t['c'], 'x': μ_tilde_t['x']}, (n_W, n_X, n_W), False)
    D2 = combine_second_order_terms(df, JX1_t, JX1_tp1_tilde, Wtp1)
    D2_coeff = np.block([[D2['c'], D2['x'], D2['w'], D2['xx'], D2['xw'], D2['ww']]])
    M_E_w = np.zeros([n_W,1])
    cov_w = np.eye(n_W)
    M_E_ww = cal_E_ww(M_E_w, cov_w)
    E_D2 = E(D2, M_E_w, M_E_ww)
    E_D2_coeff = np.block([[E_D2['c']+adj['c'], E_D2['x']+adj['x'], E_D2['xx']+adj['xx']]])
    X1X1_tilde = kron_prod(X1_tp1_tilde, X1_tp1_tilde)
    M_mat = form_M0(M_E_w, M_E_ww, X1_tp1_tilde, X1X1_tilde)
    LHS = np.eye(n_J*M_mat.shape[0]) - np.kron(M_mat.T, np.linalg.solve(schur['Λ22'], schur['Λp22']))
    RHS = vec(-np.linalg.solve(schur['Λ22'], (schur['Q'].T@E_D2_coeff)[-n_J:]))
    D_tilde_vec = np.linalg.solve(LHS, RHS)
    D_tilde = mat(D_tilde_vec, (n_J, 1+n_X+n_X**2))
    G = np.linalg.solve(schur['Z21'], D_tilde)

    ψ_x2 = X1_tp1['x']

    def solve_second_state(X1_tp1, JX1_tp1, Wtp1):

        D2 = combine_second_order_terms(df, JX1_t, JX1_tp1, Wtp1)
        D2_coeff = np.block([[D2['c'], D2['x'], D2['w'], D2['xx'], D2['xw'], D2['ww']]])

        X1X1 = kron_prod(X1_tp1, X1_tp1)
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
        
        ψ_tilde_xx = var_next['xx'] + xx_1 + xx_2
        ψ_tilde_xw = var_next['xw'] + xw_1
        ψ_tilde_xq = var_next['x'] + x_1 + x_2
        ψ_tilde_ww = var_next['ww'] + ww_1
        ψ_tilde_wq = var_next['w'] + w_1
        ψ_tilde_qq = var_next['c'] + c_1 + c_2

        X2_tp1 = LinQuadVar({'x2': ψ_x2, 'xx': ψ_tilde_xx, 'xw': ψ_tilde_xw, 'ww': ψ_tilde_ww, 'x': ψ_tilde_xq, 'w': ψ_tilde_wq, 'c': ψ_tilde_qq}, (n_X, n_X, n_W), False)

        return X2_tp1

    Wtp1 = LinQuadVar({'w': Γ_tilde, 'c':μ_tilde_t['c'], 'x': μ_tilde_t['x']}, (n_W, n_X, n_W), False)
    X2_tp1_tilde = solve_second_state(X1_tp1_tilde, JX1_tp1_tilde, Wtp1)

    Wtp1 = LinQuadVar({'w': np.eye(n_W)}, (n_W, n_X, n_W), False)
    X2_tp1 = solve_second_state(X1_tp1, JX1_tp1, Wtp1)

    J2_t = LinQuadVar({'x2': schur['N'], 'xx': G[:, 1+n_X:1+n_X+(n_X)**2], 'x': G[:, 1:1+n_X], 'c': G[:, :1]}, (n_J, n_X, n_W), False)
    X2_t = LinQuadVar({'x2': np.eye(n_X)}, (n_X, n_X, n_W), False)
    JX2_t = concat([J2_t, X2_t])
    JX2_tp1 = next_period(JX2_t, X1_tp1, X2_tp1)
    JX2_tp1_tilde = next_period(JX2_t, X1_tp1_tilde, X2_tp1_tilde)

    return J2_t, JX2_t, JX2_tp1, JX2_tp1_tilde, X2_tp1, X2_tp1_tilde

def schur_decomposition(df_tp1, df_t, var_shape):
    
    n_J, n_X, n_W = var_shape
    Λp, Λ, a, b, Q, Z = gschur(df_tp1, df_t)
    Λp22 = Λp[-n_J:, -n_J:]
    Λ22 = Λ[-n_J:, -n_J:]
    Z21 = Z.T[-n_J:, :n_J]
    Z22 = Z.T[-n_J:, n_J:]
    N = -np.linalg.solve(Z21, Z22)
    N_block = np.block([[N], [np.eye(n_X)]])
    schur_decomposition = { 
                        'N':N,
                        'N_block':N_block,
                        'Λp':Λp,
                        'Λ':Λ,
                        'Q':Q,
                        'Z':Z,
                        'Λp22':Λp22,
                        'Λ22':Λ22,
                        'Z21':Z21,
                        'Z22':Z22}
    
    return schur_decomposition

def combine_second_order_terms(df, JX1_t, JX1_tp1, Wtp1):
    """
    This function combines terms from second-order f, except those for JX2_t
    and JX2_{t+1}. 

    Parameters
    ----------
    df : dict
        Partial derivatives of f.
    JX1_t : LinQuadVar
        Vector of Jump and State Variables first order approximation, time t
    JX1_tp1 : LinQuadVar
        Vector of Jump and State Variables first order approximation, time t+1
    W_tp1 : LinQuadVar
        Vector of shocks

    Returns
    -------
    res : LinQuadVar
        Combined second-order terms (except JX2_t and JX2_{t+1}) from df.

    """
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
    """
    Get M0_mat, which satisfies E[M0 A [1 ztp1 ztp1ztp1]] = A M0_mat[1 zt ztzt]

    """
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

    Parameters
    ----------
    fun : callable
        Returns the variable to be approximated as a function of state and jump variables.
    ss : callable
        Steady states of state and jump variables 
        If `Var_t` and `Var_tp1` contain `q_t` or `q_tp1`, the output of the steady state 
        function needs to be augmented with a float 0 as the first component, which 
        is the steady state of `q_t`.
    var_shape : tuple of ints
        (1, n_X, n_W)
    JX1_t : LinQuadVar
        Vector of first order expansion results for state and jump variables.
    JX2_t : LinQuadVar
        Vector of second order expansion results for state and jump variables.
    X1_tp1 : LinQuadVar
        Vector of first order expansion results for state evolution equations.
    X2_tp1 : LinQuadVar
        Vector of second order expansion results for state evolution equations.
    args : tuple of floats/ndarray
        Model parameters, the first three elements are fixed recursive 
        utility parameters, γ, β, ρ
    second_order : Boolean
        If `True`, return approximated results, zeroth order, first order, and second 
            order results.
        If `False`, return approximated results, zeroth order, and first order results.
    zeroth_order : Boolean
        If `True`, return zeroth order approximated results.
        If `False`, return results depend on the option of `second_order`.

    Return
    ----------
    output: float / tuple
        If `zeroth_order` is `True`: 
            return zeroth order approximated result, float.
        If `zeroth_order` is `False` and `second_order` is `False`: 
            return a tuple containing approximated results, zeroth order, and first order 
            results. Approximated results and first order results are LinQuadVar. Zeroth 
            order approximated result is float.
        If `zeroth_order` is `False` and `second_order` is `True`: 
            return a tuple containing approximated results, zeroth order, first order, and
            second order results. Approximated results, first order results, and second order
            results are LinQuadVar. Zeroth order approximated result is float.
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

def solve_utility(ss, var_shape, args, X1_tp1, X2_tp1, JX1_t, JX2_t, gc_tp1_fun, tol = 1e-10):
    """
    Solves continuation values and forms approximation of change of measure

    Parameters
    ----------
    ss : callable
        Steady states of q_t, state and jump variables, 
        The default input of the `Var_t` and `Var_tp1` in `gc_tp1_fun` include
        `q_t` and `q_tp1`, the output of the steady state function is augmented 
        with a float 0 as the first component, which is the steady state of q_t.
    var_shape : tuple of ints
        (n_J, n_X, n_W)
    args : tuple of floats/ndarray
        Model parameters, the first three elements are fixed recursive 
        utility parameters, γ, β, ρ
    X1_tp1 : LinQuadVar
        Vector of first order expansion results for state evolution equations.
    X2_tp1 : LinQuadVar
        Vector of second order expansion results for state evolution equations.
    JX1_t : LinQuadVar
        Vector of first order expansion results for state and jump variables.
    JX2_t : LinQuadVar
        Vector of second order expansion results for state and jump variables.
    gc_tp1_fun : callable
        Returns the consumption growth as a function of state and jump variables.
    tol : float
        The tolerance for the eqution solver.
    
    Return
    ----------
    util_sol : dict
        A dictionary contains solved continuation values.
        μ_0 : ndarray
            The mean of shock under the change of measure N0_{t+1}
        Upsilon_2 : ndarray
            Transformed `ww` coeffients of V2_{t+1}-R2_{t}
        Upsilon_1 : ndarray
            Transformed `xw` coeffients of V2_{t+1}-R2_{t}
        Upsilon_0 : ndarray
            Transformed `w` coeffients of V2_{t+1}-R2_{t}
        log_N0 : LinQuadVar
            Log Change of measure N0_{t+1}
        log_N_tilde : LinQuadVar
            Log Change of measure N_{t+1}_tilde
        gc_tp1 : LinQuadVar
            Log growth of consumption
        gc0_tp1 : LinQuadVar
            Zeroth order expansion of log growth of consumption
        gc1_tp1 : LinQuadVar
            First order expansion of log growth of consumption
        gc2_tp1 : LinQuadVar
            Second order expansion of log growth of consumption
        vmc1_t : LinQuadVar
            First order expansion of V_{t}-C_{t}
        vmc2_t : LinQuadVar
            Second order expansion of V_{t}-C_{t}
        rmc1_t : LinQuadVar
            First order expansion of R_{t}-C_{t}
        rmc2_t : LinQuadVar
            Second order expansion of R_{t}-C_{t}
        vmr1_tp1 : LinQuadVar
            First order expansion of V_{t+1}-R_{t}
        vmr2_tp1 : LinQuadVar
            Second order expansion of V_{t+1}-R_{t}
        Σ_tilde : ndarray
            The covariance matrix of shock under the change of measure N_{t+1}_tilde
        Γ_tilde : ndarray
            The covariance matrix square root of shock under the change of measure N_{t+1}_tilde
        μ_tilde_t : LinQuadVar
            The mean of shock under the change of measure N_{t+1}_tilde
    """
    _, n_X, n_W = var_shape
    γ = args[0]
    β = args[1]
    ρ = args[2]
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
    Ew0 = μ_0.copy()
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
    rmc2_t = E(next_period(vmc2_t, X1_tp1, X2_tp1) + gc2_tp1, Ew0, Eww0)
    vmc2_tp1 = next_period(vmc2_t, X1_tp1, X2_tp1)
    vmr2_tp1 = vmc2_tp1 + gc2_tp1 - rmc2_t
    log_N_tilde = (1-γ)*(vmc1_tp1+gc1_tp1 + 0.5*(vmc2_tp1+gc2_tp1))-log_E_exp((1-γ)*(vmc1_tp1+gc1_tp1 + 0.5*(vmc2_tp1+gc2_tp1)))
    Upsilon_2 = vmr2_tp1['ww'].reshape(n_W,n_W).T*2
    Upsilon_1 = vmr2_tp1['xw'].reshape((n_X,n_W)).T
    Upsilon_0 = vmr2_tp1['w'].T + (Upsilon_2@μ_0)
    Σ_tilde = np.linalg.inv(np.eye(n_W)-0.5*Upsilon_2*(1-γ))
    Γ_tilde = sp.linalg.sqrtm(Σ_tilde)
    μ_tilde_t = LinQuadVar({'x':0.5*(1-γ)*Σ_tilde@Upsilon_1, 'c':(1-γ)*Σ_tilde@(1/(1-γ)*μ_0+0.5*(Upsilon_0-Upsilon_2@μ_0))},shape=(n_W,n_X,n_W))

    vmc0_t = (np.log(1-β) - np.log(1-β*np.exp((1-ρ)*gc0_tp1)))/(1-ρ)
    rmc0_t = vmc0_t + gc0_tp1

    vmc_t = LinQuadVar({'c':np.array([[vmc0_t]])},(1, n_X, n_W))+vmc1_t+0.5*vmc2_t
    rmc_t = LinQuadVar({'c':np.array([[rmc0_t]])},(1, n_X, n_W))+rmc1_t+0.5*rmc2_t

    util_sol = {'μ_0': μ_0, 
                'λ' : λ,
                'Upsilon_2': Upsilon_2, 
                'Upsilon_1': Upsilon_1, 
                'Upsilon_0': Upsilon_0, 
                'log_N0': log_N0,
                'log_N_tilde': log_N_tilde,
                'gc_tp1': gc_tp1,
                'gc0_tp1': gc0_tp1,
                'gc1_tp1': gc1_tp1,
                'gc2_tp1': gc2_tp1,
                'vmc0_t':vmc0_t,
                'rmc0_t':rmc0_t,
                'vmc1_t': vmc1_t,
                'vmc2_t': vmc2_t,
                'rmc1_t': rmc1_t,
                'rmc2_t': rmc2_t,
                'vmc_t':vmc_t,
                'rmc_t':rmc_t,
                'vmr1_tp1': vmr1_tp1, 
                'vmr2_tp1': vmr2_tp1,
                'Σ_tilde':Σ_tilde,
                'Γ_tilde':Γ_tilde,
                'μ_tilde_t':μ_tilde_t}

    return util_sol


class ModelSolution(dict):
    """
    Represents the model solution.

    Attributes
    ----------

    X0_t : LinQuadVar
        State Variables zeroth order approximation, time t
    X1_t : LinQuadVar
        State Variables first order approximation, time t
    X2_t : LinQuadVar
        State Variables second order approximation, time t
    X1_tp1 : LinQuadVar
        State Variables first order approximation, time t+1, original measure
    X2_tp1 : LinQuadVar
        State Variables second order approximation, time t+1, original measure
    X1_tp1_tilde : LinQuadVar
        State Variables first order approximation, time t+1, the distorted measure
    X2_tp1_tilde : LinQuadVar
        State Variables second order approximation, time t+1, the distorted measure
    J0_t : LinQuadVar
        Jump Variables zeroth order approximation, time t
    J1_t : LinQuadVar
        Jump Variables first order approximation, time t
    J2_t : LinQuadVar
        Jump Variables second order approximation, time t
    JX0_t : LinQuadVar
        Vector of Jump and State Variables zeroth order approximation, time t
    JX1_t : LinQuadVar
        Vector of Jump and State Variables first order approximation, time t
    JX2_t : LinQuadVar
        Vector of Jump and State Variables second order approximation, time t
    JX1_tp1 : LinQuadVar
        Vector of Jump and State Variables first order approximation, time t+1, original measure
    JX2_tp1 : LinQuadVar
        Vector of Jump and State Variables second order approximation, time t+1, original measure
    JX_t : LinQuadVar
        Vector of Jump and State Variables approximation, time t
    JX_tp1 : LinQuadVar
        Vector of Jump and State Variables approximation, time t+1, original measure
    JX1_tp1_tilde : LinQuadVar
        Vector of Jump and State Variables first order approximation, time t+1, the distorted measure
    JX2_tp1_tilde : LinQuadVar
        Vector of Jump and State Variables second order approximation, time t+1, the distorted measure
    util_sol : dict
        Solutions of the continuation values
    log_N0 : LinQuadVar
        logN0_t+1 change of measure     
    log_N_tilde : LinQuadVar
        logNtilde_t+1 change of measure  
    vmr1_tp1 : LinQuadVar
        First order approximation of log V_t+1 - log R_t = log V1_t+1 - log R1_t  
    vmr2_tp1 : LinQuadVar
        Second order approximation of log V_t+1 - log R_t = log V2_t+1 - log R2_t  
    gc_tp1 : LinQuadVar
        Approximation of consumption growth
    gc0_tp1 : LinQuadVar
        Zeroth order approximation of consumption growth
    gc1_tp1 : LinQuadVar
        First order approximation of consumption growth
    gc2_tp1 : LinQuadVar
        Second order approximation of consumption growth
    second_order : bool
        If True, the solution is in second-order.
    var_shape : tuple of ints
        (n_J, n_X, n_W). Number of jump variables, states and shocks
        respectively.
    ss : (n_JX, ) ndarray
        Steady states.
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

    def simulate(self, Ws):
        """
        Simulates stochastiic path for JX by generating iid normal shocks,
        or deterministic path for JX by generating zero-valued shocks.

        Parameters
        ----------
        Ws : (T, n_W) ndarray
            n_W dimensional shocks for T periods to be fed into the system.        
        T : int
            Time horizon.

        Returns
        -------
        sim_result : (T, n_J) ndarray
            Simulated Ys.

        """
        n_J, n_X, n_W = self.var_shape

        sim_result = simulate(self.JX_t,
                              self.X1_tp1,
                              self.X2_tp1,
                              Ws)

        return sim_result

    def approximate(self, fun, args=()):
        """
        Approximates function given state evolutions and jump varibles

        Parameters
        ----------
        fun : callable
            Returns the variable to be approximated as a function of state and jump variables.
        args : tuple of floats/ndarray
            Model parameters, the first three elements are fixed recursive 
            utility parameters, γ, β, ρ

        Return
        ----------
        output: tuple
            return a tuple containing approximated results, zeroth order, first order, and
            second order results. Approximated results, first order results, and second order
            results are LinQuadVar. Zeroth order approximated result is float.
        """

        _, n_X, n_W = self.var_shape
    
        W_0 = np.zeros(n_W)
        q_0 = 0.
        
        dfun = compute_derivatives(f=lambda JX_t, JX_tp1, W_tp1, q:
                                    anp.atleast_1d(fun(JX_t, JX_tp1, W_tp1, q, *args)),
                                    X=[self.ss, self.ss, W_0, q_0],
                                    second_order=True)
        
        fun_zero_order = fun(self.ss, self.ss, W_0, q_0, *args)
        JX1_tp1 = next_period(self.JX1_t, self.X1_tp1)
        fun_first_order = matmul(dfun['xtp1'], JX1_tp1)\
            + matmul(dfun['xt'], self.JX1_t)\
            + LinQuadVar({'w': dfun['wtp1'], 'c': dfun['q'].reshape(-1, 1)},
                        (1, n_X, n_W), False)
        fun_approx = fun_zero_order + fun_first_order

        JX2_tp1 = next_period(self.JX2_t, self.X1_tp1, self.X2_tp1)
        Wtp1 = LinQuadVar({'w': np.eye(n_W)}, (n_W, n_X, n_W), False)

        temp1 = combine_second_order_terms(dfun, self.JX1_t, JX1_tp1, Wtp1)
        temp2 = matmul(dfun['xt'], self.JX2_t)\
            + matmul(dfun['xtp1'], JX2_tp1)
        fun_second_order = temp1 + temp2
        fun_approx = fun_approx + fun_second_order*0.5

        return fun_approx, fun_zero_order, fun_first_order, fun_second_order

    def elasticities(self, log_SDF_ex, args=None, locs=None, T=400, shock=0, percentile=0.5):
        """
        Computes shock exposure and price elasticities for JX.

        Parameters
        ----------
        log_SDF_ex : callable
            Log stochastic discount factor exclusive of the
            change of measure N and Q.

            ``log_SDF_ex(X_t, X_tp1, W_tp1, q, *args) -> scalar``
        args : tuple of floats/ints
            Additional parameters passed to log_SDF_ex.
        locs : None or tuple of ints
            Positions of variables of interest.
            If None, all variables will be selected.
        T : int
            Time horizon.
        shock : int
            Position of the initial shock, starting from 0.
        percentile : float
            Specifies the percentile of the elasticities.

        Returns
        -------
        elasticities : (T, n_J) ndarray
            Elasticities for M.

        """
        n_J, n_X, n_W = self.var_shape
        if args == None:
            args = self.args 
        log_SDF_ex = self.approximate(log_SDF_ex, args)
        ρ = args[2]
        self.log_SDF = log_SDF_ex[0] + (ρ-1)*self.util_sol['vmr1_tp1'] + 0.5*(ρ-1)*self.util_sol['vmr2_tp1']+self.log_N_tilde
        JX_growth = self.JX_tp1 - self.JX_t
        JX_growth_list = JX_growth.split()
        if locs is not None:
            JX_growth_list = [JX_growth_list[i] for i in locs]
        exposure_all = np.zeros((T, len(JX_growth_list)))
        price_all = np.zeros((T, len(JX_growth_list)))
        for i, x in enumerate(JX_growth_list):
            exposure = exposure_elasticity(x,
                                           self.JX1_tp1[n_J: n_J+n_X],
                                           self.JX2_tp1[n_J: n_J+n_X],
                                           T,
                                           shock,
                                           percentile)
            price = price_elasticity(x,
                                     self.log_SDF,
                                     self.JX1_tp1[n_J: n_J+n_X],
                                     self.JX2_tp1[n_J: n_J+n_X],
                                     T,
                                     shock,
                                     percentile)
            exposure_all[:, i] = exposure.reshape(-1)
            price_all[:, i] = price.reshape(-1)
        return exposure_all, price_all

    def IRF(self, T, shock):
        """
        Computes impulse response functions for each component in JX to each shock.

        Parameters
        ----------
        T : int
            Time horizon.
        shock : int
            Position of the initial shock, starting from 0.

        Returns
        -------
        states : (T, n_X) ndarray
            IRF of all state variables to the designated shock.
        controls : (T, n_J) ndarray
            IRF of all control variables to the designated shock.

        """
    
        n_J, n_X, n_W = self.var_shape
        # Build the first order impulse response for each of the shocks in the system
        states1 = np.zeros((T, n_X))
        controls1 = np.zeros((T, n_J))
        
        W_0 = np.zeros(n_W)
        W_0[shock] = 1
        B = self.JX1_tp1['w'][n_J:,:]
        F = self.JX1_tp1['w'][:n_J,:]
        A = self.JX1_tp1['x'][n_J:,:]
        D = self.JX1_tp1['x'][:n_J,:]
        N = self.JX1_t['x'][:n_J,:]
        states1[0, :] = B@W_0
        controls1[0, :] = F@W_0
        for i in range(1,T):
            states1[i, :] = A@states1[i-1, :]
            controls1[i, :] = D@states1[i-1, :]
        if not self.second_order:
            states = states1
            controls = controls1
        else:
            # Define the evolutions of the states in second order
            # X_{t+1}^2 = Ψ_0 + Ψ_1 @ X_t^1 + Ψ_2 @ W_{t+1} + Ψ_3 @ X_t^2 +
            # Ψ_4 @ (X_t^1 ⊗ X_t^1) + Ψ_5 @ (X_t^1 ⊗ W_{t+1}) + Ψ_6 @ (W_{t+1} ⊗ W_{t+1})
            Ψ_0 = self.JX2_tp1['c'][n_J:,:]
            Ψ_1 = self.JX2_tp1['x'][n_J:,:]
            Ψ_2 = self.JX2_tp1['w'][n_J:,:]
            Ψ_3 = self.JX2_tp1['x2'][n_J:,:]
            Ψ_4 = self.JX2_tp1['xx'][n_J:,:]
            Ψ_5 = self.JX2_tp1['xw'][n_J:,:]
            Ψ_6 = self.JX2_tp1['ww'][n_J:,:]
            
            Φ_0 = self.JX2_tp1['c'][:n_J,:]
            Φ_1 = self.JX2_tp1['x'][:n_J,:]
            Φ_2 = self.JX2_tp1['w'][:n_J,:]
            Φ_3 = self.JX2_tp1['x2'][:n_J,:]
            Φ_4 = self.JX2_tp1['xx'][:n_J,:]
            Φ_5 = self.JX2_tp1['xw'][:n_J,:]
            Φ_6 = self.JX2_tp1['ww'][:n_J,:]
            
            states2 = np.zeros((T, n_X))
            controls2 = np.zeros((T, n_J))
            X_1_0 = np.zeros(n_X)

            # Build the second order impulse response for each shock
            W_0 = np.zeros(n_W)
            W_0[shock] = 1
            states2[0, :] = Ψ_2 @ W_0 + Ψ_5 @ np.kron(X_1_0, W_0) + Ψ_6 @ np.kron(W_0, W_0)
            controls2[0, :] = Φ_2 @ W_0 + Φ_5 @ np.kron(X_1_0, W_0) + Φ_6 @ np.kron(W_0, W_0)
            for i in range(1,T):
                states2[i, :] = Ψ_1 @ states1[i-1, :] + Ψ_3 @ states2[i-1, :] + \
                    Ψ_4 @ np.kron(states1[i-1, :], states1[i-1, :])
                controls2[i, :] = Φ_1 @ states1[i-1, :] + Φ_3 @ states2[i-1, :] + \
                    Φ_4 @ np.kron(states1[i-1, :], states1[i-1, :])
            states = states1 + .5 * states2
            controls = controls1 + .5 * controls2
            
        return states, controls

    
    