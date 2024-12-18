import numpy as np
import autograd.numpy as anp
import sympy as sp
from sympy import lambdify
import scipy as sci
import seaborn as sns
import pickle
import copy
from scipy import optimize
from scipy.optimize import fsolve

from lin_quad_util import E, cal_E_ww, matmul, concat, next_period, kron_prod, log_E_exp, lq_sum, simulate
from utilities import mat, vec, gschur
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
def split_variables(variables,var_shape):
    n_J, n_X, n_W = var_shape
    n_G = n_J+n_X+1
    n_B = n_G + 2
    # n_G = n_J+2  #index of growth variable
    # n_B = n_G+n_X+1 #index of shock 1
    # n_Q = n_G + n_X

    recursive_variables = variables[:2]
    main_variables = variables[2:n_G+1]
    q_variable = variables[n_G+1]
    shock_variables = variables[n_B:n_B+n_W]

    return recursive_variables,main_variables,q_variable,shock_variables

def uncertain_expansion(control_variables, state_variables, shock_variables, variables, variables_tp1,
                        output_constraint, capital_growth, state_equations, initial_guess, parameter_names,
        args, approach = '1', init_util = None, iter_tol = 1e-8, max_iter = 50, savepath=None):
    """
    This function solves a system with recursive utility via small-noise
    expansion, given a set of equilibrium conditions, steady states,
    log consumption growth, and other model configurations. 

    The solver returns a class storing solved variables. In particular, 
    it stores the solved variables represented by a linear or
    linear-quadratic function of the first- and second-order derivatives
    of states. It also stores laws of motion for these state derivatives.

    Returns
    -------
    res : ModelSolution
        The model solution represented as a ModelSolution object. Important
        attributes are: JX_t the approximated variables as a linear or linear-
        quadratic function of state derivatives; X1_tp1 (and X2_tp1) the laws of
        motion for the first-order (and second-order) derivatives of states. 

    """

    approach = '1'
    n_X = len(state_variables)+1
    n_W = len(shock_variables)
    n_J = len(control_variables) + n_X + 1
    n_JX = n_J + n_X
    var_shape = [n_J, n_X, n_W]

    γ = get_parameter_value('gamma', parameter_names, args)
    β = get_parameter_value('beta', parameter_names, args)
    ρ = get_parameter_value('rho', parameter_names, args)

    ss_equations, ss_variables, ss_variables_tp1, H, L = compile_equations(
        parameter_names = parameter_names,
        variables=variables,  # List of variables, such as [imk_t, Z_t, Y_t, ...]
        variables_tp1=variables_tp1,  # List of variables, such as [imk_t, Z_t, Y_t, ...]
        state_variables=state_variables,  # List of state variables, such as [imk_t, Z_t, Y_t, ...]
        control_variables=control_variables,  # List of control variables, such as [imk_t, Z_t, Y_t, ...]
        output_constraint=output_constraint,  # Defined symbolic equation
        capital_growth=capital_growth,  # Defined symbolic equation
        state_equations=state_equations, # Initial guess for solving the system
        var_shape=var_shape  # Dimensionality of shocks
    )
    # print(ss_equations)
    # print(ss_variables)

    # print(ss_variables)
    # print(type(ss_variables))
    ss_onecap_3d = generate_ss_function(ss_equations, ss_variables, ss_variables_tp1, initial_guess, var_shape, parameter_names)
    recursive_ss = ss_onecap_3d(args,return_recursive=True)[:2]

    # eq_onecap_3d = generate_evaluation_function(ss_equations, ss_variables, ss_variables_tp1, H, L, var_shape, parameter_names)
    # eq_onecap_3d(ss_onecap_3d(args),ss_onecap_3d(args),np.zeros(n_W),0.,'None',recursive_ss,args)

    f_cmk_app = func_cmk_app(ss_variables,ss_variables_tp1,var_shape)
    f_gc = func_gc(ss_variables,ss_variables_tp1,var_shape)
    f_gk = func_gk(ss_variables,ss_variables_tp1,var_shape)

    

    df, ss = take_derivatives(ss_equations[1:], ss_onecap_3d, ss_variables, ss_variables_tp1, parameter_names, var_shape, args, recursive_ss,second_order=True)
    
    
    ## Zeroth-order steady state
    ## LinQuadVar is a class that represents a linear or linear-quadratic function of the first- and second-order derivatives of states.
    ## X = X0_t + qX1_t + 0.5q^2X2_t where X0_t is computed by the steady-state and X1_t = X1_t, X2_t = X2_t by definition
    n_J, n_X, n_W = var_shape
    n_JX = n_J + n_X
    X0_t = LinQuadVar({'c': ss[n_J:].reshape(-1, 1)}, (n_X, n_X, n_W), False)
    J0_t = LinQuadVar({'c': ss[:n_J].reshape(-1, 1)}, (n_J, n_X, n_W), False)
    JX0_t = concat([J0_t, X0_t])
    X1_t = LinQuadVar({'x': np.eye(n_X)}, (n_X, n_X, n_W), False)
    X2_t = LinQuadVar({'x2': np.eye(n_X)}, (n_X, n_X, n_W), False)

    H_fun_list = H
    H0_t = [approximate_fun(H_fun, ss, ss_variables, ss_variables_tp1, parameter_names, args, var_shape, None, None, None, None, recursive_ss, second_order = False, zeroth_order = True) for H_fun in H_fun_list]

    L_fun_list = L
    L0_t = [approximate_fun(L_fun, ss, ss_variables, ss_variables_tp1, parameter_names, args, var_shape, None, None, None, None, recursive_ss, second_order = False, zeroth_order = True) for L_fun in L_fun_list]

    ## Initialize the change of measure
    if init_util == None:
        util_sol = {'μ_0': np.zeros([n_W,1]),
                    'Upsilon_2':np.zeros([n_W,n_W]),
                    'Upsilon_1':np.zeros([n_W,n_X]),
                    'Upsilon_0':np.zeros([n_W,1]),
                    'rmk1_t':LinQuadVar({'c':np.zeros([1,1]), 'x':np.zeros([1,n_X])},(1,n_X,n_W)),
                    'rmv1_t':LinQuadVar({'c':np.zeros([1,1]), 'x':np.zeros([1,n_X])},(1,n_X,n_W)),
                    'rmv2_t':LinQuadVar({'c':np.zeros([1,1]), 'x':np.zeros([1,n_X])},(1,n_X,n_W)),
                    'vmk1_t':LinQuadVar({'c':np.zeros([1,1]), 'x':np.zeros([1,n_X])},(1,n_X,n_W)),
                    'vmk2_t':LinQuadVar({'c':np.zeros([1,1]), 'x':np.zeros([1,n_X])},(1,n_X,n_W))}
    else:
        util_sol = init_util

    util_sol['Σ_tilde'] = np.linalg.inv(np.eye(n_W)-0.5*util_sol['Upsilon_2']*(1-γ))
    util_sol['Γ_tilde'] = sci.linalg.sqrtm(util_sol['Σ_tilde'])
    util_sol['μ_tilde_t'] = LinQuadVar({'x':0.5*(1-γ)*util_sol['Σ_tilde']@util_sol['Upsilon_1'], 'c':(1-γ)*util_sol['Σ_tilde']@(1/(1-γ)*util_sol['μ_0']+0.5*(util_sol['Upsilon_0']-util_sol['Upsilon_2']@util_sol['μ_0']))},shape=(n_W,n_X,n_W))
        
    μ_0_series = []
    J1_t_series = []
    J2_t_series = []
    error_series = []

    i = 0
    error = 1

    ## Iterate change of measure and model solutions to convergence
    while error > iter_tol and i < max_iter:
        # if len(error_series)>1:
        #     # if error_series[-1] > error_series[-2]:  # Compare the last two elements
        #     return first_order_expansion_approach_1(df, ss, util_sol, var_shape, H0_t, L0_t, recursive_ss, parameter_names, args, return_df=True)
    
        # if len(error_series)>10:
        #     if error_series[-1] > error_series[-2]:  # Compare the last two elements
        #         return first_order_expansion_approach_1(df, ss, util_sol, var_shape, H0_t, L0_t, recursive_ss, parameter_names, args, return_df=True)
    
        if approach == '1':
            J1_t, JX1_t, JX1_tp1, JX1_tp1_tilde, X1_tp1, X1_tp1_tilde = first_order_expansion_approach_1(df, ss, util_sol, var_shape, H0_t, L0_t, recursive_ss, parameter_names,args)
            adj = compute_adj_approach_1(H_fun_list, H, L, ss, ss_variables, ss_variables_tp1, var_shape, util_sol, JX1_t, X1_tp1, recursive_ss, parameter_names,args)
            J2_t, JX2_t, JX2_tp1, JX2_tp1_tilde, X2_tp1, X2_tp1_tilde = second_order_expansion_approach_1(df, util_sol, X1_tp1, X1_tp1_tilde, JX1_t, JX1_tp1, JX1_tp1_tilde, adj, var_shape,parameter_names)
        elif approach == '2':
            J1_t, JX1_t, JX1_tp1, JX1_tp1_tilde, X1_tp1, X1_tp1_tilde = first_order_expansion_approach_2(df, util_sol, var_shape, H0_t, args)
            adj = compute_adj_approach_2(H_fun_list, ss, var_shape, util_sol, JX1_t, X1_tp1, H0_t, args)
            J2_t, JX2_t, JX2_tp1, JX2_tp1_tilde, X2_tp1, X2_tp1_tilde = second_order_expansion_approach_2(df, util_sol, X1_tp1, X1_tp1_tilde, JX1_t, JX1_tp1, JX1_tp1_tilde, adj, var_shape)
        else: 
            raise ValueError('Please input approach 1 or 2 with string.')
        util_sol = solve_utility(ss,ss_variables, ss_variables_tp1, parameter_names, var_shape, 
                  args, X1_tp1, X2_tp1, JX1_t, JX2_t, f_cmk_app, f_gk, f_gc, recursive_ss, tol = 1e-10)
        μ_0_series.append(util_sol['μ_0'])
        J1_t_series.append(J1_t(*(np.ones([n_X,1]),np.ones([n_X,1]),np.ones([n_X,1]))))
        J2_t_series.append(J2_t(*(np.ones([n_X,1]),np.ones([n_X,1]),np.ones([n_X,1]))))

        if i > 0:
            error_1 = np.max(np.abs(μ_0_series[i] - μ_0_series[i-1]))
            print('Iteration {}: mu_0 error = {:.9g}'.format(i, error_1))
            error_2 = np.max(np.abs(J1_t_series[i] - J1_t_series[i-1])) 
            print('Iteration {}: J1 error = {:.9g}'.format(i, error_2))
            error_3 = np.max(np.abs(J2_t_series[i] - J2_t_series[i-1])) 
            print('Iteration {}: J2 error = {:.9g}'.format(i, error_3))
            error = np.max([error_1, error_2, error_3])
            print('Iteration {}: error = {:.9g}'.format(i, error))
            error_series.append(error)
        
        i+=1

    ## Formulate the solution
    JX_t = JX0_t + JX1_t + 0.5*JX2_t
    JX_tp1 = next_period(JX_t, X1_tp1, X2_tp1)
    JX_tp1_tilde = next_period(JX_t, X1_tp1_tilde, X2_tp1_tilde)

    result = ModelSolution({
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
                        'parameter_names':parameter_names,
                        'ss': ss, 
                        'ss_equations': ss_equations,
                        'second_order': True})
    if savepath:
        with open(savepath, 'wb') as file:
            pickle.dump(result, file)

    return result
    
def take_derivatives(eq, ss, ss_variables, ss_variables_tp1, parameter_names, var_shape, args, recursive_ss,second_order):
    from derivatives import compute_derivatives
    """
    Take first- or second-order derivatives.

    """
    n_J, n_X, n_W = var_shape
    n_JX = n_J + n_X
    W_0 = np.zeros(n_W)
    q_0 = 0.
    ss = ss(args)
    X=[ss, ss, W_0, q_0]
    
    dfq = compute_derivatives(eq, ss_variables, ss_variables_tp1, parameter_names, X, args, var_shape, recursive_ss,second_order=second_order)
    
           
    df = {'xt':dfq['xt'][:,:],\
        'xtp1':dfq['xtp1'][:,:],\
        'wtp1':dfq['wtp1'],\
        'q':dfq['q'],\
        'xtxt':dfq['xtxt'][:,:],\
        'xtxtp1':dfq['xtxtp1'][:,:],\
        'xtwtp1':dfq['xtwtp1'][:,:],\
        'xtq':dfq['xtq'][:,:],\
        'xtp1xtp1':dfq['xtp1xtp1'][:,:],\
        'xtp1wtp1':dfq['xtp1wtp1'][:,:],\
        'xtp1q':dfq['xtp1q'][:,:],\
        'wtp1wtp1':dfq['wtp1wtp1'],\
        'wtp1q':dfq['wtp1q'],\
        'qq':dfq['qq']}     
        
                           
    return df, ss

def first_order_expansion_approach_1(df, ss, util_sol, var_shape, H0, L0, recursive_ss, parameter_names, args,return_df=False):
    """
    Implements first-order expansion using approach 1.

    """
    n_J, n_X, n_W = var_shape
    n_JX = n_J + n_X
    HQ_loc_list = [i for i in range(n_J)]
    
    
    γ = get_parameter_value('gamma', parameter_names, args)
    β = get_parameter_value('beta', parameter_names, args)
    ρ = get_parameter_value('rho', parameter_names, args)

    μ_0 = util_sol['μ_0']

    # H_1 = np.zeros([n_JX,1])
    # HQ_loc_list = [i for i in range(n_J)]
    # for i in range(len(H0)):
    #     H_1[HQ_loc_list[i]] = μ_0.T@μ_0*(ρ-1)/2/(1-γ)*H0[i]
        
    Q0 = β*anp.exp((1.-ρ)*recursive_ss[0])
    Q1 = (1.-ρ)*Q0*(util_sol['rmv1_t'])
    P0 = (1.-β)*anp.exp((ρ-1.)*recursive_ss[1])
    P1 = (ρ-1.)*P0*util_sol['vmk1_t'] 
    Q1H0x = np.matmul(np.array(H0).reshape(n_J,1),Q1['x'])
    P1L0x = np.matmul(np.array(L0).reshape(n_J,1),P1['x'])
    Q1H0x = np.array([[float(val) for val in row] for row in Q1H0x], dtype=float)
    P1L0x = np.array([[float(val) for val in row] for row in P1L0x], dtype=float)

    Q1H0c = np.concatenate((np.matmul(np.array(H0).reshape(n_J,1),Q1['c']),np.zeros(n_X).reshape(n_X,1)))
    P1L0c = np.concatenate((np.matmul(np.array(L0).reshape(n_J,1),P1['c']),np.zeros(n_X).reshape(n_X,1)))
    Q1H0c = np.array([[float(val) for val in row] for row in Q1H0c], dtype=float)
    P1L0c = np.array([[float(val) for val in row] for row in P1L0c], dtype=float)


    dfp = copy.deepcopy(df)
    dfp['xt'][:n_J, n_J:] += P1L0x + Q1H0x 

    if return_df==True:
        return [dfp,df]

    schur = schur_decomposition(-df['xtp1'], dfp['xt'], (n_J, n_X, n_W))

    μ_0 = util_sol['μ_0']
    RHS = - (df['wtp1']@μ_0 + Q1H0c + P1L0c)
    LHS = df['xtp1'] + dfp['xt']
    D = np.linalg.solve(LHS, RHS)
    C = D[:n_J] - schur['N']@D[n_J:]

    ## Solve for the constant term in the first-order expansion under the distorted measure 
    f_1_xtp1 = df['xtp1'][:n_J] 
    f_1_wtp1 = df['wtp1'][:n_J]
    f_2_xtp1 = df['xtp1'][n_J:]
    f_2_xt = dfp['xt'][n_J:]
    f_2_wtp1 = df['wtp1'][n_J:]


    ψ_tilde_x = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_xt@schur['N_block'])
    ψ_tilde_w = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_wtp1)
    ψ_tilde_q = D[n_J:] - ψ_tilde_x@D[n_J:]

    RHS_org = - (np.block([[(f_1_xtp1@schur['N_block']@ψ_tilde_w+ f_1_wtp1)@μ_0], [np.zeros((n_X, 1))]])+Q1H0c+P1L0c)
    LHS_org = dfp['xt'] + dfp['xtp1']
    D_org = np.linalg.solve(LHS_org, RHS_org)
    ψ_q = D_org[n_J:] - ψ_tilde_x@D_org[n_J:]

    ## Formulate the solution for the first-order expansion
    X1_tp1 = LinQuadVar({'x': ψ_tilde_x, 'w': ψ_tilde_w, 'c': ψ_q}, (n_X, n_X, n_W), False)
    X1_tp1_tilde = LinQuadVar({'x': ψ_tilde_x, 'w': ψ_tilde_w, 'c': ψ_tilde_q}, (n_X, n_X, n_W), False)
    J1_t = LinQuadVar({'x': schur['N'], 'c': C}, (n_J,n_X,n_W), False)
    X1_t = LinQuadVar({'x': np.eye(n_X)}, (n_X, n_X, n_W), False)
    JX1_t = concat([J1_t, X1_t])
    JX1_tp1 = next_period(JX1_t, X1_tp1)
    JX1_tp1_tilde = next_period(JX1_t, X1_tp1_tilde)
    return J1_t, JX1_t, JX1_tp1, JX1_tp1_tilde, X1_tp1, X1_tp1_tilde


def compute_adj_approach_1(H_fun_list, H, L, ss, ss_variables, ss_variables_tp1, var_shape, util_sol, JX1_t, X1_tp1, recursive_ss, parameter_names, args):
    """
    Computes additional recursive utility adjustment using approach 1.

    """

    n_J, n_X, n_W = var_shape
    n_JX = n_J + n_X
    HQ_loc_list = [i for i in range(n_J)]

    γ = get_parameter_value('gamma', parameter_names, args)
    β = get_parameter_value('beta', parameter_names, args)
    ρ = get_parameter_value('rho', parameter_names, args)

    Q0 = β*anp.exp((1.-ρ)*recursive_ss[0])
    P0 = (1.-β)*anp.exp((ρ-1.)*recursive_ss[1])
    Q1 = (1.-ρ)*Q0*(util_sol['rmv1_t'])
    P1 = (ρ-1.)*P0*util_sol['vmk1_t'] 
    Q2 = (1.-ρ)*kron_prod(Q1, util_sol['rmv1_t']) + (1.-ρ)*Q0*(util_sol['rmv2_t'])
    P2 = (ρ-1.)*kron_prod(P1,util_sol['vmk1_t']) + (ρ-1.)*P0*util_sol['vmk2_t']


    #Approximation of zero and first-order equilibrium conditionsH_fun_list = [(lambda y: (lambda JX_t, JX_tp1, W_tp1, q, *args: eq(JX_t, JX_tp1, W_tp1, q, 'H', recursive_ss, *args)[y]))(i) for i in range(n_J)]
    H_fun_list = H
    H0_t = [approximate_fun(H_fun, ss, ss_variables, ss_variables_tp1, parameter_names, args, var_shape, None, None, None, None, recursive_ss,second_order = False, zeroth_order = True) for H_fun in H_fun_list]
    H1_t = [approximate_fun(H_fun, ss, ss_variables, ss_variables_tp1, parameter_names, args, var_shape, JX1_t, None, X1_tp1, None, recursive_ss,second_order = False, zeroth_order= False) for H_fun in H_fun_list]

    L_fun_list = L
    L0_t = [approximate_fun(L_fun, ss, ss_variables, ss_variables_tp1, parameter_names, args, var_shape, None, None, None, None, recursive_ss,second_order = False, zeroth_order = True) for L_fun in L_fun_list]
    L1_t = [approximate_fun(L_fun, ss, ss_variables, ss_variables_tp1, parameter_names, args, var_shape, JX1_t, None, X1_tp1, None, recursive_ss,second_order = False, zeroth_order= False) for L_fun in L_fun_list]


    #Distorted mean of W_{t+1}
    μ_0 = util_sol['μ_0']
    Upsilon_2 = util_sol['Upsilon_2']
    Upsilon_1 = util_sol['Upsilon_1']
    Upsilon_0 = util_sol['Upsilon_0']
    Theta_0 = [app[2]['c']+app[2]['w']@μ_0 for app in H1_t]
    Theta_1 = [app[2]['x'] for app in H1_t]
    Theta_2 = [app[2]['w'] for app in H1_t]

    lq1 = [LinQuadVar({'x': (1-γ) * Q0 * (Theta_2[i]) @ Upsilon_1,\
                'c': (1-γ) * Q0 * (Theta_2[i]) @ Upsilon_0}, (1, n_X, n_W)) for i in range(len(HQ_loc_list))]

    lq2 = [2*kron_prod(Q1,H1_t[i][2])for i in range(len(HQ_loc_list))]

    lq3 = [H0_t[i] * Q2 for i in range(len(HQ_loc_list))]

    lq4 = [2*kron_prod(P1,L1_t[i][2]) for i in range(len(HQ_loc_list))]

    lq5 = [L0_t[i] * P2 for i in range(len(HQ_loc_list))]



    adj = [lq_sum([lq1[i], lq2[i], lq3[i], lq4[i], lq5[i]]) for i in range(len(HQ_loc_list))]
    adj_aug = {'x':np.zeros([n_JX,n_X]),'c':np.zeros([n_JX,1]),'xx':np.zeros([n_JX,n_X**2]),'x2':np.zeros([n_JX,n_X])}

    for i in range(len(HQ_loc_list)):
        adj_aug['x'][HQ_loc_list[i]] = adj[i]['x']
        adj_aug['c'][HQ_loc_list[i]] = adj[i]['c']
        adj_aug['xx'][HQ_loc_list[i]] = adj[i]['xx']
        adj_aug['x2'][HQ_loc_list[i]] = adj[i]['x2']
    adj = adj_aug
    return adj_aug


def second_order_expansion_approach_1(df, util_sol, X1_tp1, X1_tp1_tilde, JX1_t, JX1_tp1, JX1_tp1_tilde, adj, var_shape,parameter_names):
    """
    Implements second-order expansion using approach 1.

    """


    n_J, n_X, n_W = var_shape

    ## Compute the Schur decomposition of the derivative matrix
    dfp = copy.deepcopy(df)
    dfp['xt'][:,n_J:] += adj['x2']

    schur = schur_decomposition(-df['xtp1'], dfp['xt'], (n_J, n_X, n_W))
    μ_0 = util_sol['μ_0']

    ## Extract the contribution from the first order expansion
    Wtp1 = LinQuadVar({'w': np.eye(n_W),'c':μ_0}, (n_W, n_X, n_W), False)
    D2 = combine_second_order_terms(dfp, JX1_t,JX1_tp1_tilde, Wtp1)
    D2_coeff = np.block([[D2['c'], D2['x'], D2['w'], D2['xx'], D2['xw'], D2['ww']]])
    M_E_w = np.zeros([n_W,1])
    cov_w = np.eye(n_W)
    M_E_ww = cal_E_ww(M_E_w, cov_w)
    E_D2 = E(D2, M_E_w, M_E_ww)
    E_D2_coeff = np.block([[E_D2['c']+adj['c'], E_D2['x']+adj['x'], E_D2['xx']+adj['xx']]])
   
    ## Solve for the first-order contribution in the jump variables
    X1X1_tilde = kron_prod(X1_tp1_tilde, X1_tp1_tilde)
    M_mat = form_M0(M_E_w, M_E_ww, X1_tp1_tilde, X1X1_tilde)
    LHS = np.eye(n_J*M_mat.shape[0]) - np.kron(M_mat.T, np.linalg.solve(schur['Λ22'], schur['Λp22'])) #I - A2' \kron A1
    RHS = vec(-np.linalg.solve(schur['Λ22'], (schur['Q'].T@E_D2_coeff)[-n_J:])) #A3
    D_tilde_vec = np.linalg.solve(LHS, RHS)
    D_tilde = mat(D_tilde_vec, (n_J, 1+n_X+n_X**2))
    G = np.linalg.solve(schur['Z21'], D_tilde)

    def solve_second_state(X1_tp1, JX1_tp1, Wtp1):
        '''
        Solve for the first-order contribution in the state variable evolution
        '''
        D2 = combine_second_order_terms(dfp, JX1_t, JX1_tp1, Wtp1)
        D2_coeff = np.block([[D2['c'], D2['x'], D2['w'], D2['xx'], D2['xw'], D2['ww']]])

        X1X1 = kron_prod(X1_tp1, X1_tp1)
        Y2_coeff = -dfp['xtp1'][n_J:]@schur['N_block']
        C_hat = np.linalg.solve(Y2_coeff, D2_coeff[n_J:])
        C_hat_coeff = np.split(C_hat, np.cumsum([1, n_X, n_W, n_X**2, n_X*n_W]),axis=1)
        G_block = np.block([[G], [np.zeros((n_X, 1+n_X+n_X**2))]])
        Gp_hat = np.linalg.solve(Y2_coeff, dfp['xtp1'][n_J:]@G_block)
        G_hat = np.linalg.solve(Y2_coeff, dfp['xt'][n_J:]@G_block)
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

    ## Solve for the state evolution under the distorted measure
    Wtp1 = LinQuadVar({'w': np.eye(n_W),'c':μ_0}, (n_W, n_X, n_W), False)
    X2_tp1_tilde = solve_second_state(X1_tp1_tilde, JX1_tp1_tilde, Wtp1)

    ## Solve for the state evolution under the original measure
    Wtp1 = LinQuadVar({'w': np.eye(n_W)}, (n_W, n_X, n_W), False)
    X2_tp1 = solve_second_state(X1_tp1, JX1_tp1, Wtp1)

    ## Formulate the solution for the second-order expansion
    J2_t = LinQuadVar({'x2': schur['N'], 'xx': G[:, 1+n_X:1+n_X+(n_X)**2], 'x': G[:, 1:1+n_X], 'c': G[:, :1]}, (n_J, n_X, n_W), False)
    X2_t = LinQuadVar({'x2': np.eye(n_X)}, (n_X, n_X, n_W), False)
    JX2_t = concat([J2_t, X2_t])
    JX2_tp1 = next_period(JX2_t, X1_tp1, X2_tp1)
    JX2_tp1_tilde = next_period(JX2_t, X1_tp1_tilde, X2_tp1_tilde)
    return J2_t, JX2_t, JX2_tp1, JX2_tp1_tilde, X2_tp1, X2_tp1_tilde

def first_order_expansion_approach_2(df, ss, util_sol, var_shape, H0, L0, recursive_ss, args):
    n_J, n_X, n_W = var_shape
    n_JX = n_J + n_X
    HQ_loc_list = [i for i in range(n_J)]


    γ = get_parameter_value('gamma', parameter_names, args)
    β = get_parameter_value('beta', parameter_names, args)
    ρ = get_parameter_value('rho', parameter_names, args)

    ## Compute the adjustment term 
    μ_0 = util_sol['μ_0']
    Σ_tilde = util_sol['Σ_tilde']
    Γ_tilde = util_sol['Γ_tilde']
    μ_tilde_t = util_sol['μ_tilde_t']
    H_1_x = np.zeros([n_JX,n_X])
    H_1_c = np.zeros([n_JX,1])
    for i in range(len(H0)):
        H_1_x[HQ_loc_list[i]] = (ρ-1)/(1-γ)*μ_0.T@μ_tilde_t['x']*H0[i]
        H_1_c[HQ_loc_list[i]] = μ_0.T@μ_0*(ρ-1)/2/(1-γ)*H0[i]+(ρ-1)/(1-γ)*μ_0.T@(μ_tilde_t['c']-μ_0)*H0[i]

    H_1_c = df['wtp1']@μ_tilde_t['c']
    df_mix = np.block([[np.zeros([n_JX,n_J]),df['wtp1']@μ_tilde_t['x']]])

    H_1 = np.zeros([n_JX,1])
    HQ_loc_list = [i for i in range(n_J)]
    for i in range(len(H0)):
        H_1[HQ_loc_list[i]] = μ_0.T@μ_0*(ρ-1)/2/(1-γ)*H0[i]
        
    Q0 = β*anp.exp((1.-ρ)*recursive_ss[0])
    Q1 = (1.-ρ)*Q0*(util_sol['rmv1_t'])
    P0 = (1.-β)*anp.exp((ρ-1.)*recursive_ss[1])
    P1 = (ρ-1.)*P0*util_sol['vmk1_t'] 
    Q1H0x = np.matmul(np.array(H0).reshape(n_J,1),Q1['x'])
    P1L0x = np.matmul(np.array(L0).reshape(n_J,1),P1['x'])
    Q1H0c = np.concatenate((np.matmul(np.array(H0).reshape(n_J,1),Q1['c']),np.array((0.,0.,0.)).reshape(n_X,1)))
    P1L0c = np.concatenate((np.matmul(np.array(L0).reshape(n_J,1),P1['c']),np.array((0.,0.,0.)).reshape(n_X,1)))


    dfp = copy.deepcopy(df)
    dfp['xt'][:n_J, n_J:] += P1L0x + Q1H0x 

    ## Compute the Schur decomposition of the derivative matrix
    # schur = schur_decomposition(-df['xtp1'], dfp['xt']+Q0*df_adj+Q0*df_mix, (n_J, n_X, n_W))
    schur = schur_decomposition(-df['xtp1'], dfp['xt']+df_mix, (n_J, n_X, n_W))

    RHS = - (df['wtp1']@μ_tilde_t['c']+ Q1H0c + P1L0c)
    LHS = df['xtp1'] + dfp['xt'] + df_mix
    D = np.linalg.solve(LHS, RHS)
    C = D[:n_J] - schur['N']@D[n_J:]

    ## Solve for the constant term in the first-order expansion under the distorted measure
    f_1_xtp1 = df['xtp1'][:n_J]
    f_1_wtp1 = df['wtp1'][:n_J]
    f_2_xtp1 = df['xtp1'][n_J:]
    f_2_xt_tilde = (dfp['xt']+df_adj+df_mix)[n_J:]
    # f_2_xt_tilde = (dfp['xt'])[n_J:]
    f_2_wtp1 = df['wtp1'][n_J:]


    ## Solve for the constant term in the first-order expansion under the original measure
    ψ_tilde_x = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_xt_tilde@schur['N_block'])
    ψ_tilde_w = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_wtp1)@Γ_tilde
    ψ_tilde_q = D[n_J:] - ψ_tilde_x@D[n_J:]

    f_2_xt_org = dfp['xt'][n_J:]
    ψ_x = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_xt_org@schur['N_block'])
    ψ_w = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_wtp1)

    df_mix_org = df_mix.copy()
    df_mix_org[n_J:] = 0

    RHS_org = - (np.block([[(f_1_xtp1@schur['N_block']@ψ_w + f_1_wtp1)@μ_tilde_t['c']], [np.zeros((n_X, 1))]])+Q1H0c+P1L0c)
    LHS_org = df['xtp1'] + dfp['xt']  + df_mix_org
    D_org = np.linalg.solve(LHS_org, RHS_org)
    ψ_q = D_org[n_J:] - ψ_tilde_x@D_org[n_J:]

    ## Formulate the solution for the first-order expansion
    X1_tp1 = LinQuadVar({'x': ψ_x, 'w': ψ_w, 'c': ψ_q}, (n_X, n_X, n_W), False)
    X1_tp1_tilde = LinQuadVar({'x': ψ_tilde_x, 'w': ψ_tilde_w, 'c': ψ_tilde_q}, (n_X, n_X, n_W), False)
    J1_t = LinQuadVar({'x': schur['N'], 'c': C}, (n_J,n_X,n_W), False)
    X1_t = LinQuadVar({'x': np.eye(n_X)}, (n_X, n_X, n_W), False)
    JX1_t = concat([J1_t, X1_t])
    JX1_tp1 = next_period(JX1_t, X1_tp1)
    JX1_tp1_tilde = next_period(JX1_t, X1_tp1_tilde)

    return J1_t, JX1_t, JX1_tp1, JX1_tp1_tilde, X1_tp1, X1_tp1_tilde



def compute_adj_approach_2(H_fun_list, ss, eq, var_shape, util_sol, JX1_t, X1_tp1, recursive_ss, args):
    n_J, n_X, n_W = var_shape
    n_JX = n_J + n_X
    HQ_loc_list = [i for i in range(n_J)]

    γ = get_parameter_value('gamma', parameter_names, args)
    β = get_parameter_value('beta', parameter_names, args)
    ρ = get_parameter_value('rho', parameter_names, args)

    Q0 = β*anp.exp((1.-ρ)*recursive_ss[0])
    P0 = (1.-β)*anp.exp((ρ-1.)*recursive_ss[1])
    Q1 = (1.-ρ)*Q0*(util_sol['rmv1_t'])
    P1 = (ρ-1.)*P0*util_sol['vmk1_t'] 
    Q2 = (1.-ρ)*kron_prod(Q1, util_sol['rmv1_t']) + (1.-ρ)*Q0*(util_sol['rmv2_t'])
    P2 = (ρ-1.)*kron_prod(P1,util_sol['vmk1_t']) + (ρ-1.)*P0*util_sol['vmk2_t']

    #Approximation of zero and first-order equilibrium conditionsH_fun_list = [(lambda y: (lambda JX_t, JX_tp1, W_tp1, q, *args: eq(JX_t, JX_tp1, W_tp1, q, 'H', recursive_ss, *args)[y]))(i) for i in range(n_J)]
    H0_t = [approximate_fun(H_fun, ss, (1, n_X, n_W), None, None, None, None, args,recursive_ss, second_order = False, zeroth_order= True) for H_fun in H_fun_list]
    H1_t = [approximate_fun(H_fun, ss, (1, n_X, n_W), JX1_t, None, X1_tp1, None, args, recursive_ss,second_order = False, zeroth_order= False) for H_fun in H_fun_list]

    L_fun_list = [(lambda y: (lambda JX_t, JX_tp1, W_tp1, q, *args: eq(JX_t, JX_tp1, W_tp1, q, 'L', recursive_ss, *args)[y]))(i) for i in range(n_J)]
    L0_t = [approximate_fun(L_fun, ss, (1, n_X, n_W), None, None, None, None, args, recursive_ss,second_order = False, zeroth_order= True) for L_fun in L_fun_list]
    L1_t = [approximate_fun(L_fun, ss, (1, n_X, n_W), JX1_t, None, X1_tp1, None, args, recursive_ss,second_order = False, zeroth_order= False) for L_fun in L_fun_list]


    #Distorted mean of W_{t+1}
    μ_0 = util_sol['μ_0']

    #2Q1N0H1
    lq2 = [2*kron_prod(Q1,H1_t[i][2])for i in range(len(HQ_loc_list))] 
        

    #Q2N0H0
    lq3 = [H0_t[i] * Q2 for i in range(len(HQ_loc_list))]

    #2P1L1
    lq4 = [2*kron_prod(P1,L1_t[i][2]) for i in range(len(HQ_loc_list))]

    #P2L0
    lq5 = [L0_t[i] * P2 for i in range(len(HQ_loc_list))]

    adj = [lq_sum([lq2[i], lq3[i], lq4[i], lq5[i]]) for i in range(len(HQ_loc_list))]
    adj_aug = {'x':np.zeros([n_JX,n_X]),'c':np.zeros([n_JX,1]),'xx':np.zeros([n_JX,n_X**2]),'x2':np.zeros([n_JX,n_X])}

    for i in range(len(HQ_loc_list)):
        adj_aug['x'][HQ_loc_list[i]] = adj[i]['x']
        adj_aug['c'][HQ_loc_list[i]] = adj[i]['c']
        adj_aug['xx'][HQ_loc_list[i]] = adj[i]['xx']
        adj_aug['x2'][HQ_loc_list[i]] = adj[i]['x2']
    adj = adj_aug

    return adj_aug

def second_order_expansion_approach_2(df, util_sol, X1_tp1, X1_tp1_tilde, JX1_t, JX1_tp1, JX1_tp1_tilde, adj, var_shape):
    n_J, n_X, n_W = var_shape

    ## Compute the Schur decomposition of the derivative matrix
    dfp = copy.deepcopy(df)
    dfp['xt'][:,n_J:] += adj['x2']

    ## Compute the Schur decomposition of the derivative matrix
    schur = schur_decomposition(-df['xtp1'], dfp['xt'], (n_J, n_X, n_W))

    ## Extract the contribution from the first order expansion
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

    ## Solve for the first-order contribution in the jump variables
    X1X1_tilde = kron_prod(X1_tp1_tilde, X1_tp1_tilde)
    M_mat = form_M0(M_E_w, M_E_ww, X1_tp1_tilde, X1X1_tilde)
    LHS = np.eye(n_J*M_mat.shape[0]) - np.kron(M_mat.T, np.linalg.solve(schur['Λ22'], schur['Λp22']))
    RHS = vec(-np.linalg.solve(schur['Λ22'], (schur['Q'].T@E_D2_coeff)[-n_J:]))
    D_tilde_vec = np.linalg.solve(LHS, RHS)
    D_tilde = mat(D_tilde_vec, (n_J, 1+n_X+n_X**2))
    G = np.linalg.solve(schur['Z21'], D_tilde)

    ψ_x2 = X1_tp1['x']

    def solve_second_state(X1_tp1, JX1_tp1, Wtp1):
        '''
        Solve for the first-order contribution in the state variable evolution
        '''
        D2 = combine_second_order_terms(df, JX1_t, JX1_tp1, Wtp1)
        D2_coeff = np.block([[D2['c'], D2['x'], D2['w'], D2['xx'], D2['xw'], D2['ww']]])

        X1X1 = kron_prod(X1_tp1, X1_tp1)
        Y2_coeff = -df['xtp1'][n_J:]@schur['N_block']
        C_hat = np.linalg.solve(Y2_coeff, D2_coeff[n_J:])
        C_hat_coeff = np.split(C_hat, np.cumsum([1, n_X, n_W, n_X**2, n_X*n_W]),axis=1)
        G_block = np.block([[G], [np.zeros((n_X, 1+n_X+n_X**2))]])
        Gp_hat = np.linalg.solve(Y2_coeff, df['xtp1'][n_J:]@G_block)
        G_hat = np.linalg.solve(Y2_coeff, dfp['xt'][n_J:]@G_block)
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

    ## Solve for the state evolution under the distorted measure
    Wtp1 = LinQuadVar({'w': Γ_tilde, 'c':μ_tilde_t['c'], 'x': μ_tilde_t['x']}, (n_W, n_X, n_W), False)
    X2_tp1_tilde = solve_second_state(X1_tp1_tilde, JX1_tp1_tilde, Wtp1)

    ## Solve for the state evolution under the original measure
    Wtp1 = LinQuadVar({'w': np.eye(n_W)}, (n_W, n_X, n_W), False)
    X2_tp1 = solve_second_state(X1_tp1, JX1_tp1, Wtp1)

    ## Formulate the solution for the second-order expansion
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
    n_J, n_X, n_W = JX1_tp1.shape

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

def approximate_fun(fun, ss, ss_variables, ss_variables_tp1, parameter_names, args,
                    var_shape, JX1_t, JX2_t, X1_tp1, X2_tp1, recursive_ss, second_order = True, zeroth_order = False):
    from derivatives import compute_derivatives

    n_J, n_X, n_W = var_shape
    n_JX = n_J + n_X
    W_0 = np.zeros(n_W)
    q_0 = 0.
    X=[ss, ss, W_0, q_0]

    recursive_variables, main_variables, q_variable, shock_variables = split_variables(ss_variables,var_shape)
    recursive_variables_tp1, main_variables_tp1, q_variable_tp1, shock_variables_tp1 = split_variables(ss_variables_tp1,var_shape)

    dfun = compute_derivatives(fun, ss_variables, ss_variables_tp1, parameter_names, X, args, var_shape, recursive_ss,second_order=second_order)
    
    fun_zero_order = fun.subs({param: val for param, val in zip(parameter_names, args)})
    fun_zero_order = fun_zero_order.subs({var: val for var, val in zip(main_variables, ss)})
    fun_zero_order = fun_zero_order.subs({var: val for var, val in zip(main_variables_tp1, ss)})
    fun_zero_order = fun_zero_order.subs({var: val for var, val in zip(shock_variables_tp1, W_0)})
    fun_zero_order = fun_zero_order.subs({q_variable: q_0})
    fun_zero_order = float(fun_zero_order)
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

def solve_utility(ss, ss_variables, ss_variables_tp1, parameter_names, var_shape, 
                  args, X1_tp1, X2_tp1, JX1_t, JX2_t, f_cmk_app, f_gk, f_gc, recursive_ss, tol = 1e-10):
    
    # print("Inputs:")
    # print(f"ss: {ss}")
    # print(f"ss_variables: {ss_variables}")
    # print(f"ss_variables_tp1: {ss_variables_tp1}")
    # print(f"parameter_names: {parameter_names}")
    # print(f"var_shape: {var_shape}")
    # print(f"args: {args}")
    # print(f"X1_tp1: {X1_tp1.coeffs}")
    # print(f"X2_tp1: {X2_tp1.coeffs}")
    # print(f"JX1_t: {JX1_t.coeffs}")
    # print(f"JX2_t: {JX2_t.coeffs}")
    # print(f"f_gc: {f_gc}")
    # print(f"f_gk: {f_gk}")
    # print(f"f_cmk_app: {f_cmk_app}")
    # print(f"recursive_ss: {recursive_ss}")
    # print(f"tol: {tol}")

    
    _, n_X, n_W = var_shape
    γ = get_parameter_value('gamma', parameter_names, args)
    β = get_parameter_value('beta', parameter_names, args)
    ρ = get_parameter_value('rho', parameter_names, args)


    ##Approximate log consumption-capital
    cmk_app = approximate_fun(f_cmk_app, ss, ss_variables, ss_variables_tp1, parameter_names, args, var_shape,  JX1_t, JX2_t, X1_tp1, X2_tp1, recursive_ss,second_order = True, zeroth_order=False)
    cmk1_t = cmk_app[2]
    cmk2_t = cmk_app[3]
    ##Consumption Growth

    ## Approximate the growth of consumption
    gc_tp1_approx = approximate_fun(f_gc, ss, ss_variables, ss_variables_tp1, parameter_names, args, var_shape,  JX1_t, JX2_t, X1_tp1, X2_tp1, recursive_ss,second_order = True, zeroth_order=False)
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
    
    ## Solve for the first order expansion of the continuation values
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

    rmv1_t = rmc1_t - vmc1_t
    
    vmc1_tp1 = next_period(vmc1_t,X1_tp1)
    vmr1_tp1 = vmc1_tp1 + gc1_tp1 - rmc1_t
    
    log_N0 = ((1-γ)*(vmc1_tp1 + gc1_tp1)-log_E_exp((1-γ)*(vmc1_tp1 + gc1_tp1)))
    μ_0 = log_N0['w'].T
    Ew0 = μ_0.copy()
    Eww0 = cal_E_ww(Ew0,np.eye(Ew0.shape[0]))

    ## Solve for the second order expansion of the continuation values
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
    rmv2_t = rmc2_t - vmc2_t


    #Capital Growth

    ## Approximate the growth of consumption
    gk_tp1_approx = approximate_fun(f_gk, ss, ss_variables, ss_variables_tp1, parameter_names, args, var_shape,  JX1_t, JX2_t, X1_tp1, X2_tp1, recursive_ss,second_order = True, zeroth_order=False)

    gk_tp1 = gk_tp1_approx[0]
    gk0_tp1 = gk_tp1_approx[1]
    gk1_tp1 = gk_tp1_approx[2]
    gk2_tp1 = gk_tp1_approx[3]

    def return_order1_t(order1_t_coeffs):
        return LinQuadVar({'c': np.array([[order1_t_coeffs[0]]]), 'x':np.array([order1_t_coeffs[1:(1+n_X)]])},(1, n_X, n_W))
    
    def return_order2_t(order2_t_coeffs):
            return LinQuadVar({'c': np.array([[order2_t_coeffs[0]]]), 'x':np.array([order2_t_coeffs[1:(1+n_X)]]),\
                            'x2':np.array([order2_t_coeffs[(1+n_X):(1+n_X+n_X)]]), 'xx':np.array([order2_t_coeffs[(1+n_X+n_X):(1+n_X+n_X+n_X*n_X)]])},(1, n_X, n_W))

    λ = β * np.exp((1-ρ) * gk0_tp1)

    rmk1_t = rmc1_t + cmk1_t
    
    
    ## Solve for the first order expansion of the continuation values
    def solve_vmk1_t(order1_init_coeffs):
        vmk1_t = return_order1_t(order1_init_coeffs)
        LHS = (1-λ)*cmk1_t + λ/(1-γ) *log_E_exp((1-γ)*(next_period(vmk1_t,X1_tp1)+gk1_tp1))
        LHS_list = [LHS['c'].item()]+ [i for i in LHS['x'][0]] 
        return list(np.array(LHS_list) - np.array(order1_init_coeffs))   
    
    vmk1_t_sol = optimize.root(solve_vmk1_t, x0 = [0]*(1 + n_X),tol = tol)
    if vmk1_t_sol['success'] ==False:
        print(vmk1_t_sol['message'])
    vmk1_t =return_order1_t(vmk1_t_sol['x'])
    


    ## Solve for the second order expansion of the continuation values
    def solve_vmk2_t(order2_init_coeffs):

        vmk2_t = return_order2_t(order2_init_coeffs)
        LHS = λ *E(next_period(vmk2_t, X1_tp1, X2_tp1) + gk2_tp1, Ew0, Eww0) + (1-ρ)*(1-λ)/λ*kron_prod(vmk1_t,vmk1_t)
        LHS_list = [LHS['c'].item()]+ [i for i in LHS['x'][0]] + [i for i in LHS['x2'][0]] + [i for i in LHS['xx'][0]] 
        return list(np.array(LHS_list) - np.array(order2_init_coeffs))
    
    vmk2_t_sol = optimize.root(solve_vmk2_t, x0 = [0]*(1 + n_X+n_X+n_X*n_X),tol = tol)
    if vmk2_t_sol['success'] ==False:
        print(vmk2_t_sol['message'])
    vmk2_t = return_order2_t(vmk2_t_sol['x'])

    vmk2_t = vmc2_t + cmk2_t

    # print()

    # print(X1_tp1.coeffs)
    # print(X2_tp1.coeffs)
    # print(vmk2_t.coeffs)
    rmk2_t = E(next_period(vmk2_t, X1_tp1, X2_tp1) + gk2_tp1, Ew0, Eww0)
    vmk2_tp1 = next_period(vmk2_t, X1_tp1, X2_tp1)
    vmr2_tp1 = vmk2_tp1 + gk2_tp1 - rmk2_t



    ## Formulate change of measure
    log_N_tilde = (1-γ)*(vmc1_tp1+gc1_tp1 + 0.5*(vmc2_tp1+gc2_tp1))-log_E_exp((1-γ)*(vmc1_tp1+gc1_tp1 + 0.5*(vmc2_tp1+gc2_tp1)))
    Upsilon_2 = vmr2_tp1['ww'].reshape(n_W,n_W).T*2
    Upsilon_1 = vmr2_tp1['xw'].reshape((n_X,n_W)).T
    Upsilon_0 = vmr2_tp1['w'].T + (Upsilon_2@μ_0)
    Σ_tilde = np.linalg.inv(np.eye(n_W)-0.5*Upsilon_2*(1-γ))
    Γ_tilde = sci.linalg.sqrtm(Σ_tilde)
    μ_tilde_t = LinQuadVar({'x':0.5*(1-γ)*Σ_tilde@Upsilon_1, 'c':(1-γ)*Σ_tilde@(1/(1-γ)*μ_0+0.5*(Upsilon_0-Upsilon_2@μ_0))},shape=(n_W,n_X,n_W))

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
                'vmc1_t': vmc1_t,
                'vmc2_t': vmc2_t,
                'rmc1_t': rmc1_t,
                'rmc2_t': rmc2_t,
                'gk_tp1': gk_tp1,
                'gk0_tp1': gk0_tp1,
                'gk1_tp1': gk1_tp1,
                'gk2_tp1': gk2_tp1,
                'vmk1_t': vmk1_t,
                'vmk2_t': vmk2_t,
                'rmk2_t': rmk2_t,
                'vmr1_tp1': vmr1_tp1, 
                'vmr2_tp1': vmr2_tp1,
                'Σ_tilde':Σ_tilde,
                'Γ_tilde':Γ_tilde,
                'μ_tilde_t':μ_tilde_t,
                'rmv1_t':rmv1_t,
                'rmv2_t':rmv2_t,
                'rmk1_t':rmk1_t}

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
        from derivatives import compute_derivatives

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


    
def generate_symbols_and_args(func):
    """
    Extracts parameter definitions from `create_args`,
    creates SymPy symbols for parameter names,
    and saves their values into an `args` list.
    
    Returns:
    - symbols: A dictionary mapping parameter names to SymPy symbols.
    - args: A list of parameter values in the order they were defined.
    """
    # Get parameters from the create_args function
    parameters = func()

    # Filter out non-parameter items (e.g., built-in Python functions)
    parameter_names = [name for name in parameters if not name.startswith("__")]

    # Create SymPy symbols for each parameter name
    symbols = {name: sp.Symbol(name) for name in parameter_names}

    # Update the global namespace with these symbols (optional for direct access)
    globals().update(symbols)

    # Create args list of values
    args = [parameters[name] for name in parameter_names]

    # Return symbols and args
    return symbols, args

def get_parameter_value(parameter_name, parameter_names, args):
    """
    Retrieve the value of a parameter from the args list based on its name.

    Parameters
    ----------
    parameter_name : str
        The name of the parameter to retrieve (e.g., 'beta_2').
    parameter_names : dict
        A dictionary mapping parameter names (keys) to symbolic variables (values).
    args : list
        A list of parameter values corresponding to the order of parameter_names.

    Returns
    -------
    value : float
        The value of the requested parameter.

    Raises
    ------
    ValueError
        If the parameter name is not found in parameter_names.
    """
    # Convert parameter_names keys to a list to get the index
    try:
        parameter_index = list(parameter_names.keys()).index(parameter_name)
    except ValueError:
        raise ValueError(f"Parameter '{parameter_name}' not found in parameter_names.")
    
    # Retrieve the value from args
    return args[parameter_index]


# def compile_equations(parameter_names, variables, variables_tp1,control_variables,state_variables,
#                 output_constraint, capital_growth, state_equations, 
#                 var_shape):
#         n_J, n_X, n_W = var_shape
#         n_C = n_J - n_X - 1 #number of endogenous state variables apart from cons
#         symbols = {name: sp.Symbol(name) for name in parameter_names}
#         symbols.update({param: sp.Symbol(param) for param in parameter_names})
#         globals().update(symbols)

        
#         state_equations = state_equations + [capital_growth]
#         var_capital_growth_tp1 = sp.Symbol(variables_tp1[n_X+n_C-1])
#         state_variables = state_variables + ['log_gk_t']


#         #Add additional variables
#         log_cmk_t = sp.Symbol('log_cmk_t')
#         vmk_t = sp.Symbol('vmk_t')
#         rmv_t = sp.Symbol('rmv_t')

#         #Costates
#         co_states = [sp.Symbol('m'+str(i)+'_t') for i in range(n_X)]
#         co_states_tp1 = [sp.Symbol('m'+str(i)+'_tp1') for i in range(n_X)]
#         co_states[-1] = sp.Symbol('mg_t')
#         co_states_tp1[-1] = sp.Symbol('mg_tp1')
#         co_states = sp.Matrix(co_states)
#         co_states_tp1 = sp.Matrix(co_states_tp1)
        
#         # Construct p and q
#         p = (1 - beta) * sp.exp((1 - rho) * (-vmk_t))
#         rec_cons = sp.exp((1 - rho)* (log_cmk_t)) 
#         q = beta * sp.exp((1 - rho) * (rmv_t))

        
#         # Consumption derivatives
#         consumption_derivatives = sp.Matrix([
#             sp.diff(output_constraint, variables[i])
#             for i in range(len(state_variables+control_variables))
#         ])
#         consumption_derivatives[-1] = 1.0  # Ensure last derivative is 1
#         L = rec_cons*consumption_derivatives

#         # State derivatives
#         state_derivatives = sp.Matrix([
#             [sp.diff(state_equations[j], variables[i]) for j in range(len(state_equations))]
#             for i in range(len(state_variables+control_variables))
#         ]).reshape(len(state_variables+control_variables), len(state_equations))

#         # Replace SymPy.Zero with float zero
#         state_derivatives = state_derivatives.applyfunc(lambda x: 0.0 if x == sp.S.Zero else x)
#         state_derivatives[-1,-1] = 1.0

#         # Construct H
#         H = sp.Matrix([
#             sum(co_state * state_derivatives[j,i] for i, co_state in enumerate(co_states_tp1))
#             for j in range(state_derivatives.shape[0])
#         ])
#         # Recursive equilibrium equation
#         recursive = 1. - rec_cons*p - q

#         # First-order conditions for optimalityprint("Type of q:", type(q))
#         # Ensure co_states is a flat list
#         flat_co_states = [x[0] for x in co_states.tolist()] if co_states.shape[1] == 1 else co_states.tolist()
#         flat_co_states_tp1 = [x[0] for x in co_states_tp1.tolist()] if co_states_tp1.shape[1] == 1 else co_states_tp1.tolist()
#         # print(np.vstack([np.zeros(n_C), flat_co_states]))
#         # foc = q * H + p * L - np.vstack([np.zeros(n_C), flat_co_states])
#         foc = q * H + p * L - sp.Matrix([0]*n_C + flat_co_states)

#         # Output constraint
#         # print(output_constraint)
#         output_constraint =  log_cmk_t - output_constraint
#         # print(output_constraint)

#         # Steady-state conditions for state variables using state equations
#         state_equations[-1] = var_capital_growth_tp1 - state_equations[-1] 
#         state_equations = state_equations[-1:] + state_equations[:-1]

#         # print(variables)
#         variables = variables[:n_C] + [variables[n_X+n_C-1]] + variables[n_C:n_X+n_C-1] + variables[n_X+n_C:]
        
#         variables_tp1 = variables_tp1[:n_C] + [variables_tp1[n_X+n_C-1]] + variables_tp1[n_C:n_X+n_C-1] + variables_tp1[n_X+n_C:]
#         # print(variables)

#         if not isinstance(variables[:n_C], list):
#             variables = list(variables[:n_C]) + variables[n_C:]
#         full_variables = sp.symbols(['rmv_t','vmk_t','log_cmk_t']+variables[:n_C]) + flat_co_states + sp.symbols(variables[n_C:])
#         # print(full_variables)
#         full_variables_tp1 = sp.symbols(['rmv_tp1','vmk_tp1','log_cmk_tp1']+variables_tp1[:n_C]) + flat_co_states_tp1  + sp.symbols(variables_tp1[n_C:])

#         #Add output constraint
#         H = sp.Matrix.vstack(sp.Matrix([sp.Float(0)]), H)
#         L = sp.Matrix.vstack(sp.Matrix([sp.Float(0)]), L)

#         print(state_equations)
#         #Create ss function
#         return [recursive, output_constraint, *foc, *state_equations],full_variables,full_variables_tp1,[*(H)],[*(L)]
def compile_equations(parameter_names, variables, variables_tp1,control_variables,state_variables,
                output_constraint, capital_growth, state_equations, 
                var_shape):
            n_J, n_X, n_W = var_shape
            n_C = n_J - n_X - 1 #number of endogenous state variables apart from cons
            symbols = {name: sp.Symbol(name) for name in parameter_names}
            symbols.update({param: sp.Symbol(param) for param in parameter_names})
            globals().update(symbols)

            
            state_equations = state_equations + [capital_growth]
            var_capital_growth_tp1 = sp.Symbol('log_gk_tp1')
            state_variables = state_variables + ['log_gk_t']
            state_variables_tp1 = [sp.Symbol(str(state_variables[i]).replace("_t", "_tp1")) for i in range(len(state_variables))]
            print(state_variables_tp1)


            #Add additional variables
            log_cmk_t = sp.Symbol('log_cmk_t')
            vmk_t = sp.Symbol('vmk_t')
            rmv_t = sp.Symbol('rmv_t')

            #Costates
            co_states = [sp.Symbol('m'+str(i)+'_t') for i in range(n_X)]
            co_states_tp1 = [sp.Symbol('m'+str(i)+'_tp1') for i in range(n_X)]
            co_states[-1] = sp.Symbol('mg_t')
            co_states_tp1[-1] = sp.Symbol('mg_tp1')
            co_states = sp.Matrix(co_states)
            co_states_tp1 = sp.Matrix(co_states_tp1)
            
            # Construct p and q
            p = (1 - beta) * sp.exp((1 - rho) * (-vmk_t))
            rec_cons = sp.exp((1 - rho)* (log_cmk_t)) 
            q = beta * sp.exp((1 - rho) * (rmv_t))

            
            # Consumption derivatives
            consumption_derivatives = sp.Matrix([
                sp.diff(output_constraint, variables[i])
                for i in range(len(state_variables+control_variables))
            ])
            consumption_derivatives[-1] = 1.0  # Ensure last derivative is 1
            L = rec_cons*consumption_derivatives

            # State derivatives
            state_derivatives = sp.Matrix([
                [sp.diff(state_equations[j], variables[i]) for j in range(len(state_equations))]
                for i in range(len(state_variables+control_variables))
            ]).reshape(len(state_variables+control_variables), len(state_equations))

            # Replace SymPy.Zero with float zero
            state_derivatives = state_derivatives.applyfunc(lambda x: 0.0 if x == sp.S.Zero else x)
            state_derivatives[-1,-1] = 1.0

            # Construct H
            H = sp.Matrix([
                sum(co_state * state_derivatives[j,i] for i, co_state in enumerate(co_states_tp1))
                for j in range(state_derivatives.shape[0])
            ])
            # Recursive equilibrium equation
            recursive = 1. - rec_cons*p - q

            # First-order conditions for optimalityprint("Type of q:", type(q))
            # Ensure co_states is a flat list
            flat_co_states = [x[0] for x in co_states.tolist()] if co_states.shape[1] == 1 else co_states.tolist()
            flat_co_states_tp1 = [x[0] for x in co_states_tp1.tolist()] if co_states_tp1.shape[1] == 1 else co_states_tp1.tolist()
            # print(np.vstack([np.zeros(n_C), flat_co_states]))
            # foc = q * H + p * L - np.vstack([np.zeros(n_C), flat_co_states])
            foc = q * H + p * L - sp.Matrix([0]*n_C + flat_co_states)

            # Output constraint
            # print(output_constraint)
            output_constraint =  log_cmk_t - output_constraint

            # Steady-state conditions for state variables using state equations
            state_equations =  [state_variables_tp1[i] - state_equations[i] for i in range(len(state_variables))]
            # state_equations[-1] = var_capital_growth_tp1 - state_equations[-1] 
            state_equations = state_equations[-1:] + state_equations[:-1]

            # print(variables)
            variables = variables[:n_C] + [variables[n_X+n_C-1]] + variables[n_C:n_X+n_C-1] + variables[n_X+n_C:]
            variables_tp1 = variables_tp1[:n_C] + [variables_tp1[n_X+n_C-1]] + variables_tp1[n_C:n_X+n_C-1] + variables_tp1[n_X+n_C:]
            # print(variables)

            if not isinstance(variables[:n_C], list):
                variables = list(variables[:n_C]) + variables[n_C:]
            full_variables = sp.symbols(['rmv_t','vmk_t','log_cmk_t']+variables[:n_C]) + flat_co_states + sp.symbols(variables[n_C:])
            # print(full_variables)
            full_variables_tp1 = sp.symbols(['rmv_tp1','vmk_tp1','log_cmk_tp1']+variables_tp1[:n_C]) + flat_co_states_tp1  + sp.symbols(variables_tp1[n_C:])

            #Add output constraint
            H = sp.Matrix.vstack(sp.Matrix([sp.Float(0)]), H)
            L = sp.Matrix.vstack(sp.Matrix([sp.Float(0)]), L)

            # print(state_equations)
            print(state_equations)
            #Create ss function
            return [recursive, output_constraint, *foc, *state_equations],full_variables,full_variables_tp1,[*(H)],[*(L)]
            # return [recursive, output_constraint, foc[0], *foc[3:], *state_equations],full_variables,full_variables_tp1,[*(H)],[*(L)]
            
def automate_step_1(variables):
    """
    Automatically substitute all variables to their _tp1 counterparts.

    Parameters:
    - variables: List of SymPy symbols representing variables.

    Returns:
    - substitutions: Dictionary for substituting variables.
    """
    substitutions = {}
    for var in variables:
        # Check if the variable has a "_t" suffix
        if var.name.endswith("_t"):
            # Create the "_tp1" counterpart
            tp1_var = sp.Symbol(var.name.replace("_t", "_tp1"))
            # Add the substitution to the dictionary
            substitutions[tp1_var] = var
    return substitutions


def generate_ss_function(equations, variables, variables_tp1, initial_guess,var_shape,parameter_names):
        """
        Generate a function to solve the steady-state equations.

        Parameters:
        - equations: A callable that takes a list of variables and returns the equations to be solved.
        - variables: A list of variable names. Ordered as: [rmv, vmk, log_cmk, imh, *states, log_gk, q, *shocks].
        - variables_tp1: A list of variable names for the next period.

        Returns:
        - A function that solves the steady-state equations.
        """
        n_J, n_X, n_W = var_shape
        # print(var_shape)
        #Make t = t+1
        substitutions_ss = automate_step_1(variables)

        #Number of variables preceding states
        # print(variables)
        q_t = sp.symbols('q_t')
        log_gk_t = sp.symbols('log_gk_t')
        
        try:
            n_G = variables.index(log_gk_t)  # Index of growth variable
            n_Q = variables.index(q_t)      # Index of q
        except ValueError as e:
            raise ValueError(f"Variable not found in the list: {e}")
        #Substitute growth variables
        substitutions_ss[variables[0]] = variables[n_G]
        substitutions_ss[variables_tp1[0]] = variables[n_G]

        #Substitute q
        # print(variables[n_Q])
        substitutions_ss[variables[n_Q]] = 0.
        substitutions_ss[variables_tp1[n_Q]] = 0.

        #Substitute shocks
        # print(variables[n_Q+1:n_Q+n_W+1])
        for w in variables[n_Q+1:n_Q+n_W+1]:
            substitutions_ss[w] = 0.
        for w in variables_tp1[n_Q+1:n_Q+n_W+1]:
            substitutions_ss[w] = 0.

        equations = [eq.subs(substitutions_ss) for eq in equations]
        # print(equations)
        variables = variables[1:n_Q]
        def ss_solver(args,return_recursive=False):
            # Unpack parameters

            # Define the function to evaluate the equations
            def f(x):
                substituted_equations = [eq.subs({var: val for var, val in zip(parameter_names, args)}) for eq in equations]
                # Update variables dynamically
                variable_dict = {str(var): val for var, val in zip(variables, x)}
                substituted_equations = [eq.subs(variable_dict) for eq in substituted_equations]
                # Debug: Print substituted equations and variable dictionary
                print("Variable Dictionary:")
                for key, value in variable_dict.items():
                    print(f"  {key}: {value}")

                print("\nSubstituted Equations:")
                for idx, eq in enumerate(substituted_equations, start=1):
                    print(f"  Equation {idx}: {eq}")
                
                # Convert to numerical values
                return anp.array([float(eq.evalf()) for eq in substituted_equations])


            # Solve the system of equations
            root = fsolve(f, initial_guess)

            errors = f(root)
            if any(np.isnan(errors)):
                raise ValueError("Solution contains NaN values.")
            if np.linalg.norm(errors, ord=2) > 1e-6:  # Tolerance for error
                raise ValueError(f"Solution error too large: {np.linalg.norm(errors, ord=2)}")

            if return_recursive:
                # Convert root[n_G] to a 1x1 Matrix and concatenate with root as a column vector
                root = np.concatenate([[root[n_G-1]],root])
                # root = np.array(sp.Matrix.vstack(sp.Matrix([[root[n_G]]]), sp.Matrix(root)))
            else:
                # If not recursive, adjust root as needed
                root = root[1:]



            return root

        return ss_solver




def generate_evaluation_function(equations, variables, variables_tp1, qH, pL, var_shape, parameter_names):
    """
    Generate a function to evaluate the equations at specific values.
    """
    # Convert parameter names to a list if it's a dictionary
    parameter_names = list(parameter_names.keys())


    # Variables
    recursive_variables, main_variables, q_variable, shock_variables = split_variables(variables,var_shape)

    # Variables for t+1
    recursive_variables_tp1, main_variables_tp1, q_variable_tp1, shock_variables_tp1 = split_variables(variables_tp1,var_shape)

    # Combine variables
    full_variables = np.concatenate([recursive_variables, main_variables, main_variables_tp1, shock_variables_tp1, [q_variable]])
    full_variables_and_params = np.concatenate([full_variables, parameter_names])

    # print(len(full_variables_and_params))
    # print(full_variables_and_params)

    # Precompile the equations using lambdify
    compiled_equations = lambdify(
        full_variables_and_params,
        equations,
        modules="numpy"
    )

    compiled_qH = lambdify(
        full_variables_and_params,
        qH,
        modules="numpy"
    )

    compiled_pL = lambdify(
        full_variables_and_params,
        pL,
        modules="numpy"
    )

    # Define the evaluation function
    def evaluation_function(Var_t, Var_tp1, W_tp1, q, mode, recursive_ss, args):
        # Combine all inputs into a single array for lambdify
        input_variables = np.concatenate([recursive_ss, Var_t, Var_tp1, W_tp1, [q], args])

        # Evaluate based on mode
        if mode == 'H':
            return np.array(compiled_qH(*input_variables), dtype=np.float64)
        elif mode == 'L':
            return np.array(compiled_pL(*input_variables), dtype=np.float64)
        else:
            return np.array(compiled_equations(*input_variables), dtype=np.float64)

    return evaluation_function

def func_cmk_app(ss_variables,ss_variables_tp1,var_shape):

    return ss_variables[2]

def func_gc(ss_variables,ss_variables_tp1,var_shape):
    n_J, n_X, n_W = var_shape

    #Number of variables preceding states
    n_G = n_J+2  #index of growth variable
    n_B = n_G+n_X+1 #index of shock 1
    n_Q = n_G + n_X
    # Combine current and next-period variables
    return  ss_variables_tp1[2] - ss_variables[2] + ss_variables_tp1[n_G]

def func_gk(ss_variables,ss_variables_tp1,var_shape):
    n_J, n_X, n_W = var_shape

    #Number of variables preceding states
    n_G = n_J+2  #index of growth variable
    n_B = n_G+n_X+1 #index of shock 1
    n_Q = n_G + n_X
    # Combine current and next-period variables

    # Evaluate the lambdified equation
    return ss_variables_tp1[n_G]


def change_parameter_value(parameter_name, parameter_names, args, value):
    """
    Retrieve the value of a parameter from the args list based on its name.

    Parameters
    ----------
    parameter_name : str
        The name of the parameter to retrieve (e.g., 'beta_2').
    parameter_names : dict
        A dictionary mapping parameter names (keys) to symbolic variables (values).
    args : list
        A list of parameter values corresponding to the order of parameter_names.

    Returns
    -------
    value : float
        The value of the requested parameter.

    Raises
    ------
    ValueError
        If the parameter name is not found in parameter_names.
    """
    # Convert parameter_names keys to a list to get the index
    try:
        parameter_index = list(parameter_names.keys()).index(parameter_name)
    except ValueError:
        raise ValueError(f"Parameter '{parameter_name}' not found in parameter_names.")
    
    args[parameter_index] = value
    
    # Retrieve the value from args
    return args