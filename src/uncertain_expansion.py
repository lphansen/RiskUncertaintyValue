import numpy as np
import autograd.numpy as anp
import scipy as sp
import seaborn as sns
from lin_quad import LinQuadVar
from lin_quad_util import E, cal_E_ww, matmul, concat, simulate, next_period, kron_prod, log_E_exp, lq_sum
from elasticity import exposure_elasticity, price_elasticity
from utilities import mat, vec, sym, gschur
from derivatives import compute_derivatives
from numba import prange
from scipy import optimize


"""
This Python script provides functions to solve for the discrete-time dynamic macro-finance models under uncertainty, based on the perturbation method.
These models feature EZ recursive preferences.

Developed and maintained by the MFR research team.
Updated on Dec. 31, 2022, 9:34 P.M. CT
"""

def uncertain_expansion(eq, ss, var_shape, args, scale_fun_list, adj_loc_list, endo_loc_list, init_N = '1', util_adj = False, tol = 1e-8, max_iter = 50):

    ## Dimensions, derivatives, and parameters
    n_Y, n_Z, n_W = var_shape
    n_X = n_Y + n_Z
    γ = args[0]
    β = args[1]
    ρ = args[2]
    df, ss = take_derivatives(eq, ss, var_shape, True, args)
    df_state = state_derivatives(df, (n_Y, n_Z, n_W), endo_loc_list)

    ## Zeroth order expansion
    Z0_t = LinQuadVar({'c': ss[n_Y+2:].reshape(-1, 1)}, (n_Z, n_Z, n_W), False)
    Y0_t = LinQuadVar({'c': ss[2:n_Y+2].reshape(-1, 1)}, (n_Y, n_Z, n_W), False)
    X0_t = concat([Y0_t, Z0_t])
    Z1_t = LinQuadVar({'x': np.eye(n_Z)}, (n_Z, n_Z, n_W), False)
    Z2_t = LinQuadVar({'x2': np.eye(n_Z)}, (n_Z, n_Z, n_W), False)

    if init_N == '1':
        Z1_tp1, Z2_tp1, util_sol, N_cm, Y1_t, Y2_t, init_N_res = initial(df, ss, var_shape, No_N_expansion, args, scale_fun_list)
    elif init_N == 'N0':
        Z1_tp1, Z2_tp1, util_sol, N_cm, Y1_t, Y2_t, init_N_res = initial(df, ss, var_shape, N_0_expansion, args, scale_fun_list)
        
    H_tilde_0_series = [N_cm['H_tilde_0']]
    H_tilde_1_series = [N_cm['H_tilde_1']]
    Λ_tilde_inv_series = [N_cm['Λ_tilde_inv']]
    Y1_t_series = [Y1_t(*(np.ones([n_Z,1]),np.ones([n_Z,1]),np.ones([n_Z,1])))]
    Y2_t_series = [Y2_t(*(np.ones([n_Z,1]),np.ones([n_Z,1]),np.ones([n_Z,1])))]
    error_series = []

    i = 0
    error = 1

    while error > tol and i < max_iter:
        ## Step 1: Update State Evolution Equation
        Z1_tp1_tilde, Z2_tp1_tilde, ψ_tilde_x, ψ_tilde_w, ψ_tilde_q, ψ_x2, ψ_tilde_xx, ψ_tilde_xw, ψ_tilde_xq, ψ_tilde_ww, ψ_tilde_wq, ψ_tilde_qq,\
            = update_state_evolution(Z1_tp1, Z2_tp1, N_cm, var_shape)

        ## Step 2: Update Jump Variables
        Y1_t, X1_t, X1_tp1_tilde, schur_first, D, C_mix = solve_jump_first(df, util_sol, Z1_tp1_tilde, N_cm, var_shape)
        Y2_t, X2_t, X2_tp1_tilde, schur_second, D2, D_tilde, G = solve_jump_second(df, util_sol, Z1_tp1_tilde, Z2_tp1_tilde, X1_t, X1_tp1_tilde, N_cm, var_shape)
        Z1_tp1, Z2_tp1 = update_state_evolution_org(df_state, var_shape, X1_t, X2_t, endo_loc_list)

        ## Step 3: Update Continuation Values and Change of measure
        X1_tp1 = next_period(X1_t, Z1_tp1)
        X2_tp1 = next_period(X2_t, Z1_tp1, Z2_tp1)
        args = list(args)
        args[-1] = True
        args = tuple(args)
        scale_fun = [approximate_fun(fun_approx, ss[2:], (1, n_Z, n_W), X1_t, X2_t, Z1_tp1, Z2_tp1, args) for fun_approx in scale_fun_list]
        if len(adj_loc_list) == 0:
            scale1 = lq_sum([approx[2] for approx in scale_fun])
            scale2 = lq_sum([approx[3] for approx in scale_fun])
        else:
            scale1 = lq_sum([approx[2] for approx in scale_fun]) + lq_sum([X1_tp1[i-1]-X1_t[i-1] for i in adj_loc_list])
            scale2 = lq_sum([approx[3] for approx in scale_fun]) + lq_sum([X2_tp1[i-1]-X2_t[i-1] for i in adj_loc_list])
        util_sol = solve_utility(γ, β, ρ, Z1_tp1, Z2_tp1, ss, var_shape, sum([approx[1] for approx in scale_fun]), scale1, scale2, X1_t, X2_t, adj_loc_list, util_adj)
        log_N = util_sol['log_N_tilde']
        N_cm = N_tilde_measure(log_N, var_shape)
        
        ## Check convergence
        H_tilde_0_series.append(N_cm['H_tilde_0'])
        H_tilde_1_series.append(N_cm['H_tilde_1'])
        Λ_tilde_inv_series.append(N_cm['Λ_tilde_inv'])
        Y1_t_series.append(Y1_t(*(np.ones([n_Z,1]),np.ones([n_Z,1]),np.ones([n_Z,1]))))
        Y2_t_series.append(Y2_t(*(np.ones([n_Z,1]),np.ones([n_Z,1]),np.ones([n_Z,1]))))
        error_1 = np.max(np.abs(H_tilde_0_series[i+1] - H_tilde_0_series[i]))
        error_2 = np.max(np.abs(H_tilde_1_series[i+1] - H_tilde_1_series[i]))
        error_3 = np.max(np.abs(Λ_tilde_inv_series[i+1] - Λ_tilde_inv_series[i]))
        error_4 = np.max(np.abs(Y1_t_series[i+1] - Y1_t_series[i])) 
        error_5 = np.max(np.abs(Y2_t_series[i+1] - Y2_t_series[i])) 
        error = np.max([error_1, error_2, error_3, error_4, error_5])
        # print(error)
        error_series.append(error)
        
        i+=1
        
    X_t = X0_t + X1_t + 0.5*X2_t
    X_tp1 = next_period(X_t, Z1_tp1, Z2_tp1)

    return ModelSolution({
                        'Z0_t':Z0_t, 'Z1_t':Z1_t, 'Z2_t':Z2_t, 'Z1_tp1': Z1_tp1, 'Z2_tp1': Z2_tp1,
                        'Y0_t':Y0_t, 'Y1_t':Y1_t, 'Y2_t':Y2_t,
                        'X0_t':X0_t, 'X1_t':X1_t, 'X2_t':X2_t, 'X1_tp1': X1_tp1, 'X2_tp1': X2_tp1, 'X_t': X_t, 'X_tp1': X_tp1,
                        'scale_tp1': scale_fun, 'init_N_res': init_N_res,
                        'util_sol':util_sol, 'log_N': log_N, 'N_cm':N_cm,
                        'var_shape': var_shape, 'args':args,
                        'ss': ss[2:], 'ss_vmc': ss[0], 'ss_rmc': ss[1],
                        'second_order': True})

def initial(df, ss, var_shape, init_expansion, args, scale_fun_list):

    n_Y, n_Z, n_W = var_shape
    γ = args[0]
    init_N_res = init_expansion(df, ss, var_shape, γ)
    Z1_tp1 = init_N_res['Z1_tp1']
    Z2_tp1 = init_N_res['Z2_tp1']
    Y1_t = concat(init_N_res['X1_t'].split()[2:-n_Z])
    Y2_t = concat(init_N_res['X2_t'].split()[2:-n_Z])
    util_sol = {'vmc1_t':init_N_res['vmc1_t'],\
                'rmc1_t':init_N_res['rmc1_t'],\
                'vmc2_t':init_N_res['vmc2_t'],\
                'rmc2_t':init_N_res['rmc2_t']}
    args = list(args)
    args[-1] = False
    args = tuple(args)
    scale_fun = [approximate_fun(fun_approx, ss, (1, n_Z, n_W), init_N_res['X1_t'], init_N_res['X2_t'], Z1_tp1, Z2_tp1, args) for fun_approx in scale_fun_list]
    scale1 = lq_sum([approx[2] for approx in scale_fun])
    scale2 = lq_sum([approx[3] for approx in scale_fun])
    log_N = form_log_N(γ, next_period(util_sol['vmc1_t'], Z1_tp1), next_period(util_sol['vmc2_t'],Z1_tp1,Z2_tp1), scale1, scale2)
    N_cm = N_tilde_measure(log_N,var_shape)

    return Z1_tp1, Z2_tp1, util_sol, N_cm, Y1_t, Y2_t, init_N_res

def state_derivatives(df, var_shape, endo_loc_list):

    df = df.copy()
    n_Y, n_Z, n_W = var_shape
    n_X = n_Y + n_Z
    n_endo = len(endo_loc_list)
    if n_endo == 0:
        x_index = np.repeat(False,n_Y+2)
        df_s = {
            'xt':df['xt'][-n_Z:,-n_Z:],
            'xtp1':df['xtp1'][-n_Z:,-n_Z:],
            'wtp1':df['wtp1'][-n_Z:,:],
            'q':df['q'][-n_Z:],
            'xtxt':df['xtxt'][-n_Z:,-n_Z*(n_X+2):][:,np.tile(np.concatenate([x_index,np.repeat(True,n_Z)]),n_Z+n_endo)],
            'xtxtp1':df['xtxtp1'][-n_Z:,-n_Z*(n_X+2):][:,np.tile(np.concatenate([x_index,np.repeat(True,n_Z)]),n_Z+n_endo)],
            'xtwtp1':df['xtwtp1'][-n_Z:,-n_Z*n_W:],
            'xtq':df['xtq'][-n_Z:,:][:,np.concatenate((x_index,np.repeat(True,n_Z)))],
            'xtp1xtp1':df['xtp1xtp1'][-n_Z:,-n_Z*(n_X+2):][:,np.tile(np.concatenate([x_index,np.repeat(True,n_Z)]),n_Z+n_endo)],
            'xtp1wtp1':df['xtp1wtp1'][-n_Z:,-n_Z*n_W:],
            'xtp1q':df['xtp1q'][-n_Z:,:][:,np.concatenate((x_index,np.repeat(True,n_Z)))],
            'wtp1wtp1':df['wtp1wtp1'][-n_Z:,:],
            'wtp1q':df['wtp1q'][-n_Z:,:],
            'qq':df['qq'][-n_Z:,:]
        }
    elif n_endo == 1:
        x_index = np.repeat(False,n_Y+2)
        x_index[endo_loc_list[0]+1] = True
        df_s = {
            'xt':np.concatenate((df['xt'][-n_Z:,[i+1 for i in endo_loc_list]],df['xt'][-n_Z:,-n_Z:]),axis = 1),
            'xtp1':np.concatenate((df['xtp1'][-n_Z:,[i+1 for i in endo_loc_list]],df['xtp1'][-n_Z:,-n_Z:]),axis = 1),
            'wtp1':df['wtp1'][-n_Z:,:],
            'q':df['q'][-n_Z:],
            'xtxt':np.concatenate((df['xtxt'][-n_Z:,(endo_loc_list[0]+1)*(n_X+2):(endo_loc_list[0]+2)*(n_X+2)],df['xtxt'][-n_Z:,-n_Z*(n_X+2):]),axis = 1)[:,np.tile(np.concatenate([x_index,np.repeat(True,n_Z)]),n_Z+n_endo)],
            'xtxtp1':np.concatenate((df['xtxtp1'][-n_Z:,(endo_loc_list[0]+1)*(n_X+2):(endo_loc_list[0]+2)*(n_X+2)],df['xtxtp1'][-n_Z:,-n_Z*(n_X+2):]),axis = 1)[:,np.tile(np.concatenate([x_index,np.repeat(True,n_Z)]),n_Z+n_endo)],
            'xtwtp1':np.concatenate((df['xtwtp1'][-n_Z:,(endo_loc_list[0]+1)*n_W:(endo_loc_list[0]+2)*n_W],df['xtwtp1'][-n_Z:,-n_Z*n_W:]),axis = 1),
            'xtq':df['xtq'][-n_Z:,:][:,np.concatenate((x_index,np.repeat(True,n_Z)))],
            'xtp1xtp1':np.concatenate((df['xtp1xtp1'][-n_Z:,(endo_loc_list[0]+1)*(n_X+2):(endo_loc_list[0]+2)*(n_X+2)],df['xtp1xtp1'][-n_Z:,-n_Z*(n_X+2):]),axis = 1)[:,np.tile(np.concatenate([x_index,np.repeat(True,n_Z)]),n_Z+n_endo)],
            'xtp1wtp1':np.concatenate((df['xtp1wtp1'][-n_Z:,(endo_loc_list[0]+1)*n_W:(endo_loc_list[0]+2)*n_W],df['xtp1wtp1'][-n_Z:,-n_Z*n_W:]),axis = 1),
            'xtp1q':df['xtp1q'][-n_Z:,:][:,np.concatenate((x_index,np.repeat(True,n_Z)))],
            'wtp1wtp1':df['wtp1wtp1'][-n_Z:,:],
            'wtp1q':df['wtp1q'][-n_Z:,:],
            'qq':df['qq'][-n_Z:,:]
        }
    elif n_endo == 2:
        x_index = np.repeat(False,n_Y+2)
        x_index[endo_loc_list[0]+1] = True
        x_index[endo_loc_list[1]+1] = True
        df_s = {
            'xt':np.concatenate((df['xt'][-n_Z:,[i+1 for i in endo_loc_list]],df['xt'][-n_Z:,-n_Z:]),axis = 1),
            'xtp1':np.concatenate((df['xtp1'][-n_Z:,[i+1 for i in endo_loc_list]],df['xtp1'][-n_Z:,-n_Z:]),axis = 1),
            'wtp1':df['wtp1'][-n_Z:,:],
            'q':df['q'][-n_Z:],
            'xtxt':np.concatenate((df['xtxt'][-n_Z:,(endo_loc_list[0]+1)*(n_X+2):(endo_loc_list[0]+2)*(n_X+2)],\
                                df['xtxt'][-n_Z:,(endo_loc_list[1]+1)*(n_X+2):(endo_loc_list[1]+2)*(n_X+2)],\
                                df['xtxt'][-n_Z:,-n_Z*(n_X+2):]),axis = 1)[:,np.tile(np.concatenate([x_index,np.repeat(True,n_Z)]),n_Z+n_endo)],
            'xtxtp1':np.concatenate((df['xtxtp1'][-n_Z:,(endo_loc_list[0]+1)*(n_X+2):(endo_loc_list[0]+2)*(n_X+2)],\
                                    df['xtxtp1'][-n_Z:,(endo_loc_list[1]+1)*(n_X+2):(endo_loc_list[1]+2)*(n_X+2)],\
                                    df['xtxtp1'][-n_Z:,-n_Z*(n_X+2):]),axis = 1)[:,np.tile(np.concatenate([x_index,np.repeat(True,n_Z)]),n_Z+n_endo)],
            'xtwtp1':np.concatenate((df['xtwtp1'][-n_Z:,(endo_loc_list[0]+1)*n_W:(endo_loc_list[0]+2)*n_W],\
                                    df['xtwtp1'][-n_Z:,(endo_loc_list[1]+1)*n_W:(endo_loc_list[1]+2)*n_W],\
                                    df['xtwtp1'][-n_Z:,-n_Z*n_W:]),axis = 1),
            'xtq':df['xtq'][-n_Z:,:][:,np.concatenate((x_index,np.repeat(True,n_Z)))],
            'xtp1xtp1':np.concatenate((df['xtp1xtp1'][-n_Z:,(endo_loc_list[0]+1)*(n_X+2):(endo_loc_list[0]+2)*(n_X+2)],\
                                    df['xtp1xtp1'][-n_Z:,(endo_loc_list[1]+1)*(n_X+2):(endo_loc_list[1]+2)*(n_X+2)],\
                                    df['xtp1xtp1'][-n_Z:,-n_Z*(n_X+2):]),axis = 1)[:,np.tile(np.concatenate([x_index,np.repeat(True,n_Z)]),n_Z+n_endo)],
            'xtp1wtp1':np.concatenate((df['xtp1wtp1'][-n_Z:,(endo_loc_list[0]+1)*n_W:(endo_loc_list[0]+2)*n_W],\
                                    df['xtp1wtp1'][-n_Z:,(endo_loc_list[1]+1)*n_W:(endo_loc_list[1]+2)*n_W],\
                                    df['xtp1wtp1'][-n_Z:,-n_Z*n_W:]),axis = 1),
            'xtp1q':df['xtp1q'][-n_Z:,:][:,np.concatenate((x_index,np.repeat(True,n_Z)))],
            'wtp1wtp1':df['wtp1wtp1'][-n_Z:,:],
            'wtp1q':df['wtp1q'][-n_Z:,:],
            'qq':df['qq'][-n_Z:,:]
        }
    return df_s

def update_state_evolution(Z1_tp1, Z2_tp1, N_cm, var_shape):
    
    n_Y, n_Z, n_W = var_shape

    ψ_x  = Z1_tp1['x']
    ψ_w  = Z1_tp1['w']
    ψ_q  = Z1_tp1['c']
    ψ_x2 = Z2_tp1['x2']
    ψ_xx = Z2_tp1['xx']
    ψ_xw = Z2_tp1['xw']
    ψ_xq = Z2_tp1['x']
    ψ_ww = Z2_tp1['ww']
    ψ_wq = Z2_tp1['w']
    ψ_qq = Z2_tp1['c']

    ψ_tilde_x = ψ_x + ψ_w@N_cm['H_tilde_1']
    ψ_tilde_w = ψ_w@N_cm['Γ'] 
    ψ_tilde_q = ψ_q + ψ_w@N_cm['H_tilde_0']
    Z1_tp1 = LinQuadVar({'x': ψ_tilde_x, 'w': ψ_tilde_w, 'c': ψ_tilde_q}, (n_Z, n_Z, n_W), False)
    # Second order
    Ix = np.eye(n_Z)
    ψ_tilde_xq = ψ_xq + ψ_xw@np.kron(Ix,N_cm['H_tilde_0']) + ψ_ww@np.kron(N_cm['H_tilde_1'],N_cm['H_tilde_0']) + ψ_ww@np.kron(N_cm['H_tilde_0'],N_cm['H_tilde_1']) + ψ_wq@N_cm['H_tilde_1']
    ψ_tilde_wq = ψ_wq@N_cm['Γ'] + ψ_ww@np.kron(N_cm['Γ'],N_cm['H_tilde_0']) + ψ_ww@np.kron(N_cm['H_tilde_0'],N_cm['Γ'])
    ψ_tilde_qq = ψ_qq + ψ_ww@np.kron(N_cm['H_tilde_0'], N_cm['H_tilde_0']) + ψ_wq@N_cm['H_tilde_0']
    ψ_tilde_xx = ψ_xx + ψ_xw@np.kron(Ix,N_cm['H_tilde_1']) + ψ_ww@np.kron(N_cm['H_tilde_1'],N_cm['H_tilde_1'])
    ψ_tilde_xw = ψ_xw@np.kron(Ix,N_cm['Γ']) + ψ_ww@np.kron(N_cm['H_tilde_1'],N_cm['Γ']) + ψ_ww@kron_comm(np.kron(N_cm['Γ'],N_cm['H_tilde_1']),n_W,n_Z)
    ψ_tilde_ww = ψ_ww@np.kron(N_cm['Γ'],N_cm['Γ'])
    Z2_tp1 = LinQuadVar({'x2': ψ_x2, 'xx': ψ_tilde_xx, 'xw': ψ_tilde_xw, 'ww': ψ_tilde_ww, 'x': ψ_tilde_xq, 'w': ψ_tilde_wq, 'c': ψ_tilde_qq}, (n_Z, n_Z, n_W),False)
    
    return Z1_tp1, Z2_tp1, ψ_tilde_x, ψ_tilde_w, ψ_tilde_q, ψ_x2, ψ_tilde_xx, ψ_tilde_xw, ψ_tilde_xq, ψ_tilde_ww, ψ_tilde_wq, ψ_tilde_qq

def solve_jump_first(df, util_sol, Z1_tp1, change_of_measure, var_shape):
    
    n_Y, n_Z, n_W = var_shape
    n_X = n_Y + n_Z
    df_p = plug_in_df_first_order(df, var_shape, util_sol)
    H2 = np.block([[np.zeros([n_W,n_Y]),change_of_measure['H_tilde_1']]])
    df_mix = df_p['wtp1']@H2
    schur_mix = schur_decomposition(-df_p['xtp1'], df_p['xt'] + df_mix, (n_Y, n_Z, n_W))
    
    adj = np.zeros((n_X, 1))
    nonzero_vr_df = df['xtp1'][-n_X:,0].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        adj[nonzero_loc[0],0] += df['xtp1'][nonzero_loc[0]+2,0]*util_sol['vmc1_t']['c'] 
    nonzero_vr_df = df['xt'][-n_X:,1].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        adj[nonzero_loc[0],0] += df['xt'][nonzero_loc[0]+2,1]*util_sol['rmc1_t']['c']

    LHS = df_p['xt']+df_mix+df_p['xtp1']
    RHS = -(df_p['wtp1']@change_of_measure['H_tilde_0']+adj)
    ### The following steps are special for the permanent income model
    if np.linalg.matrix_rank(LHS) == len(LHS):
        D = np.linalg.solve(LHS, RHS)
    else:
        new_column = np.zeros((LHS.shape[0],1))
        new_column[3,0] = 1
        new_row = np.zeros((1, LHS.shape[1]+1))
        new_row[0,n_Y+1] = 1
        LHS = np.vstack((np.hstack((LHS, new_column)),new_row))
        RHS = np.vstack((RHS, np.zeros(1)))
        
        D = np.linalg.solve(LHS, RHS)
        discount_adj_1 = np.float(D[-1])
        D = D[:-1]
    ### Permanent income model special treatment ends here

    # D = -np.linalg.inv(df_p['xt']+df_mix+df_p['xtp1'])@(df_p['wtp1']@change_of_measure['H_tilde_0']+adj)
    D2 = D[n_Y:]
    D1 = D[:n_Y]
    C_mix = D1-schur_mix['N']@D2

    Y1_t = LinQuadVar({'x': schur_mix['N'], 'c': C_mix}, (n_Y,n_Z,n_W), False)
    Z1_t = LinQuadVar({'x': np.eye(n_Z)}, (n_Z, n_Z, n_W), False)
    X1_t = concat([Y1_t, Z1_t])
    X1_tp1 = next_period(X1_t, Z1_tp1)
    
    return Y1_t, X1_t, X1_tp1, schur_mix, D, C_mix

def solve_jump_second(df, util_sol, Z1_tp1, Z2_tp1, X1_t, X1_tp1, change_of_measure, var_shape):

    n_Y, n_Z, n_W = var_shape

    df_p = plug_in_df_second_order(df, var_shape, util_sol)
    schur = schur_decomposition(-df_p['xtp1'], df_p['xt'], (n_Y, n_Z, n_W))
    
    Wtp1 = LinQuadVar({'w': change_of_measure['Γ'], 'c':change_of_measure['H_tilde_0'], 'x': change_of_measure['H_tilde_1']}, (n_W, n_Z, n_W), False)
    D2 = combine_second_order_terms(df_p, X1_t, X1_tp1, Wtp1)   
    M_E_w = np.zeros([n_W,1]) 
    cov_w = np.eye(n_W)
    M_E_ww = cal_E_ww(M_E_w, cov_w)
    E_D2 = E(D2, M_E_w, M_E_ww)
    E_D2_coeff = np.block([[E_D2['c'], E_D2['x'], E_D2['xx']]])
    Z1Z1 = kron_prod(Z1_tp1, Z1_tp1)
    M_mat = form_M0(M_E_w, M_E_ww, Z1_tp1, Z1Z1)
    LHS = np.eye(n_Y*M_mat.shape[0]) - np.kron(M_mat.T, np.linalg.solve(schur['Λ22'], schur['Λp22']))
    RHS = vec(-np.linalg.solve(schur['Λ22'], (schur['Q'].T@E_D2_coeff)[-n_Y:]))
    D_tilde_vec = np.linalg.solve(LHS, RHS)
    D_tilde = mat(D_tilde_vec, (n_Y, 1+n_Z+n_Z**2))
    G = np.linalg.solve(schur['Z21'], D_tilde)

    Y2_t = LinQuadVar({'x2': schur['N'], 'xx': G[:, 1+n_Z:1+n_Z+(n_Z)**2], 'x': G[:, 1:1+n_Z], 'c': G[:, :1]}, (n_Y, n_Z, n_W), False)
    Z2_t = LinQuadVar({'x2': np.eye(n_Z)}, (n_Z, n_Z, n_W), False)
    X2_t = concat([Y2_t, Z2_t])
    X2_tp1 = next_period(X2_t, Z1_tp1, Z2_tp1)

    return Y2_t, X2_t, X2_tp1, schur, D2, D_tilde, G

def update_state_evolution_org(df_state, var_shape, X1_t, X2_t, endo_loc_list):

    n_Y, n_Z, n_W = var_shape
    endo_loc_list = [i-1 for i in endo_loc_list]

    Z1_t = LinQuadVar({'x': np.eye(n_Z)}, (n_Z, n_Z, n_W), False)
    if len(endo_loc_list) == 0:
        state_df1 = df_state.copy()
        state_df2 = df_state.copy()
    elif len(endo_loc_list) == 1 or len(endo_loc_list) == 2:
        state_df1 = plug_in_state_first_order(df_state, (0, n_Z, n_W), X1_t, endo_loc_list)
        state_df2 = plug_in_state_second_order(df_state, (0, n_Z, n_W), X1_t, X2_t, endo_loc_list)
    elif len(endo_loc_list) > 2:
        print('Only two endogenous variables allowed in state evolution equtions in current version of codes.')
    ψ_x = np.linalg.solve(-state_df1['xtp1'], state_df1['xt'])
    ψ_w = np.linalg.solve(-state_df1['xtp1'], state_df1['wtp1'])
    ψ_q = np.linalg.solve(-state_df1['xtp1'], state_df1['q'].reshape(n_Z,1))
    Z1_tp1 = LinQuadVar({'x': ψ_x, 'w': ψ_w, 'c': ψ_q},(n_Z, n_Z, n_W), False)
    
    Wtp1 = LinQuadVar({'w': np.eye(n_W)}, (n_W, n_Z, n_W), False)
    D2 = combine_second_order_terms(state_df2, Z1_t, Z1_tp1, Wtp1)
    ψ_x2 = np.linalg.solve(-state_df2['xtp1'], state_df2['xt'])
    ψ_xq = np.linalg.solve(-state_df2['xtp1'], D2['x'])
    ψ_wq = np.linalg.solve(-state_df2['xtp1'], D2['w'])
    ψ_qq = np.linalg.solve(-state_df2['xtp1'], D2['c'])
    ψ_xx = np.linalg.solve(-state_df2['xtp1'], D2['xx'])
    ψ_xw = np.linalg.solve(-state_df2['xtp1'], D2['xw'])
    ψ_ww = np.linalg.solve(-state_df2['xtp1'], D2['ww'])
    Z2_tp1 = LinQuadVar({'x2': ψ_x2, 'xx': ψ_xx, 'xw': ψ_xw, 'ww': ψ_ww, 'x': ψ_xq, 'w': ψ_wq, 'c': ψ_qq}, (n_Z, n_Z, n_W), False)

    return Z1_tp1, Z2_tp1

def take_derivatives(f, ss, var_shape, second_order,
                      args):
    """
    Take first- or second-order derivatives.

    """
    _, _, n_W = var_shape

    W_0 = np.zeros(n_W)
    q_0 = 0.

    if callable(ss):
        ss = ss(*args)

    df = compute_derivatives(f=lambda X_t, X_tp1, W_tp1, q:
                             anp.atleast_1d(f(X_t, X_tp1, W_tp1, q, *args)),
                             X=[ss, ss, W_0, q_0],
                             second_order=second_order)
    return df, ss

def schur_decomposition(df_tp1, df_t, var_shape):
    
    n_Y, n_Z, n_W = var_shape
    Λp, Λ, a, b, Q, Z = gschur(df_tp1, df_t)
    Λp22 = Λp[-n_Y:, -n_Y:]
    Λ22 = Λ[-n_Y:, -n_Y:]
    Z21 = Z.T[-n_Y:, :n_Y]
    Z22 = Z.T[-n_Y:, n_Y:]
    N = -np.linalg.solve(Z21, Z22)
    N_block = np.block([[N], [np.eye(n_Z)]])
    schur_decomposition = {'N':N,'N_block':N_block,'Λp':Λp,'Λ':Λ,'Q':Q,'Z':Z,'Λp22':Λp22,'Λ22':Λ22,'Z21':Z21,'Z22':Z22}
    
    return schur_decomposition

def combine_second_order_terms(df, X1_t, X1_tp1, Wtp1):

    _, n_Z, n_W = X1_tp1.shape

    xtxt = kron_prod(X1_t, X1_t)
    xtxtp1 = kron_prod(X1_t, X1_tp1)
    xtwtp1 = kron_prod(X1_t, Wtp1)
    xtp1xtp1 = kron_prod(X1_tp1, X1_tp1)
    xtp1wtp1 = kron_prod(X1_tp1, Wtp1)
    wtp1wtp1 = kron_prod(Wtp1, Wtp1)
    res = matmul(df['xtxt'], xtxt)\
        + matmul(2*df['xtxtp1'], xtxtp1)\
        + matmul(2*df['xtwtp1'], xtwtp1)\
        + matmul(2*df['xtq'], X1_t)\
        + matmul(df['xtp1xtp1'], xtp1xtp1)\
        + matmul(2*df['xtp1wtp1'], xtp1wtp1)\
        + matmul(2*df['xtp1q'], X1_tp1)\
        + matmul(df['wtp1wtp1'], wtp1wtp1)\
        + matmul(2*df['wtp1q'], Wtp1)\
        + LinQuadVar({'c': df['qq']}, (df['qq'].shape[0], n_Z, n_W), False)

    return res

def form_M0(M0_E_w, M0_E_ww, Z1_tp1, Z1Z1):

    _, n_Z, n_W = Z1_tp1.shape
    M0_mat_11 = np.eye(1)
    M0_mat_12 = np.zeros((1, n_Z))
    M0_mat_13 = np.zeros((1, n_Z**2))
    M0_mat_21 = Z1_tp1['w']@M0_E_w + Z1_tp1['c']
    M0_mat_22 = Z1_tp1['x']
    M0_mat_23 = np.zeros((n_Z, n_Z**2))
    M0_mat_31 = Z1Z1['ww']@M0_E_ww + Z1Z1['w']@M0_E_w + Z1Z1['c']
    temp = np.vstack([M0_E_w.T@mat(Z1Z1['xw'][row: row+1, :], (n_W, n_Z))
                      for row in range(Z1Z1.shape[0])])    
    M0_mat_32 = temp + Z1Z1['x']
    M0_mat_33 = Z1Z1['xx']
    M0_mat = np.block([[M0_mat_11, M0_mat_12, M0_mat_13],
                       [M0_mat_21, M0_mat_22, M0_mat_23],
                       [M0_mat_31, M0_mat_32, M0_mat_33]])
    return M0_mat

def N_tilde_measure(log_N, var_shape):

    n_Y, n_Z, n_W = var_shape

    Ψ_0 = log_N['w']
    Ψ_1 = log_N['xw']
    Ψ_2 = log_N['ww']
    
    Λ = -sym(mat(2*Ψ_2,(n_W,n_W)))
    H_0 = Ψ_0.T
    H_1 = mat(Ψ_1, (n_W, n_Z))
    Λ_tilde = np.eye(n_W) + Λ
    Λ_tilde_inv = np.linalg.inv(Λ_tilde) 
    H_tilde_0 = Λ_tilde_inv@H_0
    H_tilde_1 = Λ_tilde_inv@H_1
    Γ = sp.linalg.sqrtm(Λ_tilde_inv)
    H_tilde_1_aug = np.block([[np.zeros([n_W,n_Y]),H_tilde_1]])
    change_of_measure = {'Λ':Λ, 'H_0':H_0, 'H_1':H_1, 'Λ_tilde':Λ_tilde, 'Λ_tilde_inv':Λ_tilde_inv,'H_tilde_0':H_tilde_0,'H_tilde_1':H_tilde_1,'Γ':Γ,'H_tilde_1_aug':H_tilde_1_aug}
    
    return change_of_measure

def kron_comm(AB, nX, nW):
    if not np.any(AB):
        return AB
    kcAB = np.zeros(AB.shape)
    for i in prange(AB.shape[0]):
        kcAB[i] = vec(mat(AB[i:i+1, :].T, (nX, nW)).T).T
    return kcAB

def form_log_N(γ, vmc1_tp1,vmc2_tp1,c_scale1,c_scale2):
    
    numerator = (1-γ)*(vmc1_tp1 + c_scale1 + 0.5*(vmc2_tp1 + c_scale2))
    denominator = log_E_exp((1-γ)*(vmc1_tp1 + c_scale1 + 0.5*(vmc2_tp1 + c_scale2)))
    
    return numerator - denominator

def approximate_fun(fun, ss, var_shape, X1_t, X2_t, Z1_tp1, Z2_tp1, args):
    
    _, n_Z, n_W = var_shape
    
    W_0 = np.zeros(n_W)
    q_0 = 0.
    X1_tp1 = next_period(X1_t, Z1_tp1)
    X2_tp1 = next_period(X2_t, Z1_tp1, Z2_tp1)
    dfun = compute_derivatives(f=lambda X_t, X_tp1, W_tp1, q:
                                   anp.atleast_1d(fun(X_t, X_tp1, W_tp1, q, *args)),
                                   X=[ss, ss, W_0, q_0],
                                   second_order=True)

    fun_zero_order = fun(ss, ss, W_0, q_0, *args)
    fun_first_order = matmul(dfun['xtp1'], X1_tp1)\
        + matmul(dfun['xt'], X1_t)\
        + LinQuadVar({'w': dfun['wtp1'], 'c': dfun['q'].reshape(-1, 1)},
                     (1, n_Z, n_W), False)
    fun_approx = fun_zero_order + fun_first_order
    Wtp1 = LinQuadVar({'w': np.eye(n_W)}, (n_W, n_Z, n_W), False)

    temp1 = combine_second_order_terms(dfun, X1_t, X1_tp1, Wtp1)
    temp2 = matmul(dfun['xt'], X2_t)\
        + matmul(dfun['xtp1'], X2_tp1)
    fun_second_order = temp1 + temp2
    fun_approx = fun_approx + fun_second_order*0.5

    return fun_approx, fun_zero_order, fun_first_order, fun_second_order
    
def solve_utility(γ, β, ρ, Z1_tp1, Z2_tp1, ss, var_shape, c_scale0, c_scale1, c_scale2, X1_t, X2_t, c_adj_loc_list, util_adj = False, tol = 1e-10):

    n_Y, n_Z, n_W = var_shape

    def return_order1_t(order1_t_coeffs):
        return LinQuadVar({'c': np.array([[order1_t_coeffs[0]]]), 'x':np.array([order1_t_coeffs[1:(1+n_Z)]])},(1, n_Z, n_W))
    
    def return_order2_t(order2_t_coeffs):
            return LinQuadVar({'c': np.array([[order2_t_coeffs[0]]]), 'x':np.array([order2_t_coeffs[1:(1+n_Z)]]),\
                            'x2':np.array([order2_t_coeffs[(1+n_Z):(1+n_Z+n_Z)]]), 'xx':np.array([order2_t_coeffs[(1+n_Z+n_Z):(1+n_Z+n_Z+n_Z*n_Z)]])},(1, n_Z, n_W))
    
    λ = β * np.exp((1-ρ) * c_scale0)
    
    def solve_vmc1_t(order1_init_coeffs):
        vmc1_t = return_order1_t(order1_init_coeffs)
        LHS = λ/(1-γ) *log_E_exp((1-γ)*(next_period(vmc1_t,Z1_tp1)+c_scale1))
        LHS_list = [LHS['c'].item()]+ [i for i in LHS['x'][0]] 
        return list(np.array(LHS_list) - np.array(order1_init_coeffs))   
        
    vmc1_t_sol = optimize.root(solve_vmc1_t, x0 = [0]*(1 + n_Z),tol = tol)
    if vmc1_t_sol['success'] ==False:
        print(vmc1_t_sol['message'])
    vmc1_t = return_order1_t(vmc1_t_sol['x'])
    rmc1_t = log_E_exp((1-γ)*(next_period(vmc1_t,Z1_tp1)+c_scale1))/(1-γ)
    vmc1_tp1 = next_period(vmc1_t,Z1_tp1)
    log_N0 = ((1-γ)*(vmc1_tp1 + c_scale1)-log_E_exp((1-γ)*(vmc1_tp1 + c_scale1)))
    Ew0 = log_N0['w'].T
    Eww0 = cal_E_ww(Ew0,np.eye(Ew0.shape[0]))

    def solve_vmc2_t(order2_init_coeffs):
        vmc2_t = return_order2_t(order2_init_coeffs)
        LHS = λ *E(next_period(vmc2_t, Z1_tp1, Z2_tp1) + c_scale2, Ew0, Eww0) + (1-ρ)*(1-λ)/λ*kron_prod(vmc1_t,vmc1_t)
        LHS_list = [LHS['c'].item()]+ [i for i in LHS['x'][0]] + [i for i in LHS['x2'][0]] + [i for i in LHS['xx'][0]] 
        return list(np.array(LHS_list) - np.array(order2_init_coeffs))
    
    vmc2_t_sol = optimize.root(solve_vmc2_t, x0 = [0]*(1 + n_Z+n_Z+n_Z*n_Z),tol = tol)
    if vmc2_t_sol['success'] ==False:
        print(vmc2_t_sol['message'])
    vmc2_t = return_order2_t(vmc2_t_sol['x'])
    rmc2_t = E(next_period(vmc2_t, Z1_tp1, Z2_tp1) + c_scale2, Ew0,Eww0)
    vmc2_tp1 = next_period(vmc2_t, Z1_tp1, Z2_tp1)
    log_N_tilde = (1-γ)*(vmc1_tp1+c_scale1 + 0.5*(vmc2_tp1+c_scale2))-log_E_exp((1-γ)*(vmc1_tp1+c_scale1 + 0.5*(vmc2_tp1+c_scale2)))
    vmc0_t = LinQuadVar({'c':np.array([[ss[0]]])},(1,n_Z,n_W))
    rmc0_t = LinQuadVar({'c':np.array([[ss[1]]])},(1,n_Z,n_W))
    if util_adj == True:
        vmc0_t = LinQuadVar({'c':np.array([[ss[0]+sum(ss[2+i-1] for i in c_adj_loc_list)]])},(1,n_Z,n_W))
        rmc0_t = LinQuadVar({'c':np.array([[ss[1]+sum(ss[2+i-1] for i in c_adj_loc_list)]])},(1,n_Z,n_W))
        vmc1_t += lq_sum([X1_t[i-1] for i in c_adj_loc_list])
        vmc2_t += lq_sum([X2_t[i-1] for i in c_adj_loc_list])
        rmc1_t += lq_sum([X1_t[i-1] for i in c_adj_loc_list])
        rmc2_t += lq_sum([X2_t[i-1] for i in c_adj_loc_list])
    vmc_t = vmc0_t + vmc1_t + 0.5 * vmc2_t
    rmc_t = rmc0_t + rmc1_t + 0.5 * rmc2_t

    util_sol = {'vmc1_t':vmc1_t,'vmc2_t':vmc2_t,'rmc1_t':rmc1_t,'rmc2_t':rmc2_t,'vmc_t':vmc_t,'rmc_t':rmc_t,\
                'log_N0':log_N0,'log_N_tilde':log_N_tilde,'c_scale1':c_scale1,'c_scale2':c_scale2}

    return util_sol

def E_N_tp1(Y, change_of_measure):
    
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

def solve_pd(γ, β, ρ, util_sol, Z1_tp1, Z2_tp1, gc0, gd0, var_shape, change_of_measure, tol = 1e-10):
    
    n_Y, n_Z, n_W = var_shape
    
    def return_order1_t(order1_t_coeffs):
        return LinQuadVar({'c': np.array([[order1_t_coeffs[0]]]), 'x':np.array([order1_t_coeffs[1:(1+n_Z)]])},(1, n_Z, n_W))
    
    def return_order2_t(order2_t_coeffs):
            return LinQuadVar({'c': np.array([[order2_t_coeffs[0]]]), 'x':np.array([order2_t_coeffs[1:(1+n_Z)]]),\
                            'x2':np.array([order2_t_coeffs[(1+n_Z):(1+n_Z+n_Z)]]), 'xx':np.array([order2_t_coeffs[(1+n_Z+n_Z):(1+n_Z+n_Z+n_Z*n_Z)]])},(1, n_Z, n_W))
    
    gc1_tp1 = Z1_tp1[n_Z-1]
    gc2_tp1 = Z2_tp1[n_Z-1]
    gd1_tp1 = Z1_tp1[n_Z-2]
    gd2_tp1 = Z2_tp1[n_Z-2]
    
    η_m = (np.log(β) - ρ*gc0+gd0).item()
    dM1_tp1 = ((ρ-1)*(next_period(util_sol['vmc1_t'],Z1_tp1)-util_sol['rmc1_t'])-gc1_tp1)+gd1_tp1
    dM2_tp1 = ((ρ-1)*(next_period(util_sol['vmc2_t'],Z1_tp1,Z2_tp1)-util_sol['rmc2_t'])-gc2_tp1)+gd2_tp1

    def solve_pd1_t(order1_init_coeffs):
        pd1_t = return_order1_t(order1_init_coeffs)
        LHS = E_N_tp1(dM1_tp1 + np.exp(η_m) * next_period(pd1_t, Z1_tp1), change_of_measure)
        LHS_list = [LHS['c'].item()]+ [i for i in LHS['x'][0]] 
        return list(np.array(LHS_list) - np.array(order1_init_coeffs))
    
    pd1_t_sol = optimize.root(solve_pd1_t, x0 = [0]*(1 + n_Z),tol = tol)
    if pd1_t_sol['success'] ==False:
        print(pd1_t_sol['message'])
    pd1_t = return_order1_t(pd1_t_sol['x'])
    
    def solve_pd2_t(order2_init_coeffs):
        pd2_t = return_order2_t(order2_init_coeffs)
        LHS = E_N_tp1(dM2_tp1 + np.exp(η_m)*next_period(pd2_t,Z1_tp1,Z2_tp1),change_of_measure) - kron_prod(pd1_t,pd1_t) \
            + E_N_tp1(kron_prod(dM1_tp1,dM1_tp1)+ 2*np.exp(η_m)*kron_prod(next_period(pd1_t, Z1_tp1),dM1_tp1),change_of_measure)\
            + E_N_tp1(np.exp(η_m)*kron_prod(next_period(pd1_t, Z1_tp1),next_period(pd1_t, Z1_tp1)),change_of_measure) 
        LHS_list = [LHS['c'].item()]+ [i for i in LHS['x'][0]] + [i for i in LHS['x2'][0]] + [i for i in LHS['xx'][0]] 
        return list(np.array(LHS_list) - np.array(order2_init_coeffs))
    
    pd2_t_sol = optimize.root(solve_pd2_t, x0 = [0]*(1 + n_Z+n_Z+n_Z*n_Z),tol = tol)
        
    if pd2_t_sol['success'] ==False:
        print(pd2_t_sol['message'])
    pd2_t = return_order2_t(pd2_t_sol['x'])
    
    psi0 = np.log(np.exp(η_m)/(1-np.exp(η_m)))
    pd0_t = LinQuadVar({'c': np.array([[psi0]])},(1, n_Z, n_W))
    
    pd_t = pd0_t+pd1_t + 0.5*pd2_t
    pd_sol = {'pd_t':pd_t,'pd0_t':pd0_t,'pd1_t':pd1_t,'pd2_t':pd2_t,'η_m':η_m,'dM1_tp1':dM1_tp1,'dM2_tp1':dM2_tp1,'gc1_tp1':gc1_tp1,'gc2_tp1':gc2_tp1,'gd1_tp1':gd1_tp1,'gd2_tp1':gd2_tp1}
    
    return pd_sol

def No_N_expansion(df, ss, var_shape, γ):

    n_Y, n_Z, n_W = var_shape
    n_Y = n_Y + 2
    n_X = n_Y + n_Z

    ################################################################################################################################################################################
    ## Zeroth order expansion
    ################################################################################################################################################################################

    Z0_t = LinQuadVar({'c': ss[n_Y:].reshape(-1, 1)}, (n_Z, n_Z, n_W), False)
    Y0_t = LinQuadVar({'c': ss[:n_Y].reshape(-1, 1)}, (n_Y, n_Z, n_W), False)
    X0_t = concat([Y0_t, Z0_t])

    ################################################################################################################################################################################
    ## First order expansion
    ################################################################################################################################################################################

    ## Step 1: generalized Schur decomposition, solve for the x coefficients on jump variables
    schur = schur_decomposition(-df['xtp1'], df['xt'],(n_Y, n_Z, n_W))

    ## Step 2: solve for the coefficients on state variables
    f_1_xtp1 = df['xtp1'][:n_Y]
    f_1_wtp1 = df['wtp1'][:n_Y]
    f_2_xtp1 = df['xtp1'][n_Y:]
    f_2_xt = df['xt'][n_Y:]
    f_2_wtp1 = df['wtp1'][n_Y:]
    ψ_x = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_xt@schur['N_block'])
    ψ_w = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_wtp1)

    ## Step 3: generalized Schur decomposition, solve for constants on state and jump variables
    σ_v = (df['xtp1'][[0], :]@schur['N_block']@ψ_w + df['wtp1'][[0], :]).T
    μ_0 = (1 - γ) * σ_v
    adj = np.zeros((n_Y, 1))
    σ_v = σ_v.reshape(-1,)
    adj[0, 0] = 0.5 * (1 - γ)*σ_v.dot(σ_v)
    RHS = - np.block([[adj],[np.zeros((n_Z, 1))]])
    LHS = df['xtp1'] + df['xt']
    # D = np.linalg.solve(LHS, RHS)
    ### The following steps are special for the permanent income model
    if np.linalg.matrix_rank(LHS) == len(LHS):
        D = np.linalg.solve(LHS, RHS)
    else:
        new_column = np.zeros((LHS.shape[0],1))
        new_column[3,0] = 1
        new_row = np.zeros((1, LHS.shape[1]+1))
        new_row[0,n_Y+1] = 1
        LHS = np.vstack((np.hstack((LHS, new_column)),new_row))
        RHS = np.vstack((RHS, np.zeros(1)))
        
        D = np.linalg.solve(LHS, RHS)
        discount_adj_1 = np.float(D[-1])
        D = D[:-1]
    ### Permanent income model special treatment ends here
    C = D[:n_Y] - schur['N']@D[n_Y:]
    ψ_q = D[n_Y:] - ψ_x@D[n_Y:]

    ## Form the LinQuads for first order expansion when N = 1
    Z1_tp1 = LinQuadVar({'x': ψ_x, 'w': ψ_w, 'c': ψ_q}, (n_Z, n_Z, n_W), False)
    Z1Z1 = kron_prod(Z1_tp1, Z1_tp1)
    Z1_t = LinQuadVar({'x': np.eye(n_Z)}, (n_Z, n_Z, n_W), False)
    Y1_t = LinQuadVar({'x': schur['N'], 'c': C}, (n_Y, n_Z, n_W), False)
    X1_t = concat([Y1_t, Z1_t])
    X1_tp1 = next_period(X1_t, Z1_tp1)

    ################################################################################################################################################################################
    ## Second order expansion 
    ################################################################################################################################################################################

    ## Step 1: Collect first order contributions
    Wtp1 = LinQuadVar({'w': np.eye(n_W)}, (n_W, n_Z, n_W), False)
    D2 = combine_second_order_terms(df, X1_t, X1_tp1, Wtp1)
    D2_coeff = np.block([[D2['c'], D2['x'], D2['w'], D2['xx'], D2['xw'], D2['ww']]])
    Y2_coeff = -df['xtp1'][n_Y:]@schur['N_block']
    C_hat = np.linalg.solve(Y2_coeff, D2_coeff[n_Y:])
    C_hat_coeff = np.split(C_hat, np.cumsum([1, n_Z, n_W, n_Z**2, n_Z*n_W]), axis=1)
    
    ## Step 2: Solve the coeffcients on jump variables
    M_E_w = np.zeros([n_W,1])
    cov_w = np.eye(n_W)
    M_E_ww = cal_E_ww(M_E_w, cov_w)
    E_D2 = E(D2, M_E_w, M_E_ww)
    E_D2_coeff = np.block([[E_D2['c'], E_D2['x'], E_D2['xx']]])
    M_mat = form_M0(M_E_w, M_E_ww, Z1_tp1, Z1Z1)
    LHS = np.eye(n_Y*M_mat.shape[0]) - np.kron(M_mat.T, np.linalg.solve(schur['Λ22'], schur['Λp22']))
    RHS = vec(-np.linalg.solve(schur['Λ22'], (schur['Q'].T@E_D2_coeff)[-n_Y:]))
    D_tilde_vec = np.linalg.solve(LHS, RHS)
    D_tilde = mat(D_tilde_vec, (n_Y, 1+n_Z+n_Z**2))
    G = np.linalg.solve(schur['Z21'], D_tilde)
    Y2_t = LinQuadVar({'x2': schur['N'],
                    'xx': G[:, 1+n_Z:1+n_Z+n_Z**2],
                    'x': G[:, 1:1+n_Z],
                    'c': G[:, :1]}, (n_Y, n_Z, n_W), False)
    
    ## Step 3: Solve the coeffcients on state variables
    G_block = np.block([[G], [np.zeros((n_Z, 1+n_Z+n_Z**2))]])
    Gp_hat = np.linalg.solve(Y2_coeff, df['xtp1'][n_Y:]@G_block)
    G_hat = np.linalg.solve(Y2_coeff, df['xt'][n_Y:]@G_block)
    c_1, x_1, w_1, xx_1, xw_1, ww_1 = C_hat_coeff
    c_2, x_2, xx_2 = np.split(G_hat, np.cumsum([1, n_Z]), axis=1)
    var = LinQuadVar({'c': Gp_hat[:, :1], 'x': Gp_hat[:, 1:1+n_Z],
                    'xx': Gp_hat[:, 1+n_Z:1+n_Z+n_Z**2]},
                    (n_Z, n_Z, n_W), False)
    var_next = next_period(var, Z1_tp1, None, Z1Z1)

    ψ_x2 = ψ_x.copy()
    ψ_xq = var_next['x'] + x_1 + x_2
    ψ_wq = var_next['w'] + w_1
    ψ_qq = var_next['c'] + c_1 + c_2
    ψ_xx = var_next['xx'] + xx_1 + xx_2
    ψ_xw = var_next['xw'] + xw_1
    ψ_ww = var_next['ww'] + ww_1

    ## Form the LinQuads for second order expansion when N = 1
    Z2_tp1 = LinQuadVar({'x2': ψ_x2, 'xx': ψ_xx, 'xw': ψ_xw, 'ww': ψ_ww, 'x': ψ_xq, 'w': ψ_wq, 'c': ψ_qq}, (n_Z, n_Z, n_W), False)
    Z2_t = LinQuadVar({'x2': np.eye(n_Z)}, (n_Z, n_Z, n_W), False)
    X2_t = concat([Y2_t, Z2_t])
    X2_tp1 = next_period(X2_t, Z1_tp1, Z2_tp1)
    
    Y_t = Y0_t + Y1_t + 0.5*Y2_t
    X_t = X0_t + X1_t + 0.5*X2_t
    X_tp1 = next_period(X_t, Z1_tp1, Z2_tp1)

    res = ModelSolution({'Y_t':Y_t,
                         'Y1_t':Y1_t,
                         'Y2_t':Y2_t,
                         'X_t': X_t,
                         'X_tp1': X_tp1,
                         'X1_t': X1_t,
                         'X1_tp1': X1_tp1,
                         'X2_t': X2_t,
                         'X2_tp1': X2_tp1,
                         'Z1_t':Z1_t,
                         'Z2_t':Z2_t,
                         'Z1_tp1': Z1_tp1,
                         'Z2_tp1': Z2_tp1,
                         'vmc0_t':Y0_t[0],
                         'rmc0_t':Y0_t[1],
                         'vmc1_t':Y1_t[0],
                         'rmc1_t':Y1_t[1],
                         'vmc2_t':Y2_t[0],
                         'rmc2_t':Y2_t[1],
                         'var_shape': var_shape,
                         'schur_decomposition': schur,
                         'second_order': True})
    return res

def N_0_expansion(df, ss, var_shape, γ):

    n_Y, n_Z, n_W = var_shape
    n_Y = n_Y + 2
    n_X = n_Y + n_Z

    ################################################################################################################################################################################
    ## Zeroth order expansion
    ################################################################################################################################################################################

    Z0_t = LinQuadVar({'c': ss[n_Y:].reshape(-1, 1)}, (n_Z, n_Z, n_W), False)
    Y0_t = LinQuadVar({'c': ss[:n_Y].reshape(-1, 1)}, (n_Y, n_Z, n_W), False)
    X0_t = concat([Y0_t, Z0_t])

    ################################################################################################################################################################################
    ## First order expansion
    ################################################################################################################################################################################

    ## Step 1: generalized Schur decomposition, solve for the x coefficients on jump variables
    schur = schur_decomposition(-df['xtp1'], df['xt'],(n_Y, n_Z, n_W))

    ## Step 2: solve for the coefficients on state variables
    f_1_xtp1 = df['xtp1'][:n_Y]
    f_1_wtp1 = df['wtp1'][:n_Y]
    f_2_xtp1 = df['xtp1'][n_Y:]
    f_2_xt = df['xt'][n_Y:]
    f_2_wtp1 = df['wtp1'][n_Y:]
    ψ_x = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_xt@schur['N_block'])
    ψ_w = np.linalg.solve(-f_2_xtp1@schur['N_block'], f_2_wtp1)

    ## Step 3: generalized Schur decomposition, solve for constants on state and jump variables
    σ_v = (df['xtp1'][[0], :]@schur['N_block']@ψ_w + df['wtp1'][[0], :]).T
    μ_0 = (1 - γ) * σ_v
    adj = np.zeros((n_Y, 1))
    σ_v = σ_v.reshape(-1,)
    adj[0, 0] = - 0.5 * (1 - γ)*σ_v.dot(σ_v)
    RHS = - np.block([[(f_1_xtp1@schur['N_block']@ψ_w + f_1_wtp1)@μ_0+adj],
                      [np.zeros((n_Z, 1))]])
    LHS = df['xtp1'] + df['xt']
    # D = np.linalg.solve(LHS, RHS)
    ### The following steps are special for the permanent income model
    if np.linalg.matrix_rank(LHS) == len(LHS):
        D = np.linalg.solve(LHS, RHS)
    else:
        new_column = np.zeros((LHS.shape[0],1))
        new_column[3,0] = 1
        new_row = np.zeros((1, LHS.shape[1]+1))
        new_row[0,n_Y+1] = 1
        LHS = np.vstack((np.hstack((LHS, new_column)),new_row))
        RHS = np.vstack((RHS, np.zeros(1)))
        
        D = np.linalg.solve(LHS, RHS)
        discount_adj_1 = np.float(D[-1])
        D = D[:-1]
    ### Permanent income model special treatment ends here
    C = D[:n_Y] - schur['N']@D[n_Y:]
    ψ_q = D[n_Y:] - ψ_x@D[n_Y:]

    ## Form the LinQuads for first order expansion when N = 1
    Z1_tp1 = LinQuadVar({'x': ψ_x, 'w': ψ_w, 'c': ψ_q}, (n_Z, n_Z, n_W), False)
    Z1Z1 = kron_prod(Z1_tp1, Z1_tp1)
    Z1_t = LinQuadVar({'x': np.eye(n_Z)}, (n_Z, n_Z, n_W), False)
    Y1_t = LinQuadVar({'x': schur['N'], 'c': C}, (n_Y, n_Z, n_W), False)
    X1_t = concat([Y1_t, Z1_t])
    X1_tp1 = next_period(X1_t, Z1_tp1)

    log_M = LinQuadVar({'w': μ_0.T, 'c': -0.5 * μ_0.T @ μ_0}, (1, n_Z, n_W), False)

    ################################################################################################################################################################################
    ## Second order expansion 
    ################################################################################################################################################################################

    ## Step 1: Collect first order contributions
    Wtp1 = LinQuadVar({'w': np.eye(n_W)}, (n_W, n_Z, n_W), False)
    D2 = combine_second_order_terms(df, X1_t, X1_tp1, Wtp1)
    D2_coeff = np.block([[D2['c'], D2['x'], D2['w'], D2['xx'], D2['xw'], D2['ww']]])
    Y2_coeff = -df['xtp1'][n_Y:]@schur['N_block']
    C_hat = np.linalg.solve(Y2_coeff, D2_coeff[n_Y:])
    C_hat_coeff = np.split(C_hat, np.cumsum([1, n_Z, n_W, n_Z**2, n_Z*n_W]), axis=1)
    
    ## Step 2: Solve the coeffcients on jump variables
    M_E_w = log_M['w'].T
    cov_w = np.eye(n_W)
    M_E_ww = cal_E_ww(M_E_w, cov_w)
    E_D2 = E(D2, M_E_w, M_E_ww)
    E_D2_coeff = np.block([[E_D2['c'], E_D2['x'], E_D2['xx']]])
    M_mat = form_M0(M_E_w, M_E_ww, Z1_tp1, Z1Z1)
    LHS = np.eye(n_Y*M_mat.shape[0]) - np.kron(M_mat.T, np.linalg.solve(schur['Λ22'], schur['Λp22']))
    RHS = vec(-np.linalg.solve(schur['Λ22'], (schur['Q'].T@E_D2_coeff)[-n_Y:]))
    D_tilde_vec = np.linalg.solve(LHS, RHS)
    D_tilde = mat(D_tilde_vec, (n_Y, 1+n_Z+n_Z**2))
    G = np.linalg.solve(schur['Z21'], D_tilde)
    Y2_t = LinQuadVar({'x2': schur['N'],
                    'xx': G[:, 1+n_Z:1+n_Z+n_Z**2],
                    'x': G[:, 1:1+n_Z],
                    'c': G[:, :1]}, (n_Y, n_Z, n_W), False)
    
    ## Step 3: Solve the coeffcients on state variables
    G_block = np.block([[G], [np.zeros((n_Z, 1+n_Z+n_Z**2))]])
    Gp_hat = np.linalg.solve(Y2_coeff, df['xtp1'][n_Y:]@G_block)
    G_hat = np.linalg.solve(Y2_coeff, df['xt'][n_Y:]@G_block)
    c_1, x_1, w_1, xx_1, xw_1, ww_1 = C_hat_coeff
    c_2, x_2, xx_2 = np.split(G_hat, np.cumsum([1, n_Z]), axis=1)
    var = LinQuadVar({'c': Gp_hat[:, :1], 'x': Gp_hat[:, 1:1+n_Z],
                    'xx': Gp_hat[:, 1+n_Z:1+n_Z+n_Z**2]},
                    (n_Z, n_Z, n_W), False)
    var_next = next_period(var, Z1_tp1, None, Z1Z1)

    ψ_x2 = ψ_x.copy()
    ψ_xq = var_next['x'] + x_1 + x_2
    ψ_wq = var_next['w'] + w_1
    ψ_qq = var_next['c'] + c_1 + c_2
    ψ_xx = var_next['xx'] + xx_1 + xx_2
    ψ_xw = var_next['xw'] + xw_1
    ψ_ww = var_next['ww'] + ww_1

    ## Form the LinQuads for second order expansion when N = 1
    Z2_tp1 = LinQuadVar({'x2': ψ_x2, 'xx': ψ_xx, 'xw': ψ_xw, 'ww': ψ_ww, 'x': ψ_xq, 'w': ψ_wq, 'c': ψ_qq}, (n_Z, n_Z, n_W), False)
    Z2_t = LinQuadVar({'x2': np.eye(n_Z)}, (n_Z, n_Z, n_W), False)
    X2_t = concat([Y2_t, Z2_t])
    X2_tp1 = next_period(X2_t, Z1_tp1, Z2_tp1)
    
    Y_t = Y0_t + Y1_t + 0.5*Y2_t
    X_t = X0_t + X1_t + 0.5*X2_t
    X_tp1 = next_period(X_t, Z1_tp1, Z2_tp1)

    res = ModelSolution({'Y_t':Y_t,
                         'Y1_t':Y1_t,
                         'Y2_t':Y2_t,
                         'X_t': X_t,
                         'X_tp1': X_tp1,
                         'X1_t': X1_t,
                         'X1_tp1': X1_tp1,
                         'X2_t': X2_t,
                         'X2_tp1': X2_tp1,
                         'Z1_t':Z1_t,
                         'Z2_t':Z2_t,
                         'Z1_tp1': Z1_tp1,
                         'Z2_tp1': Z2_tp1,
                         'vmc0_t':Y0_t[0],
                         'rmc0_t':Y0_t[1],
                         'vmc1_t':Y1_t[0],
                         'rmc1_t':Y1_t[1],
                         'vmc2_t':Y2_t[0],
                         'rmc2_t':Y2_t[1],
                         'log_M':log_M,
                         'var_shape': var_shape,
                         'schur_decomposition': schur,
                         'second_order': True})
    return res

class ModelSolution(dict):
    """
    Represents the model solution.

    Attributes
    ----------
    X_t : LinQuadVar
        Approximation for :math:`X_{t}` in terms of :math:`Z_{1,t}`.
    X_tp1 : LinQuadVar
        Approximation for :math:`X_{t+1}` in terms of :math:`Z_{1,t}`
        and :math:`W_{t+1}`.
    X1_t : LinQuadVar
        Representation of :math:`X_{1, t}`.
    X2_t : LinQuadVar
        Representation of :math:`X_{2, t}`.
    X1_tp1 : LinQuadVar
        Representation of :math:`X_{1, t+1}`.
    X2_tp1 : LinQuadVar
        Representation of :math:`X_{2, t+1}`.
    log_M : LinQuadVar
        Approximation for log change of measure in terms of :math:`X_{1,t}`
        and :math:`X_{2,t}`.
    nit : int
        Number of iterations performed to get M1. For second-order expansion
        only.
    second_order : bool
        If True, the solution is in second-order.
    var_shape : tuple of ints
        (n_Y, n_Z, n_W). Number of endogenous variables, states and shocks
        respectively.
    ss : (n_X, ) ndarray
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

    def approximate_fun(self, fun, args=()):
        """
        Approximates a scalar variable as a function of X and W.

        Parameters
        ----------
        fun : callable
            Function to be approximated.
            ``var_fun(X_t, X_tp1, W_tp1, q, *args) -> scalar``

            X_t, X_tp1, W_tp1, q, and args are the same as inputs to eq_cond().
        args : tuple of floats/ints
            Additional parameters to be passed to fun.

        Returns
        -------
        fun_approx : LinQuadVar
            Approximation for the input.

        """
        _, n_Z, n_W = self.var_shape

        W_0 = np.zeros(n_W)
        q_0 = 0.
        
        dfun = compute_derivatives(f=lambda X_t, X_tp1, W_tp1, q:
                                   anp.atleast_1d(fun(X_t, X_tp1, W_tp1, q, *args)),
                                   X=[self.ss, self.ss, W_0, q_0],
                                   second_order=self.second_order)

        fun_zero_order = fun(self.ss, self.ss, W_0, q_0, *args)
        fun_first_order = matmul(dfun['xtp1'], self.X1_tp1)\
            + matmul(dfun['xt'], self.X1_t)\
            + LinQuadVar({'w': dfun['wtp1'], 'c': dfun['q'].reshape(-1, 1)},
                         (1, n_Z, n_W), False)
        fun_approx = fun_zero_order + fun_first_order
        Wtp1 = LinQuadVar({'w': np.eye(n_W)}, (n_W, n_Z, n_W), False)
        if self.second_order:
            temp1 = combine_second_order_terms(dfun, self.X1_t, self.X1_tp1, Wtp1)
            temp2 = matmul(dfun['xt'], self.X2_t)\
                + matmul(dfun['xtp1'], self.X2_tp1)
            fun_second_order = temp1 + temp2
            fun_approx = fun_approx + fun_second_order*0.5

        return fun_approx#, fun_zero_order, fun_first_order, fun_second_order

    def simulate(self, Ws):
        """
        Simulates stochastiic path for X by generating iid normal shocks,
        or deterministic path for X by generating zero-valued shocks.

        Parameters
        ----------
        Ws : (T+burn_in, n_W) ndarray
            n_W dimensional shocks for (T+burn_in) periods to be fed into the system.        
        T : int
            Time horizon.
        burn_in: int
            Throwing away some iterations at the beginning of the simulation.

        Returns
        -------
        sim_result : (T, n_Y) ndarray
            Simulated Ys.

        """
        n_Y, n_Z, n_W = self.var_shape
        Z2_tp1 = self.X2_tp1[n_Y: n_Y+n_Z] if self.second_order else None
        sim_result = simulate(self.X_t,
                              self.X1_tp1[n_Y: n_Y+n_Z],
                              Z2_tp1,
                              Ws)
        return sim_result
        
    def IRF(self, T, shock):
        r"""
        Computes impulse response functions for each component in X to each shock.

        Parameters
        ----------
        T : int
            Time horizon.
        shock : int
            Position of the initial shock, starting from 0.

        Returns
        -------
        states : (T, n_Z) ndarray
            IRF of all state variables to the designated shock.
        controls : (T, n_Y) ndarray
            IRF of all control variables to the designated shock.

        """
    
        n_Y, n_Z, n_W = self.var_shape
        # Build the first order impulse response for each of the shocks in the system
        states1 = np.zeros((T, n_Z))
        controls1 = np.zeros((T, n_Y))
        
        W_0 = np.zeros(n_W)
        W_0[shock] = 1
        B = self.X1_tp1['w'][n_Y:,:]
        F = self.X1_tp1['w'][:n_Y,:]
        A = self.X1_tp1['x'][n_Y:,:]
        D = self.X1_tp1['x'][:n_Y,:]
        N = self.X1_t['x'][:n_Y,:]
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
            Ψ_0 = self.X2_tp1['c'][n_Y:,:]
            Ψ_1 = self.X2_tp1['x'][n_Y:,:]
            Ψ_2 = self.X2_tp1['w'][n_Y:,:]
            Ψ_3 = self.X2_tp1['x2'][n_Y:,:]
            Ψ_4 = self.X2_tp1['xx'][n_Y:,:]
            Ψ_5 = self.X2_tp1['xw'][n_Y:,:]
            Ψ_6 = self.X2_tp1['ww'][n_Y:,:]
            
            Φ_0 = self.X2_tp1['c'][:n_Y,:]
            Φ_1 = self.X2_tp1['x'][:n_Y,:]
            Φ_2 = self.X2_tp1['w'][:n_Y,:]
            Φ_3 = self.X2_tp1['x2'][:n_Y,:]
            Φ_4 = self.X2_tp1['xx'][:n_Y,:]
            Φ_5 = self.X2_tp1['xw'][:n_Y,:]
            Φ_6 = self.X2_tp1['ww'][:n_Y,:]
            
            states2 = np.zeros((T, n_Z))
            controls2 = np.zeros((T, n_Y))
            X_1_0 = np.zeros(n_Z)

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

    def elasticities(self, log_SDF_ex, args, locs=None, T=400, shock=0, percentile=0.5):
        r"""
        Computes shock exposure and price elasticities for X.

        Parameters
        ----------
        log_SDF_ex : callable
            Log stochastic discount factor exclusive of the
            change of measure M.

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
        elasticities : (T, n_Y) ndarray
            Elasticities for M.

        References
        ---------
        Borovicka, Hansen (2014). See http://larspeterhansen.org/.

        """
        n_Y, n_Z, n_W = self.var_shape
        log_SDF_ex = self.approximate_fun(log_SDF_ex, args)
        self.log_SDF = log_SDF_ex + self.log_M
        Z2_tp1 = self.X2_tp1[n_Y: n_Y+n_Z] if self.second_order else None
        X_growth = self.X_tp1 - self.X_t
        X_growth_list = X_growth.split()
        if locs is not None:
            X_growth_list = [X_growth_list[i] for i in locs]
        exposure_all = np.zeros((T, len(X_growth_list)))
        price_all = np.zeros((T, len(X_growth_list)))
        for i, x in enumerate(X_growth_list):
            exposure = exposure_elasticity(x,
                                           self.X1_tp1[n_Y: n_Y+n_Z],
                                           Z2_tp1,
                                           T,
                                           shock,
                                           percentile)
            price = price_elasticity(x,
                                     self.log_SDF,
                                     self.X1_tp1[n_Y: n_Y+n_Z],
                                     Z2_tp1,
                                     T,
                                     shock,
                                     percentile)
            exposure_all[:, i] = exposure.reshape(-1)
            price_all[:, i] = price.reshape(-1)
        return exposure_all, price_all


def plug_in_df_first_order(df, var_shape, util_sol):
    n_Y, n_Z, n_W = var_shape
    n_X = n_Y + n_Z
    
    df_xt       = df['xt'][-n_X:,-n_X:].copy()
    df_xtp1     = df['xtp1'][-n_X:,-n_X:].copy()
    df_wtp1     = df['wtp1'][-n_X:,:].copy()
    df_q        = df['q'][-n_X:].copy()
    
    ################################################################################################
    ## 1. df_xt ####################################################################################
    ################################################################################################
    
    nonzero_vr_df = df['xt'][2:,0:2].copy()                    
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        if nonzero_loc[1] == 0:
            # vmc1_t to Z1_t
            df_xt[[nonzero_loc[0]],-n_Z:]  += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['vmc1_t']['x']

        elif nonzero_loc[1] == 1:
            # rmc1_t to Z1_t
            df_xt[[nonzero_loc[0]],-n_Z:]  += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['rmc1_t']['x']

    
    ################################################################################################
    ## 2. df_xtp1 ##################################################################################
    ################################################################################################ 
    
    nonzero_vr_df = df['xtp1'][2:,0:2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))

    for nonzero_loc in nonzero_vr_index:
        if nonzero_loc[1] == 0:
            # vmc1_tp1 to Z1_tp1
            df_xtp1[[nonzero_loc[0]],-n_Z:]  += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['vmc1_t']['x']

        elif nonzero_loc[1] == 1:
            # rmc1_tp1 to Z1_tp1
            df_xtp1[[nonzero_loc[0]],-n_Z:]  += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['rmc1_t']['x']
    
    df_plug_in = {'xt':df_xt,\
                  'xtp1':df_xtp1,\
                  'wtp1':df_wtp1,\
                  'q':df_q}
    return  df_plug_in


def plug_in_df_second_order(df, var_shape, util_sol):

    n_Y, n_Z, n_W = var_shape
    n_X = n_Y + n_Z
    
    df_xt       = df['xt'][-n_X:,-n_X:].copy()
    df_xtp1     = df['xtp1'][-n_X:,-n_X:].copy()
    df_wtp1     = df['wtp1'][-n_X:,:].copy()
    df_q        = df['q'][-n_X:].copy()
    
    df_xtxt     = df['xtxt'][-n_X:,2*(n_X+2):][:,np.tile(np.concatenate([np.repeat(False,2),np.repeat(True,n_X)]),n_X)].copy()
    df_xtxtp1   = df['xtxtp1'][-n_X:,2*(n_X+2):][:,np.tile(np.concatenate([np.repeat(False,2),np.repeat(True,n_X)]),n_X)].copy()
    df_xtwtp1   = df['xtwtp1'][-n_X:,2*n_W:].copy()
    df_xtq      = df['xtq'][-n_X:,-n_X:].copy() #0
    df_xtp1xtp1 = df['xtp1xtp1'][-n_X:,2*(n_X+2):][:,np.tile(np.concatenate([np.repeat(False,2),np.repeat(True,n_X)]),n_X)].copy()
    df_xtp1wtp1 = df['xtp1wtp1'][-n_X:,2*n_W:].copy()
    df_xtp1q    = df['xtp1q'][-n_X:,-n_X:].copy() #0
    df_wtp1wtp1 = df['wtp1wtp1'][-n_X:,:].copy() #0
    df_wtp1q    = df['wtp1q'][-n_X:,:].copy() #0
    df_qq       = df['qq'][-n_X:,:].copy() #0
    
    ################################################################################################
    ## df_xt #######################################################################################
    ################################################################################################
    nonzero_vr_df = df['xt'][2:,0:2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        if nonzero_loc[1] == 0:
            # vmc_t to Z_t
            df_xt[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['vmc2_t']['x2']
#             df_xt[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['vmc1_t']['x']
            # vmc_2t
            new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc2_t']['xx'].reshape([n_Z,n_Z]).T.reshape([1,-1])
            new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
            for i in range(n_Z):   
                # vmc_2t to Z_tZ_t
                df_xtxt[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
            # vmc_2t to Z_tq
            df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['vmc2_t']['x']/2
            # vmc_2t to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc2_t']['c']

        elif nonzero_loc[1] == 1:
            # rmc_t to Z_t
            df_xt[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['rmc2_t']['x2']
#             df_xt[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['rmc1_t']['x']
            # rmc_2t
            new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc2_t']['xx'].reshape([n_Z,n_Z]).T.reshape([1,-1])
            new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
            for i in range(n_Z):   
                # rmc_2t to Z_tZ_t
                df_xtxt[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
            # rmc_2t to Z_tq
            df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['rmc2_t']['x']/2
            # rmc_2t to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc2_t']['c']
    
    
    ################################################################################################
    ## df_xtp1 #####################################################################################
    ################################################################################################    
    nonzero_vr_df = df['xtp1'][2:,0:2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        if nonzero_loc[1] == 0:
            # vmc_tp1 to Z_tp1
            df_xtp1[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['vmc2_t']['x2']
#             df_xtp1[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['vmc1_t']['x']
            # vmc_2tp1
            new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc2_t']['xx'].reshape([n_Z,n_Z]).T.reshape([1,-1])
            new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
            for i in range(n_Z):   
                # vmc_2tp1 to Z_tp1Z_tp1
                df_xtp1xtp1[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
            # vmc_2tp1 to Z_tp1q
            df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['vmc2_t']['x']/2
            # vmc_2tp1 to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc2_t']['c']

        elif nonzero_loc[1] == 1:
            # rmc_tp1 to Z_tp1
            df_xtp1[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['rmc2_t']['x2']
#             df_xtp1[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['rmc1_t']['x']
            # rmc_2tp1
            new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc2_t']['xx'].reshape([n_Z,n_Z]).T.reshape([1,-1])
            new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
            for i in range(n_Z):   
                # rmc_2tp1 to Z_tp1Z_tp1
                df_xtp1xtp1[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
            # rmc_2tp1 to Z_tp1q
            df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['rmc2_t']['x']/2
            # rmc_2tp1 to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc2_t']['c']
            
    
    ################################################################################################
    ## df_xtxtp1 ###################################################################################
    ################################################################################################
    # vmc_tp1vmc_t
    nonzero_vr_df = df['xtxtp1'][-n_X:,:1].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(util_sol['vmc1_t']['x'],util_sol['vmc1_t']['x'])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):
            # to Z_tp1Z_t
            df_xtxtp1[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
        # to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'] * util_sol['vmc1_t']['c']
        # to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'] * util_sol['vmc1_t']['c']
        # to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (util_sol['vmc1_t']['c']*util_sol['vmc1_t']['c'])*2

    # rmc_tp1vmc_t
    nonzero_vr_df = df['xtxtp1'][-n_X:,1:2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(util_sol['vmc1_t']['x'],util_sol['rmc1_t']['x'])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):   
            # to Z_tp1Z_t
            df_xtxtp1[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
        # to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'] * util_sol['rmc1_t']['c']
        # to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'] * util_sol['vmc1_t']['c']
        # to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (util_sol['vmc1_t']['c']*util_sol['rmc1_t']['c'])*2

    # vmc_tp1rmc_t
    nonzero_vr_df = df['xtxtp1'][-n_X:,(n_X+2):(n_X+2)+1].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(util_sol['rmc1_t']['x'],util_sol['vmc1_t']['x'])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):   
            # to Z_tp1Z_t
            df_xtxtp1[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
        # to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'] * util_sol['vmc1_t']['c']
        # to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'] * util_sol['rmc1_t']['c']
        # to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (util_sol['vmc1_t']['c']*util_sol['rmc1_t']['c'])*2

    # rmc_tp1rmc_t
    nonzero_vr_df = df['xtxtp1'][-n_X:,(n_X+2)+1:(n_X+2)+2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(util_sol['rmc1_t']['x'],util_sol['rmc1_t']['x'])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):   
            # to Z_tp1Z_t
            df_xtxtp1[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
        # to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'] * util_sol['rmc1_t']['c']
        # to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'] * util_sol['rmc1_t']['c']
        # to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (util_sol['rmc1_t']['c']*util_sol['rmc1_t']['c'])*2


    # x_tp1vmc_t
    nonzero_vr_df = df['xtxtp1'][-n_X:,n_X+2-n_X:n_X+2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # to x_tp1Z_t
        for i in range(n_Z):
            df_xtxtp1[[nonzero_loc[0]], (i+n_Y)*n_X+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'][0,i]
        # to xtp1q
        df_xtp1q[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['c'].item()

    # x_tp1rmc_t
    nonzero_vr_df = df['xtxtp1'][-n_X:,(n_X+2)*2-n_X:(n_X+2)*2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # to x_tp1Z_t
        for i in range(n_Z):
            df_xtxtp1[[nonzero_loc[0]], (i+n_Y)*n_X+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'][0,i]
        # to xtp1q
        df_xtp1q[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['c'].item()


    # vmc_tp1x_t
    nonzero_vr_df = df['xtxtp1'][-n_X:,2*(n_X+2):][:,np.tile(np.concatenate([np.array([True,False]),np.repeat(False,n_X)]),n_X)].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # to Z_tp1x_t
        df_xtxtp1[[nonzero_loc[0]], n_X*(nonzero_loc[1]+1)-n_Z:n_X*(nonzero_loc[1]+1)] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x']
        # to x_tq
        df_xtq[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['c'].item()

    # rmc_tp1x_t
    nonzero_vr_df = df['xtxtp1'][-n_X:,2*(n_X+2):][:,np.tile(np.concatenate([np.array([False,True]),np.repeat(False,n_X)]),n_X)].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # to Z_tp1x_t
        df_xtxtp1[[nonzero_loc[0]], n_X*(nonzero_loc[1]+1)-n_Z:n_X*(nonzero_loc[1]+1)] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x']
        # to x_tq
        df_xtq[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['c'].item()
    
    ################################################################################################
    ## df_xtxt #####################################################################################
    ################################################################################################
    
    # vmc_tvmc_t
    nonzero_vr_df = df['xtxt'][-n_X:,:1].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(util_sol['vmc1_t']['x'],util_sol['vmc1_t']['x'])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):
            # to Z_tZ_t
            df_xtxt[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
        # to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'] * util_sol['vmc1_t']['c']/2
        # to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'] * util_sol['vmc1_t']['c']/2
        # to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (util_sol['vmc1_t']['c']*util_sol['vmc1_t']['c'])

    # rmc_tvmc_t
    nonzero_vr_df = df['xtxt'][-n_X:,1:2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(util_sol['vmc1_t']['x'],util_sol['rmc1_t']['x'])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):   
            # to Z_tZ_t
            df_xtxt[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
        # to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'] * util_sol['rmc1_t']['c']/2
        # to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'] * util_sol['vmc1_t']['c']/2
        # to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (util_sol['vmc1_t']['c']*util_sol['rmc1_t']['c'])

    # vmc_trmc_t
    nonzero_vr_df = df['xtxt'][-n_X:,(n_X+2):(n_X+2)+1].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(util_sol['rmc1_t']['x'],util_sol['vmc1_t']['x'])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):   
            # to Z_tZ_t
            df_xtxt[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
        # to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'] * util_sol['vmc1_t']['c']/2
        # to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'] * util_sol['rmc1_t']['c']/2
        # to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (util_sol['vmc1_t']['c']*util_sol['rmc1_t']['c'])

    # rmc_trmc_t
    nonzero_vr_df = df['xtxt'][-n_X:,(n_X+2)+1:(n_X+2)+2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(util_sol['rmc1_t']['x'],util_sol['rmc1_t']['x'])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):   
            # to Z_tZ_t
            df_xtxt[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
        # to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'] * util_sol['rmc1_t']['c']/2
        # to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'] * util_sol['rmc1_t']['c']/2
        # to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (util_sol['rmc1_t']['c']*util_sol['rmc1_t']['c'])


    # x_tvmc_t
    nonzero_vr_df = df['xtxt'][-n_X:,n_X+2-n_X:n_X+2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # to x_tZ_t
        for i in range(n_Z):
            df_xtxt[[nonzero_loc[0]], (i+n_Y)*n_X+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'][0,i]
        # to xtq
        df_xtq[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['c'].item()/2

    # x_trmc_t
    nonzero_vr_df = df['xtxt'][-n_X:,(n_X+2)*2-n_X:(n_X+2)*2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # to x_tZ_t
        for i in range(n_Z):
            df_xtxt[[nonzero_loc[0]], (i+n_Y)*n_X+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'][0,i]
        # to xtq
        df_xtq[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['c'].item()/2


    # vmc_tx_t
    nonzero_vr_df = df['xtxt'][-n_X:,2*(n_X+2):][:,np.tile(np.concatenate([np.array([True,False]),np.repeat(False,n_X)]),n_X)].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # to Z_tx_t
        df_xtxt[[nonzero_loc[0]], n_X*(nonzero_loc[1]+1)-n_Z:n_X*(nonzero_loc[1]+1)] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x']
        # to x_tq
        df_xtq[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['c'].item()/2

    # rmc_tx_t
    nonzero_vr_df = df['xtxt'][-n_X:,2*(n_X+2):][:,np.tile(np.concatenate([np.array([False,True]),np.repeat(False,n_X)]),n_X)].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # to Z_tx_t
        df_xtxt[[nonzero_loc[0]], n_X*(nonzero_loc[1]+1)-n_Z:n_X*(nonzero_loc[1]+1)] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x']
        # to x_tq
        df_xtq[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['c'].item()/2
    
    ################################################################################################
    ## df_xtp1xtp1 #################################################################################
    ################################################################################################
    
    ## df_xtp1xtp1
    # vmc_tp1vmc_tp1
    nonzero_vr_df = df['xtp1xtp1'][-n_X:,:1].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(util_sol['vmc1_t']['x'],util_sol['vmc1_t']['x'])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):
            # to Z_tp1Z_tp1
            df_xtp1xtp1[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
        # to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'] * util_sol['vmc1_t']['c']/2
        # to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'] * util_sol['vmc1_t']['c']/2
        # to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (util_sol['vmc1_t']['c']*util_sol['vmc1_t']['c'])

    # rmc_tp1vmc_tp1
    nonzero_vr_df = df['xtp1xtp1'][-n_X:,1:2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(util_sol['vmc1_t']['x'],util_sol['rmc1_t']['x'])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):   
            # to Z_tp1Z_tp1
            df_xtp1xtp1[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
        # to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'] * util_sol['rmc1_t']['c']/2
        # to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'] * util_sol['vmc1_t']['c']/2
        # to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (util_sol['vmc1_t']['c']*util_sol['rmc1_t']['c'])

    # vmc_tp1rmc_tp1
    nonzero_vr_df = df['xtp1xtp1'][-n_X:,(n_X+2):(n_X+2)+1].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(util_sol['rmc1_t']['x'],util_sol['vmc1_t']['x'])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):   
            # to Z_tp1Z_tp1
            df_xtp1xtp1[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
        # to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'] * util_sol['vmc1_t']['c']/2
        # to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'] * util_sol['rmc1_t']['c']/2
        # to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (util_sol['vmc1_t']['c']*util_sol['rmc1_t']['c'])

    # rmc_tp1rmc_tp1
    nonzero_vr_df = df['xtp1xtp1'][-n_X:,(n_X+2)+1:(n_X+2)+2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(util_sol['rmc1_t']['x'],util_sol['rmc1_t']['x'])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):   
            # to Z_tp1Z_tp1
            df_xtp1xtp1[[nonzero_loc[0]], (i+n_Y+1)*n_X-n_Z : (i+n_Y+1)*n_X] += new_zz_coefs[i]
        # to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'] * util_sol['rmc1_t']['c']/2
        # to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'] * util_sol['rmc1_t']['c']/2
        # to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (util_sol['rmc1_t']['c']*util_sol['rmc1_t']['c'])


    # x_tp1vmc_tp1
    nonzero_vr_df = df['xtp1xtp1'][-n_X:,n_X+2-n_X:n_X+2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # to x_tp1Z_tp1
        for i in range(n_Z):
            df_xtp1xtp1[[nonzero_loc[0]], (i+n_Y)*n_X+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'][0,i]
        # to xtp1q
        df_xtp1q[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['c'].item()/2

    # x_tp1rmc_tp1
    nonzero_vr_df = df['xtp1xtp1'][-n_X:,(n_X+2)*2-n_X:(n_X+2)*2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # to x_tp1Z_tp1
        for i in range(n_Z):
            df_xtp1xtp1[[nonzero_loc[0]], (i+n_Y)*n_X+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'][0,i]
        # to xtp1q
        df_xtp1q[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['c'].item()/2

    # vmc_tp1x_tp1
    nonzero_vr_df = df['xtp1xtp1'][-n_X:,2*(n_X+2):][:,np.tile(np.concatenate([np.array([True,False]),np.repeat(False,n_X)]),n_X)].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # to Z_tp1x_tp1
        df_xtp1xtp1[[nonzero_loc[0]], n_X*(nonzero_loc[1]+1)-n_Z:n_X*(nonzero_loc[1]+1)] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x']
        # to x_tp1q
        df_xtp1q[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['c'].item()/2

    # rmc_tp1x_tp1
    nonzero_vr_df = df['xtp1xtp1'][-n_X:,2*(n_X+2):][:,np.tile(np.concatenate([np.array([False,True]),np.repeat(False,n_X)]),n_X)].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # to Z_tp1x_tp1
        df_xtp1xtp1[[nonzero_loc[0]], n_X*(nonzero_loc[1]+1)-n_Z:n_X*(nonzero_loc[1]+1)] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x']
        # to x_tp1q
        df_xtp1q[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['c'].item()/2
        
    
    ################################################################################################
    ## df_xtwtp1 ###################################################################################
    ################################################################################################

    # vmc_twtp1
    nonzero_vr_df = df['xtwtp1'][2:,:n_W].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        for i in range(n_Z):
            # to Z_twtp1
            df_xtwtp1[[nonzero_loc[0]], (i+n_Y)*n_W+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'][0,i]
        # to wtp1q
        df_wtp1q[[nonzero_loc[0]],[nonzero_loc[1]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['c'].item()

    # rmc_twtp1
    nonzero_vr_df = df['xtwtp1'][2:,n_W:2*n_W].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        for i in range(n_Z):
            # to Z_twtp1
            df_xtwtp1[[nonzero_loc[0]], (i+n_Y)*n_W+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'][0,i]
        # to wtp1q
        df_wtp1q[[nonzero_loc[0]],[nonzero_loc[1]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['c'].item()
    
    ################################################################################################
    ## df_xtp1wtp1 #################################################################################
    ################################################################################################
    
    # vmc_tp1wtp1
    nonzero_vr_df = df['xtp1wtp1'][2:,:n_W].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        for i in range(n_Z):
            # to Z_tp1wtp1
            df_xtp1wtp1[[nonzero_loc[0]], (i+n_Y)*n_W+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['x'][0,i]
        # to wtp1q
        df_wtp1q[[nonzero_loc[0]],[nonzero_loc[1]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['c'].item()

    # rmc_tp1wtp1
    nonzero_vr_df = df['xtp1wtp1'][2:,n_W:2*n_W].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        for i in range(n_Z):
            # to Z_tp1wtp1
            df_xtp1wtp1[[nonzero_loc[0]], (i+n_Y)*n_W+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['x'][0,i]
        # to wtp1q
        df_wtp1q[[nonzero_loc[0]],[nonzero_loc[1]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['c'].item()
        
    ################################################################################################
    ## df_xtq ######################################################################################
    ################################################################################################
    nonzero_vr_df = df['xtq'][2:,0:2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        if nonzero_loc[1] == 0:
            # vmc_tq to Z_tq
            df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['vmc1_t']['x']
            # vmc_tq to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['c']

        elif nonzero_loc[1] == 1:
            # rmc_tq to Z_tq
            df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['rmc1_t']['x']
            # rmc_tq to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['c']
        
            
    ################################################################################################
    ## df_xtp1q ####################################################################################
    ################################################################################################
    nonzero_vr_df = df['xtp1q'][2:,0:2].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        if nonzero_loc[1] == 0:
            # vmc_tp1q to Z_tp1q
            df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['vmc1_t']['x']
            # vmc_tp1q to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['vmc1_t']['c']

        elif nonzero_loc[1] == 1:
            # rmc_tp1q to Z_tp1q
            df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * util_sol['rmc1_t']['x']
            # rmc_tp1q to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * util_sol['rmc1_t']['c']
    
    df_plug_in = {'xt':df_xt,\
                  'xtp1':df_xtp1,\
                  'wtp1':df_wtp1,\
                  'q':df_q,\
                  'xtxt':df_xtxt,\
                  'xtxtp1':df_xtxtp1,\
                  'xtwtp1':df_xtwtp1,\
                  'xtq':df_xtq,\
                  'xtp1xtp1':df_xtp1xtp1,\
                  'xtp1wtp1':df_xtp1wtp1,\
                  'xtp1q':df_xtp1q,\
                  'wtp1wtp1':df_wtp1wtp1,\
                  'wtp1q':df_wtp1q,\
                  'qq':df_qq}
    return  df_plug_in


def plug_in_state_first_order(df, var_shape, X1_t, endo_loc_list):
    n_Y, n_Z, n_W = var_shape
    n_X = n_Y + n_Z
    n_endo = len(endo_loc_list)
    df_xt       = df['xt'][-n_Z:,-n_Z:].copy()
    df_xtp1     = df['xtp1'][-n_Z:,-n_Z:].copy()
    df_wtp1     = df['wtp1'][-n_Z:,:].copy()
    df_q        = df['q'][-n_Z:].copy()

    ################################################################################################
    ## 1. df_xt ####################################################################################
    ################################################################################################

    nonzero_vr_df = df['xt'][-n_Z:,0:n_endo].copy()                    
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # endo1_t to Z1_t
        df_xt[[nonzero_loc[0]],-n_Z:]  += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * X1_t[endo_loc_list[nonzero_loc[1]]]['x']

    ################################################################################################
    ## 2. df_xtp1 ##################################################################################
    ################################################################################################ 

    nonzero_vr_df = df['xtp1'][-n_Z:,0:n_endo].copy()                   
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # endo1_tp1 to Z1_tp1
        df_xtp1[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * X1_t[endo_loc_list[nonzero_loc[1]]]['x']

    df_plug_in = {'xt':df_xt,\
                'xtp1':df_xtp1,\
                'wtp1':df_wtp1,\
                'q':df_q}
            
    return df_plug_in

def plug_in_state_second_order(df, var_shape, X1_t, X2_t, endo_loc_list):
    n_Y, n_Z, n_W = var_shape
    n_X = n_Y + n_Z
    n_endo = len(endo_loc_list)

    df_xt       = df['xt'][-n_Z:,-n_Z:].copy()
    df_xtp1     = df['xtp1'][-n_Z:,-n_Z:].copy()
    df_wtp1     = df['wtp1'][-n_Z:,:].copy()
    df_q        = df['q'][-n_Z:].copy()

    df_xtxt     = df['xtxt'][-n_Z:,n_endo*(n_Z+n_endo):][:,np.tile(np.concatenate([np.repeat(False,n_endo),np.repeat(True,n_Z)]),n_Z)].copy()
    df_xtxtp1   = df['xtxtp1'][-n_Z:,n_endo*(n_Z+n_endo):][:,np.tile(np.concatenate([np.repeat(False,n_endo),np.repeat(True,n_Z)]),n_Z)].copy()
    df_xtwtp1   = df['xtwtp1'][-n_Z:,n_endo*n_W:].copy()
    df_xtq      = df['xtq'][-n_Z:,-n_Z:].copy() 
    df_xtp1xtp1 = df['xtp1xtp1'][-n_Z:,n_endo*(n_Z+n_endo):][:,np.tile(np.concatenate([np.repeat(False,n_endo),np.repeat(True,n_Z)]),n_Z)].copy()
    df_xtp1wtp1 = df['xtp1wtp1'][-n_Z:,n_endo*n_W:].copy()
    df_xtp1q    = df['xtp1q'][-n_Z:,-n_Z:].copy() 
    df_wtp1wtp1 = df['wtp1wtp1'][-n_Z:,:].copy() 
    df_wtp1q    = df['wtp1q'][-n_Z:,:].copy() 
    df_qq       = df['qq'][-n_Z:,:].copy() 

    ################################################################################################
    ## df_xt #######################################################################################
    ################################################################################################
    nonzero_vr_df = df['xt'][-n_Z:,0:n_endo].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # endo2_t to Z2_t
        df_xt[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * X2_t[endo_loc_list[nonzero_loc[1]]]['x2']
        # endo2_t
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X2_t[endo_loc_list[nonzero_loc[1]]]['xx'].reshape([n_Z,n_Z]).T.reshape([1,-1])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):   
            # endo2_t to Z1_tZ1_t
            df_xtxt[[nonzero_loc[0]], (i+1)*n_Z-n_Z : (i+1)*n_Z] += new_zz_coefs[i]
        # endo2_t to Z1_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * X2_t[endo_loc_list[nonzero_loc[1]]]['x']/2
        # endo2_t to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X2_t[endo_loc_list[nonzero_loc[1]]]['c']

    ################################################################################################
    ## df_xtp1 #####################################################################################
    ################################################################################################    
    nonzero_vr_df = df['xtp1'][-n_Z:,0:n_endo].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # endo2_tp1 to Z2_tp1
        df_xtp1[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * X2_t[endo_loc_list[nonzero_loc[1]]]['x2']
        # endo2_tp1
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X2_t[endo_loc_list[nonzero_loc[1]]]['xx'].reshape([n_Z,n_Z]).T.reshape([1,-1])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):   
            # vmc_2tp1 to Z1_tp1Z1_tp1
            df_xtp1xtp1[[nonzero_loc[0]], (i+1)*n_Z-n_Z : (i+1)*n_Z] += new_zz_coefs[i]
        # vmc_2tp1 to Z1_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * X2_t[endo_loc_list[nonzero_loc[1]]]['x']/2
        # vmc_2tp1 to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X2_t[endo_loc_list[nonzero_loc[1]]]['c']

    ################################################################################################
    ## df_xtxtp1 ###################################################################################
    ################################################################################################
    # vmc_tp1vmc_t
    nonzero_vr_df = df['xtxtp1'][-n_Z:,:1].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(X1_t[endo_loc_list[0]]['x'],X1_t[endo_loc_list[0]]['x'])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):
            # to Z_tp1Z_t
            df_xtxtp1[[nonzero_loc[0]], (i+1)*n_Z-n_Z : (i+1)*n_Z] += new_zz_coefs[i]
        # to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'] * X1_t[endo_loc_list[0]]['c']
        # to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'] * X1_t[endo_loc_list[0]]['c']
        # to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (X1_t[endo_loc_list[0]]['c']*X1_t[endo_loc_list[0]]['c'])*2

    if n_endo == 1:
        # x_tp1vmc_t
        nonzero_vr_df = df['xtxtp1'][-n_Z:,n_Z+1-n_Z:n_Z+1].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to x_tp1Z_t
            for i in range(n_Z):
                df_xtxtp1[[nonzero_loc[0]], i*n_Z+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'][0,i]
            # to xtp1q
            df_xtp1q[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['c'].item()
        
        # vmc_tp1x_t
        nonzero_vr_df = df['xtxtp1'][-n_Z:,1*(n_Z+1):][:,np.tile(np.concatenate([np.array([True]),np.repeat(False,n_Z)]),n_Z)].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to Z_tp1x_t
            df_xtxtp1[[nonzero_loc[0]], n_Z*(nonzero_loc[1]+1)-n_Z:n_Z*(nonzero_loc[1]+1)] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x']
            # to x_tq
            df_xtq[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['c'].item()

    elif n_endo == 2:
        # rmc_tp1vmc_t
        nonzero_vr_df = df['xtxtp1'][-n_Z:,1:2].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(X1_t[endo_loc_list[0]]['x'],X1_t[endo_loc_list[1]]['x'])
            new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
            for i in range(n_Z):   
                # to Z_tp1Z_t
                df_xtxtp1[[nonzero_loc[0]], (i+1)*n_Z-n_Z : (i+1)*n_Z] += new_zz_coefs[i]
            # to Z_tq
            df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'] * X1_t[endo_loc_list[1]]['c']
            # to Z_tp1q
            df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'] * X1_t[endo_loc_list[0]]['c']
            # to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (X1_t[endo_loc_list[0]]['c']*X1_t[endo_loc_list[1]]['c'])*2

        # vmc_tp1rmc_t
        nonzero_vr_df = df['xtxtp1'][-n_Z:,(n_Z+2):(n_Z+2)+1].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(X1_t[endo_loc_list[1]]['x'],X1_t[endo_loc_list[0]]['x'])
            new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
            for i in range(n_Z):   
                # to Z_tp1Z_t
                df_xtxtp1[[nonzero_loc[0]], (i+1)*n_Z-n_Z : (i+1)*n_Z] += new_zz_coefs[i]
            # to Z_tq
            df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'] * X1_t[endo_loc_list[0]]['c']
            # to Z_tp1q
            df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'] * X1_t[endo_loc_list[1]]['c']
            # to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (X1_t[endo_loc_list[0]]['c']*X1_t[endo_loc_list[1]]['c'])*2

        # rmc_tp1rmc_t
        nonzero_vr_df = df['xtxtp1'][-n_Z:,(n_Z+2)+1:(n_Z+2)+2].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(X1_t[endo_loc_list[1]]['x'],X1_t[endo_loc_list[1]]['x'])
            new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
            for i in range(n_Z):   
                # to Z_tp1Z_t
                df_xtxtp1[[nonzero_loc[0]], (i+1)*n_Z-n_Z : (i+1)*n_Z] += new_zz_coefs[i]
            # to Z_tq
            df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'] * X1_t[endo_loc_list[1]]['c']
            # to Z_tp1q
            df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'] * X1_t[endo_loc_list[1]]['c']
            # to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (X1_t[endo_loc_list[1]]['c']*X1_t[endo_loc_list[1]]['c'])*2

        # x_tp1vmc_t
        nonzero_vr_df = df['xtxtp1'][-n_Z:,n_Z+2-n_Z:n_Z+2].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to x_tp1Z_t
            for i in range(n_Z):
                df_xtxtp1[[nonzero_loc[0]], i*n_Z+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'][0,i]
            # to xtp1q
            df_xtp1q[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['c'].item()

        # x_tp1rmc_t
        nonzero_vr_df = df['xtxtp1'][-n_Z:,(n_Z+2)*2-n_Z:(n_Z+2)*2].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to x_tp1Z_t
            for i in range(n_Z):
                df_xtxtp1[[nonzero_loc[0]], i*n_Z+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'][0,i]
            # to xtp1q
            df_xtp1q[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['c'].item()

        # vmc_tp1x_t
        nonzero_vr_df = df['xtxtp1'][-n_Z:,2*(n_Z+2):][:,np.tile(np.concatenate([np.array([True,False]),np.repeat(False,n_Z)]),n_Z)].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to Z_tp1x_t
            df_xtxtp1[[nonzero_loc[0]], n_Z*(nonzero_loc[1]+1)-n_Z:n_Z*(nonzero_loc[1]+1)] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x']
            # to x_tq
            df_xtq[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['c'].item()

        # rmc_tp1x_t
        nonzero_vr_df = df['xtxtp1'][-n_Z:,2*(n_Z+2):][:,np.tile(np.concatenate([np.array([False,True]),np.repeat(False,n_Z)]),n_Z)].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to Z_tp1x_t
            df_xtxtp1[[nonzero_loc[0]], n_Z*(nonzero_loc[1]+1)-n_Z:n_Z*(nonzero_loc[1]+1)] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x']
            # to x_tq
            df_xtq[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['c'].item()

    ################################################################################################
    ## df_xtxt #####################################################################################
    ################################################################################################

    # vmc_tvmc_t
    nonzero_vr_df = df['xtxt'][-n_Z:,:1].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(X1_t[endo_loc_list[0]]['x'],X1_t[endo_loc_list[0]]['x'])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):
            # to Z_tZ_t
            df_xtxt[[nonzero_loc[0]], (i+1)*n_Z-n_Z : (i+1)*n_Z] += new_zz_coefs[i]
        # to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'] * X1_t[endo_loc_list[0]]['c']/2
        # to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'] * X1_t[endo_loc_list[0]]['c']/2
        # to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (X1_t[endo_loc_list[0]]['c']*X1_t[endo_loc_list[0]]['c'])

    if n_endo == 1:
        # x_tvmc_t
        nonzero_vr_df = df['xtxt'][-n_Z:,n_Z+1-n_Z:n_Z+1].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to x_tZ_t
            for i in range(n_Z):
                df_xtxt[[nonzero_loc[0]], i*n_Z+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'][0,i]
            # to xtq
            df_xtq[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['c'].item()/2
        
        # vmc_tx_t
        nonzero_vr_df = df['xtxt'][-n_Z:,1*(n_Z+1):][:,np.tile(np.concatenate([np.array([True]),np.repeat(False,n_Z)]),n_Z)].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to Z_tx_t
            df_xtxt[[nonzero_loc[0]], n_Z*(nonzero_loc[1]+1)-n_Z:n_Z*(nonzero_loc[1]+1)] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x']
            # to x_tq
            df_xtq[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['c'].item()/2

    elif n_endo == 2:
        # rmc_tvmc_t
        nonzero_vr_df = df['xtxt'][-n_Z:,1:2].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(X1_t[endo_loc_list[0]]['x'],X1_t[endo_loc_list[1]]['x'])
            new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
            for i in range(n_Z):   
                # to Z_tZ_t
                df_xtxt[[nonzero_loc[0]], (i+1)*n_Z-n_Z : (i+1)*n_Z] += new_zz_coefs[i]
            # to Z_tq
            df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'] * X1_t[endo_loc_list[1]]['c']/2
            # to Z_tq
            df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'] * X1_t[endo_loc_list[0]]['c']/2
            # to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (X1_t[endo_loc_list[0]]['c']*X1_t[endo_loc_list[1]]['c'])

        # vmc_trmc_t
        nonzero_vr_df = df['xtxt'][-n_Z:,(n_Z+2):(n_Z+2)+1].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(X1_t[endo_loc_list[1]]['x'],X1_t[endo_loc_list[0]]['x'])
            new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
            for i in range(n_Z):   
                # to Z_tZ_t
                df_xtxt[[nonzero_loc[0]], (i+1)*n_Z-n_Z : (i+1)*n_Z] += new_zz_coefs[i]
            # to Z_tq
            df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'] * X1_t[endo_loc_list[0]]['c']/2
            # to Z_tq
            df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'] * X1_t[endo_loc_list[1]]['c']/2
            # to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (X1_t[endo_loc_list[0]]['c']*X1_t[endo_loc_list[1]]['c'])

        # rmc_trmc_t
        nonzero_vr_df = df['xtxt'][-n_Z:,(n_Z+2)+1:(n_Z+2)+2].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(X1_t[endo_loc_list[1]]['x'],X1_t[endo_loc_list[1]]['x'])
            new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
            for i in range(n_Z):   
                # to Z_tZ_t
                df_xtxt[[nonzero_loc[0]], (i+1)*n_Z-n_Z : (i+1)*n_Z] += new_zz_coefs[i]
            # to Z_tq
            df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'] * X1_t[endo_loc_list[1]]['c']/2
            # to Z_tq
            df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'] * X1_t[endo_loc_list[1]]['c']/2
            # to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (X1_t[endo_loc_list[1]]['c']*X1_t[endo_loc_list[1]]['c'])

        # x_tvmc_t
        nonzero_vr_df = df['xtxt'][-n_Z:,n_Z+2-n_Z:n_Z+2].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to x_tZ_t
            for i in range(n_Z):
                df_xtxt[[nonzero_loc[0]], i*n_Z+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'][0,i]
            # to xtq
            df_xtq[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['c'].item()/2

        # x_trmc_t
        nonzero_vr_df = df['xtxt'][-n_Z:,(n_Z+2)*2-n_Z:(n_Z+2)*2].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to x_tZ_t
            for i in range(n_Z):
                df_xtxt[[nonzero_loc[0]], i*n_Z+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'][0,i]
            # to xtq
            df_xtq[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['c'].item()/2

        # vmc_tx_t
        nonzero_vr_df = df['xtxt'][-n_Z:,2*(n_Z+2):][:,np.tile(np.concatenate([np.array([True,False]),np.repeat(False,n_Z)]),n_Z)].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to Z_tx_t
            df_xtxt[[nonzero_loc[0]], n_Z*(nonzero_loc[1]+1)-n_Z:n_Z*(nonzero_loc[1]+1)] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x']
            # to x_tq
            df_xtq[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['c'].item()/2

        # rmc_tx_t
        nonzero_vr_df = df['xtxt'][-n_Z:,2*(n_Z+2):][:,np.tile(np.concatenate([np.array([False,True]),np.repeat(False,n_Z)]),n_Z)].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to Z_tx_t
            df_xtxt[[nonzero_loc[0]], n_Z*(nonzero_loc[1]+1)-n_Z:n_Z*(nonzero_loc[1]+1)] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x']
            # to x_tq
            df_xtq[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['c'].item()/2

    ################################################################################################
    ## df_xtp1xtp1 #################################################################################
    ################################################################################################
    ## df_xtp1xtp1
    # vmc_tp1vmc_tp1
    nonzero_vr_df = df['xtp1xtp1'][-n_Z:,:1].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(X1_t[endo_loc_list[0]]['x'],X1_t[endo_loc_list[0]]['x'])
        new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
        for i in range(n_Z):
            # to Z_tp1Z_tp1
            df_xtp1xtp1[[nonzero_loc[0]], (i+1)*n_Z-n_Z : (i+1)*n_Z] += new_zz_coefs[i]
        # to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'] * X1_t[endo_loc_list[0]]['c']/2
        # to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'] * X1_t[endo_loc_list[0]]['c']/2
        # to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (X1_t[endo_loc_list[0]]['c']*X1_t[endo_loc_list[0]]['c'])

    if n_endo == 1:
        # x_tp1vmc_tp1
        nonzero_vr_df = df['xtp1xtp1'][-n_Z:,n_Z+1-n_Z:n_Z+1].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to x_tp1Z_tp1
            for i in range(n_Z):
                df_xtp1xtp1[[nonzero_loc[0]], i*n_Z+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'][0,i]
            # to xtp1q
            df_xtp1q[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['c'].item()/2

        # vmc_tp1x_tp1
        nonzero_vr_df = df['xtp1xtp1'][-n_Z:,1*(n_Z+1):][:,np.tile(np.concatenate([np.array([True]),np.repeat(False,n_Z)]),n_Z)].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to Z_tp1x_tp1
            df_xtp1xtp1[[nonzero_loc[0]], n_Z*(nonzero_loc[1]+1)-n_Z:n_Z*(nonzero_loc[1]+1)] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x']
            # to x_tp1q
            df_xtp1q[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['c'].item()/2

    elif n_endo == 2:
        # rmc_tp1vmc_tp1
        nonzero_vr_df = df['xtp1xtp1'][-n_Z:,1:2].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(X1_t[endo_loc_list[0]]['x'],X1_t[endo_loc_list[1]]['x'])
            new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
            for i in range(n_Z):   
                # to Z_tp1Z_tp1
                df_xtp1xtp1[[nonzero_loc[0]], (i+1)*n_Z-n_Z : (i+1)*n_Z] += new_zz_coefs[i]
            # to Z_tp1q
            df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'] * X1_t[endo_loc_list[1]]['c']/2
            # to Z_tp1q
            df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'] * X1_t[endo_loc_list[0]]['c']/2
            # to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (X1_t[endo_loc_list[0]]['c']*X1_t[endo_loc_list[1]]['c'])

        # vmc_tp1rmc_tp1
        nonzero_vr_df = df['xtp1xtp1'][-n_Z:,(n_Z+2):(n_Z+2)+1].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(X1_t[endo_loc_list[1]]['x'],X1_t[endo_loc_list[0]]['x'])
            new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
            for i in range(n_Z):   
                # to Z_tp1Z_tp1
                df_xtp1xtp1[[nonzero_loc[0]], (i+1)*n_Z-n_Z : (i+1)*n_Z] += new_zz_coefs[i]
            # to Z_tp1q
            df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'] * X1_t[endo_loc_list[0]]['c']/2
            # to Z_tp1q
            df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'] * X1_t[endo_loc_list[1]]['c']/2
            # to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (X1_t[endo_loc_list[0]]['c']*X1_t[endo_loc_list[1]]['c'])

        # rmc_tp1rmc_tp1
        nonzero_vr_df = df['xtp1xtp1'][-n_Z:,(n_Z+2)+1:(n_Z+2)+2].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            new_zz_coef = nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * np.kron(X1_t[endo_loc_list[1]]['x'],X1_t[endo_loc_list[1]]['x'])
            new_zz_coefs = np.split(new_zz_coef,n_Z,axis=1)
            for i in range(n_Z):   
                # to Z_tp1Z_tp1
                df_xtp1xtp1[[nonzero_loc[0]], (i+1)*n_Z-n_Z : (i+1)*n_Z] += new_zz_coefs[i]
            # to Z_tp1q
            df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'] * X1_t[endo_loc_list[1]]['c']/2
            # to Z_tp1q
            df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'] * X1_t[endo_loc_list[1]]['c']/2
            # to qq
            df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * (X1_t[endo_loc_list[1]]['c']*X1_t[endo_loc_list[1]]['c'])

        # x_tp1vmc_tp1
        nonzero_vr_df = df['xtp1xtp1'][-n_Z:,n_Z+2-n_Z:n_Z+2].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to x_tp1Z_tp1
            for i in range(n_Z):
                df_xtp1xtp1[[nonzero_loc[0]], i*n_Z+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'][0,i]
            # to xtp1q
            df_xtp1q[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['c'].item()/2

        # x_tp1rmc_tp1
        nonzero_vr_df = df['xtp1xtp1'][-n_Z:,(n_Z+2)*2-n_Z:(n_Z+2)*2].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to x_tp1Z_tp1
            for i in range(n_Z):
                df_xtp1xtp1[[nonzero_loc[0]], i*n_Z+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'][0,i]
            # to xtp1q
            df_xtp1q[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['c'].item()/2

        # vmc_tp1x_tp1
        nonzero_vr_df = df['xtp1xtp1'][-n_Z:,2*(n_Z+2):][:,np.tile(np.concatenate([np.array([True,False]),np.repeat(False,n_Z)]),n_Z)].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to Z_tp1x_tp1
            df_xtp1xtp1[[nonzero_loc[0]], n_Z*(nonzero_loc[1]+1)-n_Z:n_Z*(nonzero_loc[1]+1)] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x']
            # to x_tp1q
            df_xtp1q[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['c'].item()/2

        # rmc_tp1x_tp1
        nonzero_vr_df = df['xtp1xtp1'][-n_Z:,2*(n_Z+2):][:,np.tile(np.concatenate([np.array([False,True]),np.repeat(False,n_Z)]),n_Z)].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            # to Z_tp1x_tp1
            df_xtp1xtp1[[nonzero_loc[0]], n_Z*(nonzero_loc[1]+1)-n_Z:n_Z*(nonzero_loc[1]+1)] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x']
            # to x_tp1q
            df_xtp1q[[nonzero_loc[0]],nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['c'].item()/2
        

    ################################################################################################
    ## df_xtwtp1 ###################################################################################
    ################################################################################################

    # vmc_twtp1
    nonzero_vr_df = df['xtwtp1'][-n_Z:,:n_W].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        for i in range(n_Z):
            # to Z_twtp1
            df_xtwtp1[[nonzero_loc[0]], i*n_W+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'][0,i]
        # to wtp1q
        df_wtp1q[[nonzero_loc[0]],[nonzero_loc[1]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['c'].item()

    if n_endo == 2:
        # rmc_twtp1
        nonzero_vr_df = df['xtwtp1'][-n_Z:,n_W:2*n_W].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            for i in range(n_Z):
                # to Z_twtp1
                df_xtwtp1[[nonzero_loc[0]], i*n_W+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'][0,i]
            # to wtp1q
            df_wtp1q[[nonzero_loc[0]],[nonzero_loc[1]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['c'].item()

    ################################################################################################
    ## df_xtp1wtp1 #################################################################################
    ################################################################################################

    # vmc_tp1wtp1
    nonzero_vr_df = df['xtp1wtp1'][-n_Z:,:n_W].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        for i in range(n_Z):
            # to Z_tp1wtp1
            df_xtp1wtp1[[nonzero_loc[0]], i*n_W+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['x'][0,i]
        # to wtp1q
        df_wtp1q[[nonzero_loc[0]],[nonzero_loc[1]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[0]]['c'].item()

    if n_endo == 2:
        # rmc_tp1wtp1
        nonzero_vr_df = df['xtp1wtp1'][-n_Z:,n_W:2*n_W].copy()
        nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
        for nonzero_loc in nonzero_vr_index:
            for i in range(n_Z):
                # to Z_tp1wtp1
                df_xtp1wtp1[[nonzero_loc[0]], i*n_W+nonzero_loc[1]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['x'][0,i]
            # to wtp1q
            df_wtp1q[[nonzero_loc[0]],[nonzero_loc[1]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[1]]['c'].item()
        
    ################################################################################################
    ## df_xtq ######################################################################################
    ################################################################################################
    nonzero_vr_df = df['xtq'][-n_Z:,0:n_endo].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # vmc_tq to Z_tq
        df_xtq[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * X1_t[endo_loc_list[nonzero_loc[1]]]['x']
        # vmc_tq to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[nonzero_loc[1]]]['c']

    ################################################################################################
    ## df_xtp1q ####################################################################################
    ################################################################################################
    nonzero_vr_df = df['xtp1q'][-n_Z:,0:n_endo].copy()
    nonzero_vr_index = np.transpose(np.nonzero(nonzero_vr_df))
    for nonzero_loc in nonzero_vr_index:
        # vmc_tp1q to Z_tp1q
        df_xtp1q[[nonzero_loc[0]],-n_Z:] += nonzero_vr_df[nonzero_loc[0],nonzero_loc[1]] * X1_t[endo_loc_list[nonzero_loc[1]]]['x']
        # vmc_tp1q to qq
        df_qq[[nonzero_loc[0]]] += nonzero_vr_df[nonzero_loc[0], nonzero_loc[1]] * X1_t[endo_loc_list[nonzero_loc[1]]]['c']

    df_plug_in = {'xt':df_xt,\
                'xtp1':df_xtp1,\
                'wtp1':df_wtp1,\
                'q':df_q,\
                'xtxt':df_xtxt,\
                'xtxtp1':df_xtxtp1,\
                'xtwtp1':df_xtwtp1,\
                'xtq':df_xtq,\
                'xtp1xtp1':df_xtp1xtp1,\
                'xtp1wtp1':df_xtp1wtp1,\
                'xtp1q':df_xtp1q,\
                'wtp1wtp1':df_wtp1wtp1,\
                'wtp1q':df_wtp1q,\
                'qq':df_qq}
    return df_plug_in