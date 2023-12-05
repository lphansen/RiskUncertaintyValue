import numpy as np
import autograd.numpy as anp
import scipy as sp
from scipy import optimize
np.set_printoptions(suppress=True)
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display, HTML, Math
pd.options.display.float_format = '{:.3g}'.format
sns.set(font_scale = 1.5)
import warnings
warnings.filterwarnings("ignore")

from uncertain_expansion import uncertain_expansion, approximate_fun
from elasticity import exposure_elasticity, price_elasticity
from lin_quad import LinQuadVar
from lin_quad_util import next_period

def eq_cond_BY(Var_t, Var_tp1, W_tp1, q, mode, *args):

    # Parameters for the model
    γ, β, ρ, α, ϕ_e, σ_squared, ν_1, σ_w, μ, μ_d, ϕ, ϕ_d, ϕ_c, π = args

    # Variables:
    q_t, pd_t, x_t, σ_t_squared = Var_t.ravel()
    q_tp1, pd_tp1, x_tp1, σ_tp1_squared = Var_tp1.ravel()
    w1_tp1, w2_tp1, w3_tp1, w4_tp1 = W_tp1.ravel()

    σ_t = anp.sqrt(σ_t_squared)
    gc_tp1 = μ + x_t + ϕ_c*σ_t*w3_tp1
    gd_tp1 = μ_d + ϕ*x_t + π*σ_t*w3_tp1 + ϕ_d*σ_t*w4_tp1
    
    psi1_1 = anp.exp(anp.log(β)- ρ*gc_tp1 + gd_tp1)*(anp.exp(pd_tp1) + 1)
    psi2_1 =  anp.exp(pd_t)

    # State process
    phi_1 = α * x_t + ϕ_e * σ_t * w1_tp1 - x_tp1
    phi_2 = σ_squared + ν_1 * (σ_t_squared - σ_squared) + σ_w * w2_tp1 - σ_tp1_squared
    
    if mode == 'psi1':
        return np.array([psi1_1])
    elif mode == 'psi2':
        return np.array([psi2_1])
    elif mode == 'phi':
        return np.array([phi_1, phi_2])

    return anp.array([psi1_1*anp.exp(q_tp1) - psi2_1, phi_1, phi_2])

def ss_func_BY(*args):

    # Parameters for the model
    γ, β, ρ, α, ϕ_e, σ_squared, ν_1, σ_w, μ, μ_d, ϕ, ϕ_d, ϕ_c, π = args

    sdf = anp.exp(anp.log(β) - ρ*μ)

    return np.array([0, np.log((sdf * np.exp(μ_d))/(1-np.exp(μ_d)*sdf)), 0., σ_squared])

def gc_tp1_approx(Var_t, Var_tp1, W_tp1, q, *args):

    # Parameters for the model
    γ, β, ρ, α, ϕ_e, σ_squared, ν_1, σ_w, μ, μ_d, ϕ, ϕ_d, ϕ_c, π = args

    # Variables:
    q_t, pd_t, x_t, σ_t_squared = Var_t.ravel()
    q_tp1, pd_tp1, x_tp1, σ_tp1_squared = Var_tp1.ravel()
    w1_tp1, w2_tp1, w3_tp1, w4_tp1 = W_tp1.ravel()
    σ_t = anp.sqrt(σ_t_squared)
    
    gc_tp1 = μ + x_t + ϕ_c*σ_t*w3_tp1
    
    return gc_tp1

def gd_tp1_approx(Var_t, Var_tp1, W_tp1, q, *args):

    # Parameters for the model
    γ, β, ρ, α, ϕ_e, σ_squared, ν_1, σ_w, μ, μ_d, ϕ, ϕ_d, ϕ_c, π = args

    # Variables:
    pd_t, x_t, σ_t_squared = Var_t.ravel()
    pd_tp1, x_tp1, σ_tp1_squared = Var_tp1.ravel()
    w1_tp1, w2_tp1, w3_tp1, w4_tp1 = W_tp1.ravel()
    σ_t = anp.sqrt(σ_t_squared)
    
    gd_tp1 = μ_d + ϕ*x_t + π*σ_t*w3_tp1 + ϕ_d*σ_t*w4_tp1
    
    return gd_tp1

def calc_SDF(res):

    n_J, n_X, n_W = res['var_shape']
    β = res['β']
    ρ = res['ρ']

    X1_tp1 = res['X1_tp1']
    X2_tp1 = res['X2_tp1']

    gc_tp1 = res['gc_tp1']
    gc0_tp1 = res['gc0_tp1']
    gc1_tp1 = res['gc1_tp1']
    gc2_tp1 = res['gc2_tp1']

    vmr1_tp1 = res['vmr1_tp1']
    vmr2_tp1 = res['vmr2_tp1']
    log_N_tilde = res['log_N_tilde']

    S0_tp1 = LinQuadVar({'c':np.log(β)-ρ*np.array([[gc0_tp1]])}, shape = (1,n_X,n_W))
    S1_tp1 = (ρ-1)*vmr1_tp1 -ρ*gc1_tp1
    S2_tp1 = (ρ-1)*vmr2_tp1 -ρ*gc2_tp1

    log_SDF = S0_tp1 + S1_tp1 + 0.5 * S2_tp1 + log_N_tilde

    return log_SDF, gc_tp1, X1_tp1, X2_tp1

def solve_BY(ρ= 2./3):

    σ_original = 0.0078

    γ = 10
    β = .998
    # ρ = 2./3
    α = 0.979
    ϕ_e = 0.044 * σ_original
    σ_squared = 1.0
    ν_1 = 0.987
    σ_w = 0.23 * 1e-5 / σ_original**2
    μ = 0.0015
    μ_d = 0.0015
    ϕ_c = 1.0 * σ_original
    ϕ = 3.0
    ϕ_d = 4.5 * σ_original
    π = 0.0

    # Solve BY model
    eq = eq_cond_BY
    ss = ss_func_BY 
    var_shape = (1, 2, 4)
    gc_tp1_fun = gc_tp1_approx
    approach = '1'
    args = (γ, β, ρ, α, ϕ_e, σ_squared, ν_1, σ_w, μ, μ_d, ϕ, ϕ_d, ϕ_c, π)
    init_util = None
    iter_tol = 1e-8
    max_iter = 50
    ModelSol = uncertain_expansion(eq, ss, var_shape, args, gc_tp1_fun, approach, init_util, iter_tol, max_iter)

    # Approximate dividend growth process
    n_J, n_X, n_W = var_shape
    ss = ModelSol['ss']
    gd_tp1_list = approximate_fun(gd_tp1_approx, ss, (1, n_X, n_W), ModelSol['JX1_t'], ModelSol['JX2_t'], ModelSol['X1_tp1'], ModelSol['X2_tp1'], args)

    res = {'JX_tp1':ModelSol['JX_tp1'],\
           'X1_tp1':ModelSol['X1_tp1'],\
            'X2_tp1':ModelSol['X2_tp1'],\
            'log_N_tilde':ModelSol['log_N_tilde'],\
            'β':ModelSol['args'][1],\
            'ρ':ModelSol['args'][2],\
            'vmc1_t':ModelSol['util_sol']['vmc1_t'],\
            'vmc2_t':ModelSol['util_sol']['vmc2_t'],\
            'rmc1_t':ModelSol['util_sol']['rmc1_t'],\
            'rmc2_t':ModelSol['util_sol']['rmc2_t'],\
            'vmr1_tp1':ModelSol['vmr1_tp1'],\
            'vmr2_tp1':ModelSol['vmr2_tp1'],\
            'var_shape':ModelSol['var_shape'],\
            'gc_tp1':ModelSol['gc_tp1'],\
            'gc0_tp1':ModelSol['gc0_tp1'],\
            'gc1_tp1':ModelSol['gc1_tp1'],\
            'gc2_tp1':ModelSol['gc2_tp1'],\
            'gd_tp1':gd_tp1_list[0],\
            'gd0_tp1':gd_tp1_list[1],\
            'gd1_tp1':gd_tp1_list[2],\
            'gd2_tp1':gd_tp1_list[3],}

    return res

# res_006 = solve_BY(ρ= 2./3)
# res_010 = solve_BY(ρ= 1.00001)
# res_015 = solve_BY(ρ= 1.5)
# res_100 = solve_BY(ρ= 10)

# with open('data/res_006.pkl', 'wb') as f:
#     pickle.dump(res_006,f)
# with open('data/res_010.pkl', 'wb') as f:
#     pickle.dump(res_010,f)
# with open('data/res_015.pkl', 'wb') as f:
#     pickle.dump(res_015,f)
# with open('data/res_100.pkl', 'wb') as f:
#     pickle.dump(res_100,f)

def solve_BY_elas(γ=10, β=.998, ρ=2./3, α = 0.979, ϕ_e = 0.044*0.0078, ν_1=0.987, σ_w = 0.23 * 1e-5 / 0.0078**2, μ=0.0015, ϕ_c=0.0078):

    σ_original = 0.0078

    # γ = 10
    # β = .998
    # ρ = 2./3
    # α = 0.979
    # ϕ_e = 0.044 * σ_original
    σ_squared = 1.0
    # ν_1 = 0.987
    # σ_w = 0.23 * 1e-5 / σ_original**2
    # μ = 0.0015
    μ_d = 0.0015
    # ϕ_c = 1.0 * σ_original
    ϕ = 3.0
    ϕ_d = 4.5 * σ_original
    π = 0.0

    eq = eq_cond_BY
    ss = ss_func_BY 
    var_shape = (1, 2, 4)
    gc_tp1_fun = gc_tp1_approx
    args = (γ, β, ρ, α, ϕ_e, σ_squared, ν_1, σ_w, μ, μ_d, ϕ, ϕ_d, ϕ_c, π)
    approach = '1'
    init_util = None
    iter_tol = 1e-8
    max_iter = 50
    ModelSol = uncertain_expansion(eq, ss, var_shape, args, gc_tp1_fun, approach, init_util, iter_tol, max_iter)

    res = {'X1_tp1':ModelSol['X1_tp1'],\
            'X2_tp1':ModelSol['X2_tp1'],\
            'log_N_tilde':ModelSol['log_N_tilde'],\
            'β':ModelSol['args'][1],\
            'ρ':ModelSol['args'][2],\
            'vmr1_tp1':ModelSol['vmr1_tp1'],\
            'vmr2_tp1':ModelSol['vmr2_tp1'],\
            'var_shape':ModelSol['var_shape'],\
            'gc_tp1':ModelSol['gc_tp1'],\
            'gc0_tp1':ModelSol['gc0_tp1'],\
            'gc1_tp1':ModelSol['gc1_tp1'],\
            'gc2_tp1':ModelSol['gc2_tp1']}

    T = 360
    quantile = [0.25, 0.5, 0.75]

    log_SDF, gc_tp1, X1_tp1, X2_tp1 = calc_SDF(res)
    expo_elas_shock_0 = [exposure_elasticity(gc_tp1, X1_tp1, X2_tp1, T, shock=0, percentile=p) for p in quantile] 
    expo_elas_shock_1 = [exposure_elasticity(gc_tp1, X1_tp1, X2_tp1, T, shock=1, percentile=p) for p in quantile]
    expo_elas_shock_2 = [exposure_elasticity(gc_tp1, X1_tp1, X2_tp1, T, shock=2, percentile=p) for p in quantile]

    price_elas_shock_0 = [price_elasticity(gc_tp1, log_SDF, X1_tp1, X2_tp1, T, shock=0, percentile=p) for p in quantile]
    price_elas_shock_1 = [price_elasticity(gc_tp1, log_SDF, X1_tp1, X2_tp1, T, shock=1, percentile=p) for p in quantile]
    price_elas_shock_2 = [price_elasticity(gc_tp1, log_SDF, X1_tp1, X2_tp1, T, shock=2, percentile=p) for p in quantile]
    
    fig, axes = plt.subplots(2,3, figsize = (25,13))
    index = ['T','0.25 quantile','0.5 quantile','0.75 quantile']
    plot_expo_elas_shock_0 = pd.DataFrame([np.arange(T),expo_elas_shock_0[0].flatten(),expo_elas_shock_0[1].flatten(),expo_elas_shock_0[2].flatten()], index = index).T
    plot_expo_elas_shock_1 = pd.DataFrame([np.arange(T),expo_elas_shock_1[0].flatten(),expo_elas_shock_1[1].flatten(),expo_elas_shock_1[2].flatten()], index = index).T
    plot_expo_elas_shock_2 = pd.DataFrame([np.arange(T),expo_elas_shock_2[0].flatten(),expo_elas_shock_2[1].flatten(),expo_elas_shock_2[2].flatten()], index = index).T
    plot_price_elas_shock_0 = pd.DataFrame([np.arange(T),price_elas_shock_0[0].flatten(),price_elas_shock_0[1].flatten(),price_elas_shock_0[2].flatten()], index = index).T
    plot_price_elas_shock_1 = pd.DataFrame([np.arange(T),-price_elas_shock_1[0].flatten(),-price_elas_shock_1[1].flatten(),-price_elas_shock_1[2].flatten()], index = index).T
    plot_price_elas_shock_2 = pd.DataFrame([np.arange(T),price_elas_shock_2[0].flatten(),price_elas_shock_2[1].flatten(),price_elas_shock_2[2].flatten()], index = index).T

    n_qt = len(quantile)
    plot_expo_elas = [plot_expo_elas_shock_0, plot_expo_elas_shock_2, plot_expo_elas_shock_1] 
    plot_price_elas = [plot_price_elas_shock_0, plot_price_elas_shock_2, plot_price_elas_shock_1] 
    shock_name = ['growth shock', 'consumption shock', 'volatility shock']
    qt = ['0.25 quantile','0.5 quantile','0.75 quantile']
    colors = ['green','red','blue']

    for i in range(len(plot_expo_elas)):
        for j in range(n_qt):
            sns.lineplot(data = plot_expo_elas[i],  x = 'T', y = qt[j], ax=axes[0,i], color = colors[j], label = qt[j])
            axes[0,i].set_xlabel('')
            axes[0,i].set_ylabel('Exposure elasticity')
            axes[0,i].set_ylim([0,0.045])
            axes[0,i].set_title('Exposure Elasticity with respect to the ' + shock_name[i])

    for i in range(len(plot_price_elas)):
        for j in range(n_qt):
            sns.lineplot(data = plot_price_elas[i],  x = 'T', y = qt[j], ax=axes[1,i], color = colors[j], label = qt[j])
            axes[1,i].set_xlabel('')
            axes[1,i].set_ylabel('Price elasticity')
            axes[1,i].set_ylim([0,0.45])
            axes[1,i].set_title('Price Elasticity with respect to the '+ shock_name[i])
    fig.suptitle('Shock Elasticity for the Consumption Growth')
    fig.tight_layout()
    print('Current paramter settings')
    print('γ = '+str("{:.4g}".format(γ)))
    print('β = '+str("{:.4g}".format(β)))
    print('ρ = '+str("{:.4g}".format(ρ)))
    print('α = '+str("{:.4g}".format(α)))
    print('ϕ_e = '+str("{:.4g}".format(ϕ_e)))
    print('ν_1 = '+str("{:.4g}".format(ν_1)))
    print('σ_w = '+str("{:.4g}".format(σ_w)))
    print('μ = '+str("{:.4g}".format(μ)))
    print('ϕ_c = '+str("{:.4g}".format(ϕ_c)))
    plt.show()

def disp_BY(Lq, Var):
    '''
    Display Linquad in Latex analytical form
    '''
    Lq_disp = {'c': r'{:.4g}'.format(*Lq['c'].flatten().tolist()),\
    'x': r'\begin{{bmatrix}}{:.4g}&{:.4g}\end{{bmatrix}}X_t^1'.format(*Lq['x'].flatten().tolist()),\
    'w':r'\begin{{bmatrix}}{:.4g}&{:.4g}&{:.4g}&{:.4g}\end{{bmatrix}}W_{{t+1}}'.format(*Lq['w'].flatten().tolist()),\
    'x2':r'\begin{{bmatrix}}{:.4g}&{:.4g}\end{{bmatrix}}X_t^2'.format(*Lq['x2'].flatten().tolist()),\
    'xx':r'X^{{1T}}_{{t}}\begin{{bmatrix}}{:.4g}&{:.4g}\\{:.4g}&{:.4g}\end{{bmatrix}}X^1_{{t}}'.format(*Lq['xx'].flatten().tolist()),\
    'xw':r'X^{{1T}}_{{t}}\begin{{bmatrix}}{:.4g}&{:.4g}&{:.4g}&{:.4g}\\{:.4g}&{:.4g}&{:.4g}&{:.4g}\end{{bmatrix}}W_{{t+1}}'.format(*Lq['xw'].flatten().tolist()),\
    'ww':r'W_{{t+1}}^{{T}}\begin{{bmatrix}}{:.4g}&{:.4g}&{:.4g}&{:.4g}\\{:.4g}&{:.4g}&{:.4g}&{:.4g}\\{:.4g}&{:.4g}&{:.4g}&{:.4g}\\{:.4g}&{:.4g}&{:.4g}&{:.4g}\end{{bmatrix}}W_{{t+1}}'.format(*Lq['ww'].flatten().tolist())}
    if (abs(Lq['c'].item())<1e-14) and abs(Lq['c'].item())!=0:
        Lq_disp.pop('c')
        Lq.coeffs.pop('c')
    Lq_disp = Var + '='+ '+'.join([Lq_disp[i] for i in ['c','x','w','x2','xx','xw','ww'] if i in Lq.coeffs])
    display(Math(Lq_disp))

def disp(Lq, Var):
    '''
    Display adjustment cost Linquad in Latex analytical form
    '''
    Lq_disp = {'c': r'{:.4g}'.format(*Lq['c'].flatten().tolist()),\
    'x': r'\begin{{bmatrix}}{:.4g}&{:.4g}\end{{bmatrix}}X_t^1'.format(*Lq['x'].flatten().tolist()),\
    'w':r'\begin{{bmatrix}}{:.4g}&{:.4g}\end{{bmatrix}}W_{{t+1}}'.format(*Lq['w'].flatten().tolist()),\
    'x2':r'\begin{{bmatrix}}{:.4g}&{:.4g}\end{{bmatrix}}X_t^2'.format(*Lq['x2'].flatten().tolist()),\
    'xx':r'X^{{1T}}_{{t}}\begin{{bmatrix}}{:.4g}&{:.4g}\\{:.4g}&{:.4g}\end{{bmatrix}}X^1_{{t}}'.format(*Lq['xx'].flatten().tolist()),\
    'xw':r'X^{{1T}}_{{t}}\begin{{bmatrix}}{:.4g}&{:.4g}\\{:.4g}&{:.4g}\end{{bmatrix}}W_{{t+1}}'.format(*Lq['xw'].flatten().tolist()),\
    'ww':r'W_{{t+1}}^{{T}}\begin{{bmatrix}}{:.4g}&{:.4g}\\{:.4g}&{:.4g}\end{{bmatrix}}W_{{t+1}}'.format(*Lq['ww'].flatten().tolist())}
    if (abs(Lq['c'].item())<1e-14) and abs(Lq['c'].item())!=0:
        Lq_disp.pop('c')
        Lq.coeffs.pop('c')
    Lq_disp = Var + '='+ '+'.join([Lq_disp[i] for i in ['c','x','w','x2','xx','xw','ww'] if i in Lq.coeffs])
    display(Math(Lq_disp))