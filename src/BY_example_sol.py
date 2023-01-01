import numpy as np
import autograd.numpy as anp
import scipy as sp
from scipy import optimize
np.set_printoptions(suppress=True)
from tqdm import tqdm
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.options.display.float_format = '{:.3g}'.format
sns.set(font_scale = 1.5)
import warnings
warnings.filterwarnings("ignore")

from uncertain_expansion import uncertain_expansion
from elasticity import exposure_elasticity, price_elasticity
from lin_quad import LinQuadVar
from lin_quad_util import next_period

def eq_cond_BY(X_t, X_tp1, W_tp1, q, *args):
    # Parameters for the model
    γ, β, ρ, α, ϕ_e, σ_squared, ν_1, σ_w, μ, μ_d, ϕ, ϕ_d, ϕ_c, π, plug_in = args

    # Variables:
    vmc_t, rmc_t, pd_t, x_t, σ_t_squared = X_t.ravel()
    vmc_tp1, rmc_tp1, pd_tp1, x_tp1, σ_tp1_squared = X_tp1.ravel()
    w1_tp1, w2_tp1, w3_tp1, w4_tp1 = W_tp1.ravel()
    
    σ_t = anp.sqrt(σ_t_squared)
    gc_tp1 = μ + x_t + ϕ_c*σ_t*w3_tp1
    gd_tp1 = μ_d + ϕ*x_t + π*σ_t*w3_tp1 + ϕ_d*σ_t*w4_tp1
    
    m = vmc_tp1 + gc_tp1 - rmc_t

    util = (1-β) + β*anp.exp((1-ρ)*(rmc_t)) - anp.exp((1-ρ)*(vmc_t))
    
    res_1 = anp.exp(pd_t) - (anp.exp(anp.log(β) + (ρ-1)*(vmc_tp1+gc_tp1-rmc_t) - ρ*gc_tp1 + gd_tp1)*(anp.exp(pd_tp1) + 1))

    res_2 = x_tp1 - (α * x_t + ϕ_e * σ_t * w1_tp1)

    res_3 = σ_tp1_squared - (σ_squared + ν_1 * (σ_t_squared - σ_squared) + σ_w * w2_tp1)

    return anp.array([m, util, res_1, res_2, res_3])

def ss_func_BY(*args):
    # Extra parameters for the model
    γ, β, ρ, α, ϕ_e, σ_squared, ν_1, σ_w, μ, μ_d, ϕ, ϕ_d, ϕ_c, π, plug_in = args

    vmc = (np.log(1-β) - np.log(1-β*np.exp((1-ρ)*μ)))/(1-ρ)
    rmc = vmc + μ
    sdf = anp.exp(anp.log(β) - ρ*μ)
    X_0 = np.array([vmc, rmc, np.log((sdf * np.exp(μ_d))/(1-np.exp(μ_d)*sdf)), 0., σ_squared])
    return X_0

def gc_tp1_approx(X_t, X_tp1, W_tp1, q, *args):
    # Parameters for the model
    γ, β, ρ, α, ϕ_e, σ_squared, ν_1, σ_w, μ, μ_d, ϕ, ϕ_d, ϕ_c, π, plug_in = args

    # Variables:
    if plug_in:
        pd_t, x_t, σ_t_squared = X_t.ravel()
        pd_tp1, x_tp1, σ_tp1_squared = X_tp1.ravel()
        w1_tp1, w2_tp1, w3_tp1, w4_tp1 = W_tp1.ravel()
    else:
        vmc_t, rmc_t, pd_t, x_t, σ_t_squared = X_t.ravel()
        vmc_tp1, rmc_tp1, pd_tp1, x_tp1, σ_tp1_squared = X_tp1.ravel()
        w1_tp1, w2_tp1, w3_tp1, w4_tp1 = W_tp1.ravel()
    σ_t = anp.sqrt(σ_t_squared)
    
    gc_tp1 = μ + x_t + ϕ_c*σ_t*w3_tp1
    
    return gc_tp1


def calc_SDF(res):

    gc_tp1 = res['gc_tp1']  
    β = res['β']
    ρ = res['ρ']
    n_Y, n_X, n_W = res['var_shape']
    log_beta = LinQuadVar({'c':np.array([[np.log(β)]])},(1,n_X,n_W))
    Z1_tp1 = res['Z1_tp1']
    Z2_tp1 = res['Z2_tp1']
    vmc_tp1 = next_period(res['vmc_t'], Z1_tp1, Z2_tp1)
    rmc_t = res['rmc_t']
    log_N = res['log_N']
    log_SDF = log_beta +(ρ - 1)*(vmc_tp1 + gc_tp1 - rmc_t) - ρ*gc_tp1 + log_N

    return log_SDF, gc_tp1, Z1_tp1, Z2_tp1

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

    plug_in = False
    args = (γ, β, ρ, α, ϕ_e, σ_squared, ν_1, σ_w, μ, μ_d, ϕ, ϕ_d, ϕ_c, π, plug_in)

    ModelSol = uncertain_expansion(eq_cond_BY, ss_func_BY, (1, 2, 4), args, [gc_tp1_approx], adj_loc_list = [], endo_loc_list = [], init_N = 'N0', util_adj = False, tol = 1e-8, max_iter = 50)

    res = {'Z1_tp1':ModelSol['Z1_tp1'],\
            'Z2_tp1':ModelSol['Z2_tp1'],\
            'vmc_t':ModelSol['util_sol']['vmc_t'],\
            'rmc_t':ModelSol['util_sol']['rmc_t'],\
            'log_N':ModelSol['log_N'],\
            'β':ModelSol['args'][1],\
            'ρ':ModelSol['args'][2],\
            'var_shape':ModelSol['var_shape'],\
            'gc_tp1':ModelSol['scale_tp1'][0][0]}

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

    plug_in = False
    args = (γ, β, ρ, α, ϕ_e, σ_squared, ν_1, σ_w, μ, μ_d, ϕ, ϕ_d, ϕ_c, π, plug_in)

    ModelSol = uncertain_expansion(eq_cond_BY, ss_func_BY, (1, 2, 4), args, [gc_tp1_approx], adj_loc_list = [], endo_loc_list = [], init_N = 'N0', util_adj = False, tol = 1e-8, max_iter = 50)

    res = {'Z1_tp1':ModelSol['Z1_tp1'],\
            'Z2_tp1':ModelSol['Z2_tp1'],\
            'vmc_t':ModelSol['util_sol']['vmc_t'],\
            'rmc_t':ModelSol['util_sol']['rmc_t'],\
            'log_N':ModelSol['log_N'],\
            'β':ModelSol['args'][1],\
            'ρ':ModelSol['args'][2],\
            'var_shape':ModelSol['var_shape'],\
            'gc_tp1':ModelSol['scale_tp1'][0][0]}

    T = 360
    quantile = [0.25, 0.5, 0.75]

    log_SDF, gc_tp1, Z1_tp1, Z2_tp1 = calc_SDF(res)
    expo_elas_shock_0 = [exposure_elasticity(gc_tp1, Z1_tp1, Z2_tp1, T, shock=0, percentile=p) for p in quantile] 
    expo_elas_shock_1 = [exposure_elasticity(gc_tp1, Z1_tp1, Z2_tp1, T, shock=1, percentile=p) for p in quantile]
    expo_elas_shock_2 = [exposure_elasticity(gc_tp1, Z1_tp1, Z2_tp1, T, shock=2, percentile=p) for p in quantile]

    price_elas_shock_0 = [price_elasticity(gc_tp1, log_SDF, Z1_tp1, Z2_tp1, T, shock=0, percentile=p) for p in quantile]
    price_elas_shock_1 = [price_elasticity(gc_tp1, log_SDF, Z1_tp1, Z2_tp1, T, shock=1, percentile=p) for p in quantile]
    price_elas_shock_2 = [price_elasticity(gc_tp1, log_SDF, Z1_tp1, Z2_tp1, T, shock=2, percentile=p) for p in quantile]
    
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
    print(np.max(abs(np.array([expo_elas_shock_0,expo_elas_shock_2,expo_elas_shock_1]))))
    plot_price_elas = [plot_price_elas_shock_0, plot_price_elas_shock_2, plot_price_elas_shock_1] 
    print(np.max(abs(np.array([price_elas_shock_0,price_elas_shock_2,price_elas_shock_1]))))
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

    # return expo_elas_shock_0, expo_elas_shock_1, expo_elas_shock_2, price_elas_shock_0, price_elas_shock_1, price_elas_shock_2

# i=0
# for gamma in tqdm([5, 10, 20]):
#     for beta in tqdm([0.995, 0.998, 0.999]):
#         for rho in [2./3, 1.00001, 1.5, 10]:
#             for mu in [0.001, 0.0015, 0.002]:
#                 for phic in [0.007,0.0078,0.0086]:
#                     name = '_'.join(['res','gamma',str("{:.4f}".format(gamma)).replace('.',''),\
#                                         'beta',str("{:.4f}".format(beta)).replace('.',''),\
#                                         'rho',str("{:.4f}".format(rho)).replace('.',''),\
#                                         'mu',str("{:.4f}".format(mu)).replace('.',''),\
#                                         'phic',str("{:.4f}".format(phic)).replace('.','')])
#                     i+=1
#                     print(i,name)
#                     with open('data/'+name+'.pkl', 'wb') as f:
#                         pickle.dump([solve_BY_elas(γ=gamma, β=beta, ρ=rho, μ=mu, ϕ_c=phic)],f)
                    