from elasticity import exposure_elasticity, price_elasticity
from uncertain_expansion import get_parameter_value
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_exposure_elasticity(res, T, quantile, time_unit, ylimit1=None, ylimit2=None, ylimit3=None, save_path=None):
    """
    Calculate and plot exposure elasticity for different shocks.
    
    Parameters:
    - X1_tp1: Data array or DataFrame for X1.
    - X2_tp1: Data array or DataFrame for X2.
    - gc_tp1: Data array or DataFrame for consumption growth.
    - T: Integer, time horizon.
    - quantile: List of quantiles to calculate.
    - save_path: Optional, path to save the figure.
    """
    sns.set_style("darkgrid")

    X1_tp1 = res['X1_tp1']
    X2_tp1 = res['X2_tp1']
    gc_tp1 = res['gc_tp1']
    
    # Calculate exposure elasticity for different shocks
    expo_elas_shock_0 = [exposure_elasticity(gc_tp1, X1_tp1, X2_tp1, T, shock=0, percentile=p) for p in quantile]
    expo_elas_shock_1 = [exposure_elasticity(gc_tp1, X1_tp1, X2_tp1, T, shock=1, percentile=p) for p in quantile]
    expo_elas_shock_2 = [exposure_elasticity(gc_tp1, X1_tp1, X2_tp1, T, shock=2, percentile=p) for p in quantile]

    # Prepare data for plotting
    index = ['T', f'{quantile[0]} quantile', f'{quantile[1]} quantile', f'{quantile[2]} quantile']
    # index = ['T','0.1 quantile', '0.5 quantile', '0.9 quantile']
    shock_data = [
        pd.DataFrame([np.arange(T), *[e.flatten() for e in expo_elas_shock_1]], index=index).T,
        pd.DataFrame([np.arange(T), *[e.flatten() for e in expo_elas_shock_0]], index=index).T,
        pd.DataFrame([np.arange(T), *[e.flatten() for e in expo_elas_shock_2]], index=index).T
    ]
    shock_data = [
        pd.DataFrame([np.arange(T), *[e.flatten() for e in expo_elas_shock_1]], index=index).T,
        pd.DataFrame([np.arange(T), *[e.flatten() for e in expo_elas_shock_0]], index=index).T
        # pd.DataFrame([np.arange(T), *[e.flatten() for e in expo_elas_shock_2]], index=index).T
    ]
    
    # Plot settings
    # fig, axes = plt.subplots(1, 3, figsize=(25,8))
    fig, axes = plt.subplots(1, 2, figsize=(16,8))
    # shock_name = ['growth shock', 'consumption shock', 'volatility shock']
    shock_name = ['growth shock', 'consumption shock']
    qt = [f'{quantile[0]} quantile', f'{quantile[1]} quantile', f'{quantile[2]} quantile']
    colors = ['green', 'red', 'blue']
    
    for i in range(len(shock_data)):
        for j in range(len(quantile)):
            sns.lineplot(data=shock_data[i], x='T', y=qt[j], ax=axes[i], color=colors[j], label=qt[j])
        
        # Customize each subplot
        axes[i].set_xlabel('')  # Optional to clear x-axis label for each subplot
        axes[i].set_ylabel('Exposure elasticity', fontsize=16)
        axes[i].set_title('With respect to the ' + shock_name[i], fontsize=14)
        axes[i].tick_params(axis='y', labelsize=12)
        axes[i].tick_params(axis='x', labelsize=12)
        axes[i].legend(fontsize=12)
    
    # Set y-axis limits for each subplot
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    # axes[2].set_ylim(bottom=0)
    if ylimit1 is not None:
        axes[0].set_ylim([0, ylimit1])
    if ylimit2 is not None:
        axes[1].set_ylim([0, ylimit2])
    # if ylimit3 is not None:
    #     axes[2].set_ylim([0, ylimit3])
    # Set x-axis label for all subplots
    for ax in axes:
        ax.set_xlabel(f'{time_unit}', fontsize=18)

    plt.tight_layout()
    # Save and show plot
    if save_path:
        plt.savefig(save_path, dpi=500)
    plt.show()

def plot_price_elasticity(res, T, quantile, time_unit, ylimit1=None, ylimit2=None, ylimit3=None, save_path=None):
    """
    Calculate and plot price elasticity for different shocks.
    
    Parameters:
    - X1_tp1: Data array or DataFrame for X1.
    - X2_tp1: Data array or DataFrame for X2.
    - gc_tp1: Data array or DataFrame for consumption growth.
    - T: Integer, time horizon.
    - quantile: List of quantiles to calculate.
    - save_path: Optional, path to save the figure.
    """
    sns.set_style("darkgrid")
    args = res['args']
    parameter_names = res['parameter_names']
    γ = get_parameter_value('gamma', parameter_names, args)
    β = get_parameter_value('beta', parameter_names, args)
    ρ = get_parameter_value('rho', parameter_names, args)
    X1_tp1 = res['X1_tp1']
    X2_tp1 = res['X2_tp1']
    gc_tp1 = res['gc_tp1']
    vmr_tp1 = res['vmr1_tp1'] + 0.5 * res['vmr2_tp1']
    logNtilde = res['log_N_tilde']
    
    # Calculate log_SDF
    log_SDF = np.log(β) - ρ * gc_tp1 + (ρ - 1) * vmr_tp1 + logNtilde
    
    # Calculate exposure elasticity for different shocks
    price_elas_shock_0 = [price_elasticity(gc_tp1, log_SDF, X1_tp1, X2_tp1, T, shock=0, percentile=p) for p in quantile]
    price_elas_shock_1 = [price_elasticity(gc_tp1, log_SDF, X1_tp1, X2_tp1, T, shock=1, percentile=p) for p in quantile]
    price_elas_shock_2 = [price_elasticity(gc_tp1, log_SDF, X1_tp1, X2_tp1, T, shock=2, percentile=p) for p in quantile]

    # Prepare data for plotting
    index = ['T', f'{quantile[0]} quantile', f'{quantile[1]} quantile', f'{quantile[2]} quantile']
    # index = ['T','0.1 quantile', '0.5 quantile', '0.9 quantile']
    # shock_data = [
    #     pd.DataFrame([np.arange(T), *[e.flatten() for e in price_elas_shock_1]], index=index).T,
    #     pd.DataFrame([np.arange(T), *[e.flatten() for e in price_elas_shock_0]], index=index).T,
    #     pd.DataFrame([np.arange(T), *[-e.flatten() for e in price_elas_shock_2]], index=index).T
    # ]
    shock_data = [
        pd.DataFrame([np.arange(T), *[e.flatten() for e in price_elas_shock_1]], index=index).T,
        pd.DataFrame([np.arange(T), *[e.flatten() for e in price_elas_shock_0]], index=index).T
    ]
    
    # Plot settings
    # fig, axes = plt.subplots(1, 3, figsize=(25,8))
    fig, axes = plt.subplots(1, 2, figsize=(16,8))
    # shock_name = ['growth shock', 'consumption shock', 'volatility shock']
    shock_name = ['growth shock', 'consumption shock']
    qt = [f'{quantile[0]} quantile', f'{quantile[1]} quantile', f'{quantile[2]} quantile']
    colors = ['green', 'red', 'blue']
    
    for i in range(len(shock_data)):
        for j in range(len(quantile)):
            sns.lineplot(data=shock_data[i], x='T', y=qt[j], ax=axes[i], color=colors[j], label=qt[j])
        
        # Customize each subplot
        axes[i].set_xlabel('')  # Optional to clear x-axis label for each subplot
        axes[i].set_ylabel('Price elasticity', fontsize=16)
        axes[i].set_title('With respect to the ' + shock_name[i], fontsize=14)
        axes[i].tick_params(axis='y', labelsize=12)
        axes[i].tick_params(axis='x', labelsize=12)
        axes[i].legend(fontsize=12)
        
    # Set y-axis limits for each subplot
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    # axes[2].set_ylim(bottom=0)
    if ylimit1 is not None:
        axes[0].set_ylim([0, ylimit1])
    if ylimit2 is not None:
        axes[1].set_ylim([0, ylimit2])
    # if ylimit3 is not None:
    #     axes[2].set_ylim([0, ylimit3])
    # Set x-axis label for all subplots
    for ax in axes:
        ax.set_xlabel(f'{time_unit}', fontsize=18)

    plt.tight_layout()
    # Save and show plot
    if save_path:
        plt.savefig(save_path, dpi=500)
    plt.show()





def plot_price_elasticity(res, T, quantile, time_unit, ylimit1=None, ylimit2=None, ylimit3=None, save_path=None):
    """
    Calculate and plot price elasticity for different shocks.
    
    Parameters:
    - X1_tp1: Data array or DataFrame for X1.
    - X2_tp1: Data array or DataFrame for X2.
    - gc_tp1: Data array or DataFrame for consumption growth.
    - T: Integer, time horizon.
    - quantile: List of quantiles to calculate.
    - save_path: Optional, path to save the figure.
    """
    sns.set_style("darkgrid")
    args = res['args']
    parameter_names = res['parameter_names']
    γ = get_parameter_value('gamma', parameter_names, args)
    β = get_parameter_value('beta', parameter_names, args)
    ρ = get_parameter_value('rho', parameter_names, args)
    X1_tp1 = res['X1_tp1']
    X2_tp1 = res['X2_tp1']
    gc_tp1 = res['gc_tp1']
    vmr_tp1 = res['vmr1_tp1'] + 0.5 * res['vmr2_tp1']
    logNtilde = res['log_N_tilde']
    
    # Calculate log_SDF
    log_SDF = np.log(β) - ρ * gc_tp1 + (ρ - 1) * vmr_tp1 + logNtilde
    
    # Calculate exposure elasticity for different shocks
    price_elas_shock_0 = [price_elasticity(gc_tp1, log_SDF, X1_tp1, X2_tp1, T, shock=0, percentile=p) for p in quantile]
    price_elas_shock_1 = [price_elasticity(gc_tp1, log_SDF, X1_tp1, X2_tp1, T, shock=1, percentile=p) for p in quantile]
    price_elas_shock_2 = [price_elasticity(gc_tp1, log_SDF, X1_tp1, X2_tp1, T, shock=2, percentile=p) for p in quantile]

    # Prepare data for plotting
    index = ['T', f'{quantile[0]} quantile', f'{quantile[1]} quantile', f'{quantile[2]} quantile']
    # index = ['T','0.1 quantile', '0.5 quantile', '0.9 quantile']
    # shock_data = [
    #     pd.DataFrame([np.arange(T), *[e.flatten() for e in price_elas_shock_1]], index=index).T,
    #     pd.DataFrame([np.arange(T), *[e.flatten() for e in price_elas_shock_0]], index=index).T,
    #     pd.DataFrame([np.arange(T), *[-e.flatten() for e in price_elas_shock_2]], index=index).T
    # ]
    shock_data = [
        pd.DataFrame([np.arange(T), *[e.flatten() for e in price_elas_shock_1]], index=index).T,
        pd.DataFrame([np.arange(T), *[e.flatten() for e in price_elas_shock_0]], index=index).T
    ]
    
    # Plot settings
    # fig, axes = plt.subplots(1, 3, figsize=(25,8))
    fig, axes = plt.subplots(1, 2, figsize=(16,8))
    # shock_name = ['growth shock', 'consumption shock', 'volatility shock']
    shock_name = ['growth shock', 'consumption shock']
    qt = [f'{quantile[0]} quantile', f'{quantile[1]} quantile', f'{quantile[2]} quantile']
    colors = ['green', 'red', 'blue']
    
    for i in range(len(shock_data)):
        for j in range(len(quantile)):
            sns.lineplot(data=shock_data[i], x='T', y=qt[j], ax=axes[i], color=colors[j], label=qt[j])
        
        # Customize each subplot
        axes[i].set_xlabel('')  # Optional to clear x-axis label for each subplot
        axes[i].set_ylabel('Price elasticity', fontsize=16)
        axes[i].set_title('With respect to the ' + shock_name[i], fontsize=14)
        axes[i].tick_params(axis='y', labelsize=12)
        axes[i].tick_params(axis='x', labelsize=12)
        axes[i].legend(fontsize=12)
        
    # Set y-axis limits for each subplot
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    # axes[2].set_ylim(bottom=0)
    if ylimit1 is not None:
        axes[0].set_ylim([0, ylimit1])
    if ylimit2 is not None:
        axes[1].set_ylim([0, ylimit2])
    # if ylimit3 is not None:
    #     axes[2].set_ylim([0, ylimit3])
    # Set x-axis label for all subplots
    for ax in axes:
        ax.set_xlabel(f'{time_unit}', fontsize=18)

    plt.tight_layout()
    # Save and show plot
    if save_path:
        plt.savefig(save_path, dpi=500)
    plt.show()