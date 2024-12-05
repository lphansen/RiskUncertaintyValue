"""
Function for derivative computation.

"""
import numpy as np
import sympy as sp
from autograd import jacobian
import warnings
from uncertain_expansion import split_variables


def combine_derivatives(df_list):
    """
    Combines df1, df2, ... into df by row.

    Parameters
    ----------
    df_list : sequence of dict
        Each dict is of (n_fi, ?) ndarrays, which are partial derivatives of fi.
        These are returned values from compute_derivatives().

    Returns
    -------
    df_list : dict of ndarrays

    """
    df = {}
    for key in df_list[0].keys():
        df[key] = np.vstack([dfi[key] for dfi in df_list])
    return df

# Number of variables preceding states
def compute_derivatives(f,variables,variables_tp1,parameter_names,X,args,var_shape,recursive_ss,second_order=True):
    if not isinstance(f, list):
        f = [f]

    # Variables
    parameter_names_temp = list(parameter_names.keys())

    recursive_variables, main_variables, q_variable, shock_variables = split_variables(variables,var_shape)
    recursive_variables_tp1, main_variables_tp1, q_variable_tp1, shock_variables_tp1 = split_variables(variables_tp1,var_shape)


    X_0, X_1, W_0, q_0 = X
    X_t, X_tp1, W_tp1, q = main_variables, main_variables_tp1, shock_variables_tp1, q_variable
    f_substituted = [eq.subs({param: val for param, val in zip(parameter_names, args)}) for eq in f]
    f_substituted = [eq.subs({recur: val for recur, val in zip(recursive_variables, recursive_ss)}) for eq in f_substituted]
        

    # Compute first-order derivatives
    first_order_derivatives = {
        'xt': [sp.Matrix([sp.diff(eq, var) for var in X_t]) for eq in f_substituted],
        'xtp1': [sp.Matrix([sp.diff(eq, var) for var in X_tp1]) for eq in f_substituted],
        'wtp1': [sp.Matrix([sp.diff(eq, var) for var in W_tp1]) for eq in f_substituted],
        'q': [sp.Matrix([sp.diff(eq, q)]) for eq in f_substituted]
    }

    # Evaluate first-order derivatives
    derivatives = {}
    for key, derivs in first_order_derivatives.items():
        evaluated = [
            d.subs({
                **dict(zip(X_t, X_0)),
                **dict(zip(X_tp1, X_1)),
                **dict(zip(W_tp1, W_0)),
                q: q_0
            }).evalf() for d in derivs
        ]
        reshaped = np.array([np.array(d.tolist(), dtype=float) for d in evaluated])
        derivatives[f"{key}"] = reshaped.reshape(reshaped.shape[0], -1)

    # Second-order derivatives evaluation if enabled
    if second_order:
        second_order_derivatives = {
                'xtxt': [sp.hessian(eq, X_t) for eq in f_substituted],
                'xtxtp1': [sp.Matrix([[sp.diff(eq, v1, v2) for v2 in X_tp1] for v1 in X_t]) for eq in f_substituted],
                'xtwtp1': [sp.Matrix([[sp.diff(eq, v1, v2) for v2 in W_tp1] for v1 in X_t]) for eq in f_substituted],
                'xtq': [sp.Matrix([sp.diff(eq, v1, q) for v1 in X_t]) for eq in f_substituted],
                'xtp1xtp1': [sp.hessian(eq, X_tp1) for eq in f_substituted],
                'xtp1wtp1': [sp.Matrix([[sp.diff(eq, v1, v2) for v2 in W_tp1] for v1 in X_tp1]) for eq in f_substituted],
                'xtp1q': [sp.Matrix([sp.diff(eq, v1, q) for v1 in X_tp1]) for eq in f_substituted],
                'wtp1wtp1': [sp.hessian(eq, W_tp1) for eq in f_substituted],
                'wtp1q': [sp.Matrix([sp.diff(eq, v1, q) for v1 in W_tp1]) for eq in f_substituted],
                'qq': [sp.diff(eq, q, q) for eq in f_substituted],
            }

        for key, derivs in second_order_derivatives.items():
            evaluated = [
                d.subs(
                    {
                        **dict(zip(X_t, X_0)),
                        **dict(zip(X_tp1, X_1)),
                        **dict(zip(W_tp1, W_0)),
                        q: q_0,
                    }
                ).evalf()
                for d in derivs
            ]
            reshaped = np.array([np.array(d.tolist(), dtype=float) if hasattr(d, 'tolist') else float(d) for d in evaluated])
            derivatives[f"{key}"] = reshaped.reshape(reshaped.shape[0], -1)   
    return derivatives

