"""
Function for derivative computation.

"""
import numpy as np
from autograd import jacobian
import warnings


# def take_derivatives(eq, ss, var_shape, second_order, eq_cond,
    #                   args, norecursive=False):
    # """
    # Take first- or second-order derivatives.

    # """
    # n_J, n_X, n_W = var_shape
    # n_JX = n_J + n_X
    # W_0 = np.zeros(n_W)
    # q_0 = 0.

    # if norecursive==False:  
    #     if callable(ss):
    #         ss = ss(*args)

        
    # else:
    #     ss = ss(*args)[3:]
    #     print(ss)

    # dfq = compute_derivatives(f = lambda Var_t, Var_tp1, W_tp1,  q:
    #                         anp.atleast_1d(eq(Var_t, Var_tp1, W_tp1, q,  eq_cond, *args)),
    #                          X=[ss, ss, W_0, q_0],
    #                          second_order=second_order)

    # if norecursive==False:            
    #     df = {'xt':dfq['xt'][:,1:],\
    #         'xtp1':dfq['xtp1'][:,1:],\
    #         'wtp1':dfq['wtp1'],\
    #         'q':dfq['q'],\
    #         'xtxt':dfq['xtxt'][:,n_JX+1:][:,np.tile(np.concatenate([np.array([False]),np.repeat(True, n_JX)]),n_JX)],\
    #         'xtxtp1':dfq['xtxtp1'][:,n_JX+1:][:,np.tile(np.concatenate([np.array([False]),np.repeat(True, n_JX)]),n_JX)],\
    #         'xtwtp1':dfq['xtwtp1'][:,n_W:],\
    #         'xtq':dfq['xtq'][:,1:],\
    #         'xtp1xtp1':dfq['xtp1xtp1'][:,n_JX+1:][:,np.tile(np.concatenate([np.array([False]),np.repeat(True, n_JX)]),n_JX)],\
    #         'xtp1wtp1':dfq['xtp1wtp1'][:,n_W:],\
    #         'xtp1q':dfq['xtp1q'][:,1:],\
    #         'wtp1wtp1':dfq['wtp1wtp1'],\
    #         'wtp1q':dfq['wtp1q'],\
    #         'qq':dfq['qq']}      
    
    # else:
    #     df = {'xt':dfq['xt'][:,:],\
    #         'xtp1':dfq['xtp1'][:,:],\
    #         'wtp1':dfq['wtp1'],\
    #         'q':dfq['q'],\
    #         'xtxt':dfq['xtxt'][:,   :],\
    #         'xtxtp1':dfq['xtxtp1'][:,:],\
    #         'xtwtp1':dfq['xtwtp1'][:,:],\
    #         'xtq':dfq['xtq'][:,:],\
    #         'xtp1xtp1':dfq['xtp1xtp1'][:,:],\
    #         'xtp1wtp1':dfq['xtp1wtp1'][:,:],\
    #         'xtp1q':dfq['xtp1q'][:,:],\
    #         'wtp1wtp1':dfq['wtp1wtp1'],\
    #         'wtp1q':dfq['wtp1q'],\
    #         'qq':dfq['qq']}   
        
                            
    # ss = ss[1:]            
    # return df, ss


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

def compute_derivatives(f, X, second_order=True):
    r"""
    Compute second-order and/or first-order partial derivatives.

    Parameters
    ----------
    f : Callable
       It takes :math:`[X_t, X_{t+1}, W_{t+1}, q]` as input and returns multi-dimensional ndarray;
       The equations/formulas defined in the function should use autograd.numpy.
    X : list of ndarrays
       Evaluated point, e.g. :math:`[X_0, X_0, W_0, q_0]`.
    second_order : bool
       If False, it computes first-order derivatives only;
       If True, it computes first- and second-order derivatives.

    Returns
    -------
    derivatives: dict
        A dictionary containing the following ndarrays:
        
            'f_xt' : (n_f, n_x) ndarray
                First order partial derivative wrt :math:`X_t`.
            'f_xtp1' : (n_f, n_x) ndarray
                First order partial derivative wrt :math:`X_{t+1}`.
            'f_wtp1' : (n_f, n_w) ndarray
                First order partial derivative wrt :math:`W_{t+1}`.
            'f_q' : (n_f, 1) ndarray
                First order partial derivative wrt :math:`q`.
            'f_xtxt' : (n_f, n_x**2) ndarray
                Second order partial derivative wrt :math:`X_t` and :math:`X_t`.
            'f_xtxtp1' : (n_f, n_x**2) ndarray
                Second order partial derivative wrt :math:`X_t` and :math:`X_{t+1}`.
            'f_xtwtp1' : (n_f, n_x*n_w) ndarray
                Second order partial derivative wrt :math:`X_t` and :math:`W_{t+1}`.
            'f_xtq' : (n_f, n_x) ndarray
                Second order partial derivative wrt :math:`X_t` and :math:`q`.
            'f_xtp1xtp1' : (n_f, n_x**2) ndarray
                Second order partial derivative wrt :math:`X_{t+1}` and :math:`X_{t+1}`.
            'f_xtp1wtp1' : (n_f, n_x*n_w) ndarray
                Second order partial derivative wrt :math:`X_{t+1}` and :math:`W_{t+1}`.
            'f_xtp1q' : (n_f, n_x) ndarray
                Second order partial derivative wrt :math:`X_{t+1}` and :math:`q`.
            'f_wtp1wtp1' : (n_f, n_w**2) ndarray
                Second order partial derivative wrt :math:`W_{t+1}` and :math:`W_{t+1}`.
            'f_wtp1q' : (n_f, n_w) ndarray
                Second order partial derivative wrt :math:`W_{t+1}` and :math:`q`.
            'f_qq' : (n_f, 1) ndarray
                Second order partial derivative wrt :math:`q` and :math:`q`.

    """
    if any(np.issubdtype(x.dtype, np.integer) for x in X[:-1]):
        X = [x.astype(np.float) for x in X[:-1]] + [X[-1]]
        warnings.warn("Casting integer inputs to float in order to handle differentiation.")
    X_0, X_1, W_0, q_0 = X

    # 1st-order derivatives
    df_xt = jacobian(f, argnum=0)
    df_xtp1 = jacobian(f, argnum=1)
    df_wtp1 = jacobian(f, argnum=2)
    df_q = jacobian(f, argnum=3)
    # Evaluate derivatives at X_0,X_1,W_0,q_0
    f_xt = np.array(df_xt(X_0, X_1, W_0, q_0))
    f_xtp1 = np.array(df_xtp1(X_0, X_1, W_0, q_0))
    f_wtp1 = np.array(df_wtp1(X_0, X_1, W_0, q_0))
    f_q = np.array(df_q(X_0, X_1, W_0, q_0))
    f_dim = f_xt.shape[0]

    derivatives = {'xt': f_xt,
                   'xtp1': f_xtp1,
                   'wtp1': f_wtp1,
                   'q': f_q}

    # 2nd-order derivatives
    if second_order:
        # 2nd-order derivatives
        df_xtxt = jacobian(df_xt, argnum=0)
        df_xtxtp1 = jacobian(df_xt, argnum=1)
        df_xtwtp1 = jacobian(df_xt, argnum=2)
        df_xtq = jacobian(df_xt, argnum=3)

        df_xtp1xtp1 = jacobian(df_xtp1, argnum=1)
        df_xtp1wtp1 = jacobian(df_xtp1, argnum=2)
        df_xtp1q = jacobian(df_xtp1, argnum=3)

        df_wtp1wtp1 = jacobian(df_wtp1, argnum=2)
        df_wtp1q = jacobian(df_wtp1, argnum=3)

        df_qq = jacobian(df_q, argnum=3)

        # Evaluate derivatives at X_0,X_1,W_0,q_0
        f_xtxt = df_xtxt(X_0, X_1, W_0, q_0).reshape((f_dim, -1))
        f_xtxtp1 = df_xtxtp1(X_0, X_1, W_0, q_0).reshape((f_dim, -1))
        f_xtwtp1 = df_xtwtp1(X_0, X_1, W_0, q_0).reshape((f_dim, -1))
        f_xtq = df_xtq(X_0, X_1, W_0, q_0).reshape((f_dim, -1))

        f_xtp1xtp1 = df_xtp1xtp1(X_0, X_1, W_0, q_0).reshape((f_dim, -1))
        f_xtp1wtp1 = df_xtp1wtp1(X_0, X_1, W_0, q_0).reshape((f_dim, -1))
        f_xtp1q = df_xtp1q(X_0, X_1, W_0, q_0).reshape((f_dim, -1))

        f_wtp1wtp1 = df_wtp1wtp1(X_0, X_1, W_0, q_0).reshape((f_dim, -1))
        f_wtp1q = df_wtp1q(X_0, X_1, W_0, q_0).reshape((f_dim, -1))

        f_qq = df_qq(X_0, X_1, W_0, q_0).reshape((f_dim, -1))

        second_order_derivatives = {'xtxt': f_xtxt,
                                    'xtxtp1': f_xtxtp1,
                                    'xtwtp1': f_xtwtp1,
                                    'xtq': f_xtq,
                                    'xtp1xtp1': f_xtp1xtp1,
                                    'xtp1wtp1': f_xtp1wtp1,
                                    'xtp1q': f_xtp1q,
                                    'wtp1wtp1': f_wtp1wtp1,
                                    'wtp1q': f_wtp1q,
                                    'qq': f_qq}

        derivatives = {**derivatives, **second_order_derivatives}

    return derivatives