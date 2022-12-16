"""
Defines a linear-quadratic variable structure to facilitate operations in
    expansion solvers and elasticity calculation.

"""
import copy
import numpy as np
from utilities import mat, vec
from numba import njit
from collections import OrderedDict

valid_keys = ('x2', 'xx', 'xw', 'ww', 'x', 'w', 'c')


class LinQuadVar:
    r"""
    Class facilitating operations on a linear-quadratic variable structure:

    .. math::

        Y = \Gamma_0 + \Gamma_1 X_{1,t} + \Gamma_2 X_{2,t}
            + \Gamma_3 X_{1,t}\otimes X_{1,t}
            + \Psi_0 W_{t+1} + \Psi_1 X_{1,t}\otimes W_{t+1}
            + \Psi_2 W_{t+1}\otimes W_{t+1}

    Parameters
    ----------
    Y_coeffs : dict
        A dictionary containing the following ndarrays:

        'c' : (n_Y, 1) ndarray
            the constant.
        'x' : (n_Y, n_X) ndarray
            the coefficient of :math:`X_{1, t}`.
        'w' : (n_Y, n_W) ndarray
            the coefficient of :math:`W_{t+1}`.
        'x2' : (n_Y, n_X) ndarray
            the coefficient of :math:`X_{2, t}`.
        'xx' : (n_Y, n_X**2) ndarray
            the coefficient of :math:`X_{1, t} \otimes X_{1, t}`.
        'xw' : (n_Y, n_X*n_W) ndarray
            the coefficient of :math:`X_{1, t} \otimes W_{t+1}`.
        'ww' : (n_Y, n_W**2) ndarray
            the coefficient of :math:`W_{t+1} \otimes W_{t+1}`.
    shape : tuple of ints
        (n_Y, n_X, n_W) are dimensions of Y, X and W respectively.
    copy_ : bool
        If True, the class will make a copy of Y_coeffs to store the coefficients.

    Attributes
    ----------
    shape : :obj:`tuple of ints`
        (n_Y, n_X, n_W) are dimensions of Y, X and W respectively.
    second_order : :obj:`bool`
        True if Y has non-zero second-order terms, including :math:`X_{2, t}`, :math:`X_{1, t} \otimes X_{1, t}`, :math:`X_{1, t} \otimes W_{t+1}` and :math:`W_{t+1} \otimes W_{t+1}`.
    deterministic : :obj:`bool`
        True if Y is deterministic conditioned on time t information. i.e. :math:`Y = \Gamma_0 + \Gamma_1 X_{1,t} + \Gamma_2 X_{2,t} + \Gamma_3 X_{1,t}\otimes X_{1,t}`

    References
    ---------
    Borovicka, Hansen (2014). See http://larspeterhansen.org/.

    """
    def __init__(self, Y_coeffs, shape, copy_=True):
        if copy_:
            self.coeffs = copy.deepcopy(Y_coeffs)
        else:
            self.coeffs = Y_coeffs
        self.shape = shape
        self.__check_shapes()
        self.__remove_zero_coeffs()
        self.__check_deterministic()
        self.__check_second_order()

    def __check_shapes(self):
        """
        Check if the shapes of coefficients in Y_coeffs are self-consistent.

        """
        terms = self.coeffs.keys()
        for key in terms:
            key_shape = _get_key_shape(key, self.shape)
            if self.coeffs[key].shape != key_shape:
                raise ValueError('Invalid shape:', key)

    def __remove_zero_coeffs(self):
        """
        Remove coefficients that are zero ndarrays to save memory.

        """
        terms = list(self.coeffs.keys())
        for key in terms:
            # if not np.any(self.coeffs[key] != 0)
            if not self.coeffs[key].any():
                del self.coeffs[key]

    def __check_deterministic(self):
        """
        Set self.deterministic = True if there is no shock term.

        """
        self.deterministic = True
        for key in ['w', 'xw', 'ww']:
            # if key in self.coeffs and np.any(self.coeffs[key] != 0)
            if key in self.coeffs and self.coeffs[key].any():
                self.deterministic = False
                break

    def __check_second_order(self):
        """
        Set self.second_order = True if there is no second-order term.

        """
        self.second_order = False
        if self.deterministic:
            terms = ['x2', 'xx']
        else:
            terms = ['x2', 'xx', 'xw', 'ww']
        for key in terms:
            # if key in self.coeffs and np.any(self.coeffs[key] != 0)
            if key in self.coeffs and self.coeffs[key].any():
                self.second_order = True
                break

    def __getitem__(self, key):
        """
        Get coefficient of Y or slices of Y.
        
        """
        if isinstance(key, int):
            key = slice(key, key+1, None)
        if isinstance(key, slice):
            Y_shape = (key.stop-key.start, self.shape[1], self.shape[2])
            Y_coeffs = {}
            for term in self.coeffs:
                Y_coeffs[term] = self.coeffs[term][key]
            return LinQuadVar(Y_coeffs, Y_shape, False)
        elif isinstance(key, str) and key in valid_keys:
            if key in self.coeffs:
                return self.coeffs[key]
            else:
                key_shape = _get_key_shape(key, self.shape)
                return np.zeros(key_shape)
        else:
            raise ValueError('Invalid input:', key)

    def __add__(self, Y2):
        """
        Computes Y + Y2. Y2 could be LinQuadVar or scalar.

        """
        if isinstance(Y2, LinQuadVar):
            Y_coeffs = {}
            terms = set(self.coeffs.keys()) | set(Y2.coeffs.keys())
            for key in terms:
                Y_coeffs[key] = self[key] + Y2[key]
            return LinQuadVar(Y_coeffs, self.shape, False)
        elif isinstance(Y2, (int, float)):
            coeffs = copy.deepcopy(self.coeffs)
            coeffs['c'] = self['c'] + Y2
            new_Y = LinQuadVar(coeffs, self.shape, False)
            return new_Y
        else:
            raise TypeError('Invalid type.')

    def __radd__(self, Y2):
        return self.__add__(Y2)

    def __sub__(self, Y2):
        return self.__add__(Y2*(-1.))

    def __rsub__(self, Y2):
        return self.__mul__(-1.) + Y2

    def __mul__(self, multiplier):
        """
        Computes Y * multiplier. Multiplier is a scalar.

        """
        Y_coeffs = {}
        for key in self.coeffs:
            Y_coeffs[key] = self.coeffs[key] * multiplier
        return LinQuadVar(Y_coeffs, self.shape, False)

    def __rmul__(self, multiplier):
        return self.__mul__(multiplier)

    def __truediv__(self, denominator):
        """
        Computes Y / denominator. Denominator is a scalar.

        """
        return self.__mul__(1./denominator)

    def split(self):
        """
        Splits the N-dimensional Y into N 1-dimensional Ys.

        Returns
        -------
        res : list of LinQuadVar
        
        See Also
        --------
        concat : Concatenates a list of LinQuadVar.

        """
        n_Y, n_X, n_W = self.shape
        res = []
        for i in range(n_Y):
            Y_coeffs = {}
            for key in self.coeffs:
                Y_coeffs[key] = self.coeffs[key][i:i+1, :]
            Y = LinQuadVar(Y_coeffs, (1, n_X, n_W), False)
            res.append(Y)
        return res

    def drop_scale(self, loc, loc2 = 100):
        n_Y, n_Z, n_W = self.shape

        lq_new = LinQuadVar({'c':self['c'],\
                'w':self['w'],\
                'x':np.delete(self['x'],loc-1).reshape(1,-1),\
                'x2':np.delete(self['x2'],loc-1).reshape(1,-1),\
                'xx':np.delete(np.delete(self['xx'], np.arange(n_Z*(loc-1),n_Z*loc)), np.arange(loc-1,n_Z*(n_Z-1),n_Z)).reshape(1,-1),\
                'xw':np.delete(self['xw'],np.arange(n_W*(loc-1),n_W*loc)).reshape(1,-1),\
                'ww':self['ww']},(1,n_Z-1,n_W))
        
        if loc2<100:      
            n_Z = n_Z - 1             
            loc = loc2 - 1
            lq_new2 = LinQuadVar({'c':lq_new['c'],\
                        'w':lq_new['w'],\
                        'x':np.delete(lq_new['x'],loc-1).reshape(1,-1),\
                        'x2':np.delete(lq_new['x2'],loc-1).reshape(1,-1),\
                        'xx':np.delete(np.delete(lq_new['xx'], np.arange(n_Z*(loc-1),n_Z*loc)), np.arange(loc-1,n_Z*(n_Z-1),n_Z)).reshape(1,-1),\
                        'xw':np.delete(lq_new['xw'],np.arange(n_W*(loc-1),n_W*loc)).reshape(1,-1),\
                        'ww':lq_new['ww']},(1,n_Z-1,n_W))
            return lq_new2
        else:
            return lq_new

    def __call__(self, X1, X2=None, W=None, series = False):
        r"""
        Evaluates Y at three vectors :math:`X_{1, t}`, :math:`X_{2, t}`
        and :math:`W_{t+1}`.

        Parameters
        ----------
        X1 : (n, 1) ndarrays
            Value of :math:`X_{1, t}` at time t.
        X2 : (n, 1) ndarrays or None
            Value of :math:`X_{2, t}` at time t. Does not need to be specified
            when the 'x2' term of Y is zero.
        W : (m, 1) ndarrays or None
            Value of :math:`W_{t+1}` at time t+1. Does not need to be specified
            when the 'w', 'xw' and 'ww' terms of Y are all zeros.

        Returns
        -------
        res : (n_Y, 1) ndarray

        """
        n_Y, n_X, n_W = self.shape
        if X2 is None:
            X2 = np.zeros(n_X)
        if W is None:
            W = np.zeros(n_W)
        if series:
            res = _call_jit_series(self['x2'], self['x'], self['w'], self['c'],
                  self['xx'], self['xw'], self['ww'], X1, X2, W,
                  self.second_order, self.deterministic)
        else:
            res = _call_jit(self['x2'], self['x'], self['w'], self['c'],
                    self['xx'], self['xw'], self['ww'], X1, X2, W,
                    self.second_order, self.deterministic)
        return res


def _get_key_shape(key, shape):
    """
    Get the shape of Y[key] given the shape of LinQuadVar.
    
    Parameters
    ----------
    key : str
        'x2', 'x', 'w', 'c', 'xx', 'xw' or 'ww'
    shape : tuple of ints
        (n_Y, n_X, n_W) are dimensions of Y, X and W respectively.
    
    Returns
    -------
    key_shape : typle of ints
        (n_Y, n_key) are dimensions of Y[key]
    
    """
    n_Y, n_X, n_W = shape
    if key == 'x2' or key == 'x':
        key_shape = (n_Y, n_X)
    elif key == 'xx':
        key_shape = (n_Y, n_X**2)
    elif key == 'xw':
        key_shape = (n_Y, n_X*n_W)
    elif key == 'ww':
        key_shape = (n_Y, n_W**2)
    elif key == 'w':
        key_shape = (n_Y, n_W)
    elif key == 'c':
        key_shape = (n_Y, 1)
    else:
        raise ValueError('Invalid key:', key)
    return key_shape


@njit
def _call_jit(x2, x, w, c, xx, xw, ww, X1, X2, W, second_order=False, deterministic=False):
    res = x@X1 + c
    if second_order:
        res += xx@np.kron(X1, X1)
        # if np.any(x2 != 0)
        if x2.any():
            res += x2@X2
    if not deterministic:
        res += w@W
        if second_order:
            res += xw@np.kron(X1, W)
            res += ww@np.kron(W, W)
    return res

@njit
def _call_jit_series(x2, x, w, c, xx, xw, ww, X1_series, X2_series, W_series, second_order=False, deterministic=False):
    res = x@X1_series + c

    for t in range(res.shape[1]):
        X1 = X1_series[:,t:t+1].copy()
        # res = x@X1 + c
        if second_order:
            res[:,t:t+1] += xx@np.kron(X1, X1)
            # if np.any(x2 != 0)
            if x2.any():
                X2 = X2_series[:,t:t+1].copy()
                res[:,t:t+1] += x2@X2
        if not deterministic:
            W = W_series[:,t+1:t+2].copy()
            res += w@W
            if second_order:
                res[:,t:t+1] += xw@np.kron(X1, W)
                res[:,t:t+1] += ww@np.kron(W, W)
    return res