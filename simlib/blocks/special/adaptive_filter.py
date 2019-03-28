"""adaptive_filter.py

A collection of components for adaptive filters.
"""


import copy
import numpy as np
from numpy.linalg import inv
from numpy.linalg import LinAlgError
from scipy.fftpack import dct

from ...misc import as_uint
from ...simsys import BaseBlock, NA

__all__ = ['FIRLMS', 'FIRNLMS', 'FIRAPLMS', 'FIRTDLMS', 'IIRLMS', 'FIRRLS']


class FIRLMS(BaseBlock):

    """SISO FIR Wiener filter using LMS method.

    The LMS algrithm use the following tap-weight vector recursion:

        .. math:: \mathbf w(n+1)=\mathbf w(n)+2\mu e(n)\mathbf x(n)
    where **w** is the tap-weight vector, *\mu* is the step-size parameter, *e*
    is the error signal which is the difference between the desired output and
    the filter output, and **x** is the input signal vector.

    The block has two input signals. The first input signal is X and the second
    input is error. X is the input to the plant and error is the defference
    between desired response and the filter.

    The block has two outputs. The first output is the estimate output of the
    plant and the second output is the weight array.

    Parameters
    ----------
    length: int
        Length of FIR filter.

    mu: scalar
        Step-size paremeter.

    initial_weght: scalar or iterable optional
        Initial weights of the filter. If a scalar is given, all initial
        weights are set to this float number. If an iterable object is given,
        its length should be equal to the length of the filter.

    initial_input: scala or iterable, optional
        Initial inputs to the filter. If a scalar is given, all initial inputs
        are set to this float number. If an iterable object is given, its
        length should not be shorter than the length of the filter.

    dtype: data-type, optional
        The desired data-type for the filter. If not given, the type will be
        float.

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'LMS Filter')

    Ports
    -----
    In[0]: Input
        The input signal

    In[1]: Error
        The error signal (desired output - filter output)

    Out[0]: Output
        The output signal

    Out[1]: Weight
        The weight coefficients of the filter. The weight of the latest input
        signal is on the left.

    See Also
    ----------
    FIRLMSMIMO
    """

    def __init__(self, length=50, mu=0.01,  initial_weight=None,
                 initial_input=None, dtype=None, dt=None, name='LMS Filter'):
        super().__init__(nin=2, nout=2, dt=dt, name=name)
        self.inports[0].rename('Input')
        self.inports[1].rename('Error')
        self.outports[0].rename('Output')
        self.outports[1].rename('Weight')

        # init for single-in-single-out system
        length = as_uint(length, msg='length must be positive int')

        if np.isscalar(mu):
            self._mu = mu
        else:
            raise ValueError('mu must be scalar')

        # initial weight
        if initial_weight is None:
            initial_weight = 0.0
        self._weights_init = np.zeros(length, dtype=dtype)
        try:
            self._weights_init[:] = initial_weight
        except ValueError:
            raise ValueError('shape of initial_weight is not correct')

        # initial input
        if initial_input is None:
            initial_input = 0.0
        self._X_init = np.zeros(length, dtype=dtype)
        initial_input = np.asarray(initial_input)
        if np.iterable(initial_input):
            try:
                self._X_init[:] = initial_input[-length:]
            except (ValueError, IndexError,):
                raise ValueError('shape of initial_input not correct')
        else:
            self._X_init[:] = initial_input
        self._X_init = self._X_init[::-1]

    def INITFUNC(self):
        self._X = copy.copy(self._X_init)
        self._weights = copy.copy(self._weights_init)

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value

    @property
    def weights(self):
        return self._weights

    def OUTPUTSTEP(self, portid):
        if portid == 0:
            xn = self.inports[0].step()
            # update the input signal series
            self._X[1:] = self._X[:-1]
            self._X[0] = xn
            # output signal
            res = np.dot(self._weights, self._X)
            return res
        elif portid == 1:
            return self._weights

    def BLOCKSTEP(self, *xs):
        xn, error = xs
        # update weights (calculate weights for next iteration)
        self._weights = (self._weights +
                         2 * self.mu.conjugate() * error * self._X)
        return NA, NA


class FIRNLMS(BaseBlock):

    """FIR Wiener filter using Normalized LMS method.

    The NLMS algrithm use the following tap-weight vector recursion:

        .. math:: \mathbf w(n+1)=\mathbf w(n)+\mu(n) e(n)\mathbf x(n)
    and

        .. math:: \mu(n)=\\frac{\\tilde \mu}{\mathbf x^T(n)\mathbf x(n) + \psi}
    where **w** is the tap-weight vector, *e* is the error signal which is the
    difference between the desired output and the filter output, and **x** is
    the input signal vector. $\\tilde \mu$ and $\psi$ are positive constants to
    imporve the reliablity of the algrithm.

    The block has tow input signals. The first input signal is X and the second
    input is error. X is the input to the plant and error is the defference
    between desired response and the filter.

    The block has two outputs. The first output is the estimate output of the
    plant and the second output is the weight array.


    The block has tow input signals. The first input signal is X and the second
    input is error. X is the input to the plant and error is the defference
    between desired response and the filter.

    The block has two outputs. The first output is the estimate output of the
    plant and the second output is the weight array.

    Parameters
    ----------
    length: int
        length of FIR filter.

    mu: float
        Step-size paremeter. Used to control the convergence and the
        misadjestment, should be positive. (default=1)

    psi: float
        Used to prevent division by a small value when norm(x) is small.

    initial_weght: float or iterable, optional
        Initial weights of the filter. If a float is given, all initial weights
        are set to this float number. If an iterable object is given, its
        length should be equal to the length of the filter.

    initial_input: float or iterable, optional
        Initial inputs to the filter. If a float is given, all initial inputs
        are set to this float number. If an iterable object is given, its
        length should be equal to the length of the filter.

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'NLMS Filter')

    Ports
    -----
    In[0]: Input
        The input signal

    In[1]: Error
        The error signal (desired output - filter output)

    Out[0]: Output
        The output signal

    Out[1]: Weight
        The weight coefficients of the filter. The weight of the latest input
        signal is on the left.

    See Also
    ----------
    FIRLMS

    Notes
    -----
    NLMS algrithm for Complex value signals is not supported yet.
    """

    def __init__(self, length=50, mu=1, psi=0, initial_weight=None,
                 initial_input=None, dt=None, name='NLMS Filter'):
        super().__init__(nin=2, nout=2, dt=dt, name=name)
        self.inports[0].rename('Input')
        self.inports[1].rename('Error')
        self.outports[0].rename('Output')
        self.outports[1].rename('Weight')

        length = as_uint(length, msg='length must be positive int')
        self._mu = mu
        self._psi = psi

        self._weights_init = np.zeros(length)
        if initial_weight is not None:
            try:
                self._weights_init += initial_weight
            except ValueError:
                raise ValueError('size of initial_weight and length of filter '
                                 'don\'t match')

        self._X_init = np.zeros_like(self._weights_init)
        if initial_input is not None:
            try:
                self._X_init += initial_input
            except ValueError:
                raise ValueError('size of initial_input and length of filter '
                                 'don\'t match')
        self._X_init = self._X_init[::-1]

    def INITFUNC(self):
        self._X = copy.copy(self._X_init)
        self._weights = copy.copy(self._weights_init)

    @property
    def weights(self):
        return self._weights

    def OUTPUTSTEP(self, portid):
        if portid == 0:
            xn = self.inports[0].step()
            # update the input signal series
            self._X[:-1] = self._X[1:]
            self._X[-1] = xn
            # output signal
            res = self._weights.dot(self._X)
            return res
        elif portid == 1:
            return self._weights

    def BLOCKSTEP(self, *xs):
        xn, error = xs
        # update weights (calculate weights for next iteration)
        mu = self._mu / (np.dot(self._X, self._X) + self._psi)
        self._weights += mu * error * self._X
        return NA, NA


class FIRAPLMS(BaseBlock):

    """FIR Wiener filter using Affine Projection LMS method.

    The NLMS algrithm use the following tap-weight vector recursion:

        .. math:: \mathbf w(n+1)=\mathbf w(n)+\\tilde \mu \mathbf X(n)        \
        \\left(\mathbf X^T(n)\mathbf X(n)+\\psi\mathbf I\\right)^{-1}         \
        \mathbf e(n)

    where

        .. math:: \mathbf X(n)=[\mathbf x(n)\\quad \mathbf x(n-1)\\quad       \
        \cdots\\quad \mathbf x(n-M+1)]

    where **w** is the tap-weight vector, *e* is the error signal vector which
    is the difference between the desired output vector and the filter output
    vector, and **x** is the input signal vector. $\\tilde \mu$ and $\psi$ are
    positive constants to imporve the reliablity of the algrithm.

    The block has tow input signals. The first input signal is X and the second
    input is error. X is the input to the plant and error is the defference
    between desired response and the filter.

    The block has two outputs. The first output is the estimate output of the
    plant and the second output is the weight array.


    The block has tow input signals. The first input signal is X and the second
    input is error. X is the input to the plant and error is the defference
    between desired response and the filter.

    The block has two outputs. The first output is the estimate output of the
    plant and the second output is the weight array.

    Parameters
    ----------
    length: int
        length of FIR filter.

    m: int
        Number of hyperplanes. if m == 1, the APLMS algrithm reduces to the
        NLMS algrithm. (default=1)

    mu: float
        Step-size paremeter. Used to control the convergence and the
        misadjestment, should be positive. (default=1)

    psi: float
        Used to prevent division by a small value on denominator. (default=0)

    initial_weght: float or iterable, optional
        Initial weights of the filter. If a float is given, all initial weights
        are set to this float number. If an iterable object is given, its
        length should be equal to the length of the filter.

    initial_input: float or iterable, optional
        Initial inputs to the filter. If a float is given, all initial inputs
        are set to this float number. If an iterable object is given, its
        length should be equal to length + m - 1

    initial_desired: float or iterable, optional
        Initial desired output of the filter. If a float is given, all initial
        desired outputs are set to this float number. If an iterable object is
        given, its length should be equal to m

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'APLMS Filter')

    Ports
    -----
    In[0]: Input
        The input signal

    In[1]: Error
        The error signal (desired output - filter output)

    Out[0]: Output
        The output signal

    Out[1]: Weight
        The weight coefficients of the filter. The weight of the latest input
        signal is on the left.

    See Also
    ----------
    FIRLMS

    Notes
    -----
    APLMS algrithm for Complex value signals is not supported yet.
    """

    def __init__(self, length=50, m=1, mu=1, psi=0, initial_weight=None,
                 initial_input=None, initial_desired=None, dt=None,
                 name='APLMS Filter'):
        super().__init__(nin=2, nout=2, dt=dt, name=name)
        self.inports[0].rename('Input')
        self.inports[1].rename('Error')
        self.outports[0].rename('Output')
        self.outports[1].rename('Weight')

        length = as_uint(length, msg='length must be positive int')
        m = as_uint(m, msg='m must be positive int')
        self._N, self._M = length, m
        self._mu = mu
        self._psi = psi

        self._weights_init = np.zeros(length)
        if initial_weight is not None:
            try:
                self._weights_init += initial_weight
            except ValueError:
                raise ValueError('size of initial_weight and length of filter '
                                 'don\'t match')

        self._X_init = np.zeros(length + m - 1)
        if initial_input is not None:
            try:
                self._X_init += initial_input
            except ValueError:
                raise ValueError('size of initial_input and given filter '
                                 'dimensions don\'t match')
        self._X_init = self._X_init[::-1]

        self._desired_init = np.zeros(m)
        if initial_desired is not None:
            try:
                self._desired_init += initial_desired
            except ValueError:
                raise ValueError('size of initial_disired and the size of m '
                                 'don\'t match')
        self._desired_init = self._desired_init[::-1]

        self._y = None        # vector of history outputs of the filter
        self._X_matrix = None  # matrix of input signal with shape N*M

    def INITFUNC(self):
        self._X = copy.copy(self._X_init)
        self._weights = copy.copy(self._weights_init)
        self._desired = copy.copy(self._desired_init)

    @property
    def weights(self):
        return self._weights

    def _form_X(self):
        M, N = self._M, self._N
        X = np.zeros([N, M])
        for i in range(N):
            X[i] = self._X[i:i + M]
        self._X_matrix = X
        return X

    def OUTPUTSTEP(self, portid):
        if portid == 0:
            # get input signal
            xn = self.inports[0].step()
            # update the input signal series
            self._X[:-1] = self._X[1:]
            self._X[-1] = xn
            # output signal
            X_matrix = self._form_X()
            self._y = np.dot(X_matrix.T, self._weights)
            return self._y[-1]
        elif portid == 1:
            return self._weights

    def BLOCKSTEP(self, *xs):
        xn, error = xs
        # update the desired signal output vector
        desired = self._y[-1] + error
        self._desired[:-1] = self._desired[1:]
        self._desired[-1] = desired
        desired = self._desired
        # calculate error vector
        error_vector = desired - self._y
        # update weights (calculate weights for next iteration)
        X = self._X_matrix
        A = np.dot(X.T, X) + self._psi * np.eye(self._M)
        try:
            invA = inv(A)
        except LinAlgError:
            # If the initial inputs are all zero, A must be Singular.
            # The dimension M must be reduced to solve the problem at the
            # several steps at the beginning of the iteration.
            if not self.n + 1 < self._M:
                raise
            if not hasattr(self, 'warning_message'):
                self.warning_message = ('The dimenson m is reduced '
                                        'automatically to prevent singular '
                                        'matrix at the beginning of the '
                                        'iteration.')
                self.system.warn(self.warning_message)
            m_new = self.n + 1
            A = A[-m_new:, -m_new:]
            invA = inv(A)
            error_vector = error_vector[-m_new:]
            X = X[:, -m_new:]
        eta = self._mu * np.dot(X, np.dot(invA, error_vector))
        self._weights += eta
        return NA, NA


class FIRTDLMS(BaseBlock):

    """SISO FIR Wiener filter using TDLMS method.

    The LMS algrithm use the following tap-weight vector recursion:

        .. math:: \mathbf w(n+1)=\mathbf w(n)+2\mu e(n)\mathbf x_T^n(n)
    where **w** is the tap-weight vector, *\mu* is the step-size parameter, *e*
    is the error signal which is the difference between the desired output and
    the filter output, and **x_T^n** is the normalized transfered input signal
    vector.

    The block has two input signals. The first input signal is X and the second
    input is error. X is the input to the plant and error is the defference
    between desired response and the filter.

    The block has two outputs. The first output is the estimate output of the
    plant and the second output is the weight array.

    Parameters
    ----------
    length: int
        Length of FIR filter.

    mu: scalar
        Step-size paremeter.

    beta: float
        Exponential factor for estimating the power of the signal. Should be
        close to but less than 1.

    initial_weght: scalar or iterable optional
        Initial weights of the filter. If a scalar is given, all initial
        weights are set to this float number. If an iterable object is given,
        its length should be equal to the length of the filter.

    initial_input: scala or iterable, optional
        Initial inputs to the filter. If a scalar is given, all initial inputs
        are set to this float number. If an iterable object is given, its
        length should not be shorter than the length of the filter.

    dt: float, optional
        Sampling time.

    dtype: data-type, optional
        The desired data-type for the filter. If not given, the type will be
        float.

    name: string, optional
        Name of this block. (default = 'LMS Filter')

    Note
    ----
    The default transform is DCT. It can be modified by changing the method
    "transform".

    See Also
    ----------
    FIRLMS
    """

    def __init__(self, length=50, mu=0.01, beta=0.99, initial_weight=None,
                 initial_input=None, dtype=None, dt=None,
                 name='FIRTDLMS Filter'):
        super().__init__(nin=2, nout=2, dt=dt, name=name)
        self.inports[0].rename('Input')
        self.inports[1].rename('Error')
        self.outports[0].rename('Output')
        self.outports[1].rename('Weight')

        # init for single-in-single-out system
        length = as_uint(length, msg='length must be positive int')

        if np.isscalar(mu):
            self._mu = mu
        else:
            raise ValueError('mu must be scalar')

        self._beta = beta

        # initial weight
        if initial_weight is None:
            initial_weight = 0.0
        self._weights_init = np.zeros(length, dtype=dtype)
        try:
            self._weights_init[:] = initial_weight
        except ValueError:
            raise ValueError('shape of initial_weight is not correct')

        # initial input
        if initial_input is None:
            initial_input = 0.0
        self._X_init = np.zeros(length, dtype=dtype)
        initial_input = np.asarray(initial_input)
        if np.iterable(initial_input):
            try:
                self._X_init[:] = initial_input[-length:]
            except (ValueError, IndexError,):
                raise ValueError('shape of initial_input not correct')
        else:
            self._X_init[:] = initial_input
        self._X_init = self._X_init[::-1]

        # estimate of mean square values of _XTN_init
        self._sigma_init = self.transform(self._X_init) ** 2

    def INITFUNC(self):
        self._X = copy.copy(self._X_init)
        self._weights = copy.copy(self._weights_init)
        self._sigma = self._sigma_init

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value

    @property
    def weights(self):
        return self._weights

    def OUTPUTSTEP(self, portid):
        if portid == 0:
            xn = self.inports[0].step()
            # update the input signal series
            self._X[1:] = self._X[:-1]
            self._X[0] = xn
            # do transform
            XT = self.transform(self._X)
            # do normalization
            beta = self._beta
            self._sigma = self._sigma * beta + (1 - beta) * XT ** 2
            XTN = self._sigma ** -0.5 * XT
            self._XTN = XTN
            # output signal
            res = np.dot(self._weights, self._XTN)
            return res
        elif portid == 1:
            return self._weights

    def BLOCKSTEP(self, *xs):
        xn, error = xs
        # update weights (calculate weights for next iteration)
        self._weights = (self._weights +
                         2 * self.mu.conjugate() * error * self._XTN)
        return NA, NA

    def transform(self, x):
        return dct(x)


class IIRLMS(BaseBlock):

    """SISO IIR adaptive filter using Output Error Method.

    The algrithm use the following tap-weight vector recursion:

        .. math:: \mathbf w(n+1)=\mathbf w(n)+2\mu e(n)\mathbf \eta(n)
    where **w** is the tap-weight vector, *\mu* is the step-size parameter, *e*
    is the error signal which is the difference between the desired output and
    the filter output, and **\eta** is the vector of history inputs and
    outputs.

    The block has two input signals. The first input signal is X and the second
    input is error. X is the input to the plant and error is the defference
    between desired response and the filter.

    The block has three outputs. The first output is the estimate output of the
    plant, the second output is the weight of the inputs of the system and the
    last output is the weight of outputs of the system.

    Parameters
    ----------
    N: int
        Length of history inputs.

    M: int
        Length of history outputs.

    mu: scalar
        Step-size paremeter.

    initial_weight_input: scalar or iterable optional
        Initial weights of history inputs. If a scalar is given, all initial
        weights are set to this float number. If an iterable object is given,
        its length should be equal to N.

    initial_weight_output: scalar or iterable optional
        Initial weights of history outputs. If a scalar is given, all initial
        weights are set to this float number. If an iterable object is given,
        its length should be equal to M.

    initial_input: scalar or iterable, optional
        Initial inputs to the filter. If a scalar is given, all initial inputs
        are set to this float number. If an iterable object is given, its
        length should not be smaller than N.

    initial_output: scalar of iterable, optional
        Initial outputs of the system. If a scalar is given, all initial
        outputs are set to this float number. If an iterable object is given,
        its length should not be smaller than M.

    dtype: data-type, optional
        The desired data-type for the filter. If not given, the type will be
        float.

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'IIRLMS Filter')

    See Also
    ----------
    FIRLMS
    """

    def __init__(self, N=5, M=5, mu=0.01, initial_weight_input=None,
                 initial_weight_output=None, initial_input=None,
                 initial_output=None, dtype=None, dt=None,
                 name='IIRLMS Filter'):
        super().__init__(nin=2, nout=3, dt=dt, name=name)
        self.inports[0].rename('Input')
        self.inports[1].rename('Error')
        self.outports[0].rename('Output')
        self.outports[1].rename('WeightInput')
        self.outports[1].rename('WeightOutput')

        # init
        N = as_uint(N, msg='N must be positive int')
        M = as_uint(M, msg='M must be positive int')

        if np.isscalar(mu):
            self._mu = mu
        else:
            raise ValueError('mu must be scalar')

        # initial weight
        if initial_weight_input is None:
            initial_weight_input = 0.0
        self._weights_input_init = np.zeros(N, dtype=dtype)
        try:
            self._weights_input_init[:] = initial_weight_input
        except ValueError:
            raise ValueError('shape of initial_weight_input is not correct')

        if initial_weight_output is None:
            initial_weight_output = 0.0
        self._weights_output_init = np.zeros(M, dtype=dtype)
        try:
            self._weights_output_init[:] = initial_weight_output
        except ValueError:
            raise ValueError('shape of initial_weight_output is not correct')

        # initial input and output
        if initial_input is None:
            initial_input = 0.0
        self._X_init = np.zeros(N, dtype=dtype)
        initial_input = np.asarray(initial_input)
        if np.iterable(initial_input):
            try:
                self._X_init[:] = initial_input[-N:]
            except (ValueError, IndexError,):
                raise ValueError('shape of initial_input not correct')
        else:
            self._X_init[:] = initial_input
        self._X_init = self._X_init[::-1]

        if initial_output is None:
            initial_output = 0.0
        self._Y_init = np.zeros(M, dtype=dtype)
        initial_output = np.asarray(initial_output)
        if np.iterable(initial_output):
            try:
                self._Y_init[:] = initial_output[-M:]
            except (ValueError, IndexError,):
                raise ValueError('shape of initial_output not correct')
        else:
            self._Y_init[:] = initial_output
        self._Y_init = self._Y_init[::-1]

    def INITFUNC(self):
        self._X = copy.copy(self._X_init)
        self._Y = copy.copy(self._Y_init)
        self._weights_input = copy.copy(self._weights_input_init)
        self._weights_output = copy.copy(self._weights_output_init)

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value

    @property
    def weights(self):
        return self._weights_input, self._weights_output

    def OUTPUTSTEP(self, portid):
        if portid == 0:
            xn = self.inports[0].step()
            # update the input signal series
            self._X[1:] = self._X[:-1]
            self._X[0] = xn
            # output signal
            yn = (np.dot(self._weights_input, self._X) -
                  np.dot(self._weights_output, self._Y))
            # update the output signal series
            self._Y[1:] = self._Y[:-1]
            self._Y[0] = yn
            return yn
        elif portid == 1:
            return self._weights_input
        elif portid == 2:
            return self._weights_output

    def BLOCKSTEP(self, *xs):
        xn, error = xs
        # update weights (calculate weights for next iteration)
        self._weights_input = (self._weights_input +
                               2 * self.mu.conjugate() * error * self._X)
        self._weights_output = (self._weights_output -
                                2 * self.mu.conjugate() * error * self._Y)
        return NA, NA, NA


class FIRRLS(BaseBlock):

    """Adaptive filter using the standard rescursice least squares (RLS)
    algorithm.

    The details of the algorithm can be found at p421 in Ref[1]

    The block has tow input signals. The first input signal is X and the second
    input is error. X is the input to the plant and error is the defference
    between desired response and the filter.

    The block has two outputs. The first output is the estimate output of the
    plant and the second output is the weight array.


    The block has tow input signals. The first input signal is X and the second
    input is error. X is the input to the plant and error is the defference
    between desired response and the filter.

    The block has two outputs. The first output is the estimate output of the
    plant and the second output is the weight array.

    Parameters
    ----------
    length: int
        length of FIR filter.

    lambda_: float
        The weighting factor, 0 < lambda_ < 1

    initial_weight_input: scalar or iterable optional
        Initial weights of history inputs. If a scalar is given, all initial
        weights are set to this float number. If an iterable object is given,
        its length should be equal to length.

    initial_input: float or iterable, optional
        Initial inputs to the filter. If a float is given, all initial inputs
        are set to this float number. If an iterable object is given, its
        length should be equal to the length of the filter.

    initial_phi_inv: np.array with two dimensions or float, optional
        Initial value of the inverse of *\Phi_\lambda*. If an array is given,
        each dimension of the array should be the same as length and the array
        should be symmetric. If a float is given, the initial value will be an
        array with this value in the diagonal terms. (default=1e-3)
    dtype: data-type, optional
        The desired data-type for the filter. If not given, the type will be
        float.

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'FIRRLS Filter')

    Ports
    -----
    In[0]: Input
        The input signal

    In[1]: Error
        The error signal (desired output - filter output)

    Out[0]: Output
        The output signal

    Out[1]: Weight
        The weight coefficients of the filter. The weight of the latest input
        signal is on the left.

    See Also
    ----------
    FIRLMS

    References
    -----
    [1] Farhang-Boroujeny B. Adaptive filters: theory and applications[M]. John
    Wiley & Sons, 2013.
    """

    def __init__(self, length=50, lambda_=0.99, initial_weight=None,
                 initial_input=None, initial_phi_inv=1e-3,
                 dtype=None, dt=None, name='NLMS Filter'):
        super().__init__(nin=2, nout=2, dt=dt, name=name)
        self.inports[0].rename('Input')
        self.inports[1].rename('Error')
        self.outports[0].rename('Output')
        self.outports[1].rename('Weight')

        length = as_uint(length, msg='length must be positive int')

        self._lambda = lambda_

        self._weights_init = np.zeros(length, dtype=dtype)
        if initial_weight is not None:
            try:
                self._weights_init += initial_weight
            except ValueError:
                raise ValueError('size of initial_weight and length of filter '
                                 'don\'t match')

        self._X_init = np.zeros_like(self._weights_init)
        if initial_input is not None:
            try:
                self._X_init += initial_input
            except ValueError:
                raise ValueError('size of initial_input and length of filter '
                                 'don\'t match')
        self._X_init = self._X_init[::-1]

        if np.isscalar(initial_phi_inv):
            self._phi_inv_init = np.diag(
                [initial_phi_inv] * length).astype(dtype)
        else:
            self._phi_inv_init = np.zeros([length, length], dtype=dtype)
            try:
                self._phi_inv_init[:] = initial_phi_inv
            except ValueError:
                raise ValueError('size of initial_phi_inv is not correct')

    def INITFUNC(self):
        self._X = copy.copy(self._X_init)
        self._weights = copy.copy(self._weights_init)
        self._phi_inv = copy.copy(self._phi_inv_init)

    @property
    def weights(self):
        return self._weights

    def OUTPUTSTEP(self, portid):
        if portid == 0:
            xn = self.inports[0].step()
            # update the input signal series
            self._X[:-1] = self._X[1:]
            self._X[-1] = xn
            # output signal
            res = self._weights.dot(self._X)
            return res
        elif portid == 1:
            return self._weights

    def BLOCKSTEP(self, *xs):
        xn, error = xs
        # update weights (calculate weights for next iteration)
        u = self._phi_inv.dot(self._X)
        k = u / (self._lambda + np.dot(self._X, u))
        self._weights += k * error
        self._phi_inv = self._phi_inv - k.reshape(-1, 1).dot(u.reshape(1, -1))
        self._phi_inv /= self._lambda
        self._phi_inv = np.triu(self._phi_inv) + np.triu(self._phi_inv, 1).T
#        self._phi_inv = (self._phi_inv - k.reshape(-1,1).
#                         dot(self._X.reshape(1,-1).dot(self._phi_inv)))/self._lambda
        return NA, NA
