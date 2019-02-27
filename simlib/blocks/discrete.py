"""discrete.py

This module contains the basic discrete system components in control system.
"""

__version__ = '0.1.0'
# This is a major revision of the previous versions, intending to make the
# library easier to use and maintain.

import copy
from collections import deque
import numpy as np
from ..simsys import BaseBlock, NA
from ..simexceptions import SimulationError
from ..misc import as_uint, tf_dot


__all__ = ['Delay', 'TappedDelay', 'FIRFilter', 'FIRFilterTimeVarying',
           'IIRFilter', 'IIRFilterTimeVarying',
           'TransferFunction', 'StateSpace']


class Delay(BaseBlock):

    """Block of delay component.

    Delay input signal by a specified number of samples.

    Parameters
    ----------
    n_delay: int
        Number of time samples to delay, must be greater than 0. (default = 1)

    initial: float or iterable, optional
        Initial values stored in the block. If a float is given, all initial
        values are set to this float number. If a iterable object is given, its
        length should be equal to n_delay. Latest values on the right.

    dtype: data-type, optional
        The desired data-type of the block. If not given, the dtype is assumed
        to be float.

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'Delay')

    Ports
    -----
    In[0]: Input
        The input signal

    Out[0]: Output
        The output signal
    """

    def __init__(self, n_delay=1, initial=0.0, dtype=None, dt=None,
                 name='Delay'):
        super().__init__(nin=1, nout=1, dt=dt, name=name)
        self.inports[0].rename('Input')
        self.outports[0].rename('Output')

        self._n_delay = as_uint(n_delay, msg='delay must be int > 0')
        _initial = np.zeros([n_delay], dtype=dtype)
        _initial[:] = initial

        self._initial = deque(_initial)

    def INITFUNC(self):
        self._deque = deque(copy.copy(self._initial))

    def OUTPUTSTEP(self, portid):
        return self._deque.popleft()

    def BLOCKSTEP(self, *xs):
        self._deque.append(xs[0])
        return NA,


class TappedDelay(BaseBlock):

    """Block of tapped delay component.

    Delay input signal by a specified number of samples and output all delayed
    signals.

    Parameters
    ----------
    n_delay: int
        Number of time samples to delay, must be greater than 0. (default = 1)

    initial: float or iterable, optional
        Initial values stored in the block. If a float is given, all initial
        values are set to this float number. If a iterable object is given, its
        length should be equal to n_delay. Latest values on the right.

    dtype: data-type, optional
        The desired data-type of the block. If not given, the dtype is assumed
        to be float.

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'TappedDelay')

    See Also
    ----------
    BaseBlock
    """

    def __init__(self, n_delay=1, initial=0.0, dtype=None, dt=None,
                 name='TappedDelay'):
        super().__init__(nin=1, nout=1, dt=dt, name=name)
        self.inports[0].rename('Input')
        self.outports[0].rename('Output')

        self._n_delay = as_uint(n_delay, msg='delay must be int > 0')
        _initial = np.zeros([n_delay], dtype=dtype)
        _initial[:] = initial

        self._initial = deque(_initial)

    def INITFUNC(self):
        self._deque = copy.copy(self._initial)

    def OUTPUTSTEP(self, portId):
        return np.array(self._deque)

    def BLOCKSTEP(self, *xs):
        self._deque.popleft()
        self._deque.append(xs[0])
        return NA,


class FIRFilter(BaseBlock):
    """Block of a SISO discrete FIR filter.

    Build a FIR filter. The coefficient of the filter should be defined before
    initialize.

    The transfer function of FIRFilter can be written as:
        G(z) = a[0]*z^0 + a[1] * z^{-1} + ... + a[N]*z[-N]
    where a is the coefficients of the filter

    Parameters
    ----------
    coefficients: array-like object
        Coefficients of the FIR filter. The first value in the coefficients
        represents the gain of the direct output of the input while the
        remaining values represent the gains of the history inputs.

    initial_inputs: float or iterable, optional
        Initial inputs of the block. If a float is given, all initial values
        are set to this float number. If a iterable object is given, its length
        should not less than the length of coefficient - 1. Latest inputs on
        the right.

    dt: float, optional
        Sampling time.

    dtype: data-type, optional
        The desired data-type for the filter. If not given, the datatype is
        assumed to be float.

    name: string, optional
        Name of this block. (default = 'FIRFilter')

    Ports
    -----
    In[0]: Input
    The input signal.

    Out[0]: Output
    The output signal.

    Notes
    -----
    The Input signal and Output signal to the block are scalars.
    """

    def __init__(self, coefficients, initial_inputs=None, dtype=None, dt=None,
                 name='FIRFilter'):
        # coefficients should be an array

        super().__init__(nin=1, nout=1, dt=dt, name=name)

        # format coefficients
        coefficients = np.array(coefficients)
        if len(coefficients.shape) != 1:
            raise ValueError('coefficients must be 1-dimensional')
        self._coefficients = coefficients
        n_coefficient = len(coefficients)
        self.inports[0].rename('Input')
        self.outports[0].rename('Output')

        # check whether the filter has direct feedback
        if coefficients[0]:
            self._direct_feedback = True
        else:
            self._direct_feedback = False

        # initial inputs
        x = np.zeros(n_coefficient - 1, dtype=dtype)
        if initial_inputs is None:
            initial_inputs = 0.0
        x[:] = initial_inputs
        self._x_init = x[::-1]

    def INITFUNC(self):
        self._x = copy.copy(self._x_init)

    def OUTPUTSTEP(self, index):
        # if the system has direct feed-back, do nothing
        if self._direct_feedback:
            return NA
        y = np.dot(self._coefficients[1:], self._x)
        return y

    def BLOCKSTEP(self, *xs):
        coefficients = self._coefficients
        y = NA
        x_new = xs[0]

        # whether the system has direct feed-back
        if self._direct_feedback:
            y = coefficients[0] * x_new
            y += np.dot(coefficients[1:], self._x)

        self._x[1:] = self._x[:-1]
        try:
            if len(self._x):
                self._x[0] = x_new
        except ValueError:
            raise SimulationError('Input of {self} must be scalar.'
                                  .format(self=self))
        return y,


class FIRFilterTimeVarying(BaseBlock):
    """Block of a SISO discrete FIR filter of fixed length.

    Build a FIR filter. The coefficient of the filter is defined by the second
    input port.

    The transfer function of FIRFilter can be written as:
        G(z) = a[0]*z^0 + a[1] * z^{-1} + ... + a[N]*z[-N]
    where a is the coefficients of the filter

    Parameters
    ----------
    len_coefficients: int
        Length of the FIR filter.

    initial_inputs: float or iterable, optional
        Initial inputs of the block. If a float is given, all initial values
        are set to this float number. If a iterable object is given, its length
        should not less than the length of coefficient - 1. Latest inputs on
        the right.

    dt: float, optional
        Sampling time.

    dtype: data-type, optional
        The desired data-type for the filter. If not given, the datatype is
        assumed to be float.

    name: string, optional
        Name of this block. (default = 'FIRFilter')

    Ports
    -----
    In[0]: Input
    The input signal.

    In[1]: Coefficients
        The first value in the coefficients represents the gain of the direct
        output of the input while the remaining values represent the gains of
        the history inputs.

    Out[0]: Output
    The output signal.

    Notes
    -----
    The Input signal and Output signal to the block are scalars.
    """

    def __init__(self, len_coefficients, initial_inputs=None, dtype=None,
                 dt=None, name='FIRFilter'):
        # coefficients should be an int

        super().__init__(nin=2, nout=1, dt=dt, name=name)
        self.inports[0].rename('Input')
        self.inports[1].rename('Coefficients')
        self.outports[0].rename('Output')

        # format coefficients
        n_coefficients = as_uint(len_coefficients,
                                 msg='len_coefficients must positive int')
        self._n_coefficients = n_coefficients
        self._coefficients = np.zeros([n_coefficients])

        # initial inputs
        x = np.zeros(n_coefficients - 1, dtype=dtype)
        if initial_inputs is None:
            initial_inputs = 0.0
        x[:] = initial_inputs
        self._x_init = x[::-1]

    def INITFUNC(self):
        self._x = copy.copy(self._x_init)

    def OUTPUTSTEP(self, index):
        try:
            self._coefficients[:] = self.inports[1].step()
        except Exception:
            raise SimulationError("input of inports[1] of {n} is not correct"
                                  .format(s=len(self._coefficients), n=self))
        coefficients = self._coefficients

        # if the system has direct feed-back, do nothing
        if coefficients[0]:
            return NA

        y = np.dot(self._coefficients[1:], self._x)
        return y

    def BLOCKSTEP(self, *xs):
        coefficients = self._coefficients
        y = NA
        x_new = xs[0]

        # whether the system has direct feed-back
        if coefficients[0]:
            y = coefficients[0] * x_new
            y += np.dot(coefficients[1:], self._x)

        self._x[1:] = self._x[:-1]
        try:
            if len(self._x):
                self._x[0] = x_new
        except ValueError:
            raise SimulationError('Input of {self} must be scalar.'
                                  .format(self=self))
        return y,


class IIRFilter(BaseBlock):

    """Block of a SISO discrete IIR filter.

    Build a IIR filter. The coefficient of the filter can be either defined by
    initialization parameters or by the second or third input port.

    The transfer function of IIRFilter can be written as:
        G(z) = ((a[0]*z^0 + a[1]*z^{-1} + ... + a[N]*z[-N]) /
                (1 + b[0]*z^{-1} + ... + b[M]*z[-M-1]))
    where a and b are the coefficients of the filter

    Parameters
    ----------
    coefficients_input: array-like object
        Coefficients of the inputs of the IIR filter. The first value in the
        coefficients_input represents the gain of the direct output of the
        input while the remaining values represent the gains of the history
        inputs.

    coefficients_output: array-like object
        Coefficients of the outputs of the IIR filter. Note that the
        coefficient of the z^{-i} term is b[i-1] and the coefficient of the z^0
        term is always 1.

    initial_inputs: float or iterable, optional
        Initial inputs to the filter. If a float is given, all initial values
        are set to this float number. If a iterable object is given, its length
        should not less than the length of coefficient_input - 1. Latest inputs
        on the right.

    initial_outputs: float or iterable, optional
        Initial outputs of the filter. If a float is given, all initial values
        are set to this float number. If a iterable object is given, its length
        should not less than the length of coefficient_output. Latest inputs on
        the right.

    dtype: data-type, optional
        The desired data-type for the filter. If not given, the datatype is
        assumed to be float.

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'IIRFilter')

    Ports
    -----
    In[0]: Input
    The input signal.

    Out[0]: Output
    The output signal.

    Notes
    -----
    The Input signal and Output signal to the block are scalars.
    """

    def __init__(self, coefficients_input, coefficients_output,
                 initial_inputs=None, initial_outputs=None, dtype=None, dt=None,
                 name='IIRFilter'):
        super().__init__(nin=1, nout=1, dt=dt, name=name)
        self.inports[0].rename('Input')
        self.outports[0].rename('Output')

        # format coefficients
        coefficients_input = np.array(coefficients_input)
        if len(coefficients_input.shape) != 1:
            raise ValueError('coefficients_input must be 1-dimensional or int')
        self._coefficients_input = coefficients_input
        N = len(coefficients_input)
        coefficients_output = np.array(coefficients_output)
        if len(coefficients_output.shape) != 1:
            raise ValueError(
                'coefficients_output must be 1-dimensional or int')
        self._coefficients_output = coefficients_output
        M = len(coefficients_output) + 1

        # check whether the filter has direct feedback
        if coefficients_input[0]:
            self._direct_feedback = True
        else:
            self._direct_feedback = False

        # initial inputs and outputs
        x = np.zeros(N - 1, dtype=dtype)
        if initial_inputs is None:
            initial_inputs = 0.0
        x[:] = initial_inputs
        self._x_init = x[::-1]
        y = np.zeros(M - 1, dtype=dtype)
        if initial_outputs is None:
            initial_outputs = 0.0
        y[:] = initial_outputs
        self._y_init = y[::-1]

    def INITFUNC(self):
        self._x = copy.copy(self._x_init)
        self._y = copy.copy(self._y_init)

    def OUTPUTSTEP(self, index):
        # if the system has direct feed-back, do nothing
        if self._direct_feedback:
            return NA

        y = (np.dot(self._coefficients_input[1:], self._x) -
             np.dot(self._coefficients_output, self._y))
        self._y[1:] = self._y[:-1]
        if len(self._y):
            self._y[0] = y
        return y

    def BLOCKSTEP(self, *xs):
        coefficients_input = self._coefficients_input
        coefficients_output = self._coefficients_output
        y = NA
        x_new = xs[0]

        # if the system has direct feed-back
        if self._direct_feedback:
            y = coefficients_input[0] * x_new
            y += (np.dot(coefficients_input[1:], self._x) -
                  np.dot(coefficients_output, self._y))
            self._y[1:] = self._y[:-1]
            self._y[0] = y

        self._x[1:] = self._x[:-1]
        try:
            if len(self._x):
                self._x[0] = x_new
        except ValueError:
            raise SimulationError('Input of {self} must be scalar.'
                                  .format(self=self))
        return y,


class IIRFilterTimeVarying(BaseBlock):

    """Block of a SISO discrete IIR filter of fixed length of numerator and
    denominator.

    Build a IIR filter. The coefficient of the filter can be either defined by
    initialization parameters or by the second or third input port.

    The transfer function of IIRFilter can be written as:
        G(z) = ((a[0]*z^0 + a[1]*z^{-1} + ... + a[N]*z[-N]) /
                (1 + b[0]*z^{-1} + ... + b[M]*z[-M-1]))
    where a and b are the coefficients of the filter

    Parameters
    ----------
    len_coefficients_input: int
        length of the coefficients of the inputs of the IIR filter.

    len_coefficients_output: int
        Length of the coefficients of the outputs of the IIR filter. Note that
        this value is M which means it should be equal to the number of terms
        of the denominator - 1.

    initial_inputs: float or iterable, optional
        Initial inputs to the filter. If a float is given, all initial values
        are set to this float number. If a iterable object is given, its length
        should not less than the length of coefficient_input - 1. Latest inputs
        on the right.

    initial_outputs: float or iterable, optional
        Initial outputs of the filter. If a float is given, all initial values
        are set to this float number. If a iterable object is given, its length
        should not less than the length of coefficient_output. Latest inputs on
        the right.

    dtype: data-type, optional
        The desired data-type for the filter. If not given, the datatype is
        assumed to be float.

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'IIRFilter')

    Ports
    -----
    In[0]: Input
    The input signal.

    In[1]: Coefficients of Input
        The coefficients of the numerator of the filter. The first value in the
        coefficients_input represents the gain of the direct output of the
        input while the remaining values represent the gains of the history
        inputs.

    In[2]: Coefficients of Output
        The coefficients of the demoninator of the filter. Note that the
        coefficient of the z^{-i} term is b[i-1] and the coefficient of the z^0
        term is always 1.

    Out[0]: Output
    The output signal.

    Notes
    -----
    The Input signal and Output signal to the block are scalars.
    """

    def __init__(self, len_coefficients_input, len_coefficients_output,
                 initial_inputs=None, initial_outputs=None, dtype=None, dt=None,
                 name='IIRFilter'):
        super().__init__(nin=3, nout=1, dt=dt, name=name)
        self.inports[0].rename('Input')
        self.inports[1].rename('Coefficients of Input')
        self.inports[2].rename('Coefficients of Output')
        self.outports[0].rename('Output')

        self._len_coefficients_input = as_uint(len_coefficients_input,
                                               msg='len_coefficients_input should be positive int')
        self._coefficients_input = np.zeros([self._len_coefficients_input],
                                            dtype=dtype)
        self._len_coefficients_output = as_uint(len_coefficients_output,
                                                msg='len_coefficients_output should be positive int')
        self._coefficients_output = np.zeros([self._len_coefficients_output],
                                             dtype=dtype)

        # initial inputs and outputs
        x = np.zeros(self._len_coefficients_input - 1, dtype=dtype)
        if initial_inputs is None:
            initial_inputs = 0.0
        x[:] = initial_inputs
        self._x_init = x[::-1]
        y = np.zeros(self._len_coefficients_output, dtype=dtype)
        if initial_outputs is None:
            initial_outputs = 0.0
        y[:] = initial_outputs
        self._y_init = y[::-1]

    def INITFUNC(self):
        self._x = copy.copy(self._x_init)
        self._y = copy.copy(self._y_init)

    def OUTPUTSTEP(self, index):
        try:
            self._coefficients_input[:] = self.inports[1].step()
        except Exception:
            raise SimulationError("input of inports[1] of {n} is not correct"
                                  .format(s=len(self._coefficients_input), n=self))
        try:
            self._coefficients_output[:] = self.inports[2].step()
        except Exception:
            raise SimulationError("input of inports[2] of {n} is not correct"
                                  .format(s=len(self._coefficients_output), n=self))

        # if the system has direct feed-back, do nothing
        if self._coefficients_input[0]:
            return NA

        y = (np.dot(self._coefficients_input[1:], self._x) -
             np.dot(self._coefficients_output, self._y))
        self._y[1:] = self._y[:-1]
        if len(self._y):
            self._y[0] = y
        return y

    def BLOCKSTEP(self, *xs):
        coefficients_input = self._coefficients_input
        coefficients_output = self._coefficients_output
        y = NA
        x_new = xs[0]

        # if the system has direct feed-back
        if self._coefficients_input[0]:
            y = coefficients_input[0] * x_new
            y += (np.dot(coefficients_input[1:], self._x) -
                  np.dot(coefficients_output, self._y))
            self._y[1:] = self._y[:-1]
            self._y[0] = y

        self._x[1:] = self._x[:-1]
        try:
            if len(self._x):
                self._x[0] = x_new
        except ValueError:
            raise SimulationError('Input of {self} must be scalar.'
                                  .format(self=self))
        return y,


class TransferFunction(BaseBlock):

    """Block of discrete LTI system in transfer function form.

    Build a SISO discrete system in transfer function form. The system can be
    written as:
        G(z) = (b0*z^n + ... + b(n)*z^0) / (a0*z^m + ... + a(m)*z^0)
    Note n should be equal to or less than m.

    The numerator and denominator of the transfer function can be either
    defined when initialization or by input of the block. The input ports of
    the block will follow the following order: input signal, coefficients of
    numerator, coefficients of denominator.

    Parameters
    ----------
    num: array-like object
        Polynomial coefficients of the numerator. If int is given, it will be
        defined by the second input port with given size.

    den: array-like object
        Polynomial coefficients of the denominator. If int is given, it will be
        defined by the second or third input port, depending on whether num is
        array-like object or int.

    initial_inputs: float or iterable, optional
        Initial inputs of the block. If a float is given, all initial values
        are set to this float number. If a iterable object is given, its length
        should be equal to the length of denominator - 1. From older inputs to
        newer inputs.

    initial_outputs: float or iterable, optional
        Initial outputs of the block. If a float is given, all initial values
        are set to this float number. If a iterable object is given, its length
        should be equal to the length of denominator - 1. From older outputs to
        newer outputs.

    dtype: data-type, optional
        The desired data-type for the filter. If not given, the datatype is
        assumed to be float.

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'StateSpace')

    Ports
    -----
    In[0]: Input
    The input signal.

    Out[0]: Output
    The output signal.

    See Also
    ----------
    BaseBlock
    """

    def __init__(self, num, den, initial_inputs=None, initial_outputs=None,
                 dtype=None, dt=None, name='TransferFunction'):

        super().__init__(nin=1, nout=1, dt=dt, name=name)
        self.inports[0].rename('Input')
        self.outports[0].rename('Output')

        num = np.array(num)
        while len(num) > 1 and not num[0]:
            num = num[1:]
        n = len(num) - 1
        den = np.array(den)
        while len(den) > 1 and not den[0]:
            den = den[1:]
        a0 = den[0]
        if a0 == 0:
            raise ValueError('den is all zero')
        m = len(den) - 1

        if m < n:
            raise ValueError("order of numerator should be equal to or "
                             "less than denominator")

        self._num, self._den = num / a0, den[1:] / a0
        self._m, self._n_ = m, n

        # initial inputs and outputs
        x = np.zeros(m, dtype=dtype)
        if initial_inputs is None:
            initial_inputs = 0.0
        x[:] = initial_inputs
        self._x_init = x[::-1]
        y = np.zeros(m, dtype=dtype)
        if initial_outputs is None:
            initial_outputs = 0.0
        y[:] = initial_outputs
        self._y_init = y[::-1]

    def INITFUNC(self):
        self._x = copy.copy(self._x_init)
        self._y = copy.copy(self._y_init)

    def OUTPUTSTEP(self, index):

        m, n = self._m, self._n_

        # if the lti system has direct feed-back, do nothing
        if m == n:
            return NA
        x = self._x
        y = np.dot(self._num, x[-n - 1:]) - np.dot(self._den, self._y)
        self._y[1:] = self._y[:-1]
        self._y[0] = y
        return y

    def BLOCKSTEP(self, *xs):
        m, n = self._m, self._n_
        y = NA

        # check if the lti system has direct feed-back
        if m == n:
            y = self._num[0] * xs[0]
            # Do this check in case self._y is empty
            if len(self._y):
                y += np.dot(self._num[1:], self._x) - \
                    np.dot(self._den, self._y)
                self._y[1:] = self._y[:-1]
                self._y[0] = y
        # Do this check in case self._x is empty
        if len(self._x):
            self._x[1:] = self._x[:-1]
            self._x[0] = xs[0]
        return y,


class StateSpace(BaseBlock):

    """Block of discrete LTI system in state space form.

    Build a discrete system in state space form. The system can be written as:
    x(n+1) = G*x(n) + H*u(n); y(n) = C*x(n) + D*u(n), where u is the input and
    y is the output of the system.
    The input signal of the block should one-dimensional array or floating
    point number. The number of output signals of the block can be either 1 or
    2 depending on the argument 'state_output'. If state_output == True, the
    block has two output signals, they are system output y and state variable x
    respectively. If state_output == False, the block only has one output, y.

    Parameters
    ----------
    G: 2-D array-like object
        State matrix.

    H: array-like object
        Input matrix.

    C: array-like object
        Output matrix.

    D: array-like object, optional
        Feedthrough matrix.

    initial: float or iterable, optional
        Initial values stored in the block. If a float is given, all initial
        values are set to this float number. If a iterable object is given, its
        length should be equal to the number of state variables.

    state_output: bool
        Whether to output state variables. (default = True)

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'StateSpace')

    Ports
    -----
    In[0]: Input
    The input signal.

    Out[0]: Output
    The output signal.

    Notes
    -----
    The Input signal to this block can either be scalar, vector or matrix and
    they will all be converted to 1-D vector and the output of the block is
    always a 1-D array.

    See Also
    ----------
    BaseBlock
    """

    def __init__(self, G, H, C, D=None, initial=None, dt=None,
                 name='StateSpace'):
        super().__init__(nin=1, nout=2, dt=dt, name=name)
        self.inports[0].rename('Input')
        self.outports[0].rename('Output')
        self.outports[1].rename('StateVariable')

        # format G
        G = np.array(G)
        if G.shape[0] != G.shape[1]:
            raise ValueError('G must be square')
        self._G = G
        m = self._G.shape[0]

        # format H
        H = np.array(H)
        self._H = H.reshape(m, -1)
        n = self._H.shape[1]

        # format C
        C = np.array(C)
        self._C = C.reshape(-1, m)
        k = self._C.shape[0]

        # format D
        if D is None:
            self._D = np.zeros([k, n])
        else:
            D = np.array(D)
            self._D = D.reshape(k, n)

        # initial values
        if initial is None:
            initial = 0.0
        self._x_init = np.zeros([m]) + initial

    def INITFUNC(self):
        self._x = copy.copy(self._x_init)

    def OUTPUTSTEP(self, portid):
        if portid == 1:
            return self._x
        return NA

    def BLOCKSTEP(self, *xs):
        G, H = self._G, self._H
        C, D = self._C, self._D
        u, x = np.array(xs[0]).reshape(-1), self._x
        y = C.dot(x) + D.dot(u)
        self._x = G.dot(x) + H.dot(u)
        return y, NA
