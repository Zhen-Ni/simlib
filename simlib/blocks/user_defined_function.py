"""user_defined_function.py

This module contains the interface for user-defined functions.
"""

import ctypes
import numpy as np
from ..simsys import BaseBlock, NA
from ..simexceptions import SimulationError


__all__ = ['ctypes', 'UserDefinedFunction', 'PythonFunction', 'CFunction']


class UserDefinedFunction(BaseBlock):
    """The UserDefinedFunction block.

    This block runs the user defined function in each iteration.

    Parameters
    ----------
    func: callable
        The user defined function processing the input signals. The number of
        inputs to the function is defined by nin. The output of the function
        can be iterable and the length should be given by nout, and each
        element in the iterable object will be sent to the outports. However,
        if nout is set to None, the output of the function is not necessarily
        iterable and it will be sent to outports[0].

    t_as_arg: bool, optional
        Whether the user-defined function `func` accepts `t` as a keyword
        argument. If it is True, the simulation time will be an additional
        keyword argument passed to the function within each iteration. (default
        to False)

    nin: int, optional
        Number of input signals. This should also be the number of input
        parameters of func. (default to 1)

    nout: int or None, optional
        Number of output signals. If nout is None, output of the user-defined
        function will be sent to outports[0].
        (default = None)

    dt: float, optional
        Sampling time of this block. The input and output be evaluate once
        for each sampling time.

    name: string, optional
        The name of this block.
    """

    def __init__(self, func, t_as_arg=False, nin=1, nout=None, dt=None,
                 name='UserDefinedFunction'):
        if nout is None:
            super().__init__(nin=nin, nout=1, dt=dt, name=name)
        else:
            super().__init__(nin=nin, nout=nout, dt=dt, name=name)

        if t_as_arg:
            self._func = lambda _t, *_xs: func(*_xs, t=_t)
        else:
            self._func = lambda _t, *_xs: func(*_xs)
        self._direct_output = False if nout is None else True

    def BLOCKSTEP(self, *xs):
        if self._direct_output:
            return self._func(self.t, *xs)
        else:
            return self._func(self.t, *xs),


class PythonFunction(BaseBlock):
    """Provides python functions to act as a block.

    It is a class provides interface for users to highly customize their
    blocks. Users may define their routine either during initializing or
    runtime. Users may also writting their own classes by directly inheriting
    BaseBlock.


    Parameters
    ----------
    nin: int, optional
        Number of inputs of this block.

    nout: int, optional
        Number of outputs of this block.

    initfunc: callable or None, optional
        The user-defined function for INITFUNC. This function is invoked once
        the simulation system is initialized. It is usually used to set up some
        variables for the simulation. The initfunc should take no input
        arguments and returns nothing (though it is not mandatory).  If
        initfunc is set to None, the defeault version in BaseBlock will be
        used. (default to None)

    outputstep: callable or None, optional
        The user-defined function for OUTPUTSTEP. The outputstep is invoked as
        many times as nout at the beginning of each iteration during the
        simulation. This function provides the outputs of the block which is
        independent of the inputs to the block at this time step. The
        outputstep should take  exactly one input argument, which is an
        interger indicates the portid of the output. It should returns the
        corresponding output for this time step if it is independent of the
        input. However, if the output of the corresponding port is dependent on
        the input, this function should return sim.NA for this port. If it is
        set to be None, the default version in BaseBlock will be used. (default
        to None)

    blockstep: callable or None, optional
        The user-defined function for BLOCKSTEP. The blockstep is called once
        in each iteration, and it should provides output for all the ports
        unless the corresponding port has already been set by outputstep. This
        function should take the same number input arguments as that defined
        by nin, which are the inputs to the block, and returns a tuple with
        length nout. Each element of the tuple is the output of the
        corresponding port. If the output of the port has already been given
        in outputstep, it may be set to sim.NA. if blockstep is set to None,
        the default version in BaseBlock will be used. (default to None)

    dt: float, optional
        Sampling time of this block. The input and output be evaluate once
        for each sampling time.

    name: string, optional
        The name of this block.
    """

    def __init__(self, nin=1, nout=1, initfunc=None, outputstep=None,
                 blockstep=None, dt=None, name='PythonFunction'):
        super().__init__(nin=nin, nout=nout, dt=dt, name=name)

        self.set_initfunc(initfunc)
        self.set_outputstep(outputstep)
        self.set_blockstep(blockstep)

    def set_initfunc(self, initfunc):
        """Set INITFUNC for the block."""
        self._initfunc = initfunc

    def set_outputstep(self, outputstep):
        """Set OUTPUTSTEP for the block."""
        self._outputstep = outputstep

    def set_blockstep(self, blockstep):
        """Set BLOCKSTEP for the block."""
        self._blockstep = blockstep

    def INITFUNC(self):
        if self._initfunc is not None:
            return self._initfunc()
        else:
            return super().INITFUNC()

    def OUTPUTSTEP(self, portid):
        if self._outputstep is not None:
            return self._outputstep(portid)
        else:
            return super().OUTPUTSTEP(portid)

    def BLOCKSTEP(self, *xs):
        if self._blockstep is not None:
            return self._blockstep(*xs)
        else:
            return super().BLOCKSTEP(*xs)


class CFunction(BaseBlock):
    """Use C functions to act as a block.

    This block provides an easy way for users to invoke C code for simulation.

    Parameters
    ----------
    libname: str
        The name of the shared library. It will be loaded using the standard C
        calling convention provided by ctypes.

    types_in: iterable
        The datatypes of the input of the block, should be consistent with
        the functions in the dynamic library. The valid datatypes are defined
        in ctypes. If array is used, user may define its size in `sizes_in`.

    sizes_in: iterable or None, optional
        If types_in contains pointer, then the size of the array should be
        defined in `sizes_in`. The `sizes_in` is of the same length of
        `types_in`. The corresponding position of `sizes_in` indicates the
        length of the array defined in `types_in`. If a scalar value is used,
        the corresponding position in `sizes_in` should be set to sim.NA

    types_out: iterable or None
        The datatypes of the output of the block, should be consistent with
        the functions in the dynamic library. The valid datatypes are defined
        in ctypes. If array is used, user may define its size in `sizes_out`.

    sizes_out: iterable or None
        If types_out contains pointer, then the size of the array should be
        defined in `sizes_out`. The `sizes_out` is of the same length of
        types_out. The corresponding position of `sizes_out` indicates the
        length of the array defined in `types_out`. If a scalar value is used,
        the corresponding position in `sizes_out` should be set to sim.NA

    initfunc: str or None, optional
        The name of the function in `libname` for INITFUNC. This function is
        invoked once the simulation system is initialized. It is usually used
        to set up some variables for the simulation. The initfunc should follow
        the form `void initfunc(void);`, and be loaded by ctypes.  If initfunc
        is set to None, the defeault python routine in BaseBlock will be used.
        (default to None)

    outputstep: str or None, optional
        The name of the function for OUTPUTSTEP. The outputstep is invoked as
        many times as nout at the beginning of each iteration during the
        simulation. This function provides the outputs of the block which is
        independent of the inputs to the block at this time step. The
        outputstep should be loaded using ctypes and follow the form
        ```
        void outputstep(const unsigned int portid, void* output, bool* valid);
        ```
        where `portid` is the ID of the output port, `output` is a pointer
        to the output signal and `valid` shows whether the output is given in
        outputstep. Not that as void * is used, a type convension might be
        necessary in the C function. The outport will be updated only if
        `valid` is set to true. If `outputstep` is set to None, the default
        python version in BaseBlock will be used. (default to None)

    blockstep: str or None, optional
        The name of the function for BLOCKSTEP. The blockstep is called once
        in each iteration, and it should provides output for all the ports
        unless the corresponding port has already been set by outputstep. The
        `blockstep` should be loaded using ctypes and follow the form
        ```
        void blockstep(const TI0 *x0, const TI1* x1, ...,
                       TO0* y0, TO1* y1, ...,
                       bool* valid0, bool* valid1, ...);
        ```
        where `xi` is the input signal, `yi` is the output signal and `validi`
        indicates whether the output is given in this functon. TIi and TOi are
        the data types of the inputs and outputs defined in `types_in` and
        `types_out`. If `blockstep` is set to None, the default python routine
        in BaseBlock will be used. (default to None)

    dt: float, optional
        Sampling time of this block. The input and output be evaluate once
        for each sampling time.

    name: string, optional
        The name of this block.
    """

    def __init__(self, libname, types_in, sizes_in, types_out, sizes_out,
                 initfunc=None, outputstep=None, blockstep=None,
                 dt=None, name='CFunction'):

        self._lib = ctypes.cdll.LoadLibrary(libname)
        nin = len(types_in)
        nout = len(types_out)
        self._nin = nin
        self._nout = nout
        self._types_in = types_in
        self._types_out = types_out
        self._sizes_in = sizes_in
        self._sizes_out = sizes_out
        self._cinputs = None
        self._coutputs = None
        self._cstatus = None
        self._construct_c_structures()

        self.set_initfunc(initfunc)
        self.set_outputstep(outputstep)
        self.set_blockstep(blockstep)

        super().__init__(nin=nin, nout=nout, dt=dt, name=name)

    @property
    def clib(self):
        return self._lib

    def invoke(self, name, *args, restype=None):
        """Call C function with automatic type recognization.

        This method calls the C function in the dynamic library and
        automatically converts the arguments. The mechanics of the method
        is converting the args into numpy arrays and using
        numpy.ctypeslib.as_ctypes to convert them into C types. If the users
        want to pass values by reference, they should use numpy array objects
        as args directly or the changes to the input arguments will lose.

        Parameters
        ----------
        name: str
            Name of the C function.
        *args:
            Input parameters to the C function.
        restype: ctypes or None, optional
            The return type of the C functin. If None, the return type will be
            left as is. (default to None)

        Returns
        -------
        out: np.array
            The result of the C function.

        Notes
        -----
        The argtypes and restype of the C function will be modified by this
        method.
        """
        func = getattr(self._lib, name)
        if restype is not None:
            func.restype = restype
        cargs = []
        cargtypes = []
        for i in args:
            arg = np.asarray(i)
            carg = np.ctypeslib.as_ctypes(arg)
            cargtype = type(carg)
            cargs.append(carg)
            cargtypes.append(cargtype)
        func.argtypes = cargtypes
        res = func(*cargs)
        return np.asarray(res)

    def set_initfunc(self, name):
        """Set initfunc for the block.


        User can set the function for initfunc during runtime. However, the
        interface of the function should be consistent with that defined during
        initialization.
        """
        if name is None:
            self._initfunc = None
            return
        self._initfunc = getattr(self._lib, name)

    def set_outputstep(self, name):
        """Set outputstep for the block.

        User can set the function for outputstep during runtime. However, the
        interface of the function should be consistent with that defined during
        initialization.
        """
        if name is None:
            self._outputstep = None
            return
        self._outputstep = getattr(self._lib, name)
        self._outputstep.argtypes = [ctypes.c_uint,
                                     ctypes.c_void_p,
                                     ctypes.POINTER(ctypes.c_bool)]

    def set_blockstep(self, name):
        """Set blockstep for the block.

        User can set the function for blockstep during runtime. However, the
        interface of the function should be consistent with that defined during
        initialization.
        """
        if name is None:
            self._blockstep = None
            return
        self._blockstep = getattr(self._lib, name)
        argtypes = list([ctypes.POINTER(T) for T in self._types_in])
        argtypes.extend([ctypes.POINTER(T) for T in self._types_out])
        argtypes.extend([ctypes.POINTER(ctypes.c_bool)] * self._nout)
        self._blockstep.argtypes = argtypes

    def INITFUNC(self):
        if self._initfunc is None:
            return super().INITFUNC()
        self._initfunc()

    def OUTPUTSTEP(self, portid):
        if self._outputstep is None:
            return super().OUTPUTSTEP(portid)

        output = self._coutputs[portid]
        valid = self._cstatus[portid]
        self._outputstep(portid,
                         ctypes.cast(output, ctypes.c_void_p),
                         ctypes.byref(valid))
        if valid:
            size = self._sizes_out[portid]
            return self._read_c_variable(output, size)
        else:
            return NA

    def BLOCKSTEP(self, *xs):
        if self._blockstep is None:
            return super().BLOCKSTEP(*xs)

        # write inputs
        for i in range(self._nin):
            try:
                self._write_c_variable(xs[i], self._cinputs[i],
                                       self._sizes_in[i])
            except ValueError:
                raise SimulationError('fail to pass data from {inport} to C '
                                      'function, please check the size of the '
                                      'input'.format(inport=self._inports[i]))

        # call function in c
        self._blockstep(*self._cinputs, *self._coutputs,
                        *[ctypes.byref(i) for i in self._cstatus])

        # read results
        results = []
        for i in range(self._nout):
            if self._cstatus[i]:
                size = self._sizes_out[i]
                yi = self._read_c_variable(self._coutputs[i], size)
            else:
                yi = NA
            results.append(yi)
        return results

    def _construct_c_structures(self):
        """Construct structures for interacting with C code."""
        # Necessary to copy inputs as there's no guarantee the inputs are the
        # correct data types.
        cinputs = []
        for i in range(self._nin):
            cinputs.append(self._construct_c_structures_helper(
                self._types_in[i], self._sizes_in[i]))
        coutputs = []
        for i in range(self._nout):
            coutputs.append(self._construct_c_structures_helper(
                self._types_out[i], self._sizes_out[i]))
        cstatus = []
        for i in range(self._nout):
            cstatus.append(ctypes.c_bool())

        self._cinputs = cinputs
        self._coutputs = coutputs
        self._cstatus = cstatus

    def _construct_c_structures_helper(self, ctype, size):
        # Return a pointer to the C variable if size is NA.
        if size is NA:
            return ctypes.pointer(ctype())
        # Return an array of the C variable otherwise.
        elif not np.iterable(size):
            return (ctype * size)()
        else:
            for i in size:
                ctype = ctype * i
            return ctype()

    def _write_c_variable(self, value, cvariable, size):
        """Write value to cvariable."""
        if size is NA:
            cvariable.contents.value = value
        else:
            cvariable[:] = value

    def _read_c_variable(self, cvariable, size):
        """Read values from cvariable."""
        if size is NA:
            return np.asarray(cvariable.contents)
        else:
            return np.asarray(cvariable)
