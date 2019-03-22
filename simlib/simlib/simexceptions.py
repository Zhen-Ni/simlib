#!/usr/bin/env python3


class simException(Exception):

    """Base class for exceptions in this module."""
    pass


class SetupError(simException):

    """Exception raised for errors while building up the system."""

    def __init__(self, message):
        self.message = message


class InitializationError(simException):

    """Exception raised for errors while initializing up the system."""

    def __init__(self, message):
        self.message = message


class StepError(simException):

    """Exception raised for errors while stepping forward the system."""

    def __init__(self, message):
        self.message = message


class StopSimulation(simException):

    """Exception raised for errors while stepping forward the system."""
    pass


class DefinitionError(simException):

    """Exception raised for errors while programming user-defined blocks."""

    def __init__(self, message):
        self.message = message


class SimulationError(simException):

    """Exception raised for errors while running simulations."""

    def __init__(self, message):
        self.message = message
