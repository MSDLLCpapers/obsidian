"""Custom obsidian exceptions for improved error handling"""


class IncompatibleObjectiveError(Exception):
    """Exception that gets raised when the objective(s) cannot be successfully called as specified"""
    pass


class SurrogateFitError(Exception):
    """Exception that gets raised when the surrogate model fails to fit"""
    pass


class UnsupportedError(Exception):
    """Exception that gets raised when an optimization is requested on an unsupported feature"""
    pass


class UnfitError(Exception):
    """Exception that gets raised when an action is called before a model or transform has been fit"""
    pass


class DataWarning(UserWarning):
    """Warning that gets raised if there is an issue with input data"""
    pass


class OptimizerWarning(UserWarning):
    """Warning that gets raised if there is an issue with optimization configuration"""
    pass
