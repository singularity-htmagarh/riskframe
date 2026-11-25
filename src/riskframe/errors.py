"""Exception hierarchy. Everything riskframe raises derives from RiskFrameError."""
from __future__ import annotations


class RiskFrameError(Exception):
    """Base class for all riskframe errors."""


class DataValidationError(RiskFrameError, ValueError):
    """Input data failed validation (NaNs, duplicates, misalignment, ...)."""


class InsufficientDataError(DataValidationError):
    """Not enough observations to compute a statistic reliably."""


class MandateError(RiskFrameError, ValueError):
    """A mandate, limit or risk budget is mis-specified."""


class MissingInputError(RiskFrameError):
    """A calculation needs an input (benchmark, classification, ...) that was not supplied."""
