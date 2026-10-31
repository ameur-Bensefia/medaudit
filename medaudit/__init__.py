"""medaudit: a statistical audit of leakage, shortcuts and confounds in machine-learning evaluation.

Six audit modules work from predictions, splits and metadata:
  A1 split integrity · A2 identity and site shortcuts · A3 data integrity
  A4 confounds · A5 uncertainty · A6 transfer
and are summarized in a report card (medaudit.report).
"""
__version__ = "0.3.2"

from .api import audit  # noqa: E402
from .schema import Dataset  # noqa: E402

__all__ = ["audit", "Dataset"]
