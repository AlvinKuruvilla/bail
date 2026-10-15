"""A Result type for Python with exhaustive match and early return."""

from bail._catching import catching
from bail._core import Err, Null, Ok, Option, Panic, Result, Some, bails

__all__ = [
    "Err",
    "Null",
    "Ok",
    "Option",
    "Panic",
    "Result",
    "Some",
    "bails",
    "catching",
]

__version__ = "0.1.0"
