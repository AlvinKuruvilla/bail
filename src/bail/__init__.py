"""A Result type for Python with exhaustive match and early return."""

from bail._bails import bails
from bail._catching import catching
from bail._option import Null, Option, Some
from bail._panic import Panic
from bail._result import Err, Ok, Result

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
