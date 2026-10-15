"""An Option is a value that may be absent: `Some(value)` or `Null()`."""

from bail import Null, Option, Some


def divide(numerator: float, denominator: float) -> Option[float]:
    if denominator == 0.0:
        return Null()
    return Some(numerator / denominator)


match divide(2.0, 3.0):
    case Some(x):
        print(f"Result: {x}")
    case Null():
        print("Cannot divide by 0")
