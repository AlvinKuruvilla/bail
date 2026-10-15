"""The exception raised when a Result or Option is unwrapped as the wrong variant."""

__all__ = ["Panic"]


class Panic(RuntimeError):
    """Raised by `unwrap` and `expect` when the value is not the expected variant.

    When the unexpected value is an `Err` holding an exception, that exception
    is the `__cause__`.
    """
