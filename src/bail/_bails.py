"""Early return: `.Q` and the `@bails` decorator.

Python cannot make a method call return from its caller, so `.Q` raises
`Bail` carrying the value to return, and `@bails` catches it at the function
boundary and returns that value. `Bail` derives from `BaseException` so that
an `except Exception:` between the two does not swallow the early return.
Without `@bails` it propagates like any uncaught exception, and its message
names the missing decorator.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from bail._option import Null
    from bail._result import Err

__all__ = ["bails"]


class Bail(BaseException):
    def __init__(self, value: Err[object] | Null) -> None:
        super().__init__(
            f"`.Q` on {value!r} outside a `@bails` function; decorate the "
            "function that returns this Result or Option with `@bails`"
        )
        self.value = value


def bails[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    """Make `.Q` inside `f` return the `Err` or `Null()` from `f`.

    `f` must return a `Result` or an `Option`, and every `.Q` in it must be on
    a value of that kind with an error type that fits `f`'s. The type checkers
    do not check either; bail's lint does. Works on `async def` functions.
    """
    if inspect.iscoroutinefunction(f):

        @functools.wraps(f)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> object:
            try:
                return await f(*args, **kwargs)
            except Bail as bail:
                return bail.value

        return cast(Callable[P, R], async_wrapper)

    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return f(*args, **kwargs)
        except Bail as bail:
            # `R` is `f`'s declared Result or Option type, and the bailed value
            # is an `Err` or `Null()` of it, assuming the lint's rules hold.
            return cast(R, bail.value)

    return wrapper
