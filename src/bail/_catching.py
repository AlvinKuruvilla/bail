"""The bridge from code that raises to code that returns a Result.

`@catching(ValueError)` turns a function that returns `T` and raises
`ValueError` into one that returns `Result[T, ValueError]`. The `Err` holds the
exception itself, traceback included, so `unwrap` can chain it as the cause.
Exceptions not listed propagate unchanged, so a bug elsewhere still crashes.

```python
@catching(OSError, json.JSONDecodeError)
def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())

load(path)  # Result[dict[str, object], OSError | JSONDecodeError]
catching(ValueError)(int)("3")  # Ok(3), for a one-off call
```

Only `Exception` subclasses can be listed. `KeyboardInterrupt`, `SystemExit`
and the exception `.Q` uses for early return derive from `BaseException`, so
they always propagate. tryhard kept an explicit list of exceptions that pass
through; the bound makes that list unnecessary.

# Typing

mypy infers a single type parameter bound by several arguments as their join,
so `catching(OSError, JSONDecodeError)` would type its errors as `Exception`.
The overloads for one to four exception types bind one parameter per argument,
which gives the union under both checkers. Four is arbitrary. Past it, mypy
falls back to the join and pyright still infers the union.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Coroutine
from typing import Any, overload

from bail._result import Err, Ok, Result

__all__ = ["catching"]


class _Catching[E: Exception]:
    __slots__ = ("_exceptions",)

    def __init__(self, exceptions: tuple[type[E], ...]) -> None:
        self._exceptions = exceptions

    # A coroutine function also matches the plain overload, which both
    # checkers report as an overlap. Order resolves it: the async form wins.
    @overload
    def __call__[**P, T](  # type: ignore[overload-overlap]
        self, f: Callable[P, Coroutine[Any, Any, T]]
    ) -> Callable[P, Coroutine[Any, Any, Result[T, E]]]: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    def __call__[**P, T](self, f: Callable[P, T]) -> Callable[P, Result[T, E]]: ...

    def __call__(self, f: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.isgeneratorfunction(f) or inspect.isasyncgenfunction(f):
            raise TypeError(
                f"@catching cannot wrap generator function {f.__qualname__}: its "
                "exceptions are raised during iteration, after the call returns. "
                "Catch them where the generator is consumed."
            )
        exceptions = self._exceptions

        if inspect.iscoroutinefunction(f):

            @functools.wraps(f)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Result[Any, E]:
                try:
                    return Ok(await f(*args, **kwargs))
                except exceptions as error:
                    return Err(error)

            return async_wrapper

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Result[Any, E]:
            try:
                return Ok(f(*args, **kwargs))
            except exceptions as error:
                return Err(error)

        return wrapper


@overload
def catching[E1: Exception](e1: type[E1], /) -> _Catching[E1]: ...
@overload
def catching[E1: Exception, E2: Exception](
    e1: type[E1], e2: type[E2], /
) -> _Catching[E1 | E2]: ...
@overload
def catching[E1: Exception, E2: Exception, E3: Exception](
    e1: type[E1], e2: type[E2], e3: type[E3], /
) -> _Catching[E1 | E2 | E3]: ...
@overload
def catching[E1: Exception, E2: Exception, E3: Exception, E4: Exception](
    e1: type[E1], e2: type[E2], e3: type[E3], e4: type[E4], /
) -> _Catching[E1 | E2 | E3 | E4]: ...
@overload
def catching[E: Exception](first: type[E], /, *rest: type[E]) -> _Catching[E]: ...


def catching(first: object, /, *rest: object) -> _Catching[Any]:
    """Wrap a function so the listed exceptions become an `Err`.

    Works as a decorator, on `def` and `async def`, or inline:
    `catching(ValueError)(int)("3")`. At least one exception type is required;
    pass `Exception` to catch every ordinary exception.

    Raises `TypeError` if an argument is not an `Exception` subclass.
    """
    # The overloads already require Exception subclasses. This check covers
    # untyped callers and the common slip of passing an instance, ValueError().
    exceptions: list[type[Exception]] = []
    for candidate in (first, *rest):
        if not (isinstance(candidate, type) and issubclass(candidate, Exception)):
            raise TypeError(
                f"catching() takes Exception subclasses, got {candidate!r}; "
                "pass the class, as in catching(ValueError)"
            )
        exceptions.append(candidate)
    return _Catching(tuple(exceptions))
