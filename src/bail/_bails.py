"""Early return and declared exceptions: `.Q` and the `@bails` decorator.

`@bails` does two things, both driven by the function's signature:

- **Early return.** Python cannot make a method call return from its caller,
  so `.Q` raises `Bail` carrying the value to return, and `@bails` catches it
  at the function boundary and returns that value. `Bail` derives from
  `BaseException` so that an `except Exception:` between the two does not
  swallow the early return. Without `@bails` it propagates like any uncaught
  exception, and its message names the missing decorator.
- **Declared exceptions.** When the function is declared to return
  `Result[T, E]`, an exception raised inside it whose class is named in `E`
  comes back as `Err(exception)`. The signature already says the function
  returns that error; `@bails` makes a raised one arrive the same way.
  Exceptions not named in `E` propagate unchanged.

```python
@bails
def load(path: Path) -> Result[Config, OSError | JSONDecodeError]:
    return Ok(Config(**json.loads(path.read_text())))  # OSError -> Err(OSError)
```

# Reading the declared error type

The return annotation is read on the first call and cached, not when the
decorator runs. A decorator runs as soon as its `def` executes, before the
rest of the module exists, so an exception class defined further down would
not yet be resolvable. Resolving at decoration time would also force a
Python 3.15 `lazy` import (PEP 810) to load as soon as the module is
imported, which undoes the laziness.

Only the return annotation is evaluated, never the parameters', so a
parameter type imported only under `TYPE_CHECKING` does no harm. A name in the
return annotation that cannot be resolved at runtime, typically an exception
imported only under `TYPE_CHECKING`, raises `TypeError` on the first call
with the fix in the message. bail-lint's BAIL003 reports the same mistake
before the code runs.

On Python 3.12 and 3.13, a string annotation (from
`from __future__ import annotations`) is evaluated against the module's
globals and the function's type parameters, so it cannot name a class defined
inside an enclosing function. Python 3.14's `annotationlib` lifts that limit.
"""

from __future__ import annotations

import functools
import inspect
import sys
import types
import typing
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAliasType, cast

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
    """Make `.Q` and the exceptions `f` declares return an `Err` from `f`.

    `.Q` inside `f` returns the `Err` or `Null()` it was called on. If `f` is
    declared to return `Result[T, E]`, an exception raised in `f` whose class
    is named in `E` is returned as `Err(exception)`; other exceptions
    propagate. Works on `async def` functions.

    Raises `TypeError` on the first call if `f`'s return annotation names
    something that does not exist at runtime, or names an exception that is
    not an `Exception` subclass.
    """
    # Filled on the first call; see the module docs for why not now.
    declared: list[tuple[type[Exception], ...]] = []

    def caught() -> tuple[type[Exception], ...]:
        if not declared:
            declared.append(_declared_exceptions(f))
        return declared[0]

    if inspect.iscoroutinefunction(f):

        @functools.wraps(f)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> object:
            exceptions = caught()
            try:
                return await f(*args, **kwargs)
            except Bail as bail:
                return bail.value
            except exceptions as error:
                return _err(error)

        return cast(Callable[P, R], async_wrapper)

    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        exceptions = caught()
        try:
            return f(*args, **kwargs)
        except Bail as bail:
            # `R` is `f`'s declared Result or Option type, and the bailed value
            # is an `Err` or `Null()` of it, assuming the lint's rules hold.
            return cast(R, bail.value)
        except exceptions as error:
            # `error`'s class is named in `R`'s error type, so `Err(error)` is
            # one of `R`'s values.
            return cast(R, _err(error))

    return wrapper


def _err(error: Exception) -> object:
    # `bail._result` imports this module when it loads, so `Err` is imported
    # when first needed rather than at the top.
    from bail._result import Err

    return Err(error)


def _declared_exceptions(f: Callable[..., object]) -> tuple[type[Exception], ...]:
    """Return the `Exception` classes named in `f`'s declared `Result` error type.

    Returns an empty tuple when `f` has no return annotation or does not
    return a `Result`.
    """
    annotation = _return_annotation(f)
    if annotation is None:
        return ()
    error_type = _result_error_type(annotation)
    if error_type is None:
        return ()

    exceptions: list[type[Exception]] = []
    for member in _union_members(error_type):
        if not isinstance(member, type) or not issubclass(member, BaseException):
            continue  # a value type such as `str`, or a type variable
        if not issubclass(member, Exception):
            raise TypeError(
                f"{f.__qualname__} declares {member.__name__} in its error type, "
                "but @bails only turns Exception subclasses into an Err; "
                f"{member.__name__} must keep propagating"
            )
        exceptions.append(member)
    return tuple(exceptions)


def _return_annotation(f: Callable[..., object]) -> object | None:
    """Evaluate `f`'s return annotation alone, leaving the parameters' untouched."""
    if sys.version_info >= (3, 14):
        import annotationlib

        # FORWARDREF leaves names that do not resolve as ForwardRef objects
        # instead of raising, so parameters annotated with TYPE_CHECKING-only
        # types do not stop us reading the return annotation.
        annotations = annotationlib.get_annotations(
            f, format=annotationlib.Format.FORWARDREF
        )
    else:
        # A plain dict: indexing it evaluates nothing.
        annotations = getattr(f, "__annotations__", {})
    if "return" not in annotations:
        return None
    annotation: object = annotations["return"]

    source: str | None = None
    if isinstance(annotation, str):
        source = annotation
    elif isinstance(annotation, typing.ForwardRef):
        source = annotation.__forward_arg__
    if source is None:
        return annotation

    names: dict[str, Any] = {
        param.__name__: param for param in getattr(f, "__type_params__", ())
    }
    try:
        # The same evaluation `typing.get_type_hints` performs: the source is
        # the annotation text the function's author wrote.
        evaluated: object = eval(source, getattr(f, "__globals__", {}), names)
    except NameError as error:
        raise TypeError(
            f"@bails cannot read {f.__qualname__}'s return type {source!r}: "
            f"{error}. Import the name at runtime rather than under "
            '`if TYPE_CHECKING:`, or, if ruff moved it there, add "bail.bails" '
            "to ruff's lint.flake8-type-checking.runtime-evaluated-decorators"
        ) from error
    return evaluated


def _result_error_type(annotation: object) -> object | None:
    """Return `E` if `annotation` is `Result[T, E]` or a `type` alias of one."""
    from bail._result import Result

    while True:
        origin = typing.get_origin(annotation)
        if origin is Result:
            args = typing.get_args(annotation)
            return args[1] if len(args) == 2 else None
        if isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__  # `type Loaded = Result[...]`
        elif isinstance(origin, TypeAliasType):
            annotation = origin.__value__  # a subscripted generic alias
        else:
            return None


def _union_members(error_type: object) -> list[object]:
    """Flatten `error_type` into its union members, expanding `type` aliases."""
    if isinstance(error_type, TypeAliasType):
        return _union_members(error_type.__value__)
    if typing.get_origin(error_type) in (typing.Union, types.UnionType):
        return [
            member
            for arg in typing.get_args(error_type)
            for member in _union_members(arg)
        ]
    return [error_type]
