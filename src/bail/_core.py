"""Option and Result: values that either hold something or say why not.

`Option[T]` is `Some[T] | Null`: a value that may be absent. `Result[T, E]` is
`Ok[T] | Err[E]`: the outcome of an operation that may fail. Both are plain
unions of two `@final` classes, so `match` narrows each arm to the right type
and mypy and pyright report an arm that is missing:

```python
match parse_port(raw):
    case Ok(port):
        connect(port)
    case Err(reason):
        log.warning(reason)
```

The methods follow Rust's `Option` and `Result`, whose documentation most
docstrings below are adapted from.

# Variance

`Some`, `Ok` and `Err` are covariant, so an `Ok[int]` is a `Result[int | str,
E]`. `T` and `E` are therefore declared `covariant=True`, and both checkers
reject a method that takes `T` or `E` as a parameter. Rust's
`unwrap_or(default: T)` becomes `unwrap_or[U](default: U) -> T | U`, and
`and_then` accepts a function with a different error type and returns the
union of the two.

The classes declare their variance with old-style `TypeVar`s instead of PEP 695
class syntax, which infers it. Inference failed under mypy 2.4: the classes
and aliases refer to each other (`Some` subclasses `_Option`, whose methods
return `Option`, which names `Some`), and mypy inferred every class here as
invariant. pyright inferred them correctly. Declaring the variance also turns
a misplaced `T` into an error at the method instead of silent invariance.
Methods still use PEP 695 syntax for their own type parameters.

`Null` and `Ok` subclass their base with `Never` in the slot they lack: `Null`
is an `_Option[Never]`, an `Ok[T]` is a `_Result[T, Never]`. Under covariance,
`Never` is assignable to every type, so the subclasses' signatures check
against the base's for any `T` and `E`.

`Some`, `Ok` and `Err` are plain classes, not dataclasses. From Python 3.13 a
dataclass gets a generated `__replace__(self, *, value: T)`, which puts `T` in
a parameter, and pyright treats every generic dataclass as invariant when
targeting 3.13 or later (by design: microsoft/pyright#10464). The payload sits
in a `Final` slot behind a read-only property. `test_ok_is_covariant` fails
under either checker if any of this breaks.

# Provenance

Ported from rustshed 0.5.0 (MIT, Copyright (c) 2022 Paweł Rubin) at
https://github.com/pawelrubin/rustshed/tree/125993ac118d8dcb6d6a3c29bfc9b13f2d5b6989

Changes from rustshed: the variance rules above, plain
classes in place of dataclasses, `Null()` is a class rather than a
module-level instance, and `Panic` chains the `Err` value when it is an
exception.
"""

from __future__ import annotations

import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, Final, Generic, Never, NoReturn, TypeVar, cast, final

__all__ = ["Err", "Null", "Ok", "Option", "Panic", "Result", "Some", "bails"]

# Declared, not inferred: see the variance section above.
T = TypeVar("T", covariant=True)
E = TypeVar("E", covariant=True)


class Panic(RuntimeError):
    """Raised by `unwrap` and `expect` when the value is not the expected variant.

    When the unexpected value is an `Err` holding an exception, that exception
    is the `__cause__`.
    """


class _Option(ABC, Generic[T]):  # noqa: UP046 -- declared variance, see module docs
    """The methods every `Option` variant implements."""

    __slots__ = ()

    @abstractmethod
    def is_some(self) -> bool:
        """Return true if the option is a `Some`.

        This does not narrow the type. Use `match` or `isinstance(opt, Some)`.
        """

    @abstractmethod
    def is_some_and(self, f: Callable[[T], bool]) -> bool:
        """Return true if the option is a `Some` whose value satisfies `f`."""

    @abstractmethod
    def is_null(self) -> bool:
        """Return true if the option is `Null()`.

        This does not narrow the type. Use `match` or `isinstance(opt, Null)`.
        """

    @abstractmethod
    def expect(self, msg: str) -> T:
        """Return the `Some` value, or raise `Panic(msg)` if the option is `Null()`."""

    @abstractmethod
    def unwrap(self) -> T:
        """Return the `Some` value, or raise `Panic` if the option is `Null()`.

        Prefer `match` or `unwrap_or` so that the `Null()` case is handled.
        """

    @abstractmethod
    def unwrap_or[U](self, default: U) -> T | U:
        """Return the `Some` value, or `default`.

        `default` is evaluated eagerly. Use `unwrap_or_else` to compute it only
        when needed.
        """

    @abstractmethod
    def unwrap_or_else[U](self, f: Callable[[], U]) -> T | U:
        """Return the `Some` value, or the result of calling `f`."""

    @abstractmethod
    def map[U](self, f: Callable[[T], U]) -> Option[U]:
        """Apply `f` to the `Some` value, leaving `Null()` untouched."""

    @abstractmethod
    def inspect(self, f: Callable[[T], object]) -> Option[T]:
        """Call `f` with the `Some` value, if any, and return the option unchanged."""

    @abstractmethod
    def map_or[U](self, default: U, f: Callable[[T], U]) -> U:
        """Apply `f` to the `Some` value, or return `default` for `Null()`.

        `default` is evaluated eagerly. Use `map_or_else` to compute it only
        when needed.
        """

    @abstractmethod
    def map_or_else[U](self, default: Callable[[], U], f: Callable[[T], U]) -> U:
        """Apply `f` to the `Some` value, or return `default()` for `Null()`."""

    @abstractmethod
    def ok_or[E](self, err: E) -> Result[T, E]:
        """Map `Some(v)` to `Ok(v)` and `Null()` to `Err(err)`.

        `err` is evaluated eagerly. Use `ok_or_else` to compute it only when
        needed.
        """

    @abstractmethod
    def ok_or_else[E](self, err: Callable[[], E]) -> Result[T, E]:
        """Map `Some(v)` to `Ok(v)` and `Null()` to `Err(err())`."""

    @abstractmethod
    def and_[U](self, optb: Option[U]) -> Option[U]:
        """Return `Null()` if the option is `Null()`, otherwise `optb`."""

    @abstractmethod
    def and_then[U](self, f: Callable[[T], Option[U]]) -> Option[U]:
        """Return `Null()` if the option is `Null()`, otherwise `f(value)`.

        Some languages call this flatmap.
        """

    @abstractmethod
    def filter(self, predicate: Callable[[T], bool]) -> Option[T]:
        """Keep the `Some` value only if it satisfies `predicate`."""

    @abstractmethod
    def or_[U](self, optb: Option[U]) -> Option[T | U]:
        """Return the option if it is a `Some`, otherwise `optb`.

        `optb` is evaluated eagerly. Use `or_else` to compute it only when
        needed.
        """

    @abstractmethod
    def or_else[U](self, f: Callable[[], Option[U]]) -> Option[T | U]:
        """Return the option if it is a `Some`, otherwise `f()`."""

    @abstractmethod
    def xor[U](self, optb: Option[U]) -> Option[T | U]:
        """Return whichever of the option and `optb` is a `Some`, if exactly one is."""

    @abstractmethod
    def zip[U](self, other: Option[U]) -> Option[tuple[T, U]]:
        """Return `Some((a, b))` if both options are `Some`, otherwise `Null()`."""

    @abstractmethod
    def zip_with[U, R](self, other: Option[U], f: Callable[[T, U], R]) -> Option[R]:
        """Return `Some(f(a, b))` if both options are `Some`, otherwise `Null()`."""

    @abstractmethod
    def unzip[A, B](
        self: _Option[tuple[A, B]],
    ) -> tuple[Option[A], Option[B]]:
        """Split `Some((a, b))` into `(Some(a), Some(b))`, and `Null()` into two."""

    @abstractmethod
    def transpose[A, F](self: _Option[Result[A, F]]) -> Result[Option[A], F]:
        """Turn an Option of a Result into a Result of an Option.

        `Null()` becomes `Ok(Null())`, `Some(Ok(v))` becomes `Ok(Some(v))`, and
        `Some(Err(e))` becomes `Err(e)`.
        """

    @abstractmethod
    def flatten[U](self: _Option[Option[U]]) -> Option[U]:
        """Remove one level of nesting: `Some(Some(v))` becomes `Some(v)`."""

    @property
    @abstractmethod
    def Q(self) -> T:
        """Return the `Some` value, or return `Null()` from the enclosing `@bails`.

        Outside a `@bails` function, `.Q` on `Null()` raises an exception that
        names the missing decorator.
        """


@final
class Some(_Option[T]):
    """An `Option` that holds a value."""

    __slots__ = ("_value",)
    __match_args__ = ("value",)

    def __init__(self, value: T) -> None:
        self._value: Final = value

    @property
    def value(self) -> T:
        return self._value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Some):
            return False
        theirs = cast("Some[object]", other)  # isinstance leaves T unknown
        return bool(self._value == theirs._value)

    def __hash__(self) -> int:
        return hash((Some, self._value))

    def __repr__(self) -> str:
        return f"Some({self._value!r})"

    def is_some(self) -> bool:
        return True

    def is_some_and(self, f: Callable[[T], bool]) -> bool:
        return f(self.value)

    def is_null(self) -> bool:
        return False

    def expect(self, msg: str) -> T:
        return self.value

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: object) -> T:
        return self.value

    def unwrap_or_else[U](self, f: Callable[[], U]) -> T:
        return self.value

    def map[U](self, f: Callable[[T], U]) -> Some[U]:
        return Some(f(self.value))

    def inspect(self, f: Callable[[T], object]) -> Some[T]:
        f(self.value)
        return self

    def map_or[U](self, default: U, f: Callable[[T], U]) -> U:
        return f(self.value)

    def map_or_else[U](self, default: Callable[[], U], f: Callable[[T], U]) -> U:
        return f(self.value)

    def ok_or(self, err: object) -> Ok[T]:
        return Ok(self.value)

    def ok_or_else[E](self, err: Callable[[], E]) -> Ok[T]:
        return Ok(self.value)

    def and_[U](self, optb: Option[U]) -> Option[U]:
        return optb

    def and_then[U](self, f: Callable[[T], Option[U]]) -> Option[U]:
        return f(self.value)

    def filter(self, predicate: Callable[[T], bool]) -> Option[T]:
        return self if predicate(self.value) else Null()

    def or_[U](self, optb: Option[U]) -> Some[T]:
        return self

    def or_else[U](self, f: Callable[[], Option[U]]) -> Some[T]:
        return self

    def xor[U](self, optb: Option[U]) -> Option[T | U]:
        match optb:
            case Some():
                return Null()
            case Null():
                return self

    def zip[U](self, other: Option[U]) -> Option[tuple[T, U]]:
        match other:
            case Some(value):
                return Some((self.value, value))
            case Null():
                return Null()

    def zip_with[U, R](self, other: Option[U], f: Callable[[T, U], R]) -> Option[R]:
        match other:
            case Some(value):
                return Some(f(self.value, value))
            case Null():
                return Null()

    def unzip[A, B](self: Some[tuple[A, B]]) -> tuple[Some[A], Some[B]]:
        left, right = self.value
        return Some(left), Some(right)

    def transpose[A, F](self: Some[Result[A, F]]) -> Result[Option[A], F]:
        match self.value:
            case Ok(value):
                return Ok(Some(value))
            case Err(error):
                return Err(error)

    def flatten[U](self: Some[Option[U]]) -> Option[U]:
        return self.value

    @property
    def Q(self) -> T:
        return self._value


@final
class Null(_Option[Never]):
    """An `Option` that holds nothing.

    `Null()` always returns the same instance, the way `type(None)()` returns
    `None`, so it is safe to call wherever an empty Option is needed. Match it
    with `case Null():`. A bare `case Null:` is a capture pattern and matches
    anything.
    """

    __slots__ = ()
    _instance: ClassVar[Null | None] = None

    def __new__(cls) -> Null:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Null()"

    def is_some(self) -> bool:
        return False

    def is_some_and(self, f: Callable[[Never], bool]) -> bool:
        return False

    def is_null(self) -> bool:
        return True

    def expect(self, msg: str) -> NoReturn:
        raise Panic(msg)

    def unwrap(self) -> NoReturn:
        raise Panic("called `unwrap` on `Null()`")

    def unwrap_or[U](self, default: U) -> U:
        return default

    def unwrap_or_else[U](self, f: Callable[[], U]) -> U:
        return f()

    def map[U](self, f: Callable[[Never], U]) -> Null:
        return self

    def inspect(self, f: Callable[[Never], object]) -> Null:
        return self

    def map_or[U](self, default: U, f: Callable[[Never], U]) -> U:
        return default

    def map_or_else[U](self, default: Callable[[], U], f: Callable[[Never], U]) -> U:
        return default()

    def ok_or[E](self, err: E) -> Err[E]:
        return Err(err)

    def ok_or_else[E](self, err: Callable[[], E]) -> Err[E]:
        return Err(err())

    def and_[U](self, optb: Option[U]) -> Null:
        return self

    def and_then[U](self, f: Callable[[Never], Option[U]]) -> Null:
        return self

    def filter(self, predicate: Callable[[Never], bool]) -> Null:
        return self

    def or_[U](self, optb: Option[U]) -> Option[U]:
        return optb

    def or_else[U](self, f: Callable[[], Option[U]]) -> Option[U]:
        return f()

    def xor[U](self, optb: Option[U]) -> Option[U]:
        return optb

    def zip[U](self, other: Option[U]) -> Null:
        return self

    def zip_with[U, R](self, other: Option[U], f: Callable[[Never, U], R]) -> Null:
        return self

    def unzip(self) -> tuple[Null, Null]:
        return self, self

    def transpose(self) -> Ok[Null]:
        return Ok(self)

    def flatten(self) -> Null:
        return self

    @property
    def Q(self) -> NoReturn:
        raise _Bail(self)


type Option[T] = Some[T] | Null


class _Result(ABC, Generic[T, E]):  # noqa: UP046 -- declared variance, see module docs
    """The methods every `Result` variant implements."""

    __slots__ = ()

    @abstractmethod
    def is_ok(self) -> bool:
        """Return true if the result is an `Ok`.

        This does not narrow the type. Use `match` or `isinstance(res, Ok)`.
        """

    @abstractmethod
    def is_ok_and(self, f: Callable[[T], bool]) -> bool:
        """Return true if the result is an `Ok` whose value satisfies `f`."""

    @abstractmethod
    def is_err(self) -> bool:
        """Return true if the result is an `Err`.

        This does not narrow the type. Use `match` or `isinstance(res, Err)`.
        """

    @abstractmethod
    def is_err_and(self, f: Callable[[E], bool]) -> bool:
        """Return true if the result is an `Err` whose error satisfies `f`."""

    @abstractmethod
    def ok(self) -> Option[T]:
        """Return the `Ok` value as a `Some`, discarding any error."""

    @abstractmethod
    def err(self) -> Option[E]:
        """Return the `Err` value as a `Some`, discarding any success value."""

    @abstractmethod
    def map[U](self, f: Callable[[T], U]) -> Result[U, E]:
        """Apply `f` to the `Ok` value, leaving an `Err` untouched."""

    @abstractmethod
    def map_or[U](self, default: U, f: Callable[[T], U]) -> U:
        """Apply `f` to the `Ok` value, or return `default` for an `Err`.

        `default` is evaluated eagerly. Use `map_or_else` to compute it only
        when needed.
        """

    @abstractmethod
    def map_or_else[U](self, default: Callable[[E], U], f: Callable[[T], U]) -> U:
        """Apply `f` to the `Ok` value, or `default` to the `Err` value."""

    @abstractmethod
    def map_err[F](self, f: Callable[[E], F]) -> Result[T, F]:
        """Apply `f` to the `Err` value, leaving an `Ok` untouched."""

    @abstractmethod
    def inspect(self, f: Callable[[T], object]) -> Result[T, E]:
        """Call `f` with the `Ok` value, if any, and return the result unchanged."""

    @abstractmethod
    def inspect_err(self, f: Callable[[E], object]) -> Result[T, E]:
        """Call `f` with the `Err` value, if any, and return the result unchanged."""

    @abstractmethod
    def expect(self, msg: str) -> T:
        """Return the `Ok` value, or raise `Panic` naming `msg` and the error."""

    @abstractmethod
    def unwrap(self) -> T:
        """Return the `Ok` value, or raise `Panic` naming the error.

        Prefer `match` or `unwrap_or` so that the `Err` case is handled.
        """

    @abstractmethod
    def expect_err(self, msg: str) -> E:
        """Return the `Err` value, or raise `Panic` naming `msg` and the value."""

    @abstractmethod
    def unwrap_err(self) -> E:
        """Return the `Err` value, or raise `Panic` naming the value."""

    @abstractmethod
    def and_[U, F](self, res: Result[U, F]) -> Result[U, E | F]:
        """Return `res` if the result is an `Ok`, otherwise the result's `Err`."""

    @abstractmethod
    def and_then[U, F](self, f: Callable[[T], Result[U, F]]) -> Result[U, E | F]:
        """Return `f(value)` if the result is an `Ok`, otherwise the result's `Err`.

        The error type of the chain is the union of both error types.
        """

    @abstractmethod
    def or_[U, F](self, res: Result[U, F]) -> Result[T | U, F]:
        """Return the result if it is an `Ok`, otherwise `res`.

        `res` is evaluated eagerly. Use `or_else` to compute it only when
        needed.
        """

    @abstractmethod
    def or_else[U, F](self, f: Callable[[E], Result[U, F]]) -> Result[T | U, F]:
        """Return the result if it is an `Ok`, otherwise `f(error)`."""

    @abstractmethod
    def unwrap_or[U](self, default: U) -> T | U:
        """Return the `Ok` value, or `default`.

        `default` is evaluated eagerly. Use `unwrap_or_else` to compute it only
        when needed.
        """

    @abstractmethod
    def unwrap_or_else[U](self, f: Callable[[E], U]) -> T | U:
        """Return the `Ok` value, or the result of calling `f` with the error."""

    @abstractmethod
    def transpose[A](self: _Result[Option[A], E]) -> Option[Result[A, E]]:
        """Turn a Result of an Option into an Option of a Result.

        `Ok(Null())` becomes `Null()`, `Ok(Some(v))` becomes `Some(Ok(v))`, and
        `Err(e)` becomes `Some(Err(e))`.
        """

    @property
    @abstractmethod
    def Q(self) -> T:
        """Return the `Ok` value, or return the `Err` from the enclosing `@bails`.

        Outside a `@bails` function, `.Q` on an `Err` raises an exception that
        names the missing decorator.
        """


@final
class Ok(_Result[T, Never]):
    """A `Result` that succeeded."""

    __slots__ = ("_value",)
    __match_args__ = ("value",)

    def __init__(self, value: T) -> None:
        self._value: Final = value

    @property
    def value(self) -> T:
        return self._value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ok):
            return False
        theirs = cast("Ok[object]", other)  # isinstance leaves T unknown
        return bool(self._value == theirs._value)

    def __hash__(self) -> int:
        return hash((Ok, self._value))

    def __repr__(self) -> str:
        return f"Ok({self._value!r})"

    def is_ok(self) -> bool:
        return True

    def is_ok_and(self, f: Callable[[T], bool]) -> bool:
        return f(self.value)

    def is_err(self) -> bool:
        return False

    def is_err_and(self, f: Callable[[Never], bool]) -> bool:
        return False

    def ok(self) -> Some[T]:
        return Some(self.value)

    def err(self) -> Null:
        return Null()

    def map[U](self, f: Callable[[T], U]) -> Ok[U]:
        return Ok(f(self.value))

    def map_or[U](self, default: U, f: Callable[[T], U]) -> U:
        return f(self.value)

    def map_or_else[U](self, default: Callable[[Never], U], f: Callable[[T], U]) -> U:
        return f(self.value)

    def map_err[F](self, f: Callable[[Never], F]) -> Ok[T]:
        return self

    def inspect(self, f: Callable[[T], object]) -> Ok[T]:
        f(self.value)
        return self

    def inspect_err(self, f: Callable[[Never], object]) -> Ok[T]:
        return self

    def expect(self, msg: str) -> T:
        return self.value

    def unwrap(self) -> T:
        return self.value

    def expect_err(self, msg: str) -> NoReturn:
        raise Panic(f"{msg}: {self.value!r}")

    def unwrap_err(self) -> NoReturn:
        raise Panic(f"called `unwrap_err` on an `Ok` value: {self.value!r}")

    def and_[U, F](self, res: Result[U, F]) -> Result[U, F]:
        return res

    def and_then[U, F](self, f: Callable[[T], Result[U, F]]) -> Result[U, F]:
        return f(self.value)

    def or_[U, F](self, res: Result[U, F]) -> Ok[T]:
        return self

    def or_else[U, F](self, f: Callable[[Never], Result[U, F]]) -> Ok[T]:
        return self

    def unwrap_or(self, default: object) -> T:
        return self.value

    def unwrap_or_else[U](self, f: Callable[[Never], U]) -> T:
        return self.value

    def transpose[A](self: Ok[Option[A]]) -> Option[Ok[A]]:
        match self.value:
            case Some(value):
                return Some(Ok(value))
            case Null():
                return Null()

    @property
    def Q(self) -> T:
        return self._value


@final
class Err(_Result[Never, E]):
    """A `Result` that failed."""

    __slots__ = ("_error",)
    __match_args__ = ("error",)

    def __init__(self, error: E) -> None:
        self._error: Final = error

    @property
    def error(self) -> E:
        return self._error

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Err):
            return False
        theirs = cast("Err[object]", other)  # isinstance leaves T unknown
        return bool(self._error == theirs._error)

    def __hash__(self) -> int:
        return hash((Err, self._error))

    def __repr__(self) -> str:
        return f"Err({self._error!r})"

    def is_ok(self) -> bool:
        return False

    def is_ok_and(self, f: Callable[[Never], bool]) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def is_err_and(self, f: Callable[[E], bool]) -> bool:
        return f(self.error)

    def ok(self) -> Null:
        return Null()

    def err(self) -> Some[E]:
        return Some(self.error)

    def map[U](self, f: Callable[[Never], U]) -> Err[E]:
        return self

    def map_or[U](self, default: U, f: Callable[[Never], U]) -> U:
        return default

    def map_or_else[U](self, default: Callable[[E], U], f: Callable[[Never], U]) -> U:
        return default(self.error)

    def map_err[F](self, f: Callable[[E], F]) -> Err[F]:
        return Err(f(self.error))

    def inspect(self, f: Callable[[Never], object]) -> Err[E]:
        return self

    def inspect_err(self, f: Callable[[E], object]) -> Err[E]:
        f(self.error)
        return self

    def expect(self, msg: str) -> NoReturn:
        self._panic(f"{msg}: {self.error!r}")

    def unwrap(self) -> NoReturn:
        self._panic(f"called `unwrap` on an `Err` value: {self.error!r}")

    def expect_err(self, msg: str) -> E:
        return self.error

    def unwrap_err(self) -> E:
        return self.error

    def and_[U, F](self, res: Result[U, F]) -> Err[E]:
        return self

    def and_then[U, F](self, f: Callable[[Never], Result[U, F]]) -> Err[E]:
        return self

    def or_[U, F](self, res: Result[U, F]) -> Result[U, F]:
        return res

    def or_else[U, F](self, f: Callable[[E], Result[U, F]]) -> Result[U, F]:
        return f(self.error)

    def unwrap_or[U](self, default: U) -> U:
        return default

    def unwrap_or_else[U](self, f: Callable[[E], U]) -> U:
        return f(self.error)

    def transpose(self) -> Some[Err[E]]:
        return Some(self)

    @property
    def Q(self) -> NoReturn:
        raise _Bail(self)

    def _panic(self, message: str) -> NoReturn:
        # Chaining keeps the original traceback when the Err came from a
        # caught exception, which is the common case for the exception bridge.
        if isinstance(self.error, BaseException):
            raise Panic(message) from self.error
        raise Panic(message)


type Result[T, E] = Ok[T] | Err[E]


# Early return
#
# Python cannot make a method call return from its caller, so `.Q` raises
# `_Bail` carrying the value to return, and `@bails` catches it at the function
# boundary and returns that value. `_Bail` derives from `BaseException` so that
# an `except Exception:` between the two does not swallow the early return.
# Without `@bails` it propagates like any uncaught exception, and its message
# names the missing decorator.


class _Bail(BaseException):
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
            except _Bail as bail:
                return bail.value

        return cast(Callable[P, R], async_wrapper)

    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return f(*args, **kwargs)
        except _Bail as bail:
            # `R` is `f`'s declared Result or Option type, and the bailed value
            # is an `Err` or `Null()` of it, assuming the lint's rules hold.
            return cast(R, bail.value)

    return wrapper
