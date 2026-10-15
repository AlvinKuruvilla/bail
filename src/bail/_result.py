"""The Result type: `Ok(value)` for an operation that succeeded, `Err(error)` for
one that failed.

The variance rules and provenance in `bail._option` apply here unchanged:
`T` and `E` are declared covariant, every method takes a fresh type parameter
where Rust takes `T` or `E`, and `Ok` and `Err` are plain classes because a
generic dataclass is invariant under pyright on Python 3.13+.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Final, Generic, Never, NoReturn, TypeVar, cast, final

from bail._bails import Bail
from bail._option import Null, Option, Some
from bail._panic import Panic

__all__ = ["Err", "Ok", "Result"]

# Declared, not inferred: see the variance section of `bail._option`.
T = TypeVar("T", covariant=True)
E = TypeVar("E", covariant=True)


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

    @abstractmethod
    def _bail_lint_q(self, sink: Callable[[Err[E]], None], /) -> T:
        """`.Q` as bail-lint's BAIL002 rewrite spells it. Not part of the API.

        `sink` is never called. Its type is the check: the rewrite passes a
        sink typed to accept the enclosing function's declared return type, so
        the call type-checks only if every `Err` this `.Q` can return fits it.
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

    def _bail_lint_q(self, sink: Callable[[Err[Never]], None], /) -> T:
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
        raise Bail(self)

    def _bail_lint_q(self, sink: Callable[[Err[E]], None], /) -> NoReturn:
        raise Bail(self)

    def _panic(self, message: str) -> NoReturn:
        # Chaining keeps the original traceback when the Err came from a
        # caught exception, which is the common case for the exception bridge.
        if isinstance(self.error, BaseException):
            raise Panic(message) from self.error
        raise Panic(message)


type Result[T, E] = Ok[T] | Err[E]
