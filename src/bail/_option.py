"""The Option type: `Some(value)` for a value that is present, `Null()` for one
that is absent.

`Option[T]` and `Result[T, E]` (in `bail._result`) are plain unions of two
`@final` classes, so `match` narrows each arm to the right type and mypy and
pyright report an arm that is missing. The methods follow Rust's `Option` and
`Result`, whose documentation most docstrings here are adapted from.

The two types refer to each other at runtime: `Some.ok_or` builds an `Ok`, and
`Ok.ok` builds a `Some`. `bail._result` imports this module when it loads, so
the six methods here that build an `Ok` or `Err` import `bail._result` when
they are called.

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

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Final,
    Generic,
    Never,
    NoReturn,
    TypeVar,
    cast,
    final,
)

from bail._bails import Bail
from bail._panic import Panic

if TYPE_CHECKING:
    from bail._result import Err, Ok, Result

__all__ = ["Null", "Option", "Some"]

# Declared, not inferred: see the variance section above.
T = TypeVar("T", covariant=True)


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

    @abstractmethod
    def _bail_lint_q(self, sink: Callable[[Null], None], /) -> T:
        """`.Q` as bail-lint's BAIL002 rewrite spells it. Not part of the API.

        `sink` is never called. Its type is the check: the rewrite passes a
        sink typed to accept the enclosing function's declared return type, so
        the call type-checks only if every `Null()` this `.Q` can return fits it.
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
        from bail._result import Ok

        return Ok(self.value)

    def ok_or_else[E](self, err: Callable[[], E]) -> Ok[T]:
        from bail._result import Ok

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
        from bail._result import Err, Ok

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

    def _bail_lint_q(self, sink: Callable[[Never], None], /) -> T:
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
        from bail._result import Err

        return Err(err)

    def ok_or_else[E](self, err: Callable[[], E]) -> Err[E]:
        from bail._result import Err

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
        from bail._result import Ok

        return Ok(self)

    def flatten(self) -> Null:
        return self

    @property
    def Q(self) -> NoReturn:
        raise Bail(self)

    def _bail_lint_q(self, sink: Callable[[Null], None], /) -> NoReturn:
        raise Bail(self)


type Option[T] = Some[T] | Null
