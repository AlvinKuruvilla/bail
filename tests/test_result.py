"""Result behaviour, mostly ported from rustshed's tests of Rust's doc examples."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, auto
from math import sqrt
from pathlib import Path

import pytest

from bail import Err, Null, Ok, Option, Panic, Result, Some


def parse_int(raw: str) -> Result[int, ValueError]:
    try:
        return Ok(int(raw))
    except ValueError as error:
        return Err(error)


def checked_sqrt(x: float) -> Result[float, str]:
    return Ok(sqrt(x)) if x >= 0 else Err("math domain error")


def path_stat(path: str | Path) -> Result[os.stat_result, FileNotFoundError]:
    try:
        return Ok(Path(path).stat())
    except FileNotFoundError as error:
        return Err(error)


def test_match_narrows_both_arms() -> None:
    def describe(res: Result[int, str]) -> str:
        match res:
            case Ok(value):
                return f"ok {value + 1}"
            case Err(error):
                return f"err {error.upper()}"

    assert describe(Ok(1)) == "ok 2"
    assert describe(Err("bad")) == "err BAD"


def test_ok_is_covariant() -> None:
    # Fails to type-check, not at runtime, if Ok or Err become invariant.
    def widen(res: Result[int, str]) -> Result[int | str, object]:
        return res

    assert widen(Ok(1)) == Ok(1)


def test_is_ok() -> None:
    x: Result[int, str] = Ok(-3)
    assert x.is_ok() is True

    x = Err("Some error message")
    assert x.is_ok() is False


def test_is_ok_and() -> None:
    x: Result[int, str] = Ok(2)
    assert x.is_ok_and(lambda v: v > 1) is True

    x = Ok(0)
    assert x.is_ok_and(lambda v: v > 1) is False

    x = Err("hey")
    assert x.is_ok_and(lambda v: v > 1) is False


def test_is_err() -> None:
    x: Result[int, str] = Ok(-3)
    assert x.is_err() is False

    x = Err("Some error message")
    assert x.is_err() is True


def test_is_err_and() -> None:
    class ErrorKind(Enum):
        NOT_FOUND = auto()
        PERMISSION_DENIED = auto()

    @dataclass
    class Error:
        kind: ErrorKind

    def is_not_found(error: Error) -> bool:
        return error.kind == ErrorKind.NOT_FOUND

    x: Result[int, Error] = Err(Error(ErrorKind.NOT_FOUND))
    assert x.is_err_and(is_not_found) is True

    x = Err(Error(ErrorKind.PERMISSION_DENIED))
    assert x.is_err_and(is_not_found) is False

    x = Ok(123)
    assert x.is_err_and(is_not_found) is False


def test_ok() -> None:
    x: Result[int, str] = Ok(2)
    assert x.ok() == Some(2)

    x = Err("Nothing here")
    assert x.ok() == Null()


def test_err() -> None:
    x: Result[int, str] = Ok(2)
    assert x.err() == Null()

    x = Err("Nothing here")
    assert x.err() == Some("Nothing here")


def test_map() -> None:
    numbers: list[int] = []
    for raw in ["1", "2", "E", "A", "5"]:
        match parse_int(raw).map(lambda x: x**2):
            case Ok(value):
                numbers.append(value)
            case Err(_):
                continue

    assert numbers == [1, 4, 25]


def test_map_or() -> None:
    x: Result[str, str] = Ok("foo")
    assert x.map_or(42, len) == 3

    x = Err("bar")
    assert x.map_or(42, len) == 42


def test_map_or_else() -> None:
    k = 21

    x: Result[str, str] = Ok("foo")
    assert x.map_or_else(lambda _: k * 2, len) == 3

    x = Err("bar")
    assert x.map_or_else(lambda _: k * 2, len) == 42


def test_map_err() -> None:
    def stringify(x: int) -> str:
        return f"error code: {x}"

    x: Result[int, int] = Ok(2)
    assert x.map_err(stringify) == Ok(2)

    x = Err(13)
    assert x.map_err(stringify) == Err("error code: 13")


def test_inspect(capsys: pytest.CaptureFixture[str]) -> None:
    cubed = (
        parse_int("4")
        .inspect(lambda x: print(f"original: {x}"))
        .map(lambda x: pow(x, 3))
        .expect("failed to parse number")
    )
    assert cubed == 64
    assert capsys.readouterr().out == "original: 4\n"

    assert Err("foo").inspect(print) == Err("foo")


def test_inspect_err(capsys: pytest.CaptureFixture[str]) -> None:
    path_stat("/bad/path").inspect_err(lambda e: print(f"failed: {e.strerror}"))
    assert capsys.readouterr().out == "failed: No such file or directory\n"

    assert Ok(5).inspect_err(print) == Ok(5)


def test_expect() -> None:
    x: Result[int, str] = Err("emergency failure")
    with pytest.raises(Panic, match=r"^Testing expect: 'emergency failure'$"):
        x.expect("Testing expect")

    x = Ok(42)
    assert x.expect("Should succeed") == 42


def test_unwrap() -> None:
    x: Result[int, str] = Ok(2)
    assert x.unwrap() == 2

    x = Err("emergency failure")
    with pytest.raises(Panic, match=r"'emergency failure'$"):
        x.unwrap()


def test_unwrap_chains_an_exception_error() -> None:
    # tryhard preserved exception chains; an Err built from a caught exception
    # must keep its traceback when unwrapped.
    res = parse_int("x")
    with pytest.raises(Panic) as panic:
        res.unwrap()

    assert isinstance(panic.value.__cause__, ValueError)


def test_expect_err() -> None:
    x: Result[int, str] = Ok(10)
    with pytest.raises(Panic, match=r"^Testing expect_err: 10$"):
        x.expect_err("Testing expect_err")

    x = Err("error message")
    assert x.expect_err("should fail") == "error message"


def test_unwrap_err() -> None:
    x: Result[int, str] = Ok(2)
    with pytest.raises(Panic, match=r": 2$"):
        x.unwrap_err()

    x = Err("error message")
    assert x.unwrap_err() == "error message"


def test_and_() -> None:
    x: Result[int, str] = Ok(2)
    y: Result[str, str] = Err("late error")
    assert x.and_(y) == Err("late error")

    x = Err("early error")
    y = Ok("foo")
    assert x.and_(y) == Err("early error")

    x = Err("not a 2")
    y = Err("late error")
    assert x.and_(y) == Err("not a 2")

    x = Ok(2)
    y = Ok("different result type")
    assert x.and_(y) == Ok("different result type")


def test_and_then() -> None:
    def sqrt_then_to_string(x: float) -> Result[str, str]:
        return checked_sqrt(x).map(str)

    assert Ok(16).and_then(sqrt_then_to_string) == Ok("4.0")
    assert Ok(-1).and_then(sqrt_then_to_string) == Err("math domain error")
    assert Err("not a number").and_then(sqrt_then_to_string) == Err("not a number")

    # A function that only returns Ok leaves the second error type unsolved;
    # pyright reports it as Unknown unless the target is annotated.
    root_mtime: Result[float, FileNotFoundError] = path_stat("/").and_then(
        lambda stat: Ok(stat.st_mtime)
    )
    assert root_mtime.is_ok()

    missing_mtime: Result[float, FileNotFoundError] = path_stat("/bad/path").and_then(
        lambda stat: Ok(stat.st_mtime)
    )
    assert missing_mtime.is_err()


def test_and_then_unions_the_error_types() -> None:
    def positive(x: int) -> Result[int, str]:
        return Ok(x) if x > 0 else Err("not positive")

    chained: Result[int, ValueError | str] = parse_int("-3").and_then(positive)
    assert chained == Err("not positive")


def test_or() -> None:
    x: Result[int, str] = Ok(2)
    y: Result[int, str] = Err("late error")
    assert x.or_(y) == Ok(2)

    x = Err("Early error")
    y = Ok(2)
    assert x.or_(y) == Ok(2)

    x = Err("not a 2")
    y = Err("late error")
    assert x.or_(y) == Err("late error")

    x = Ok(2)
    y = Ok(100)
    assert x.or_(y) == Ok(2)


def test_or_else() -> None:
    def sq(x: int) -> Result[int, int]:
        return Ok(x * x)

    def err(x: int) -> Result[int, int]:
        return Err(x)

    assert Ok(2).or_else(sq).or_else(sq) == Ok(2)
    assert Ok(2).or_else(err).or_else(sq) == Ok(2)
    assert Err(3).or_else(sq).or_else(err) == Ok(9)
    assert Err(3).or_else(err).or_else(err) == Err(3)


def test_unwrap_or() -> None:
    x: Result[int, str] = Ok(9)
    assert x.unwrap_or(2) == 9

    x = Err("error")
    assert x.unwrap_or(2) == 2


def test_unwrap_or_else() -> None:
    assert Ok(2).unwrap_or_else(len) == 2
    assert Err("foo").unwrap_or_else(len) == 3


def test_transpose() -> None:
    x: Result[Option[int], str] = Ok(Some(5))
    y: Option[Result[int, str]] = Some(Ok(5))
    assert x.transpose() == y

    x = Ok(Null())
    assert x.transpose() == Null()

    x = Err("foo")
    assert x.transpose() == Some(x)
