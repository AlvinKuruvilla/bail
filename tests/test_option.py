"""Option behaviour, mostly ported from rustshed's tests of Rust's doc examples."""

from __future__ import annotations

import copy
import pickle
import re
from dataclasses import dataclass
from math import sqrt

import pytest

from bail import Err, Null, Ok, Option, Panic, Result, Some


def checked_sqrt(x: float) -> Option[float]:
    return Some(sqrt(x)) if x >= 0 else Null()


def get[T](xs: list[T], index: int) -> Option[T]:
    return Some(xs[index]) if 0 <= index < len(xs) else Null()


def test_null_is_a_singleton() -> None:
    assert Null() is Null()
    assert copy.copy(Null()) is Null()
    assert copy.deepcopy(Null()) is Null()
    assert pickle.loads(pickle.dumps(Null())) is Null()


def test_null_repr_matches_how_it_is_written() -> None:
    assert repr(Null()) == "Null()"
    assert repr(Some(Null())) == "Some(Null())"


def test_match_narrows_both_arms() -> None:
    def describe(opt: Option[int]) -> str:
        match opt:
            case Some(value):
                return f"some {value + 1}"
            case Null():
                return "null"

    assert describe(Some(1)) == "some 2"
    assert describe(Null()) == "null"


def test_is_some() -> None:
    x: Option[int] = Some(2)
    assert x.is_some() is True

    x = Null()
    assert x.is_some() is False


def test_is_some_and() -> None:
    def greater_than_one(x: int) -> bool:
        return x > 1

    x: Option[int] = Some(2)
    assert x.is_some_and(greater_than_one) is True

    x = Some(0)
    assert x.is_some_and(greater_than_one) is False

    x = Null()
    assert x.is_some_and(greater_than_one) is False


def test_is_null() -> None:
    x: Option[int] = Some(2)
    assert x.is_null() is False

    x = Null()
    assert x.is_null() is True


def test_expect() -> None:
    x: Option[str] = Some("value")
    assert x.expect("fruits are healthy") == "value"

    x = Null()
    with pytest.raises(Panic, match=r"^fruits are healthy$"):
        x.expect("fruits are healthy")


def test_unwrap() -> None:
    x: Option[str] = Some("air")
    assert x.unwrap() == "air"

    x = Null()
    with pytest.raises(Panic, match=re.escape("`unwrap` on `Null()`")):
        x.unwrap()


def test_unwrap_or() -> None:
    assert Some("car").unwrap_or("bike") == "car"
    assert Null().unwrap_or("bike") == "bike"


def test_unwrap_or_else() -> None:
    k = 10
    assert Some(4).unwrap_or_else(lambda: 2 * k) == 4
    assert Null().unwrap_or_else(lambda: 2 * k) == 20


def test_map() -> None:
    assert Some("Hello, World!").map(len) == Some(13)
    assert Null().map(len) == Null()


def test_inspect(capsys: pytest.CaptureFixture[str]) -> None:
    v = [1, 2, 3, 4, 5]

    _x: Option[int] = get(v, 3).inspect(lambda x: print(f"got {x}"))
    _x = get(v, 5).inspect(lambda x: print(f"got {x}"))

    assert capsys.readouterr().out == "got 4\n"


def test_map_or() -> None:
    x: Option[str] = Some("foo")
    assert x.map_or(42, len) == 3

    x = Null()
    assert x.map_or(42, len) == 42


def test_map_or_else() -> None:
    k = 21

    x: Option[str] = Some("foo")
    assert x.map_or_else(lambda: 2 * k, len) == 3

    x = Null()
    assert x.map_or_else(lambda: 2 * k, len) == 42


def test_ok_or() -> None:
    x: Option[str] = Some("foo")
    assert x.ok_or(0) == Ok("foo")

    x = Null()
    assert x.ok_or(0) == Err(0)


def test_ok_or_else() -> None:
    x: Option[str] = Some("foo")
    assert x.ok_or_else(lambda: 0) == Ok("foo")

    x = Null()
    assert x.ok_or_else(lambda: 0) == Err(0)


def test_and() -> None:
    x: Option[int] = Some(2)
    y: Option[str] = Null()
    assert x.and_(y) == Null()

    x = Null()
    y = Some("foo")
    assert x.and_(y) == Null()

    x = Some(2)
    y = Some("foo")
    assert x.and_(y) == Some("foo")

    x = Null()
    y = Null()
    assert x.and_(y) == Null()


def test_and_then() -> None:
    def sqrt_then_to_string(x: float) -> Option[str]:
        return checked_sqrt(x).map(str)

    assert Some(16).and_then(sqrt_then_to_string) == Some("4.0")
    assert Some(-1).and_then(sqrt_then_to_string) == Null()
    assert Null().and_then(sqrt_then_to_string) == Null()

    arr_2d = [["A0", "A1"], ["B0", "B1"]]

    item_0_1 = get(arr_2d, 0).and_then(lambda row: get(row, 1))
    assert item_0_1 == Some("A1")

    item_2_0 = get(arr_2d, 2).and_then(lambda row: get(row, 0))
    assert item_2_0 == Null()


def test_filter() -> None:
    def is_even(n: int) -> bool:
        return n % 2 == 0

    assert Null().filter(is_even) == Null()
    assert Some(3).filter(is_even) == Null()
    assert Some(4).filter(is_even) == Some(4)


def test_or() -> None:
    x: Option[int] = Some(2)
    y: Option[int] = Null()
    assert x.or_(y) == Some(2)

    x = Null()
    y = Some(100)
    assert x.or_(y) == Some(100)

    x = Some(2)
    y = Some(100)
    assert x.or_(y) == Some(2)

    x = Null()
    y = Null()
    assert x.or_(y) == Null()


def test_or_else() -> None:
    def nobody() -> Option[str]:
        return Null()

    def vikings() -> Option[str]:
        return Some("vikings")

    assert Some("barbarians").or_else(vikings) == Some("barbarians")
    assert Null().or_else(vikings) == Some("vikings")
    assert Null().or_else(nobody) == Null()


def test_xor() -> None:
    x: Option[int] = Some(2)
    y: Option[int] = Null()
    assert x.xor(y) == Some(2)

    x = Null()
    y = Some(2)
    assert x.xor(y) == Some(2)

    x = Some(2)
    y = Some(2)
    assert x.xor(y) == Null()

    x = Null()
    y = Null()
    assert x.xor(y) == Null()


def test_zip() -> None:
    x = Some(1)
    y = Some("hi")
    z = Null()

    assert x.zip(y) == Some((1, "hi"))
    assert x.zip(z) == Null()
    assert z.zip(x) == Null()


def test_zip_with() -> None:
    @dataclass
    class Point:
        x: float
        y: float

    x = Some(17.5)
    y = Some(42.5)

    assert x.zip_with(y, Point) == Some(Point(17.5, 42.5))
    assert x.zip_with(Null(), Point) == Null()
    assert Null().zip_with(x, Point) == Null()
    assert Null().zip_with(Null(), Point) == Null()


def test_unzip() -> None:
    assert Some((1, "hi")).unzip() == (Some(1), Some("hi"))
    assert Null().unzip() == (Null(), Null())


def test_transpose() -> None:
    x: Result[Option[int], str] = Ok(Some(5))
    y: Option[Result[int, str]] = Some(Ok(5))
    assert y.transpose() == x

    y = Some(Err("foo"))
    x = Err("foo")
    assert y.transpose() == x

    y = Null()
    assert y.transpose() == Ok(Null())


def test_flatten() -> None:
    x: Option[Option[int]] = Some(Some(6))
    assert x.flatten() == Some(6)

    x = Some(Null())
    assert x.flatten() == Null()

    x = Null()
    assert x.flatten() == Null()

    z: Option[Option[Option[int]]] = Some(Some(Some(6)))
    assert z.flatten() == Some(Some(6))
    assert z.flatten().flatten() == Some(6)
