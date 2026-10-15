"""Early return with `.Q` inside `@bails`."""

from __future__ import annotations

import asyncio

import pytest

from bail import Err, Null, Ok, Option, Result, Some, bails


def parse_digit(raw: str) -> Result[int, str]:
    return Ok(int(raw)) if raw.isdigit() else Err(f"not a digit: {raw!r}")


def first_char(raw: str) -> Option[str]:
    return Some(raw[0]) if raw else Null()


@bails
def add(a: str, b: str) -> Result[int, str]:
    return Ok(parse_digit(a).Q + parse_digit(b).Q)


def test_q_returns_the_ok_value() -> None:
    assert add("1", "2") == Ok(3)


def test_q_returns_the_first_err_from_the_function() -> None:
    assert add("x", "y") == Err("not a digit: 'x'")
    assert add("1", "y") == Err("not a digit: 'y'")


def test_q_returns_the_same_err_object() -> None:
    failure: Result[int, str] = Err("boom")

    @bails
    def passthrough() -> Result[int, str]:
        return Ok(failure.Q)

    assert passthrough() is failure


def test_q_on_option_returns_null() -> None:
    @bails
    def initials(first: str, last: str) -> Option[str]:
        return Some(first_char(first).Q + first_char(last).Q)

    assert initials("Ada", "Lovelace") == Some("AL")
    assert initials("", "Lovelace") is Null()


def test_except_exception_does_not_swallow_the_early_return() -> None:
    # The early return travels as an exception. If it derived from Exception,
    # this handler would catch it and the function would return Ok(-1).
    @bails
    def guarded(raw: str) -> Result[int, str]:
        try:
            value = parse_digit(raw).Q
        except Exception:
            return Ok(-1)
        return Ok(value)

    assert guarded("x") == Err("not a digit: 'x'")


def test_q_without_bails_names_the_missing_decorator() -> None:
    def undecorated() -> Result[int, str]:
        return Ok(parse_digit("x").Q)  # bail: ignore

    with pytest.raises(BaseException, match="outside a `@bails` function"):
        undecorated()


def test_q_in_an_undecorated_helper_returns_from_the_nearest_bails() -> None:
    # Known behaviour, shared with rustshed: the early return goes to the
    # nearest `@bails` up the stack, not to the function containing `.Q`. The
    # lint reports the `.Q` in `helper` as missing its decorator.
    def helper(raw: str) -> int:
        return parse_digit(raw).Q  # bail: ignore

    @bails
    def outer(raw: str) -> Result[int, str]:
        return Ok(helper(raw) * 10)

    assert outer("x") == Err("not a digit: 'x'")


def test_bails_keeps_the_function_metadata() -> None:
    assert add.__name__ == "add"
    assert add.__wrapped__ is not None  # type: ignore[attr-defined]


def test_bails_on_async_functions() -> None:
    async def fetch(raw: str) -> Result[int, str]:
        await asyncio.sleep(0)
        return parse_digit(raw)

    @bails
    async def total(a: str, b: str) -> Result[int, str]:
        return Ok((await fetch(a)).Q + (await fetch(b)).Q)

    assert asyncio.run(total("1", "2")) == Ok(3)
    assert asyncio.run(total("1", "z")) == Err("not a digit: 'z'")
