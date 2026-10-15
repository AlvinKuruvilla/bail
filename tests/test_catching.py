"""Turning raised exceptions into a Result with `@catching`."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest

from bail import Err, Ok, Panic, Result, catching


@catching(ValueError)
def parse(raw: str) -> int:
    return int(raw)


@catching(KeyError, json.JSONDecodeError)
def port_from(raw: str) -> int:
    config: dict[str, Any] = json.loads(raw)
    return int(config["port"])


def test_returns_ok_when_nothing_raises() -> None:
    assert parse("3") == Ok(3)


def test_returns_err_holding_the_exception() -> None:
    match parse("x"):
        case Err(error):
            assert isinstance(error, ValueError)
            assert error.__traceback__ is not None
        case Ok(_):
            pytest.fail("expected an Err")


def test_error_type_is_the_union_of_the_listed_exceptions() -> None:
    # Fails to type-check under mypy if the overloads stop binding one type
    # parameter per argument: mypy would join KeyError and JSONDecodeError to
    # Exception, which is not assignable to this annotation.
    missing: Result[int, KeyError | json.JSONDecodeError] = port_from("{}")
    malformed: Result[int, KeyError | json.JSONDecodeError] = port_from("{")

    assert isinstance(missing.unwrap_err(), KeyError)
    assert isinstance(malformed.unwrap_err(), json.JSONDecodeError)
    assert port_from('{"port": 5432}') == Ok(5432)


def test_unlisted_exceptions_propagate() -> None:
    with pytest.raises(TypeError):
        port_from('{"port": [1]}')


def test_base_exceptions_propagate_even_when_catching_exception() -> None:
    @catching(Exception)
    def interrupted() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted()


def test_unwrap_chains_the_caught_exception() -> None:
    with pytest.raises(Panic) as panic:
        parse("x").unwrap()

    assert isinstance(panic.value.__cause__, ValueError)


def test_inline_form_for_a_one_off_call() -> None:
    assert catching(ValueError)(int)("7") == Ok(7)
    assert catching(ValueError)(int)("seven").is_err()


def test_async_functions() -> None:
    @catching(ValueError)
    async def fetch(raw: str) -> int:
        await asyncio.sleep(0)
        return int(raw)

    assert asyncio.run(fetch("4")) == Ok(4)
    assert asyncio.run(fetch("four")).is_err()


def test_keeps_the_function_metadata() -> None:
    assert parse.__name__ == "parse"


def test_more_than_four_exception_types() -> None:
    # Past four types the overloads fall back to a single parameter; the
    # runtime behaviour must not change.
    @catching(KeyError, IndexError, ValueError, TypeError, ZeroDivisionError)
    def divide(a: int, b: int) -> float:
        return a / b

    assert divide(1, 0).is_err()


def test_rejects_generator_functions() -> None:
    def numbers() -> Iterator[int]:
        yield 1

    with pytest.raises(TypeError, match="generator function"):
        catching(ValueError)(numbers)


def test_rejects_an_exception_instance() -> None:
    with pytest.raises(TypeError, match="pass the class"):
        catching(ValueError())  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue, reportArgumentType]
