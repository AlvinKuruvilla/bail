"""`@bails` returning the exceptions a function's `Result` error type declares.

This module uses `from __future__ import annotations`, so every annotation is
a string that `@bails` evaluates itself. test_declared_errors_evaluated.py
covers annotations Python evaluates.
"""

# The read-once test counts calls to the private resolver.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from bail import Err, Null, Ok, Option, Result, bails

if TYPE_CHECKING:
    from decimal import InvalidOperation  # deliberately type-checking only

type IOErrors = OSError | ValueError
type Loaded = Result[int, OSError]
type Parsed[T] = Result[T, ValueError]


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


@bails
def load_number(path: str) -> Result[int, OSError | json.JSONDecodeError]:
    return Ok(int(json.loads(read(path))))


def test_declared_exception_becomes_err() -> None:
    match load_number("/no/such/file"):
        case Err(error):
            # FileNotFoundError is caught because it subclasses OSError.
            assert isinstance(error, FileNotFoundError)
            assert error.__traceback__ is not None
        case Ok(_):
            pytest.fail("expected an Err")


def test_undeclared_exception_still_propagates() -> None:
    # Pins the behaviour from before declared exceptions existed: only the
    # classes named in the error type are caught.
    @bails
    def parse(raw: str) -> Result[int, OSError]:
        return Ok(int(raw))

    with pytest.raises(ValueError, match="invalid literal"):
        parse("x")


def test_value_error_types_are_not_caught_and_exceptions_beside_them_are() -> None:
    @bails
    def parse(raw: str) -> Result[int, str | ValueError]:
        if not raw:
            return Err("empty")
        return Ok(int(raw))

    assert parse("") == Err("empty")
    assert isinstance(parse("x").unwrap_err(), ValueError)


def test_exception_catches_every_ordinary_exception() -> None:
    @bails
    def anything() -> Result[int, Exception]:
        raise LookupError("gone")

    assert isinstance(anything().unwrap_err(), LookupError)


def test_type_aliases_are_expanded() -> None:
    @bails
    def via_error_alias(raw: str) -> Result[int, IOErrors]:
        return Ok(int(raw))

    @bails
    def via_result_alias() -> Loaded:
        raise PermissionError("denied")

    @bails
    def via_generic_alias(raw: str) -> Parsed[int]:
        return Ok(int(raw))

    assert isinstance(via_error_alias("x").unwrap_err(), ValueError)
    assert isinstance(via_result_alias().unwrap_err(), PermissionError)
    assert isinstance(via_generic_alias("x").unwrap_err(), ValueError)


def test_generic_error_type_catches_nothing() -> None:
    @bails
    def fail[E: Exception](error: E) -> Result[int, E]:
        raise error

    with pytest.raises(KeyError):
        fail(KeyError("k"))


def test_non_result_returns_catch_nothing() -> None:
    @bails
    def maybe(raw: str) -> Option[int]:
        return Null() if not raw else Ok(int(raw)).ok()

    @bails
    def optional(raw: str) -> Result[int, ValueError] | None:
        return Ok(int(raw))

    with pytest.raises(ValueError):
        maybe("x")
    with pytest.raises(ValueError):
        optional("x")


def test_forward_reference_to_a_class_defined_later() -> None:
    assert isinstance(load_config().unwrap_err(), ConfigError)


@bails
def load_config() -> Result[int, ConfigError]:
    raise ConfigError("missing port")


class ConfigError(Exception):
    pass


def test_type_checking_only_name_raises_a_type_error_naming_the_fix() -> None:
    @bails
    def divide(raw: str) -> Result[int, InvalidOperation]:
        return Ok(int(raw))

    with pytest.raises(TypeError, match="runtime-evaluated-decorators") as raised:
        divide("1")
    assert "InvalidOperation" in str(raised.value)


def test_type_checking_only_parameter_types_are_not_evaluated() -> None:
    @bails
    def parse(
        raw: str, hint: InvalidOperation | None = None
    ) -> Result[int, ValueError]:
        return Ok(int(raw))

    assert isinstance(parse("x").unwrap_err(), ValueError)


def test_base_exception_in_the_error_type_is_refused() -> None:
    @bails
    def interrupted() -> Result[int, KeyboardInterrupt]:
        return Ok(1)

    with pytest.raises(TypeError, match="must keep propagating"):
        interrupted()


def test_the_annotation_is_read_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import bail._bails

    calls: list[object] = []
    original = bail._bails._declared_exceptions

    def counting(f: object) -> tuple[type[Exception], ...]:
        calls.append(f)
        return original(f)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(bail._bails, "_declared_exceptions", counting)

    @bails
    def parse(raw: str) -> Result[int, ValueError]:
        return Ok(int(raw))

    assert calls == []  # not at decoration time
    parse("1")
    parse("x")
    parse("2")
    assert len(calls) == 1


def test_q_and_declared_exceptions_in_one_function() -> None:
    def positive(n: int) -> Result[int, str]:
        return Ok(n) if n > 0 else Err("not positive")

    @bails
    def parse_positive(raw: str) -> Result[int, str | ValueError]:
        return Ok(positive(int(raw)).Q)

    assert parse_positive("3") == Ok(3)
    assert parse_positive("-3") == Err("not positive")
    assert isinstance(parse_positive("x").unwrap_err(), ValueError)


def test_async_functions() -> None:
    @bails
    async def fetch(raw: str) -> Result[int, ValueError]:
        await asyncio.sleep(0)
        return Ok(int(raw))

    assert asyncio.run(fetch("4")) == Ok(4)
    assert isinstance(asyncio.run(fetch("four")).unwrap_err(), ValueError)
