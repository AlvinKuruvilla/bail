from __future__ import annotations

from bail import Err, Null, Ok, Option, Result, Some, bails


def parse(s: str) -> Result[int, str]:
    return Ok(int(s)) if s.isdigit() else Err(s)


def lookup(k: str) -> Result[int, KeyError]:
    return Err(KeyError(k))


def find(k: str) -> Option[int]:
    return Null()


@bails
def fits(a: str) -> Result[int, str]:
    return Ok(parse(a).Q + 1)


@bails
def widens(a: str) -> Result[int, str | KeyError]:
    return Ok(parse(a).Q + lookup(a).Q)


@bails
def leaks(a: str) -> Result[int, str]:
    return Ok(parse(a).Q + lookup(a).Q)


@bails
def option_in_result(a: str) -> Result[int, str]:
    return Ok(find(a).Q)


@bails
def some_in_result(a: str) -> Result[int, str]:
    return Ok(Some(1).Q)


@bails
def option_fits(a: str) -> Option[int]:
    return Some(find(a).Q)


@bails
async def async_leaks(a: str) -> Result[int, str]:
    return Ok(lookup(a).Q)


@bails
def ignored(a: str) -> Result[int, str]:
    return Ok(lookup(a).Q)  # bail: ignore


@bails
def unannotated(a: str):  # type: ignore[no-untyped-def]
    return Ok(lookup(a).Q)


class Service:
    @bails
    def method_leaks(self, a: str) -> "Result[int, str]":  # noqa: UP037
        return Ok(lookup(a).Q)
