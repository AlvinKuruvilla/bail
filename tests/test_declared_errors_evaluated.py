"""Declared exceptions with annotations Python evaluates itself.

No `from __future__ import annotations` here, so on 3.12 and 3.13 the return
annotation is a real object by the time `@bails` sees it, and on 3.14 it is
read through `annotationlib`. test_declared_errors.py covers string
annotations.
"""

import json

from bail import Ok, Result, bails


@bails
def parse(raw: str) -> Result[int, json.JSONDecodeError | ValueError]:
    return Ok(int(json.loads(raw)))


def test_declared_exceptions_from_an_evaluated_annotation() -> None:
    assert parse("3") == Ok(3)
    assert isinstance(parse("{").unwrap_err(), json.JSONDecodeError)
    assert isinstance(parse('"x"').unwrap_err(), ValueError)
