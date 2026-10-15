# bail

A Result type for Python: `Ok(value)` or `Err(error)`, handled with `match`
and chained with `.Q`, Rust's `?` operator.

```python
from bail import Err, Ok, Result, bails


def parse_number(text: str) -> Result[int, str]:
    return Ok(int(text)) if text.isdigit() else Err(f"not a number: {text!r}")


@bails
def add(a: str, b: str) -> Result[int, str]:
    return Ok(parse_number(a).Q + parse_number(b).Q)


match add("1", "x"):
    case Ok(total):
        print(total)
    case Err(reason):
        print(reason)  # not a number: 'x'
```

mypy and pyright narrow each `case` arm and report a `match` that leaves one
out. `Option` (`Some(value)` or `Null()`) works the same way.

There is one example per feature in [`examples/`](examples/). The design and
its trade-offs are in [`DESIGN.md`](DESIGN.md).

Requires Python 3.12 or later and has no runtime dependencies. Not yet on PyPI.

## Development

```bash
uv sync                                # create .venv with the dev tools
uv run pytest                          # tests
uv run ruff check . && uv run ruff format .
uv run mypy src tests examples         # both checkers: the tests assert
uv run pyright src tests examples      # narrowing and variance
uv run pre-commit install              # git hooks, once
```

## Licence

MIT. The Option and Result core is ported from
[rustshed](https://github.com/pawelrubin/rustshed) (MIT, Copyright (c) 2022
Paweł Rubin).
