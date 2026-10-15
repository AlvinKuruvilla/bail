<div align="center">

<img src="docs/logo.svg" alt="bail" width="112">

# bail

**Rust's `Result` and `Option` for Python, checked by mypy and pyright.**

[![CI](https://github.com/AlvinKuruvilla/bail/actions/workflows/ci.yml/badge.svg)](https://github.com/AlvinKuruvilla/bail/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-3776ab)](https://github.com/AlvinKuruvilla/bail/blob/main/pyproject.toml)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)
[![Checked with pyright](https://img.shields.io/badge/pyright-strict-2a6db2)](https://github.com/microsoft/pyright)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

- [Installation](#installation)
- [Usage](#usage)
  - [Result](#result)
  - [Option](#option)
  - [Early return with `.Q`](#early-return-with-q)
- [Why bail](#why-bail)
- [Examples](#examples)
- [Development](#development)
- [License](#license)

## Installation

bail needs Python 3.12 or later and has no dependencies. It is not on PyPI yet:

```bash
uv add git+https://github.com/AlvinKuruvilla/bail
```

## Usage

### Result

A function that can fail returns `Ok(value)` or `Err(error)`. The caller
handles both with `match`, and each arm is narrowed to its type.

```python
from bail import Err, Ok, Result


def parse_version(header: bytes) -> Result[int, str]:
    if not header:
        return Err("invalid header length")
    if header[0] not in (1, 2):
        return Err("invalid version")
    return Ok(header[0])


match parse_version(bytes([1, 2, 3, 4])):
    case Ok(version):
        print(f"working with version: {version}")
    case Err(error):
        print(f"error parsing header: {error}")
```

Leave out the `Err` arm and mypy and pyright both report it.

### Option

A value that may be absent is `Some(value)` or `Null()`.

```python
from bail import Null, Option, Some


def divide(numerator: float, denominator: float) -> Option[float]:
    if denominator == 0.0:
        return Null()
    return Some(numerator / denominator)
```

`Null()` always returns the same object, the way `type(None)()` returns `None`.

### Early return with `.Q`

`.Q` is Rust's `?` operator. It returns the `Ok` value, or makes the enclosing
`@bails` function return the `Err`.

```python
from bail import Err, Ok, Result, bails


def parse_number(text: str) -> Result[int, str]:
    return Ok(int(text)) if text.isdigit() else Err(f"not a number: {text!r}")


@bails
def add(a: str, b: str) -> Result[int, str]:
    return Ok(parse_number(a).Q + parse_number(b).Q)
```

`@bails` also works on `async def` functions.

## Why bail

Python Result libraries differ in whether the type checker sees an exhaustive
`match`:

| Library | `match` narrows | Missing arm reported | Maintained |
|---|---|---|---|
| **bail** | mypy, pyright | mypy, pyright | yes |
| [returns](https://github.com/dry-python/returns) 0.29 | pyright only | no | yes |
| [expression](https://github.com/dbrattli/Expression) 5.7 | via `tag=` only | no | yes |
| [rustshed](https://github.com/pawelrubin/rustshed) 0.5 | mypy, pyright | mypy, pyright | no release since 2022 |
| [result](https://github.com/rustedpy/result) 0.17 | mypy, pyright | mypy, pyright | archived |

Measured 2026-10-14 with mypy 2.4.0 and pyright 1.1.414. [`DESIGN.md`](DESIGN.md)
covers the method and the trade-offs, including what the type checkers cannot
verify about `.Q`.

## Examples

Each file in [`examples/`](examples/) shows one feature and runs as-is:

| File | Shows |
|---|---|
| [`result.py`](examples/result.py) | returning and matching a `Result` |
| [`option.py`](examples/option.py) | returning and matching an `Option` |
| [`early_return.py`](examples/early_return.py) | `.Q` and `@bails` |

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests examples
uv run pyright src tests examples
```

Both type checkers run in CI on Python 3.12–3.14. The tests check type narrowing
and variance, so a regression there fails type-checking, not pytest.

## License

MIT. The Option and Result core is ported from
[rustshed](https://github.com/pawelrubin/rustshed) (MIT, Copyright (c) 2022
Paweł Rubin). See [LICENSE](LICENSE).
