# bail

A Result type for Python: `Result[T, E] = Ok[T] | Err[E]`, matched with `match`
and chained with `.Q`, checked by mypy and pyright without plugins.

```python
from bail import Ok, Err, Result, bails


def parse_port(raw: str) -> Result[int, str]:
    return Ok(int(raw)) if raw.isdigit() else Err(f"not a port: {raw!r}")


@bails
def endpoint(host: str, raw_port: str) -> Result[str, str]:
    port = parse_port(raw_port).Q  # an Err returns from endpoint here
    return Ok(f"{host}:{port}")


match endpoint("db", "54x2"):
    case Ok(address):
        connect(address)
    case Err(reason):
        log.warning(reason)
```

bail succeeds [tryhard](https://github.com/AISecurityAssurance/tryhard). tryhard's
helpers either raise or return a default. `TRY_WITH_ERROR` returns a
`(value, error)` tuple, which leaves `if error:` checks at every call site. bail
replaces the tuple with a type that the checker narrows.

# Goals

- **Exhaustive `match`**: a missing `Err` arm is a type error under
  `mypy --strict` and pyright, with no plugin.
- **Flat chaining**: a sequence of fallible steps reads as straight-line code.
- **A bridge from exceptions**: wrap a raising call into a `Result`, keeping
  tryhard's `catch=` narrowing and `KeyboardInterrupt`/`SystemExit` passthrough.
- **Zero runtime dependencies.**

# Non-goals

- Higher-kinded types, IO/Reader containers, or anything that needs a checker
  plugin. dry-python/returns covers that space.
- More than one way to collapse a chain. Combinators (`map`, `and_then`,
  `unwrap_or`) stay for single-step transforms.

# Design

## The type is a union of two final classes

`Ok` and `Err` are `@final` frozen dataclasses, and `Result` is a type alias for
their union. This is the only shape tested where `match` both narrows and
detects a missing case:

| Library (version) | narrows in `match` | detects a missing case |
|---|---|---|
| union of `@final` dataclasses | mypy, pyright | mypy, pyright |
| returns 0.29 | pyright only; mypy gives `Any` even with its plugin | no |
| expression 5.7 | only via `Result(tag="ok", ok=v)` | no |
| rustshed 0.5, montasaurus/result | mypy, pyright | mypy, pyright |

Measured 2026-10-14 with mypy 2.4.0 and pyright 1.1.414 on Python 3.13. The probe
files were in `~/Dev/tmp/monad-research/probe/`, which is outside this repo.

## Chaining: `.Q` inside `@bails`

`.Q` on an `Ok` returns its value. On an `Err` it raises a private exception
carrying the `Err`, and the nearest `@bails` decorator catches it and returns
the `Err`. This is Rust's `?` operator, implemented the way rustshed implements it.

Alternatives considered:

| Technique | value types | error types checked | async | 3 steps, ok / err |
|---|---|---|---|---|
| `.Q` + `@bails` | exact | no | yes | 0.83 / 0.60 µs |
| generator do: `x = yield from bind(r)` | exact, but `bind()` needed for pyright | yes | no | 1.20 / 0.57 µs |
| expression `@effect.result` | exact | no | — | 18.7 / 4.8 µs |
| `match` ladder (baseline) | exact | yes | yes | 1.07 / 0.25 µs |

Timings were measured with `timeit`, best of 5, on one machine.

Generator do-notation is the only technique where the checker verifies the
error type, but it lost on friction. A bare `yield from r` gives `Unknown` in
pyright (microsoft/pyright#11881), so users would have to write
`yield from bind(r)`. It also cannot work in async code, because an async
generator cannot `yield from`. Users would reach for `.Q` anyway.

## The cost of `.Q`: unchecked error types

No checker sees which error types escape through `.Q`. This probe passes both
checkers:

```python
def lookup(key: str) -> Result[int, KeyError]: ...


@bails
def leaks(key: str) -> Result[int, str]:
    return Ok(lookup(key).Q)  # returns Err(KeyError) from a Result[int, str]
```

bail closes this gap with a lint run outside the type system. It has two rules:

- **Error-type leak**: for each `.Q` inside a `@bails` function, the operand's
  `E` must be assignable to the function's declared `E`.
- **Missing decorator**: a `.Q` in a function without `@bails` is an error.
  The checker sees `.Q` as returning `T`, so the code type-checks, but at
  runtime the `Err` escapes as an exception.

The implementation is open: a mypy plugin, or a script built on pyright's type
output. Users have to opt in to it.

## Exception bridge

tryhard's `MUST`/`TRY` become one decorator and one call form, both returning
`Result`:

```python
@catching(json.JSONDecodeError)
def load(raw: str) -> dict[str, object]:
    return json.loads(raw)  # -> Result[dict[str, object], JSONDecodeError]


config = attempt(json.loads, raw, catch=(json.JSONDecodeError,))
```

`VERIFY` and `CHECK` have no Result equivalent. They assert invariants and should
still raise. Whether they stay in bail or are dropped is an open question.

# Provenance

The core (`Ok`, `Err`, combinators, `.Q`, the `Option` type, and its tests) is
ported from [rustshed](https://github.com/pawelrubin/rustshed) 0.5.0, MIT, at
commit `125993a`. rustshed has had no human commits since 2022-12. bail copies
the code instead of depending on it so the lint and API changes can be made in
one place. The licence and copyright notice travel with the ported files.

# Trade-offs

- **Unchecked `.Q` error types**: the friction-free chaining style gives up the
  one guarantee generator do-notation had. The lint gets some of that guarantee
  back, but only for users who run it.
- **Two exception paths**: `.Q` uses an exception for control flow, so a `.Q`
  outside any `@bails` escapes as an exception, not as an `Err`.
  rustshed has the same behaviour.
- **Fork maintenance**: copying rustshed means owning its bugs.

# Open questions

1. **Option type**: rustshed ships `Some | Null` with its own `.Q`. Keep it, or
   ship Result only.
2. **`VERIFY`/`CHECK`**: keep them as tryhard-compatible assertion helpers, or
   drop them.
3. **Do-notation as an optional mode** if pyright#11881 is fixed.
4. **Lint implementation**: mypy plugin or a standalone checker.
