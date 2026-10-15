# Session notes

- rustshed's `to_result[IndexError]` catches every `Exception`: `__getitem__`
  throws the exception type away. A function typed `Result[int, IndexError]`
  returned `Err(ValueError)` and pyright reported no error. Don't port it as-is
  in step 4; tryhard's `catch=` narrowing has to really narrow.
- rustshed documents `case Null:` for matching an empty Option. That is a
  capture pattern, which binds any value, so it only works because it's the
  last case.
- The aria Python template's CI has three gaps: it tests only one Python
  version, runs `mypy src` without `tests`, and supports only one type checker.
  bail's ci.yml works around them; the template is unchanged.
- mypy 2.4 inferred every class in `_core.py` as invariant under PEP 695 syntax.
  pyright got them right. A small standalone repro hasn't reproduced it yet.
  It might be worth a mypy issue once it's minimised. The pre-change version
  is in tmp/core_before_variance.py.
- `and_then` with a function that only ever returns `Ok` leaves `F`
  unsolved, so pyright reports the result as `Err[Unknown]` unless the
  target is annotated. mypy infers `Never`. A PEP 696 default
  (`F = Never`) would fix it, but that syntax needs 3.13 and the
  `TypeVar(default=)` form needs typing_extensions on 3.12.
- The template's pre-commit hook pins ruff v0.9.0, which is older than the
  ruff in uv.lock. The newer one also formats Python blocks inside
  Markdown, so the hook and CI can disagree.
