"""A checker for the rules about `.Q` that the type checkers cannot enforce.

`.Q` raises an exception that the nearest `@bails` up the call stack turns
back into a return value. Two mistakes type-check cleanly and go wrong at
runtime, and this module reports them:

- **BAIL001**: a `.Q` whose enclosing function is not decorated with
  `@bails`. The `Err` escapes as an exception, or returns from some other
  function further up the stack.
- **BAIL002** (not yet implemented): a `.Q` whose `Err` type is not part of
  the enclosing function's declared error type.

Run it on files or directories:

    python -m bail.lint src tests

It prints one line per finding and exits 1 if there were any. A finding on a
line containing `# bail: ignore` is suppressed.

# Which function encloses a `.Q`

The function that `.Q` returns from is the innermost one that owns the frame
the `.Q` runs in. That is any `def`, `async def` or `lambda`, and also a
generator expression: its body runs when the generator is consumed, which
can be after the enclosing function has returned. List, set and dict
comprehensions are not counted, because since Python 3.12 they run inline in
the enclosing function's frame (PEP 709).

BAIL001 is syntactic: it does not know the type of the value before `.Q`, so
any attribute named `Q` counts. A `.Q` on an unrelated library's object is a
false positive, which the ignore comment covers.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Finding", "check_source", "main"]

_IGNORE_MARKER = "# bail: ignore"

# Directories never worth descending into when given a directory to check.
_SKIPPED_DIRS = frozenset({"__pycache__", "node_modules", "venv", "build", "dist"})

_Scope = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.GeneratorExp


@dataclass(frozen=True, slots=True)
class Finding:
    """One rule violation, positioned at the `Q` of a `.Q`."""

    path: str
    line: int
    column: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.code} {self.message}"


def check_source(source: str, path: str = "<string>") -> list[Finding]:
    """Return the findings for one module's source text, in source order.

    Raises `SyntaxError` if `source` does not parse.
    """
    tree = ast.parse(source, filename=path)
    visitor = _Visitor(path, _bails_names(tree))
    visitor.visit(tree)
    lines = source.splitlines()
    return [
        finding
        for finding in visitor.findings
        if _IGNORE_MARKER not in lines[finding.line - 1]
    ]


def _bails_names(tree: ast.Module) -> tuple[frozenset[str], frozenset[str]]:
    """Return the names the decorator is visible under in this module.

    The first set holds bare names (`bails`, or its alias from
    `from bail import bails as early`). The second holds module names for the
    attribute form (`bail`, or `b` from `import bail as b`).
    """
    bare: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "bail":
            bare.update(a.asname or a.name for a in node.names if a.name == "bails")
        elif isinstance(node, ast.Import):
            modules.update(a.asname or a.name for a in node.names if a.name == "bail")
    return frozenset(bare), frozenset(modules)


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str, names: tuple[frozenset[str], frozenset[str]]) -> None:
        self.path = path
        self.bare_names, self.module_names = names
        self.findings: list[Finding] = []
        # Innermost scope last; empty at module and class level.
        self.scopes: list[_Scope] = []

    def _in_scope(self, node: _Scope) -> None:
        self.scopes.append(node)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Decorators and defaults are evaluated in the enclosing scope, so
        # visit them before entering the function.
        for expr in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]:
            if expr is not None:
                self.visit(expr)
        self.scopes.append(node)
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._in_scope(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        # The first iterable is evaluated immediately, in the enclosing scope.
        self.visit(node.generators[0].iter)
        self.scopes.append(node)
        self.visit(node.elt)
        for index, comprehension in enumerate(node.generators):
            if index > 0:
                self.visit(comprehension.iter)
            self.visit(comprehension.target)
            for condition in comprehension.ifs:
                self.visit(condition)
        self.scopes.pop()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "Q" and isinstance(node.ctx, ast.Load):
            self._check_q(node)
        self.generic_visit(node)

    def _check_q(self, node: ast.Attribute) -> None:
        scope = self.scopes[-1] if self.scopes else None
        if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
            if self._is_bails(scope):
                return
            where = f"`{scope.name}`, which is not decorated with `@bails`"
        elif isinstance(scope, ast.Lambda):
            where = "a lambda, which cannot be decorated with `@bails`"
        elif isinstance(scope, ast.GeneratorExp):
            where = "a generator expression, which runs after the function returns"
        else:
            where = "module or class level, outside any function"
        # `end_col_offset` is the 0-based offset just past `Q`, which is also
        # the 1-based column of `Q` itself.
        assert node.end_lineno is not None and node.end_col_offset is not None
        self.findings.append(
            Finding(
                self.path,
                node.end_lineno,
                node.end_col_offset,
                "BAIL001",
                f"`.Q` in {where}; an Err here escapes as an exception",
            )
        )

    def _is_bails(self, function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        for decorator in function.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id in self.bare_names:
                return True
            if (
                isinstance(decorator, ast.Attribute)
                and decorator.attr == "bails"
                and isinstance(decorator.value, ast.Name)
                and decorator.value.id in self.module_names
            ):
                return True
        return False


def _python_files(paths: Sequence[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*.py")):
                parts = child.relative_to(path).parts[:-1]
                if not any(p.startswith(".") or p in _SKIPPED_DIRS for p in parts):
                    yield child
        else:
            yield path


def main(argv: Sequence[str] | None = None) -> int:
    """Check the given files and directories; return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="bail-lint", description="Check that every `.Q` is inside `@bails`."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="files or directories")
    args = parser.parse_args(argv)

    findings: list[Finding] = []
    for path in _python_files(args.paths):
        try:
            source = path.read_text(encoding="utf-8")
            findings.extend(check_source(source, str(path)))
        except (OSError, SyntaxError, UnicodeDecodeError) as error:
            print(f"{path}: cannot check: {error}", file=sys.stderr)
            return 2
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
