"""A checker for the rules about `.Q` that the type checkers cannot enforce.

`.Q` raises an exception that the nearest `@bails` up the call stack turns
back into a return value. Two mistakes type-check cleanly and go wrong at
runtime, and this module reports them:

- **BAIL001**: a `.Q` whose enclosing function is not decorated with
  `@bails`. The `Err` escapes as an exception, or returns from some other
  function further up the stack.
- **BAIL002**: a `.Q` that can return an `Err` (or `Null()`) the enclosing
  function's declared return type does not allow. Only checked with
  `--checker`, because it runs a type checker.
- **BAIL003**: a name in a `@bails` function's return annotation that is
  imported only under `if TYPE_CHECKING:`. `@bails` evaluates the return
  annotation at runtime to find the exceptions it declares, so the name must
  exist then; otherwise the first call raises `TypeError`.

Run it on files or directories:

    bail-lint src tests                     # BAIL001 and BAIL003
    bail-lint --checker pyright src tests   # all three

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

# How BAIL002 works

The question is whether `Err[E]`, for the `E` of the value before `.Q`, is
assignable to the function's declared return type. bail does not answer it;
the project's own type checker does, after a rewrite. In each `@bails`
function with a return annotation, every `X.Q` becomes

    X._bail_lint_q(_bail_typing.cast("_bail_abc.Callable[[<return type>], None]", None))

`_bail_lint_q` on `Err[E]` takes a `Callable[[Err[E]], None]`, so the
argument type-checks only if a function accepting the declared return type
also accepts `Err[E]`: exactly the assignability in question. `Null` does the
same with `Callable[[Null], None]`, and `Ok` and `Some`, which never bail,
accept any sink. Because the method is on the classes, a union operand is
checked member by member.

The rewrite replaces only the `Q` token and appends the two imports at the
end of the file, so every original line keeps its number. The rewritten
files go into a temporary copy of the project, the checker runs there, and
an argument-type error that lands exactly on an inserted sink becomes a
BAIL002 at the original `.Q`. Every other diagnostic is the ordinary type
check's business and is dropped.

The copy holds the project's Python sources and checker configuration and
nothing else. The checker runs under the interpreter running the lint, so it
sees the same installed packages. A project whose imports only resolve
through settings outside those files, such as a `MYPYPATH` pointing
elsewhere, may not check the same way in the copy.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Finding", "check_source", "check_types", "main"]

_IGNORE_MARKER = "# bail: ignore"

# Directories never worth descending into, when walking a directory to check
# or copying a project.
_SKIPPED_DIRS = frozenset({"__pycache__", "node_modules", "venv", "build", "dist"})

# Files copied into the temporary project so the checker sees the same
# sources and configuration.
_PROJECT_FILES = ("*.py", "*.pyi", "py.typed")
_CONFIG_FILES = (
    "pyproject.toml",
    "setup.cfg",
    "mypy.ini",
    ".mypy.ini",
    "pyrightconfig.json",
)

_REWRITE_IMPORTS = (
    "\nimport typing as _bail_typing\nimport collections.abc as _bail_abc\n"
)

_Scope = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.GeneratorExp
_Function = ast.FunctionDef | ast.AsyncFunctionDef


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
    """Return the BAIL001 and BAIL003 findings for one module's source text.

    Findings are in source order. Raises `SyntaxError` if `source` does not
    parse.
    """
    visitor = _visit(source, path)
    findings = [*visitor.findings, *_type_checking_only_findings(visitor)]
    findings.sort(key=lambda f: (f.line, f.column, f.code))
    return _unignored(findings, source)


def _visit(source: str, path: str) -> _Visitor:
    tree = ast.parse(source, filename=path)
    visitor = _Visitor(path, _bails_names(tree), tree)
    visitor.visit(tree)
    return visitor


def _unignored(findings: list[Finding], source: str) -> list[Finding]:
    lines = source.splitlines()
    return [f for f in findings if _IGNORE_MARKER not in lines[f.line - 1]]


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
    def __init__(
        self,
        path: str,
        names: tuple[frozenset[str], frozenset[str]],
        tree: ast.Module,
    ) -> None:
        self.path = path
        self.tree = tree
        # Every function decorated with `@bails`, for BAIL003.
        self.bails_functions: list[_Function] = []
        self.bare_names, self.module_names = names
        self.findings: list[Finding] = []
        # Each `.Q` that is correctly inside `@bails`, with its function, for
        # BAIL002.
        self.bailing: list[tuple[ast.Attribute, _Function]] = []
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

    def _visit_function(self, node: _Function) -> None:
        # Decorators and defaults are evaluated in the enclosing scope, so
        # visit them before entering the function.
        for expr in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]:
            if expr is not None:
                self.visit(expr)
        if self._is_bails(node):
            self.bails_functions.append(node)
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
                self.bailing.append((node, scope))
                return
            where = f"`{scope.name}`, which is not decorated with `@bails`"
        elif isinstance(scope, ast.Lambda):
            where = "a lambda, which cannot be decorated with `@bails`"
        elif isinstance(scope, ast.GeneratorExp):
            where = "a generator expression, which runs after the function returns"
        else:
            where = "module or class level, outside any function"
        line, column = _q_position(node)
        self.findings.append(
            Finding(
                self.path,
                line,
                column,
                "BAIL001",
                f"`.Q` in {where}; an Err here escapes as an exception",
            )
        )

    def _is_bails(self, function: _Function) -> bool:
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


def _q_position(node: ast.Attribute) -> tuple[int, int]:
    """Return the 1-based line and column of the `Q` in `node`.

    `end_col_offset` is the 0-based offset just past `Q`, which is also the
    1-based column of `Q` itself. Both count UTF-8 bytes, which equal
    characters on the ASCII lines nearly all code is written in.
    """
    assert node.end_lineno is not None and node.end_col_offset is not None
    return node.end_lineno, node.end_col_offset


# BAIL003


def _type_checking_only_findings(visitor: _Visitor) -> list[Finding]:
    """Report names in `@bails` return annotations that only exist for checkers."""
    hidden = _type_checking_only_names(visitor.tree)
    if not hidden:
        return []
    findings: list[Finding] = []
    for function in visitor.bails_functions:
        if function.returns is None:
            continue
        for name, line, column in _annotation_names(function.returns):
            if name in hidden:
                findings.append(
                    Finding(
                        visitor.path,
                        line,
                        column,
                        "BAIL003",
                        f"`{name}` is imported only under `TYPE_CHECKING`, but "
                        f"`@bails` reads `{function.name}`'s return type at runtime; "
                        'import it normally, or add "bail.bails" to ruff\'s '
                        "runtime-evaluated-decorators if ruff moved it",
                    )
                )
    return findings


def _type_checking_only_names(tree: ast.Module) -> frozenset[str]:
    """Return module-level names bound under `if TYPE_CHECKING:` and nowhere else.

    A name also bound outside such a block, including in its `else:` branch,
    exists at runtime and is not returned.
    """
    hidden: set[str] = set()
    hidden_blocks: list[ast.stmt] = []
    for statement in tree.body:
        if isinstance(statement, ast.If) and _is_type_checking(statement.test):
            hidden_blocks.extend(statement.body)
            for hidden_statement in statement.body:
                hidden.update(_imported_names(hidden_statement))

    inside_hidden = {id(n) for block in hidden_blocks for n in ast.walk(block)}
    visible: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in inside_hidden:
            continue
        visible.update(_imported_names(node))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            visible.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            visible.add(node.id)
        elif isinstance(node, ast.TypeAlias):
            visible.add(node.name.id)
    return frozenset(hidden - visible)


def _is_type_checking(test: ast.expr) -> bool:
    # `TYPE_CHECKING`, or `typing.TYPE_CHECKING` under any module alias.
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _imported_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        # `import json.decoder` binds `json`.
        return [a.asname or a.name.split(".")[0] for a in node.names]
    if isinstance(node, ast.ImportFrom):
        return [a.asname or a.name for a in node.names if a.name != "*"]
    return []


def _annotation_names(annotation: ast.expr) -> list[tuple[str, int, int]]:
    """Return each name an annotation reads, with its 1-based line and column.

    A string annotation is parsed, and its names are reported at the string.
    """
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            parsed = ast.parse(annotation.value, mode="eval")
        except SyntaxError:
            return []
        return [
            (node.id, annotation.lineno, annotation.col_offset + 1)
            for node in ast.walk(parsed)
            if isinstance(node, ast.Name)
        ]
    return [
        (node.id, node.lineno, node.col_offset + 1)
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name)
    ]


# BAIL002


@dataclass(frozen=True, slots=True)
class _Site:
    """A rewritten `.Q`: where it was, and where its sink argument now is."""

    line: int
    column: int
    sink_column: int  # 1-based, on the same line in the rewritten file
    function: str
    declared: str


def _rewrite(source: str, visitor: _Visitor) -> tuple[str, list[_Site]]:
    """Rewrite each `.Q` inside an annotated `@bails` function for BAIL002."""
    lines = source.splitlines(keepends=True)
    sites: list[_Site] = []
    # Per line, the `Q` columns to replace, rightmost first so earlier columns
    # stay valid while editing.
    edits: dict[int, list[tuple[int, str, str]]] = {}
    for node, function in visitor.bailing:
        if function.returns is None:
            continue
        declared = _annotation_text(source, function.returns)
        line, column = _q_position(node)
        edits.setdefault(line, []).append((column, function.name, declared))

    for line, line_edits in edits.items():
        original = lines[line - 1]
        text = original
        # Apply left to right. `shift` counts the characters earlier edits on
        # this line added, so each `Q`'s position in the edited text is its
        # position in the original plus `shift`.
        shift = 0
        for column, name, declared in sorted(line_edits):
            start = _char_offset(original, column - 1) + shift
            replacement = f"_bail_lint_q({_sink_expression(declared)})"
            text = text[:start] + replacement + text[start + 1 :]
            sink_column = start + len("_bail_lint_q(") + 1
            sites.append(_Site(line, column, sink_column, name, declared))
            shift += len(replacement) - 1
        lines[line - 1] = text

    rewritten = "".join(lines)
    if not rewritten.endswith("\n"):
        rewritten += "\n"
    return rewritten + _REWRITE_IMPORTS, sites


def _char_offset(text: str, byte_offset: int) -> int:
    """Convert an ast column, which counts UTF-8 bytes, to a string index."""
    return len(text.encode("utf-8")[:byte_offset].decode("utf-8", errors="ignore"))


def _annotation_text(source: str, annotation: ast.expr) -> str:
    """Return a return annotation as one line of type-expression text."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        text = annotation.value  # a string annotation, `-> "Result[int, str]"`
    else:
        segment = ast.get_source_segment(source, annotation)
        assert segment is not None, "parsed from this source"
        text = segment
    return " ".join(text.split())


def _sink_expression(declared: str) -> str:
    return f"_bail_typing.cast({f'_bail_abc.Callable[[{declared}], None]'!r}, None)"


@dataclass(frozen=True, slots=True)
class _Diagnostic:
    path: str  # relative to the project root, with forward slashes
    line: int
    column: int  # 1-based
    message: str


def check_types(paths: Sequence[Path], checker: str, root: Path) -> list[Finding]:
    """Return the BAIL002 findings for `paths`, using `checker` on a copy of `root`.

    `checker` is "mypy" or "pyright". Raises `CheckerError` if the checker
    cannot run.
    """
    root = root.resolve()
    rewrites: dict[str, tuple[str, list[_Site], str]] = {}
    sources: dict[str, str] = {}
    for path in _python_files(paths):
        source = path.read_text(encoding="utf-8")
        sources[str(path)] = source
        rewritten, sites = _rewrite(source, _visit(source, str(path)))
        if sites:
            relative = path.resolve().relative_to(root).as_posix()
            rewrites[relative] = (rewritten, sites, str(path))
    if not rewrites:
        return []

    with tempfile.TemporaryDirectory(prefix="bail-lint-") as scratch:
        copy = Path(scratch)
        _copy_project(root, copy)
        for relative, (rewritten, _, _) in rewrites.items():
            (copy / relative).write_text(rewritten, encoding="utf-8")
        diagnostics = _run_checker(checker, copy, sorted(rewrites))

    findings: list[Finding] = []
    for diagnostic in diagnostics:
        if diagnostic.path not in rewrites:
            continue
        _, sites, original_path = rewrites[diagnostic.path]
        for site in sites:
            if (site.line, site.sink_column) == (diagnostic.line, diagnostic.column):
                leak = _leak_finding(original_path, site, diagnostic)
                findings.extend(_unignored([leak], sources[original_path]))
    findings.sort(key=lambda f: (f.path, f.line, f.column))
    return findings


def _leak_finding(path: str, site: _Site, diagnostic: _Diagnostic) -> Finding:
    bailed = _bailed_type(diagnostic.message)
    what = f"`{bailed}`" if bailed else "a value"
    if bailed == "Null":
        what = "`Null()`"
    return Finding(
        path,
        site.line,
        site.column,
        "BAIL002",
        f"`.Q` can return {what}, which `{site.function}` does not declare: "
        f"it returns `{site.declared}`",
    )


# The bailed type, as each checker phrases the sink mismatch. From mypy,
#   ...; expected "Callable[[Err[KeyError]], None]"
# and from pyright,
#   ... parameter "sink" of type "(Err[KeyError]) -> None" ...
_MYPY_EXPECTED = re.compile(r'expected "Callable\[\[(?P<bailed>.+)\], None\]"')
_PYRIGHT_EXPECTED = re.compile(
    r'parameter "sink" of type "\((?P<bailed>.+?)\) -> None"'
)


def _bailed_type(message: str) -> str | None:
    for pattern in (_MYPY_EXPECTED, _PYRIGHT_EXPECTED):
        match = pattern.search(message)
        if match:
            return match["bailed"]
    return None


def _copy_project(root: Path, copy: Path) -> None:
    for pattern in _PROJECT_FILES:
        for source in root.rglob(pattern):
            parts = source.relative_to(root).parts[:-1]
            if any(p.startswith(".") or p in _SKIPPED_DIRS for p in parts):
                continue
            target = copy / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in _CONFIG_FILES:
        if (root / name).is_file():
            shutil.copy2(root / name, copy / name)


class CheckerError(Exception):
    """The type checker could not run, so BAIL002 could not be checked."""


def _run_checker(checker: str, copy: Path, files: list[str]) -> list[_Diagnostic]:
    if checker == "mypy":
        command = [sys.executable, "-m", "mypy", "--show-column-numbers"]
        command += ["--no-error-summary", "--no-pretty", "--no-color-output", *files]
    elif checker == "pyright":
        command = [sys.executable, "-m", "pyright", "--outputjson"]
        command += ["--pythonpath", sys.executable, *files]
    else:
        raise CheckerError(f"unknown checker {checker!r}; use mypy or pyright")
    try:
        done = subprocess.run(
            command, cwd=copy, capture_output=True, text=True, check=False
        )
    except OSError as error:
        raise CheckerError(f"cannot run {checker}: {error}") from error
    # Both checkers exit 1 when they report errors; anything higher means they
    # did not run to completion.
    if done.returncode > 1:
        detail = (done.stderr or done.stdout).strip()
        raise CheckerError(
            f"{checker} failed (exit {done.returncode}); is it installed in "
            f"this environment?\n{detail}"
        )
    if checker == "mypy":
        return _mypy_diagnostics(done.stdout)
    return _pyright_diagnostics(done.stdout, copy)


_MYPY_LINE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?P<column>\d+): error: (?P<message>.*)"
    r"  \[arg-type\]$"
)


def _mypy_diagnostics(output: str) -> list[_Diagnostic]:
    diagnostics: list[_Diagnostic] = []
    for line in output.splitlines():
        match = _MYPY_LINE.match(line)
        if match:
            diagnostics.append(
                _Diagnostic(
                    Path(match["path"]).as_posix(),
                    int(match["line"]),
                    int(match["column"]),
                    match["message"],
                )
            )
    return diagnostics


def _pyright_diagnostics(output: str, copy: Path) -> list[_Diagnostic]:
    report = json.loads(output)
    diagnostics: list[_Diagnostic] = []
    for item in report.get("generalDiagnostics", []):
        if item.get("rule") != "reportArgumentType":
            continue
        start = item["range"]["start"]
        path = Path(item["file"]).resolve().relative_to(copy.resolve()).as_posix()
        diagnostics.append(
            _Diagnostic(
                path, start["line"] + 1, start["character"] + 1, item["message"]
            )
        )
    return diagnostics


def _python_files(paths: Sequence[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*.py")):
                parts = child.relative_to(path).parts[:-1]
                if not any(p.startswith(".") or p in _SKIPPED_DIRS for p in parts):
                    yield child
        else:
            yield path


def _project_root(start: Path) -> Path:
    """Return the nearest directory at or above `start` with checker config."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if any((candidate / name).is_file() for name in _CONFIG_FILES):
            return candidate
    return start


def main(argv: Sequence[str] | None = None) -> int:
    """Check the given files and directories; return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="bail-lint", description="Check the rules for `.Q` and `@bails`."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="files or directories")
    parser.add_argument(
        "--checker",
        choices=["mypy", "pyright"],
        help="also check BAIL002 by running this type checker",
    )
    args = parser.parse_args(argv)

    findings: list[Finding] = []
    for path in _python_files(args.paths):
        try:
            source = path.read_text(encoding="utf-8")
            findings.extend(check_source(source, str(path)))
        except (OSError, SyntaxError, UnicodeDecodeError) as error:
            print(f"{path}: cannot check: {error}", file=sys.stderr)
            return 2

    if args.checker:
        try:
            leaks = check_types(args.paths, args.checker, _project_root(Path.cwd()))
        except (CheckerError, ValueError) as error:
            # ValueError: a path outside the project root.
            print(f"bail-lint: BAIL002 not checked: {error}", file=sys.stderr)
            return 2
        findings.extend(leaks)

    findings.sort(key=lambda f: (f.path, f.line, f.column, f.code))
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
