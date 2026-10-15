"""bail-lint: BAIL001 (`.Q` outside `@bails`) and BAIL002 (undeclared `Err`)."""

# The rewrite test checks line numbers and sink positions, which only the
# private helpers expose.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bail.lint import CheckerError, _rewrite, _visit, check_source, check_types, main


def codes(source: str) -> list[tuple[int, str]]:
    findings = check_source(textwrap.dedent(source))
    return [(finding.line, finding.code) for finding in findings]


def test_q_inside_bails_is_fine() -> None:
    assert (
        codes("""
            from bail import bails

            @bails
            def add(a, b):
                return Ok(parse(a).Q + parse(b).Q)

            @bails
            async def fetch(url):
                return Ok((await get(url)).Q)
            """)
        == []
    )


def test_q_in_an_undecorated_function() -> None:
    assert codes("""
        def add(a, b):
            return Ok(parse(a).Q)
        """) == [(3, "BAIL001")]


def test_q_at_module_and_class_level() -> None:
    assert codes("""
        port = parse("80").Q

        class Settings:
            port = parse("80").Q
        """) == [(2, "BAIL001"), (5, "BAIL001")]


def test_nested_function_needs_its_own_decorator() -> None:
    # `.Q` returns from the innermost function, so the outer decorator does
    # not cover a helper defined inside it.
    assert codes("""
        from bail import bails

        @bails
        def outer():
            def helper():
                return parse("x").Q
            return Ok(helper())
        """) == [(7, "BAIL001")]


def test_lambda_is_its_own_scope() -> None:
    assert codes("""
        from bail import bails

        @bails
        def total(items):
            return Ok(sum(map(lambda item: parse(item).Q, items)))
        """) == [(6, "BAIL001")]


def test_list_comprehension_runs_in_the_function() -> None:
    # PEP 709: since 3.12 list, set and dict comprehensions run inline in the
    # enclosing frame, so `@bails` catches a `.Q` inside them.
    assert (
        codes("""
            from bail import bails

            @bails
            def parse_all(items):
                listed = [parse(i).Q for i in items]
                return Ok(listed + list({parse(i).Q for i in items}))
            """)
        == []
    )


def test_generator_expression_runs_later() -> None:
    assert codes("""
        from bail import bails

        @bails
        def parse_all(items):
            return Ok(list(parse(i).Q for i in items))
        """) == [(6, "BAIL001")]


def test_generator_expression_first_iterable_runs_in_the_function() -> None:
    assert (
        codes("""
            from bail import bails

            @bails
            def parse_all(raw):
                return Ok(list(i for i in split(raw).Q))
            """)
        == []
    )


def test_decorator_arguments_belong_to_the_enclosing_scope() -> None:
    # A default value is evaluated where the function is defined, so a `.Q`
    # there is outside the decorated function.
    assert codes("""
        from bail import bails

        @bails
        def connect(port=parse("80").Q):
            return Ok(port)
        """) == [(5, "BAIL001")]


def test_aliased_and_module_forms_of_the_decorator() -> None:
    assert (
        codes("""
            import bail
            import bail as b
            from bail import bails as early

            @bail.bails
            def one():
                return parse("1").Q

            @b.bails
            def two():
                return parse("2").Q

            @early
            def three():
                return parse("3").Q
            """)
        == []
    )


def test_a_decorator_named_bails_from_elsewhere_does_not_count() -> None:
    assert codes("""
        from somewhere_else import bails

        @bails
        def add():
            return parse("1").Q
        """) == [(6, "BAIL001")]


def test_ignore_comment_suppresses_the_finding() -> None:
    assert (
        codes("""
            def legacy():
                return thing.Q  # bail: ignore
            """)
        == []
    )


def test_assigning_to_q_is_not_a_use() -> None:
    assert codes("record.Q = 1\n") == []


def test_finding_points_at_q() -> None:
    (finding,) = check_source("value = parse(raw).Q\n", "app.py")
    assert str(finding).startswith("app.py:1:20: BAIL001 ")


def test_main_reports_and_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "good.py").write_text("x = 1\n")
    (tmp_path / "bad.py").write_text("x = parse('1').Q\n")
    hidden = tmp_path / ".venv"
    hidden.mkdir()
    (hidden / "vendored.py").write_text("x = parse('1').Q\n")

    assert main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "bad.py:1:16: BAIL001" in out
    assert "vendored" not in out


def test_main_exits_zero_when_clean(tmp_path: Path) -> None:
    (tmp_path / "good.py").write_text("x = 1\n")
    assert main([str(tmp_path)]) == 0


def test_main_reports_unparseable_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "broken.py").write_text("def (:\n")
    assert main([str(tmp_path)]) == 2
    assert "cannot check" in capsys.readouterr().err


# BAIL002: `.Q` returning an Err the function does not declare.
#
# These run a real type checker on the project in tests/fixtures/bail002, so
# they take a second or two each. Every function in its app.py is named for
# the outcome it expects.

FIXTURE = Path(__file__).parent / "fixtures" / "bail002"

EXPECTED_LEAKS = [
    (30, 38, "`Err[KeyError]`", "leaks"),
    (35, 23, "`Null()`", "option_in_result"),
    (50, 25, "`Err[KeyError]`", "async_leaks"),
    (66, 29, "`Err[KeyError]`", "method_leaks"),
]


@pytest.mark.parametrize("checker", ["mypy", "pyright"])
def test_bail002_reports_exactly_the_leaks(checker: str) -> None:
    findings = check_types([FIXTURE / "pkg"], checker, FIXTURE)

    assert [
        (f.line, f.column, f.message.split(" can return ")[1].split(",")[0], f.code)
        for f in findings
    ] == [
        (line, column, bailed, "BAIL002") for line, column, bailed, _ in EXPECTED_LEAKS
    ]
    for finding, (_, _, _, function) in zip(findings, EXPECTED_LEAKS, strict=True):
        assert f"`{function}` does not declare" in finding.message


def test_bail002_respects_the_ignore_comment(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(FIXTURE)
    assert main(["--checker", "pyright", "pkg"]) == 1
    out = capsys.readouterr().out
    assert "ignored" not in out
    assert out.count("BAIL002") == len(EXPECTED_LEAKS)


def test_rewrite_keeps_line_numbers_and_finds_the_sink() -> None:
    source = textwrap.dedent("""\
        from bail import bails

        @bails
        def pair(a, b) -> Result[int, str]:
            return Ok((parse(a).Q, parse("é").Q))
        """)
    rewritten, sites = _rewrite(source, _visit(source, "app.py"))
    original_lines = source.splitlines()
    new_lines = rewritten.splitlines()

    assert new_lines[: len(original_lines) - 1] == original_lines[:-1]
    assert [site.line for site in sites] == [5, 5]
    for site in sites:
        sink = new_lines[site.line - 1][site.sink_column - 1 :]
        assert sink.startswith(
            "_bail_typing.cast('_bail_abc.Callable[[Result[int, str]]"
        )


def test_unknown_checker_is_an_error() -> None:
    with pytest.raises(CheckerError, match="unknown checker"):
        check_types([FIXTURE / "pkg"], "pytype", FIXTURE)
