"""`.Q` returns the value, or returns the Err from the `@bails` function early."""

from bail import Err, Ok, Result, bails


def parse_number(text: str) -> Result[int, str]:
    return Ok(int(text)) if text.isdigit() else Err(f"not a number: {text!r}")


@bails
def add(a: str, b: str) -> Result[int, str]:
    return Ok(parse_number(a).Q + parse_number(b).Q)


print(add("1", "2"))  # Ok(3)
print(add("1", "x"))  # Err("not a number: 'x'")
