"""`@catching` turns the listed exceptions into an `Err`; others still raise."""

from bail import Err, Ok, catching


@catching(ValueError)
def parse_port(text: str) -> int:
    return int(text)


match parse_port("80x"):
    case Ok(port):
        print(f"port {port}")
    case Err(error):
        print(f"bad port: {error}")  # bad port: invalid literal for int() ...
