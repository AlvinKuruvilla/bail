"""A Result is the outcome of something that can fail: `Ok(value)` or `Err(error)`."""

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
