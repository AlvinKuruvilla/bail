"""An exception named in the declared error type comes back as an `Err`."""

import json

from bail import Err, Ok, Result, bails


@bails
def load_port(raw: str) -> Result[int, KeyError | json.JSONDecodeError]:
    return Ok(int(json.loads(raw)["port"]))


match load_port('{"host": "db"}'):
    case Ok(port):
        print(f"port {port}")
    case Err(error):
        print(f"bad config: {error!r}")  # bad config: KeyError('port')
