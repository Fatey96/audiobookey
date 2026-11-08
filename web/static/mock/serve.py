#!/usr/bin/env python3
"""Mock static server for testing the reader frontend without the backend.

  python web/static/mock/serve.py [port]

Serves web/static/ and then open http://127.0.0.1:PORT/?mock .
The ?mock flag makes app.js dynamic-import mock/mock.js, a fetch shim, so
nothing here touches the real app path. Exists because Windows mimetypes
registry sometimes serves .js as text/plain, which breaks ES modules.
"""

import mimetypes
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


def run(port: int) -> None:
    mimetypes.add_type("application/javascript", ".js")
    handler = partial(SimpleHTTPRequestHandler, directory="web/static")
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"serving web/static on http://127.0.0.1:{port}/?mock (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 8900)
