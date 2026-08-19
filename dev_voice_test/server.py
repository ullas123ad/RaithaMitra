"""
Temporary Dev Server for RaithaMitra Voice Tester
=================================================
Serves dev_voice_test/index.html on http://127.0.0.1:8080.
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

PORT = 8080
DIRECTORY = Path(__file__).parent.resolve()


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def log_message(self, format, *args):
        # Keep dev server logs clean
        sys.stderr.write(f"[DevUI] {self.address_string()} - {format%args}\n")


def run():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"=================================================================")
        print(f"🌾 RaithaMitra Temporary Voice Tester UI Running!")
        print(f"👉 Open in your browser: http://127.0.0.1:{PORT}")
        print(f"👉 Backend API endpoint: http://127.0.0.1:5000/api/v1/advisory/audio")
        print(f"=================================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down dev UI server.")


if __name__ == "__main__":
    run()
