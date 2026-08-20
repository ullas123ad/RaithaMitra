"""
Temporary Dev Server for RaithaMitra Voice Tester
=================================================
Serves dev_voice_test/index.html on http://127.0.0.1:8080 using ThreadingHTTPServer.
Handles concurrent requests cleanly with no hanging connections.
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

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Connection", "close")
        super().end_headers()

    def log_message(self, format, *args):
        # Clean console log format without emoji crashes
        sys.stderr.write(f"[DevUI] {self.address_string()} - {format%args}\n")


def run():
    server_address = ("127.0.0.1", PORT)
    httpd = http.server.ThreadingHTTPServer(server_address, Handler)
    print("=================================================================")
    print(f"RaithaMitra Temporary Voice Tester UI Running!")
    print(f"Open in your browser: http://127.0.0.1:{PORT}")
    print(f"Backend API endpoint: http://127.0.0.1:5000/api/v1/advisory/audio")
    print("=================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dev UI server.")


if __name__ == "__main__":
    run()
