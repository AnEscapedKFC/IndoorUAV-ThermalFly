"""Run many independent JSONL clients against one local IndoorUAV policy process."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import socketserver
import subprocess
import sys


DEFAULT_SOCKET = Path(__file__).resolve().parent / "runtime" / "policy.sock"


def encode(message):
    return (json.dumps(message, allow_nan=False) + "\n").encode()


def exchange(address, message):
    with socket.socket(socket.AF_UNIX) as conn:
        conn.connect(str(address))
        conn.sendall(encode(message))
        with conn.makefile("rb") as stream:
            answer = stream.readline()
        if not answer:
            raise RuntimeError("Policy service disconnected before responding")
        return json.loads(answer)


def serve(address, command):
    address.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if address.exists():
        raise FileExistsError(f"Socket already exists: {address}")
    old_umask = os.umask(0o077)
    try:
        server = socketserver.UnixStreamServer(str(address), Request)
    finally:
        os.umask(old_umask)
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as worker:
            server.worker = worker
            server.completed = 0
            try:
                print(f"Policy socket {address}; worker PID {worker.pid}", file=sys.stderr, flush=True)
                server.serve_forever(poll_interval=0.25)
            finally:
                worker.stdin.close()
                if worker.poll() is None:
                    worker.terminate()
                    try:
                        worker.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        worker.kill()
    finally:
        server.server_close()
        address.unlink(missing_ok=True)


class Request(socketserver.StreamRequestHandler):
    def handle(self):
        request = {}
        try:
            request = json.loads(self.rfile.readline())
            worker = self.server.worker
            if request == {"broker": "status"}:
                reply = {"pid": worker.pid, "exit_code": worker.poll(), "responses": self.server.completed}
            else:
                if worker.poll() is not None:
                    raise RuntimeError(f"Policy worker exited with code {worker.returncode}")
                worker.stdin.write(encode(request))
                worker.stdin.flush()
                line = worker.stdout.readline()
                if not line:
                    raise RuntimeError(f"Policy worker closed stdout (exit {worker.poll()})")
                reply = json.loads(line)
                if "id" in request and reply.get("id") != request["id"]:
                    raise ValueError("Worker response ID differs from request ID")
                self.server.completed += 1
        except (OSError, ValueError, RuntimeError) as error:
            reply = {"id": request.get("id") if isinstance(request, dict) else None, "error": str(error)}
        try:
            self.wfile.write(encode(reply))
        except BrokenPipeError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    sub = parser.add_subparsers(dest="mode")
    broker = sub.add_parser("serve")
    broker.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.mode == "serve":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("serve requires the model worker command")
        def stop_service(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop_service)
        try:
            serve(args.socket, command)
        except KeyboardInterrupt:
            pass
    elif args.mode == "status":
        print(json.dumps(exchange(args.socket, {"broker": "status"})))
    else:
        for line in sys.stdin:
            if line.strip():
                print(json.dumps(exchange(args.socket, json.loads(line))), flush=True)


if __name__ == "__main__":
    main()
