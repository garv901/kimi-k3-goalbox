#!/usr/bin/env python3
"""Credential relay for scaffold-search runs: the real Anthropic API key never leaves this host.

Runs on a host the run users and the shared `dc` account cannot reach (laptop, or rack 3 tray01 as kevin).
Claude Code in the run container is pointed at it with ANTHROPIC_BASE_URL=https://<relay>:<port> and
ANTHROPIC_API_KEY=<per-run token>; the relay authenticates the run by that token, checks the client IP
against an allowlist, enforces per-run request/byte caps, swaps in the real key and forwards the request
to https://api.anthropic.com unchanged (streaming passthrough). Bodies are never logged; one JSONL line per
request: time, run, method, path, status, bytes in/out, duration.

  python3 claude_relay.py --listen 0.0.0.0:8443 --tokens tokens.json --key-file ./relay.key \\
      --allow-ip 10.13.84.14 --cert relay.pem --certkey relay-key.pem [--max-requests 4000] [--max-mb 2000]
  tokens.json: {"ss-run-r01-<random>": "r01", ...}   (one token per run id; delete the line to revoke)
  The real key is read from --key-file (0600) once at start and kept only in memory.

Test without the network: --upstream http://127.0.0.1:PORT (plain http stub).
"""
import argparse, http.client, json, os, ssl, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailers',
       'transfer-encoding', 'upgrade', 'host', 'x-api-key', 'authorization'}


class State:
    def __init__(self, a):
        self.a = a
        self.key = open(a.key_file).read().strip()
        self.tokens = json.load(open(a.tokens))          # token -> run id
        self.allow_ip = set(a.allow_ip or [])
        self.use = {}                                    # run id -> {'requests': n, 'bytes': n}
        self.lock = threading.Lock()
        self.log = open(a.log, 'a')
        u = urlsplit(a.upstream)
        self.up_host, self.up_port, self.up_tls = u.hostname, u.port or (443 if u.scheme == 'https' else 80), u.scheme == 'https'

    def reload_tokens(self):
        try:
            self.tokens = json.load(open(self.a.tokens))
        except (OSError, ValueError):
            pass

    def record(self, **row):
        with self.lock:
            self.log.write(json.dumps({'t': round(time.time(), 3), **row}) + '\n'); self.log.flush()

    def charge(self, run, nbytes):
        with self.lock:
            u = self.use.setdefault(run, {'requests': 0, 'bytes': 0})
            u['requests'] += 1; u['bytes'] += nbytes
            return u['requests'] <= self.a.max_requests and u['bytes'] <= self.a.max_mb * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'ss-relay/1'

    def log_message(self, *a):   # quiet; we write our own log
        pass

    def _deny(self, code, why, run=None):
        self.send_response(code); self.send_header('Content-Length', '0'); self.end_headers()
        self.server.state.record(run=run, ip=self.client_address[0], method=self.command, path=self.path[:120],
                                 status=code, deny=why)

    def do_ANY(self):
        st = self.server.state
        ip = self.client_address[0]
        if st.allow_ip and ip not in st.allow_ip:
            return self._deny(403, 'ip not allowed')
        tok = self.headers.get('x-api-key') or (self.headers.get('authorization') or '').removeprefix('Bearer ').strip()
        st.reload_tokens()
        run = st.tokens.get(tok)
        if not run:
            return self._deny(401, 'unknown run token')
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n) if n else b''
        if not st.charge(run, n):
            return self._deny(429, 'run cap exceeded', run)
        t0 = time.time()
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        hdrs['x-api-key'] = st.key
        hdrs['Host'] = st.up_host
        try:
            conn = (http.client.HTTPSConnection if st.up_tls else http.client.HTTPConnection)(st.up_host, st.up_port, timeout=600)
            conn.request(self.command, self.path, body=body, headers=hdrs)
            r = conn.getresponse()
        except OSError as e:
            return self._deny(502, f'upstream: {e.__class__.__name__}', run)
        self.send_response(r.status)
        chunked = r.getheader('Transfer-Encoding', '').lower() == 'chunked'
        for k, v in r.getheaders():
            if k.lower() not in ('connection', 'transfer-encoding', 'content-length'):
                self.send_header(k, v)
        out = 0
        if chunked:
            self.send_header('Transfer-Encoding', 'chunked'); self.end_headers()
            while True:
                chunk = r.read1(65536) if hasattr(r, 'read1') else r.read(65536)
                if not chunk:
                    break
                self.wfile.write(b'%x\r\n%s\r\n' % (len(chunk), chunk)); self.wfile.flush(); out += len(chunk)
            self.wfile.write(b'0\r\n\r\n')
        else:
            data = r.read(); out = len(data)
            self.send_header('Content-Length', str(out)); self.end_headers()
            self.wfile.write(data)
        conn.close()
        st.record(run=run, ip=ip, method=self.command, path=self.path[:120], status=r.status, bytes_in=n,
                  bytes_out=out, ms=round((time.time() - t0) * 1000), stream=chunked)

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = do_ANY


def serve(a):
    st = State(a)
    host, port = a.listen.rsplit(':', 1)
    srv = ThreadingHTTPServer((host, int(port)), Handler)
    srv.daemon_threads = True
    srv.state = st
    if a.cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(a.cert, a.certkey)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    print(f'relay on {a.listen} -> {a.upstream}; {len(st.tokens)} run tokens; ip allowlist {sorted(st.allow_ip) or "any"}; '
          f'caps {a.max_requests} req / {a.max_mb} MB per run; log {a.log}', flush=True)
    return srv


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--listen', default='127.0.0.1:8443')
    ap.add_argument('--tokens', required=True); ap.add_argument('--key-file', required=True)
    ap.add_argument('--allow-ip', action='append')
    ap.add_argument('--cert'); ap.add_argument('--certkey')
    ap.add_argument('--upstream', default='https://api.anthropic.com')
    ap.add_argument('--max-requests', type=int, default=4000); ap.add_argument('--max-mb', type=int, default=2000)
    ap.add_argument('--log', default='relay.jsonl')
    return ap.parse_args(argv)


if __name__ == '__main__':
    serve(parse(sys.argv[1:])).serve_forever()
