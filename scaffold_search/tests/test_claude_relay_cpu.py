"""CPU: credential relay (tools/claude_relay.py) against a local stub upstream: the real key never reaches
the client side, per-run tokens authenticate, unknown tokens / other IPs / exhausted caps are refused,
streaming (chunked) responses pass through, and the log carries no bodies or keys.

  python3.12 -m pytest -q scaffold_search/tests/test_claude_relay_cpu.py
"""
import http.client, importlib.util, json, os, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location('claude_relay', f'{ROOT}/tools/claude_relay.py')
relay = importlib.util.module_from_spec(spec); spec.loader.exec_module(relay)

REAL_KEY = 'sk-ant-REAL-0000000000000000000000000000'
SEEN = []


class Stub(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n)
        SEEN.append({'key': self.headers.get('x-api-key'), 'auth': self.headers.get('authorization'),
                     'path': self.path, 'body': body, 'host': self.headers.get('Host')})
        if self.path.endswith('stream'):
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Transfer-Encoding', 'chunked'); self.end_headers()
            for ev in (b'event: a\ndata: {}\n\n', b'event: b\ndata: {"done":1}\n\n'):
                self.wfile.write(b'%x\r\n%s\r\n' % (len(ev), ev)); self.wfile.flush()
            self.wfile.write(b'0\r\n\r\n')
        else:
            out = json.dumps({'ok': True, 'echo_len': n}).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(out))); self.end_headers(); self.wfile.write(out)


def _start():
    stub = ThreadingHTTPServer(('127.0.0.1', 0), Stub); stub.daemon_threads = True
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    d = tempfile.mkdtemp()
    open(f'{d}/key', 'w').write(REAL_KEY + '\n'); os.chmod(f'{d}/key', 0o600)
    import time
    json.dump({'ss-run-r01-aaaa': 'r01', 'ss-run-r02-bbbb': {'run': 'r02', 'exp': int(time.time()) + 3600},
               'ss-run-r03-cccc': {'run': 'r03', 'exp': int(time.time()) - 1}}, open(f'{d}/tokens.json', 'w'))
    a = relay.parse(['--listen', '127.0.0.1:0', '--tokens', f'{d}/tokens.json', '--key-file', f'{d}/key',
                     '--allow-ip', '127.0.0.1', '--upstream', f'http://127.0.0.1:{stub.server_port}',
                     '--max-requests', '3', '--max-mb', '1', '--log', f'{d}/relay.jsonl'])
    srv = relay.serve(a)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, stub, d


def _post(port, path, token, body=b'{"model":"claude-opus-5-5"}', auth=None):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    h = {'Content-Type': 'application/json'}
    if token:
        h['x-api-key'] = token
    if auth:
        h['Authorization'] = auth
    c.request('POST', path, body=body, headers=h)
    r = c.getresponse(); data = r.read(); c.close()
    return r.status, data, dict(r.getheaders())


def test_relay_swaps_key_authenticates_runs_and_enforces_caps():
    srv, stub, d = _start()
    port = srv.server_address[1]
    SEEN.clear()
    st, data, _ = _post(port, '/v1/messages', 'ss-run-r01-aaaa')
    assert st == 200 and json.loads(data)['ok'] and SEEN[-1]['key'] == REAL_KEY and SEEN[-1]['auth'] is None
    assert SEEN[-1]['host'] == '127.0.0.1' and SEEN[-1]['body'] == b'{"model":"claude-opus-5-5"}'
    # bearer form of the token works too
    st, _, _ = _post(port, '/v1/messages', None, auth='Bearer ss-run-r02-bbbb')
    assert st == 200 and SEEN[-1]['key'] == REAL_KEY
    # unknown token / no token / expired token
    assert _post(port, '/v1/messages', 'ss-run-r09-zzzz')[0] == 401
    assert _post(port, '/v1/messages', None)[0] == 401
    assert _post(port, '/v1/messages', 'ss-run-r03-cccc')[0] == 401
    assert len(SEEN) == 2
    # streaming passthrough
    st, data, hdr = _post(port, '/v1/messages?stream', 'ss-run-r01-aaaa')
    assert st == 200 and b'"done":1' in data and 'text/event-stream' in hdr.get('Content-Type', '')
    # cap: r01 has used 2 of 3 requests; third ok, fourth refused; r02 unaffected
    assert _post(port, '/v1/messages', 'ss-run-r01-aaaa')[0] == 200
    assert _post(port, '/v1/messages', 'ss-run-r01-aaaa')[0] == 429
    assert _post(port, '/v1/messages', 'ss-run-r02-bbbb')[0] == 200
    # revocation: remove r02's token from the file -> next request refused
    json.dump({'ss-run-r01-aaaa': 'r01'}, open(f'{d}/tokens.json', 'w'))
    assert _post(port, '/v1/messages', 'ss-run-r02-bbbb')[0] == 401
    log = open(f'{d}/relay.jsonl').read()
    assert REAL_KEY not in log and 'claude-opus-5-5' not in log and '"deny": "run cap exceeded"' in log
    assert '"deny": "expired run token"' in log and 'ss-run-' not in log   # tokens are not logged either
    rows = [json.loads(l) for l in log.splitlines()]
    assert {r.get('run') for r in rows if r.get('status') == 200} == {'r01', 'r02'}
    assert all('bytes_in' in r for r in rows if r.get('status') == 200)
    srv.shutdown(); stub.shutdown()


def test_relay_refuses_other_source_ips():
    srv, stub, d = _start()
    srv.state.allow_ip = {'10.13.84.14'}
    assert _post(srv.server_address[1], '/v1/messages', 'ss-run-r01-aaaa')[0] == 403
    rows = [json.loads(l) for l in open(f'{d}/relay.jsonl')]
    assert rows[-1]['deny'] == 'ip not allowed' and rows[-1]['ip'] == '127.0.0.1'
    srv.shutdown(); stub.shutdown()
