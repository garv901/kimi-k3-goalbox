#!/usr/bin/env python3
"""Allowlist proxy for one goalbox run. Listens on RUN_DIR/sock/net.sock (host side).

- CONNECT host:443 is tunneled only for hosts in ALLOW_CONNECT or ALLOW_PATTERNS (Hugging Face).
- http://pypi.goalbox/simple/<name>/ is served from https://pypi.org/simple/<name>/
  unless <name> matches a pattern in goalbox/blocked-packages.txt (then 403).
  Wheel files come from files.pythonhosted.org through CONNECT; pypi.org itself is not
  reachable, so the index can't be bypassed.
- Everything else is refused. Every decision is logged to RUN_DIR/net.log.
"""
import asyncio, fnmatch, re, sys, time, urllib.request
from pathlib import Path

RUN = Path(sys.argv[1]).resolve()
GB = Path(__file__).resolve().parents[1]
ALLOW_CONNECT = {"api.anthropic.com", "files.pythonhosted.org"}
# Hugging Face: model cards, configs, modeling code and weights (hub API, LFS and Xet storage).
ALLOW_PATTERNS = ("huggingface.co", "*.huggingface.co", "hf.co", "*.hf.co")
BLOCKED = [l.split("#")[0].strip() for l in (GB / "blocked-packages.txt").read_text().splitlines()]
BLOCKED = [p for p in BLOCKED if p]
LOG = (RUN / "net.log").open("a", buffering=1)


def log(verdict, what):
    LOG.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {verdict:5} {what}\n")


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def blocked(name):
    return any(fnmatch.fnmatch(norm(name), p) for p in BLOCKED)


async def pipe(r, w):
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try: w.close()
        except Exception: pass


def reply(w, code, reason, body=b"", ctype="text/plain"):
    w.write(f"HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)


def fetch_index(name, accept):
    req = urllib.request.Request(f"https://pypi.org/simple/{name}/",
                                 headers={"Accept": accept or "text/html", "User-Agent": "goalbox-proxy"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.headers.get("Content-Type", "text/html"), r.read()


async def handle(cr, cw):
    try:
        head = await asyncio.wait_for(cr.readuntil(b"\r\n\r\n"), 60)
    except Exception:
        cw.close(); return
    lines = head.decode("latin-1").split("\r\n")
    method, target, _ = (lines[0].split(" ") + ["", ""])[:3]
    headers = {k.lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:] if l)}
    if method == "CONNECT":
        host, _, port = target.rpartition(":")
        if (host in ALLOW_CONNECT or any(fnmatch.fnmatch(host.lower(), p) for p in ALLOW_PATTERNS)) and port == "443":
            try:
                ur, uw = await asyncio.open_connection(host, 443)
            except OSError as e:
                log("ERROR", f"CONNECT {target} {e}"); reply(cw, 502, "Bad Gateway"); cw.close(); return
            log("allow", f"CONNECT {target}")
            cw.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await cw.drain()
            await asyncio.gather(pipe(cr, uw), pipe(ur, cw))
        else:
            log("DENY", f"CONNECT {target}")
            reply(cw, 403, "Forbidden", f"goalbox: {host} is not on the network allowlist\n".encode())
            cw.close()
        return
    m = re.match(r"http://pypi\.goalbox(/simple/([^/?]+)/?)?", target)
    if method == "GET" and m and m.group(2):
        name = m.group(2)
        if blocked(name):
            log("DENY", f"pypi {name}")
            reply(cw, 403, "Forbidden", f"goalbox: package {name} is blocked for this benchmark\n".encode())
        else:
            try:
                status, ctype, body = await asyncio.to_thread(fetch_index, norm(name), headers.get("accept"))
                log("allow", f"pypi {name}")
                reply(cw, status, "OK", body, ctype)
            except urllib.error.HTTPError as e:
                log("allow", f"pypi {name} -> {e.code}")
                reply(cw, e.code, "Upstream")
            except Exception as e:
                log("ERROR", f"pypi {name} {e}")
                reply(cw, 502, "Bad Gateway")
    else:
        log("DENY", f"{method} {target}")
        reply(cw, 403, "Forbidden", b"goalbox: only the PyPI index is served over plain HTTP\n")
    await cw.drain()
    cw.close()


async def main():
    sock = RUN / "sock/net.sock"
    sock.unlink(missing_ok=True)
    server = await asyncio.start_unix_server(handle, path=str(sock))
    log("start", f"allow={sorted(ALLOW_CONNECT)}+{ALLOW_PATTERNS} blocked={len(BLOCKED)} patterns")
    async with server:
        await server.serve_forever()

asyncio.run(main())
