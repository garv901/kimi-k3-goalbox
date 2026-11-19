#!/usr/bin/env python3
"""Network and API boundary tests run *inside* a disposable C1 test pod (stdlib only; python:3-slim).

    python3 /c1/c1_net_tests.py --role agent|unlabeled [--proxy goalbox-proxy:3128] [--json]

Each test states the expected outcome for the role; the script prints one line per test and a JSON
summary, exit 0 only if every outcome matched. "blocked" means the connection could not be made
(connect timeout or refused/unreachable within TIMEOUT), which is how a default-deny NetworkPolicy shows up.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
import sys
import time

TIMEOUT = 6.0


def tcp(host: str, port: int) -> str:
    try:
        with socket.create_connection((host, port), timeout=TIMEOUT):
            return "open"
    except socket.timeout:
        return "blocked"
    except (ConnectionRefusedError, OSError) as exc:
        return f"blocked({type(exc).__name__})"


def resolve(name: str, server: str | None = None) -> str:
    """DNS lookup through the resolver (server=None) or a raw UDP query to `server`."""
    if server is None:
        try:
            socket.getaddrinfo(name, 443, proto=socket.IPPROTO_TCP)
            return "resolved"
        except socket.gaierror as exc:
            return f"failed({exc.errno})"
    # minimal DNS A query
    import random
    import struct
    qid = random.randint(0, 65535)
    labels = b"".join(bytes([len(p)]) + p.encode() for p in name.split("."))
    packet = struct.pack(">HHHHHH", qid, 0x0100, 1, 0, 0, 0) + labels + b"\x00" + struct.pack(">HH", 1, 1)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(TIMEOUT)
    try:
        s.sendto(packet, (server, 53))
        s.recvfrom(512)
        return "answered"
    except socket.timeout:
        return "blocked"
    except OSError as exc:
        return f"blocked({type(exc).__name__})"
    finally:
        s.close()


def via_proxy(proxy: str, method: str, target: str, extra: str = "") -> str:
    host, port = proxy.rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port)), timeout=TIMEOUT) as s:
            if method == "CONNECT":
                s.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
            else:
                s.sendall(f"{method} {target} HTTP/1.1\r\nHost: {target.split('/')[2]}\r\n{extra}\r\n".encode())
            s.settimeout(TIMEOUT)
            head = s.recv(1024).decode("latin-1", "replace").split("\r\n")[0]
            return head.split(" ")[1] if head.startswith("HTTP/") else f"noreply({head[:40]!r})"
    except socket.timeout:
        return "blocked"
    except OSError as exc:
        return f"blocked({type(exc).__name__})"


def tls_through_proxy(proxy: str, target_host: str) -> str:
    """CONNECT then a TLS handshake to the allowed host: proves the tunnel carries TLS to that host only."""
    host, port = proxy.rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port)), timeout=TIMEOUT) as s:
            s.sendall(f"CONNECT {target_host}:443 HTTP/1.1\r\nHost: {target_host}:443\r\n\r\n".encode())
            head = s.recv(1024).decode("latin-1", "replace").split("\r\n")[0]
            if " 200 " not in head:
                return f"proxy:{head.split(' ')[1] if head.startswith('HTTP/') else head[:30]}"
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(s, server_hostname=target_host) as tls:
                tls.sendall(f"GET / HTTP/1.1\r\nHost: {target_host}\r\nConnection: close\r\n\r\n".encode())
                reply = tls.recv(200).decode("latin-1", "replace").split("\r\n")[0]
                return f"tls_ok:{reply.split(' ')[1] if reply.startswith('HTTP/') else reply[:30]}"
    except ssl.SSLCertVerificationError as exc:
        return f"tls_cert_error({exc.verify_message})"
    except socket.timeout:
        return "blocked"
    except OSError as exc:
        return f"blocked({type(exc).__name__})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["agent", "unlabeled"], required=True)
    parser.add_argument("--proxy", default=os.environ.get("C1_PROXY", "goalbox-proxy:3128"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    agent = args.role == "agent"
    blocked = lambda r: r.startswith("blocked")        # noqa: E731
    api_host = os.environ.get("KUBERNETES_SERVICE_HOST") or "kubernetes.default.svc.cluster.local"
    proxy_host = args.proxy.rsplit(":", 1)[0]
    tests = [
        # name, callable, expectation for this role (predicate), description
        ("api_token_absent", lambda: "absent" if not os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token") else "PRESENT",
         lambda r: r == "absent", "no ServiceAccount token mounted"),
        ("dns_cluster", lambda: resolve(proxy_host + ".goalbox-k3.svc.cluster.local"),
         (lambda r: r == "resolved") if agent else (lambda r: r.startswith("failed")), "cluster DNS through kube-dns"),
        ("dns_external_direct", lambda: resolve("example.com", server="8.8.8.8"), blocked, "raw DNS to a public resolver"),
        ("dns_external_direct_2", lambda: resolve("example.com", server="1.1.1.1"), blocked, "raw DNS to a second public resolver"),
        ("direct_https_external", lambda: tcp("1.1.1.1", 443), blocked, "direct external HTTPS by IP"),
        ("direct_https_external_name", lambda: tcp("api.anthropic.com", 443) if agent else "skipped(no dns)",
         (blocked if agent else (lambda r: r.startswith("skipped"))), "direct external HTTPS by name"),
        ("kubernetes_api", lambda: tcp(api_host, 443), blocked, "Kubernetes API server"),
        ("kubernetes_api_6443", lambda: tcp(api_host, 6443), blocked, "Kubernetes API server (6443)"),
        ("metadata_endpoint", lambda: tcp("169.254.169.254", 80), blocked, "cloud metadata endpoint"),
        ("link_local_metadata_https", lambda: tcp("169.254.169.254", 443), blocked, "cloud metadata endpoint (443)"),
        ("node_ssh", lambda: tcp(os.environ.get("C1_NODE_IP", "10.13.84.1"), 22), blocked, "a cluster node address (ssh)"),
        ("proxy_other_port", lambda: tcp(proxy_host, 8080), blocked, "proxy pod on a port other than 3128"),
        ("proxy_reachable", lambda: via_proxy(args.proxy, "CONNECT", "api.anthropic.com:443"),
         (lambda r: r == "200") if agent else blocked, "CONNECT api.anthropic.com:443 through the proxy"),
    ]
    if agent:
        tests += [
            ("proxy_tls_anthropic", lambda: tls_through_proxy(args.proxy, "api.anthropic.com"), lambda r: r.startswith("tls_ok"),
             "TLS handshake to api.anthropic.com inside the tunnel"),
            ("proxy_connect_huggingface", lambda: via_proxy(args.proxy, "CONNECT", "huggingface.co:443"), lambda r: r == "200", "CONNECT huggingface.co:443"),
            ("proxy_connect_pypi_files", lambda: via_proxy(args.proxy, "CONNECT", "files.pythonhosted.org:443"), lambda r: r == "200", "CONNECT files.pythonhosted.org:443"),
            ("proxy_connect_github", lambda: via_proxy(args.proxy, "CONNECT", "github.com:443"), lambda r: r == "403", "CONNECT github.com:443 (not allowlisted)"),
            ("proxy_connect_pypi_org", lambda: via_proxy(args.proxy, "CONNECT", "pypi.org:443"), lambda r: r == "403", "CONNECT pypi.org:443 (index only via pypi.goalbox)"),
            ("proxy_connect_wildcard_hf", lambda: via_proxy(args.proxy, "CONNECT", "exfil-data.huggingface.co:443"), lambda r: r == "403", "CONNECT to an arbitrary huggingface.co subdomain"),
            ("proxy_connect_allowed_host_other_port", lambda: via_proxy(args.proxy, "CONNECT", "api.anthropic.com:22"), lambda r: r == "403", "CONNECT allowed host on port 22"),
            ("proxy_connect_allowed_host_port_80", lambda: via_proxy(args.proxy, "CONNECT", "api.anthropic.com:80"), lambda r: r == "403", "CONNECT allowed host on port 80"),
            ("proxy_connect_ip", lambda: via_proxy(args.proxy, "CONNECT", "1.1.1.1:443"), lambda r: r == "403", "CONNECT to a raw IP"),
            ("proxy_connect_metadata", lambda: via_proxy(args.proxy, "CONNECT", "169.254.169.254:443"), lambda r: r == "403", "CONNECT to the metadata address"),
            ("proxy_plain_http_external", lambda: via_proxy(args.proxy, "GET", "http://example.com/"), lambda r: r == "403", "plain HTTP GET to an external site"),
            ("proxy_plain_http_anthropic", lambda: via_proxy(args.proxy, "GET", "http://api.anthropic.com/"), lambda r: r == "403", "plain HTTP GET to an allowed host"),
            ("proxy_post_index", lambda: via_proxy(args.proxy, "POST", "http://pypi.goalbox/simple/numpy/", "Content-Length: 0\r\n"), lambda r: r == "403", "POST to the package index"),
            ("proxy_pypi_index_allowed", lambda: via_proxy(args.proxy, "GET", "http://pypi.goalbox/simple/numpy/"), lambda r: r == "200", "GET pypi.goalbox/simple/numpy/"),
            ("proxy_pypi_index_blocked_pkg", lambda: via_proxy(args.proxy, "GET", "http://pypi.goalbox/simple/vllm/"), lambda r: r == "403", "GET pypi.goalbox/simple/vllm/ (blocked package)"),
            ("proxy_pypi_index_blocked_pkg2", lambda: via_proxy(args.proxy, "GET", "http://pypi.goalbox/simple/flashinfer-python/"), lambda r: r == "403", "GET blocked package flashinfer"),
        ]
    results = []
    ok_all = True
    for name, fn, expect, desc in tests:
        start = time.time()
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001
            result = f"error({type(exc).__name__}: {exc})"
        ok = bool(expect(result))
        ok_all &= ok
        results.append({"test": name, "result": result, "ok": ok, "s": round(time.time() - start, 1), "what": desc})
        print(f"{'ok  ' if ok else 'FAIL'} {name:40s} {result:30s} {desc}", flush=True)
    summary = {"role": args.role, "ok": ok_all, "passed": sum(r["ok"] for r in results), "total": len(results), "results": results}
    print(json.dumps(summary) if args.json else f"{summary['passed']}/{summary['total']} as expected; ok={ok_all}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
