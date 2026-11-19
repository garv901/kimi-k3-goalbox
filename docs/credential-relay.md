# Credential relay: the real Anthropic key never enters rack 10

Problem: every rack-10 tray is reached through the shared `dc` account, which has passwordless sudo and
docker-group membership, i.e. root on the node. Nothing stored on the node (container permissions, run
users, Kubernetes Secrets) is hidden from a dc user. So the run container must hold **no reusable
credential**.

Design (`tools/claude_relay.py`, CPU-tested in `tests/test_claude_relay_cpu.py`):
- The real API key (a dedicated key in its own Anthropic workspace with a hard spend cap, created for the
  pilot/campaign and revoked afterwards) lives on a relay host that dc users cannot reach: the laptop or
  rack 3 tray01 under `kevin`. The relay reads it once from a 0600 file.
- The container gets, via `/srv/ss/creds/claude.env` (root-only, copied into each run user's env by
  `isolation.agent_env`): `ANTHROPIC_BASE_URL=https://<relay>:8443`, `ANTHROPIC_API_KEY=<per-run token>`
  (`ss-run-<run id>-<32 hex>`, generated per run, listed in the relay's `tokens.json`, deleted after the run),
  `NODE_EXTRA_CA_CERTS=/srv/ss/relay-ca.pem` (the relay's private CA, public material; it must live outside the root-only creds dir because the run user reads it).
- `isolation.minimal_claude_json` pre-approves the token in `.claude.json` (`customApiKeyResponses`), and
  the allowlist proxy admits the relay host:port in addition to `*.anthropic.com` (`_allowed(host, port)`).
- The relay accepts only the tray's source IP(s), only known tokens, enforces per-run request and byte caps,
  replaces `x-api-key`/`Authorization` with the real key, forwards to `https://api.anthropic.com` with
  streaming passthrough, and logs one line per request (run, path, status, bytes, ms; **no bodies, no keys**).
- Per run: `run_arm` could later create the token itself (it would need to write `tokens.json` on the relay
  host over ssh); for the pilot the operator creates one token before the run and removes it after.

What a dc user can obtain: a per-run token, usable only from the tray's IP, only while the relay is up,
only within the caps, and only for the Anthropic API (not the account). The relay log shows every use.
What stays open: node root can read the agent's prompts/transcripts (not secret), and can use the token
during the run window from that tray (visible in the relay log as extra requests; the cap bounds the cost).

Billing: the relay path is API billing; Claude Code's OAuth subscription login cannot be relayed this way
(and must never be placed on rack 10). Expected cost per 120-min run is bounded by the per-run cap.

Operations (pilot): `python3 tools/claude_relay.py --listen 0.0.0.0:8443 --tokens tokens.json --key-file
relay.key --allow-ip <tray ip> --cert relay.pem --certkey relay-key.pem --log relay-<date>.jsonl` in a tmux on
the relay host; smoke with `curl -sS https://<relay>:8443/v1/models -H "x-api-key: <token>"` from the tray
(through the container proxy) before the run; stop the relay and delete the token after the run.

## Deployment used for Gate I (2026-11-19)
Relay process on the operator's laptop (`127.0.0.1:8443`, TLS, CN `ss-relay`, SAN DNS:ss-relay + IP:10.13.84.11,
private CA valid 30 days) → reverse ssh tunnel to a kevin-only unix socket on rack 3 tray01
(`ssh -R /tmp/ss-relay-kevin/relay.sock:127.0.0.1:8443`, dir 0700; tray01's sshd forces TCP `-R` listeners onto
0.0.0.0, so a unix socket is used instead) → `socat TCP4-LISTEN:8443,bind=10.13.84.11,range=<tray ip>/32
UNIX-CONNECT:…/relay.sock` as kevin on tray01 (source-host restriction) → container `--add-host ss-relay:10.13.84.11`
→ allowlist proxy admits `ss-relay:8443` → `claude -p` as the run user with `ANTHROPIC_BASE_URL=https://ss-relay:8443`.
Verified with the real Claude Code 2.1.326 binary against a stub upstream: the stub received the relay's key, never
the run token; hostname/CA failures and non-approved source hosts are refused; expired and revoked tokens get 401.
