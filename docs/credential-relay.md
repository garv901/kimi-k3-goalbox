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
  `NODE_EXTRA_CA_CERTS=/srv/ss/creds/relay-ca.pem` (the relay's private CA, public material).
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
