# Logins: Claude Code and GitHub

All commands as root inside the container. Never commit or copy credentials into this repo.

## Claude Code (required)
The agents run `claude -p` (2.1.326, model `claude-opus-5-5`) as per-run Unix users. The harness
gives each run user a copy of root's login and keeps the copies in sync every minute (OAuth tokens
refresh/rotate; `isolation.sync_creds`). The account needs enough usage for **3 concurrent
agents** for 2 h each, plus the subagents A3 starts (up to 4 engine subagents per A3 run). Expect
roughly $10-120 of API-equivalent usage per run. Rate-limit stalls count as lost time
(PROTOCOL.md) and are not rerun, so a plan that rate-limits will depress scores.

**A. OAuth login (recommended)**
```sh
claude            # interactive; run /login, pick the subscription account, finish in a browser
                  # (copy the URL if the container has no browser), then /exit
ls -l /root/.claude/.credentials.json /root/.claude.json    # both must exist
bash scripts/bootstrap_node.sh --check-login                # runs one tiny `claude -p`
```

**B. Long-lived token or API key**
```sh
claude setup-token                      # prints a CLAUDE_CODE_OAUTH_TOKEN (subscription), or use an API key
install -m 600 /dev/null /srv/ss/creds/claude.env
echo 'CLAUDE_CODE_OAUTH_TOKEN=...' >> /srv/ss/creds/claude.env   # or ANTHROPIC_API_KEY=sk-...
```
`isolation.agent_env` adds these variables to every run user's environment. If
`/root/.claude/.credentials.json` also exists, Claude Code may prefer one or the other; use one
method only. With an API key, usage is billed per token (see costs in the README).

The run users' homes contain only the credential copy, a minimal `settings.json` (WebFetch and
WebSearch denied, no auto-update/telemetry) and a minimal `.claude.json`; root's history,
projects and settings are not visible to them.

## GitHub (needed for private clones and for snapshot pushes)
```sh
gh auth login                 # account with read access to kamahori/vibe-megakernel
gh auth setup-git             # bootstrap_node.sh does this too when gh is logged in
```
Read access needed:
- `kamahori/vibe-megakernel` (benchmark, commit 886eeb3e28db071f6c43b31fd0b6a9c26dbeaf27).
  The seed kernel is vendored in `seeds/` and needs no access.

## Where snapshots go (optional but recommended)
Set `SS_SNAPSHOT_REPO` to a repo you can push to (an empty private repo is best), e.g.
`git@github.com:<you>/<repo>.git` (needs an SSH key in `/root/.ssh`) or
`https://github.com/<you>/<repo>.git` (uses the gh credential helper). Then rerun
`bash scripts/bootstrap_node.sh`, which stores it in `/srv/ss/snapshot_repo` and creates the
`results` branch. The harness (root, never the agent) pushes:
- `run/<run_id>-<arm>`: the agent's workspace (no build outputs or files > 50 MB), transcript,
  telemetry, snapshot index; every 10 min, at each passing check and at the end;
- `results`: `results/<run_id>/result.json` (+ telemetry, run.json, checks), `results/_ledger/`,
  `results/_schedules/`.
Without `SS_SNAPSHOT_REPO` nothing is pushed; everything stays under `/srv/ss/runs/<run_id>/`
(back it up yourself). Agents never hold GitHub credentials: run users cannot read `/root`.
