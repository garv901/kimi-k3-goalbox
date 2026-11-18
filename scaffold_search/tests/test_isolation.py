"""Isolation tests (node, root, needs the ss-proxy and ss-lease daemons running).

  python3.12 scaffold_search/tests/test_isolation.py [--claude]

Creates run users ss-isoA / ss-isoB with run dirs under /srv/ss/runs, then checks what ss-isoA
can and cannot do. --claude also runs `claude -p` as the user (costs a few cents).
"""
import os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scaffold_search import isolation as iso  # noqa: E402

RUNS = '/srv/ss/runs'


def setup(run_id):
    d = f'{RUNS}/{run_id}'
    uid, gid = iso.create_user(run_id, d)
    os.makedirs(f'{d}/ws', exist_ok=True)
    os.chown(f'{d}/ws', uid, gid); os.chmod(f'{d}/ws', 0o700)
    open(f'{d}/ws/secret.txt', 'w').write('B-only\n'); os.chown(f'{d}/ws/secret.txt', uid, gid)
    return d


def as_user(run_id, cmd, timeout=120):
    d = f'{RUNS}/{run_id}'
    p = subprocess.run(['bash', '-c', cmd], cwd=f'{d}/ws', env=iso.agent_env(run_id, d),
                       preexec_fn=iso.demote(run_id), capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


CASES = [
    # (description, command, expect_success)
    ('read other run NOTES under /root', 'cat /root/megabench-runs/*/NOTES.md', False),
    ('list /root', 'ls /root', False),
    ('read gh config', 'ls /root/.config/gh', False),
    ('read root claude history', 'cat /root/.claude/history.jsonl', False),
    ('list /srv/ss/runs', 'ls /srv/ss/runs', False),
    ('read other run ws', f'cat {RUNS}/isoB2/ws/secret.txt', False),
    ('read other run home', f'ls {RUNS}/isoB2/home', False),
    ('read builder trees', 'ls /srv/ss/build', False),
    ('read harness checkout', 'ls /srv/ss/harness', False),
    ('read lease log', 'cat /srv/ss/lease.jsonl', False),
    ('list /tmp', 'ls /tmp', False),
    ('read a root file left in /tmp', 'cat /tmp/ss_leak_test.log', False),
    ('read trusted benchmark', 'ls /srv/ss/trusted/vibe-megakernel/megabench | head -3', True),
    ('write trusted benchmark', 'touch /srv/ss/trusted/vibe-megakernel/x', False),
    ('import megabench', 'python3.12 -c "import megabench, sys; print(megabench.__file__)"', True),
    ('own workspace write', 'echo hi > mine.txt && cat mine.txt', True),
    ('GPU without lease', 'python3.12 -c "import torch; assert torch.cuda.device_count() > 0 and torch.cuda.is_available(); torch.zeros(1).cuda()"', False),
    ('GPU through lease', '/srv/ss/bin/gpu_lease run --gpus 1 -- python3.12 -c "import torch; x=torch.ones(4).cuda(); print(torch.cuda.device_count(), float(x.sum()))"', True),
    ('curl github through proxy env', 'curl -sS -m 20 -o /dev/null -w "%{http_code}" https://github.com', False),
    ('pip download', 'timeout 60 python3.12 -m pip download --no-deps -d $TMPDIR/p requests', False),
    ('anthropic API reachable', 'curl -sS -m 20 -o /dev/null -w "%{http_code}" https://api.anthropic.com/v1/models', True),
]


def main():
    open('/tmp/ss_leak_test.log', 'w').write('old check log\n'); os.chmod('/tmp/ss_leak_test.log', 0o644)
    iso.setup_node()
    for r in ('isoA2', 'isoB2'):
        setup(r)
    fails = 0
    for desc, cmd, want in CASES:
        rc, out = as_user('isoA2', cmd)
        ok = (rc == 0) == want
        if desc == 'anthropic API reachable':  # any HTTP status from Anthropic means the path works
            ok = out.strip()[-3:].isdigit() and out.strip()[-3:] != '000'
        fails += not ok
        print(f'{"PASS" if ok else "FAIL"} {desc}: rc={rc} want_success={want} :: {out.strip()[-160:]!r}')
    # lease identity: run id comes from the user name, not SS_RUN_ID
    rc, out = as_user('isoA2', 'SS_RUN_ID=forged /srv/ss/bin/gpu_lease run -v --gpus 1 -- true')
    ok = 'run=isoA2' in out
    fails += not ok
    print(f'{"PASS" if ok else "FAIL"} lease run id from user name :: {out.strip()[-120:]!r}')
    if '--claude' in sys.argv:
        rc, out = as_user('isoA2', 'claude -p "Reply with exactly: ISO_OK" --disallowedTools WebFetch,WebSearch '
                                  '--output-format text', timeout=300)
        ok = 'ISO_OK' in out
        fails += not ok
        print(f'{"PASS" if ok else "FAIL"} claude -p as run user: rc={rc} :: {out.strip()[-200:]!r}')
        rc, out = as_user('isoA2', 'claude -p "Run this exact bash command and print its output: '
                                  'curl -sS -m 15 https://github.com | head -c 200" --permission-mode auto '
                                  '--disallowedTools WebFetch,WebSearch --output-format text', timeout=300)
        print(f'INFO claude bash curl github: rc={rc} :: {out.strip()[-300:]!r}')
    print('FAILS', fails)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
