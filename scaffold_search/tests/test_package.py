"""CPU tests for the packaging layer (scripts/, configurable snapshot repo, optional token auth).

  python3.12 -m pytest -q scaffold_search/tests/test_package.py
"""
import importlib, json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from scaffold_search import schedule, isolation, workspace  # noqa: E402


def test_protocol_manifest_matches():
    """arms, overlays, prompt, harness_files, gpu_lease, patch and seed manifest = PROTOCOL.md"""
    p = subprocess.run([sys.executable, f'{ROOT}/scripts/verify_protocol.py'], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'all match' in p.stdout


def test_schedule_is_the_protocol_one():
    s = schedule.make({'A0S': 4, 'A1': 4, 'A2': 4, 'A3': 4}, 3, 20261118, 'r')
    got = ' '.join(f"{r['run_id']}:{r['arm']}" for r in s['order'])
    assert got == ('r01:A2 r02:A2 r03:A0S r04:A1 r05:A3 r06:A2 r07:A2 r08:A1 r09:A3 r10:A3 '
                   'r11:A0S r12:A3 r13:A0S r14:A0S r15:A1 r16:A1'), got


def test_arms_overlays():
    want = {'A0S': ['S'], 'A1': ['S', 'A1'], 'A2': ['S', 'A1', 'A2'], 'A3': ['S', 'A1', 'A3']}
    for a, ov in want.items():
        arm = workspace.load_arm(f'{ROOT}/arms/{a}.toml')
        assert arm['overlays'] == ov and arm['budget_minutes'] == 120 and arm['task'] == 'improve-b8'


def test_overlays_do_not_collide():
    seen = {}
    for ov in ('S', 'A1', 'A2', 'A3'):
        for r, _, fs in os.walk(f'{ROOT}/overlays/{ov}'):
            for f in fs:
                rel = os.path.relpath(os.path.join(r, f), f'{ROOT}/overlays/{ov}')
                assert rel not in seen, (rel, ov, seen.get(rel))
                seen[rel] = ov


def test_snapshot_remote_configurable(monkeypatch):
    from scaffold_search import snapshot
    monkeypatch.setenv('SS_SNAPSHOT_REPO', 'git@example.com:me/ss.git')
    assert importlib.reload(snapshot).REMOTE == 'git@example.com:me/ss.git'
    monkeypatch.delenv('SS_SNAPSHOT_REPO')
    r = importlib.reload(snapshot).REMOTE
    assert r == snapshot.PLACEHOLDER_REMOTE or os.path.exists('/srv/ss/snapshot_repo')


def test_claude_env_file(monkeypatch):
    d = tempfile.mkdtemp()
    f = f'{d}/claude.env'
    open(f, 'w').write('# comment\nCLAUDE_CODE_OAUTH_TOKEN="tok-123"\nOTHER=x\nANTHROPIC_API_KEY=\n')
    monkeypatch.setattr(isolation, 'CLAUDE_ENV_FILE', f)
    assert isolation.claude_env() == {'CLAUDE_CODE_OAUTH_TOKEN': 'tok-123'}
    env = isolation.agent_env('r01', '/srv/ss/runs/r01')
    assert env['CLAUDE_CODE_OAUTH_TOKEN'] == 'tok-123' and env['SS_RUN_ID'] == 'r01'
    monkeypatch.setattr(isolation, 'CLAUDE_ENV_FILE', f'{d}/missing')
    assert 'CLAUDE_CODE_OAUTH_TOKEN' not in isolation.agent_env('r01', '/srv/ss/runs/r01')


def test_shell_scripts_parse():
    for f in sorted(os.listdir(f'{ROOT}/scripts')):
        if f.endswith('.sh'):
            p = subprocess.run(['bash', '-n', f'{ROOT}/scripts/{f}'], capture_output=True, text=True)
            assert p.returncode == 0, (f, p.stderr)
