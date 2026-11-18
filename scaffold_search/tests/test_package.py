"""CPU tests for the packaging layer (scripts/, configurable snapshot repo, optional token auth).

  python3.12 -m pytest -q scaffold_search/tests/test_package.py
"""
import importlib, json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from scaffold_search import schedule, isolation, workspace  # noqa: E402


def test_protocol_manifest_matches():
    """this branch runs PROTOCOL_TP4.md: everything but the TBD rows (TP4 seed, b3) matches. PROTOCOL.md (TP8)
    is frozen on branch scaffold-search-dsv41; here gpu_lease.py, harness_files and overlays A1/A3 differ
    (GPU-count neutral, byte-identical behaviour on 8 GPUs: test_tp8_workspace_texts_unchanged)."""
    p = subprocess.run([sys.executable, f'{ROOT}/scripts/verify_protocol.py', '--protocol', 'PROTOCOL_TP4.md',
                        '--allow-tbd'], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'all match' in p.stdout


def test_tp8_workspace_texts_unchanged():
    """harness_files rendered for 8 GPUs = the tree frozen in PROTOCOL.md (TP8)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('vp', f'{ROOT}/scripts/verify_protocol.py')
    vp = importlib.util.module_from_spec(spec); spec.loader.exec_module(vp)
    want = vp.protocol_table(f'{ROOT}/PROTOCOL.md')['scaffold_search/harness_files (tree)']
    d = tempfile.mkdtemp()
    for r, _, fs in os.walk(f'{ROOT}/scaffold_search/harness_files'):
        for f in fs:
            src = os.path.join(r, f)
            dst = os.path.join(d, os.path.relpath(src, f'{ROOT}/scaffold_search/harness_files'))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, 'w').write(workspace.render(open(src).read(), 8))
    assert vp.th(d) == want


def test_workspace_wrappers_use_task_gpus():
    for n in (8, 4):
        ws = tempfile.mkdtemp()
        for t in ('check_candidate', 'profile_candidate', 'timeline_candidate'):
            open(f'{ws}/{t}', 'w').write('#!/bin/sh\npython -m megabench.integrations.gpu_noise check --case X\n')
        workspace.harness_files(ws, workspace.SEED_PREBUILD, n)
        for f in ('check_candidate', 'profile_candidate', 'timeline_candidate'):
            assert f'./gpu_run -g {n} -m 30' in open(f'{ws}/{f}').read(), f
        notes = open(f'{ws}/TASK.d/harness.md').read()
        assert '@' not in notes and f'--nproc_per_node {n} devtest.py' in notes
        assert f'or {n})' in open(f'{ws}/gpu_run').read()


def test_tp4_arms_and_prompt():
    want = {'A0S': ['S'], 'A1': ['S', 'A1'], 'A2': ['S', 'A1', 'A2'], 'A3': ['S', 'A1', 'A3']}
    for a, ov in want.items():
        t8 = workspace.load_arm(f'{ROOT}/arms/{a}.toml')
        t4 = workspace.load_arm(f'{ROOT}/arms/tp4/{a}.toml')
        assert t4['name'] == a and t4['overlays'] == ov and t4['task'] == 'improve-b8-tp4'
        assert t4['prompt'] == 'goal_improve_tp4.md' and t8['prompt'] == 'goal_improve.md'
        assert {k: v for k, v in t4.items() if k not in ('task', 'prompt')} == \
               {k: v for k, v in t8.items() if k not in ('task', 'prompt')}
    p8 = open(f'{ROOT}/prompts/goal_improve.md').read()
    p4 = open(f'{ROOT}/prompts/goal_improve_tp4.md').read()
    assert p8.replace('TP8 on the 8 local B300 GPUs', 'TP4 on the 4 local B300 GPUs') == p4 and p4 != p8
    port = workspace.load_arm(f'{ROOT}/arms/tp4/PORT.toml')
    assert port['task'] == 'port-b8-tp4' and workspace.TASKS['port-b8-tp4']['seed'] == 'dsv41_b8_nopf'
    assert workspace.TASKS['improve-b8-tp4']['gpus'] == 4 and workspace.TASKS['improve-b8']['gpus'] == 8


def test_tp4_schedules_match_run_script():
    txt = open(f'{ROOT}/scripts/run_main_tp4.sh').read()
    e4a = schedule.make_latin({'tp4/A0S': 6, 'tp4/A1': 6, 'tp4/A2': 6, 'tp4/A3': 6}, 4, 20261119, 't')
    e4b = schedule.make({'tp4/A1': 4, 'tp4/A3': 4}, 3, 20261120, 'c')
    for name, s in (('E4A', e4a), ('E4B', e4b)):
        got = ' '.join(f"{r['run_id']}:{r['arm']}" for r in s['order'])
        assert f"{name}='{got}'" in txt, (name, got)
    # Latin blocks: every wave has each arm once; per slot no arm appears more than twice
    for w in e4a['waves']:
        assert len({r['arm'] for r in w}) == 4
    for i in range(4):
        col = [w[i]['arm'] for w in e4a['waves']]
        assert max(col.count(a) for a in set(col)) <= 2 and len(set(col)) == 4


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
