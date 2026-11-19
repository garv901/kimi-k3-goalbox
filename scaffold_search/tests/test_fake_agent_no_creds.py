"""CPU: a --fake-agent run must not need any Claude credential, and the relay settings in claude.env
reach the agent environment, the proxy allowlist and the pre-approved API key list.

  python3.12 -m pytest -q scaffold_search/tests/test_fake_agent_no_creds.py
"""
import json, os, sys, tempfile
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scaffold_search import isolation  # noqa: E402


class _PW:
    pw_uid, pw_gid, pw_dir = 4242, 4243, '/nonexistent'


def _fake_env(tmp, text):
    f = os.path.join(tmp, 'claude.env')
    open(f, 'w').write(text)
    return mock.patch.object(isolation, 'CLAUDE_ENV_FILE', f)


def _create(run_dir, require_creds, root_creds_exists=False):
    calls = []

    def getpwnam(name):   # the user does not exist until useradd ran
        if not any(c[0] == 'useradd' for c in calls):
            raise KeyError(name)
        return _PW()
    with mock.patch.object(isolation, 'sh', lambda *a, **k: calls.append(a)), \
         mock.patch.object(isolation, 'ensure_group', lambda n: 4243), \
         mock.patch.object(isolation.pwd, 'getpwnam', getpwnam), \
         mock.patch.object(isolation.os, 'chown', lambda *a, **k: None), \
         mock.patch.object(isolation, 'ROOT_CREDS', run_dir + '/no-such-root-creds' if not root_creds_exists else __file__):
        return isolation.create_user('smoke1', run_dir, require_creds=require_creds), calls


def test_fake_agent_user_without_any_credential():
    tmp = tempfile.mkdtemp()
    with _fake_env(tmp, ''):
        (uid, gid), calls = _create(f'{tmp}/run', require_creds=False)
    assert (uid, gid) == (4242, 4243)
    assert any(c[0] == 'useradd' for c in calls)
    assert not os.path.exists(f'{tmp}/run/home/.claude/.credentials.json')
    env = None
    with _fake_env(tmp, ''):
        env = isolation.agent_env('smoke1', f'{tmp}/run')
    assert not any(k in env for k in ('ANTHROPIC_API_KEY', 'CLAUDE_CODE_OAUTH_TOKEN', 'ANTHROPIC_BASE_URL'))
    assert env['HTTPS_PROXY'] == 'http://127.0.0.1:3128'


def test_real_agent_user_still_requires_a_credential():
    tmp = tempfile.mkdtemp()
    with _fake_env(tmp, ''), pytest.raises(SystemExit, match='no Claude credentials'):
        _create(f'{tmp}/run', require_creds=True)


def test_relay_settings_reach_env_proxy_and_key_approval():
    tmp = tempfile.mkdtemp()
    text = ('ANTHROPIC_BASE_URL=https://relay.example.internal:8443\n'
            'ANTHROPIC_API_KEY=ss-run-r01-0123456789abcdef0123456789abcdef\n'
            'NODE_EXTRA_CA_CERTS=/srv/ss/creds/relay-ca.pem\nIGNORED_KEY=x\n')
    with _fake_env(tmp, text):
        env = isolation.agent_env('r01', f'{tmp}/run')
        assert env['ANTHROPIC_BASE_URL'] == 'https://relay.example.internal:8443'
        assert env['ANTHROPIC_API_KEY'].startswith('ss-run-r01-') and 'IGNORED_KEY' not in env
        assert isolation.relay_endpoint() == ('relay.example.internal', 8443)
        assert isolation._allowed('relay.example.internal', 8443)
        assert not isolation._allowed('relay.example.internal', 443)      # only the configured port
        assert isolation._allowed('api.anthropic.com', 443) and not isolation._allowed('api.anthropic.com', 8443)
        assert not isolation._allowed('github.com', 443)
        with mock.patch.object(os.path, 'exists', lambda p: False):
            cj = isolation.minimal_claude_json('/srv/ss/runs/r01/ws')
        assert cj['customApiKeyResponses'] == {'approved': ['cdef0123456789abcdef'], 'rejected': []}
        assert cj['hasCompletedOnboarding'] is True
    with _fake_env(tmp, ''):
        assert isolation.relay_endpoint() is None and not isolation._allowed('relay.example.internal', 8443)
