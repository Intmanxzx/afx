"""Public control commands preserve their owners and parser semantics."""
import pytest
from autofusex import cli
from autofusex.client.sync_cli import parser
from autofusex.server.cli import build_parser


def test_server_has_no_client_subcommand():
    assert build_parser().parse_args(['--data', '/tmp/db', 'serve']).command == 'serve'
    with pytest.raises(SystemExit):
        build_parser().parse_args(['client'])


@pytest.mark.parametrize('command', ['sync', 'retest'])
def test_sync_and_retest_route_to_client(command, monkeypatch):
    from autofusex.client import sync_cli
    seen = []
    monkeypatch.setattr(sync_cli, 'main', lambda argv, **kw: seen.append((argv, kw)) or 0)
    assert cli.main([command, 'once']) == 0
    assert seen == [(['once'], {'retest': command == 'retest'})]


def test_sync_retest_capabilities_are_distinct():
    common = ['--endpoint', 'http://localhost', '--token-file', 'token', '--results', 'results']
    assert parser().parse_args(['setup', *common]).retest is False
    assert parser(retest=True).parse_args(['run', *common]).retest is True
    with pytest.raises(SystemExit):
        parser(retest=True).parse_args(['setup', *common])
