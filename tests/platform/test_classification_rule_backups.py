import pytest
from test_platform import env
from autofusex.server.classification_rules import BUILTIN_RULES, KEY, load, history, merge_builtins
from autofusex.server.ingest import update_settings, UploadError


def test_missing_builtins_are_retained_and_can_be_disabled():
    rules = merge_builtins([])
    assert len(rules) == len(BUILTIN_RULES)
    override = dict(rules[0], enabled=False)
    merged = merge_builtins([override])
    assert merged[0]['enabled'] is False
    assert len(merged) == len(BUILTIN_RULES)


def test_save_backup_restore_and_revision_conflict(env):
    store, _, _ = env
    original = load(store)
    rev = store.settings()[KEY]['revision']
    custom = dict(original[0], id='custom')
    update_settings(store, 'admin', {KEY: [custom]}, {KEY: rev})
    assert len(load(store)) == len(original) + 1
    backup = history(store)[0]
    assert backup['revision'] == rev
    with pytest.raises(UploadError):
        update_settings(store, 'other', {KEY: []}, {KEY: rev})
    assert len(history(store)) == 1
    update_settings(store, 'admin', {KEY: backup['rules']}, {KEY: rev + 1})
    assert load(store) == original
    assert len(history(store)) == 2


def test_old_editor_payload_can_receive_explicit_conflict():
    from autofusex.server.api_models import ClassificationRules
    assert ClassificationRules(rules=[]).base_revision is None
