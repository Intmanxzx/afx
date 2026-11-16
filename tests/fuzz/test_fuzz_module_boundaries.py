from pathlib import Path


def test_three_runtime_facades_are_importable_without_crossing_roles():
    from autofusex.fuzz import client, daemon, server
    assert callable(client.materialize)
    assert callable(daemon.request)
    assert callable(server.serve)


def test_catalog_has_short_display_names_and_every_rule_is_individual():
    from autofusex.fuzz.client.nnsmith.aten_catalog import RULES, BY_TARGET
    assert len(RULES) == len(BY_TARGET)
    assert all(rule.display_name and len(rule.display_name) < len(rule.target) for rule in RULES)
    assert any(rule.display_name == 'reducesum' for rule in RULES)
    assert any(rule.display_name == 'transpose' for rule in RULES)


def test_sampling_default_node_budget_is_ten():
    from autofusex.fuzz.client.generation.sampling import SamplingConfig
    assert SamplingConfig().max_nodes == 10
