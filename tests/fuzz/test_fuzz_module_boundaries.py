from pathlib import Path


def test_three_runtime_facades_are_importable_without_crossing_roles():
    from autofusex.workloads.fuzz import generation as client
    import autofusex.client as daemon
    from autofusex.viewer import server
    assert callable(client.materialize)
    assert callable(daemon.request)
    assert callable(server.serve)


def test_catalog_has_short_display_names_and_every_rule_is_individual():
    from autofusex.workloads.fuzz.nnsmith.aten_catalog import RULES, BY_TARGET
    assert len(RULES) == len(BY_TARGET)
    assert all(rule.display_name and len(rule.display_name) < len(rule.target) for rule in RULES)
    assert any(rule.display_name == 'reducesum' for rule in RULES)
    assert any(rule.display_name == 'permute' for rule in RULES)


def test_sampling_default_node_budget_is_ten():
    from autofusex.workloads.fuzz.generation.sampling import SamplingConfig
    assert SamplingConfig().max_nodes == 10


def test_ops_display_groups_and_categories_select_all_overloads():
    from autofusex.workloads.fuzz.generation.operator_catalog import normalize_ops, matches_ops
    from autofusex.workloads.fuzz.nnsmith.aten_catalog import RULES
    expected = {
        'broadcast': {'aten.broadcast_to.default', 'aten.broadcast_tensors.default'},
        'permute': {'aten.transpose.int', 'aten.permute.default', 'aten.transpose_copy.int'},
        'reduceall': {'aten.all.default', 'aten.all.dim', 'aten.all.dims'},
        'reduceany': {'aten.any.default', 'aten.any.dim', 'aten.any.dims'},
        'lshift': {'aten.__lshift__.Scalar', 'aten.__lshift__.Tensor'},
        'rshift': {'aten.__rshift__.Scalar', 'aten.__rshift__.Tensor'},
        'xor': {'aten.__xor__.Scalar', 'aten.__xor__.Tensor'},
        'reshape': {'aten.reshape.default', 'aten.view.default', 'aten.ravel.default'},
        'split': {'aten.split.Tensor', 'aten.split_with_sizes.default', 'aten.chunk.default'},
    }
    for name, targets in expected.items():
        accepted = normalize_ops(name)
        assert all(matches_ops(target, accepted) for target in targets)
        assert not matches_ops('aten.add.Tensor', accepted)
    for category in {r.category for r in RULES}:
        for spelling in (category, category.lower(), category.upper()):
            accepted = normalize_ops(spelling)
            assert {r.target for r in RULES if matches_ops(r.target, accepted)} == {
                r.target for r in RULES if r.category == category}
    assert [r.target for r in RULES if matches_ops(r.target, normalize_ops('aten.transpose.int'))] == ['aten.transpose.int']


def test_documented_operator_metadata_matches_catalog():
    from autofusex.workloads.fuzz.nnsmith.aten_catalog import RULES
    path = Path(__file__).parents[2] / 'docs/fuzz/aten_op_list.md'
    rows = [line.split('|') for line in path.read_text().splitlines() if line.startswith('| ') and 'aten.' in line]
    assert len(rows) == len(RULES)
    for row, rule in zip(rows, RULES):
        assert [value.strip().strip(chr(96)) for value in row[2:6]] == [
            rule.target, rule.display_name, rule.category, rule.family]


def test_renamed_display_labels_keep_stored_cases_readable():
    import json
    from autofusex.workloads.fuzz.contracts import canonical_json
    from autofusex.workloads.fuzz.case_spec import decode
    from autofusex.execution.reporting.report import spec_metadata
    path = Path(__file__).parent / 'fixtures/current-query.json'
    value = json.loads(path.read_text())
    graph = json.loads(value['graph_json'])
    node = next(n for n in graph['nodes'] if n['op'] == 'aten.transpose.int')
    node['display_name'] = 'transpose'
    value['graph_json'] = canonical_json(graph)
    spec = decode(json.dumps(value))
    reported = spec_metadata(spec)
    assert next(n for n in reported['graph']['nodes'] if n['op'] == node['op'])['display_name'] == 'permute'
    node['display_name'] = 'permute'
    value['graph_json'] = canonical_json(graph)
    decode(json.dumps(value))
