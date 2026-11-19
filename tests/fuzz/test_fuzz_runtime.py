"""Seed/case-spec metadata boundaries unrelated to the fleet runtime dashboard."""
import json


def test_retired_sampled_metadata_cannot_be_attached_to_current_seed():
    from dataclasses import replace, asdict
    from autofusex.fuzz.client.case_spec import decode, encode
    from autofusex.fuzz.client.generation.sampling import SamplingConfig
    from fuzz_fixtures import three_shapes_spec
    import pytest
    spec=three_shapes_spec();config=SamplingConfig();graph=json.loads(spec.graph_json)
    graph.update(dialect='nnsmith-sampled/5',sampling=asdict(config))
    spec=replace(spec,seed=config.seed(42,True),graph_json=json.dumps(graph,sort_keys=True,separators=(',',':')))
    with pytest.raises(ValueError,match='current graph metadata differs from seed'):
        decode(encode(spec))
    graph['sampling']['dim_max']=2
    with pytest.raises(ValueError):decode(encode(replace(spec,graph_json=json.dumps(graph,sort_keys=True,separators=(',',':')))))
