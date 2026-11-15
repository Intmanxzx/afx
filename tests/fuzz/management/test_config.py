import pytest
from autofusex.fuzz.management.config import parse,validate,fingerprint,merge,quantity


def task():
    return dict(schema_version=1,kind='fuzz_task',environment='dev',workload=dict(mode='replay',seeds=['C000000016']))


def test_yaml_json_same_defaults():
    a=validate(parse('schema_version: 1\nkind: fuzz_task\nenvironment: dev\nworkload:\n  mode: replay\n  seeds: [C000000016]\n'))
    b=validate(task())
    assert fingerprint(a)==fingerprint(b)
    assert a['execution']['timeout']=='120s'
    assert parse('value: on')['value']=='on'
    assert parse('value: true')['value'] is True
    assert quantity('16GiB')==16*1024**3


@pytest.mark.parametrize('text',['a: 1\na: 2','a: &a []\nb: *a','a: !!python/object:os.system {}','a: .nan','1: x','a: '+ '['*30+'0'+']'*30])
def test_reject_ambiguous_or_unsafe_yaml(text):
    with pytest.raises(ValueError):parse(text)


def test_strict_validation():
    value=task();value['unknown']=1
    with pytest.raises(ValueError):validate(value)
    value=task();value['execution']={'resources':{'policy':'auto','workers':2}}
    with pytest.raises(ValueError):validate(value)
    value=task();value['results']={'save':False,'upload':True}
    with pytest.raises(ValueError):validate(value)
    value=task();value['termination']={'mode':'continuous'}
    with pytest.raises(ValueError):validate(value)
    value=task();value['schema_version']=True
    with pytest.raises(ValueError):validate(value)


def test_list_replace_no_mutation():
    base={'x':{'a':1,'b':[1,2]}}
    assert merge(base,{'x':{'b':[3]}})=={'x':{'a':1,'b':[3]}}
    assert base['x']['b']==[1,2]
