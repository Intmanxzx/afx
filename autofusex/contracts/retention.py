"""Bounded retention for the client-owned debug and run directories."""
from autofusex.contracts.configuration import obj

DAYS = dict(type='integer', minimum=0, maximum=3650)
CLEANUP = obj(dict(debug_days=DAYS, runs_days=DAYS))
DEFAULT_CLEANUP = dict(debug_days=2, runs_days=2)


def policy(value):
    from jsonschema import Draft202012Validator
    Draft202012Validator(CLEANUP).validate(value)
    return dict(DEFAULT_CLEANUP, **value)
