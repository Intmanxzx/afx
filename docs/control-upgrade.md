# Control upgrade lifecycle

The resident manager is the replaceable control agent. It owns platform sync,
commands, task admission and release staging. The supervisor owns process
replacement and the activation journal. Execution uses the registered environment
manifest's interpreter; active-client.json never overrides worker Python.

A release is prepared in a clean control-venv without system site packages.
Preparation failure does not stop tasks or change active-client.json. A successful
preparation records client_switch=requested and leaves the operation activating.
The supervisor drains work, shuts down the old manager, switches the pointer and
starts the new manager. Success requires the expected Python and version, a fresh
managed heartbeat, and a running restored task if the old session was running.
Only then does the operation succeed and permit the upgrade acknowledgement.
Failed startup restores the old pointer. If task ownership cannot be resolved,
the supervisor refuses to start another manager rather than risking duplicate
workers. Rollback is supported only for the current compatible manager schema.

## Legacy migration

Legacy 0.2.2–0.2.2.6 launchers cannot acquire these semantics from a wheel update.
Install the new control build separately, stop existing tasks, load the site's
working CANN environment and run:

    NEW_CONTROL_PYTHON -m autofusex.fuzz.daemon.management.migrate \
      --data-dir EXISTING_DATA_HOME --execution-python EXISTING_EXECUTION_PYTHON

The command probes the execution stack before changing state, creates a private
SQLite backup plus configuration copies, requires tasks to be stopped, requests
old manager shutdown and waits for the launcher lock. It preserves identity,
credentials, results and task configuration. It registers the explicit execution
interpreter and starts the new supervisor. If takeover fails, retain the backup
and diagnose; do not kill PIDs recovered from the database.

Control upgrades do not upgrade the execution environment. Torch, CANN and the
pinned NNSmith fork must be provisioned and verified in that environment separately.

## Validation boundaries

Process switching and startup rollback have subprocess regression coverage.
Production migration, restoration with real NPU work, failure during draining,
and network outage during activation require end-to-end validation before release.
Do not publish or label this implementation production-ready based only on unit
or subprocess tests.
