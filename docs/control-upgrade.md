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

A one-time manual migration is the supported baseline for legacy installations.
Automatic migration from 0.2.2 is not a release acceptance requirement.
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

## Required acceptance after manual migration

1. Preserve installation identity, credentials, historical results and task config.
2. Start and stop through the platform; stopping leaves control and heartbeat alive.
3. Upgrade between two builds of the new architecture while a task is running.
   Verify the new control PID, interpreter and imported package, unchanged execution
   environment, restored task and matching server acknowledgement.
4. Repeat while stopped; upgrade must not start an unwanted task.
5. Fail download/install before activation; old work must continue.
6. Fail new control startup or heartbeat readiness; roll back without duplicate workers.
7. Restart the supervisor during activation; recover its durable transaction rather
   than report success from a version pointer alone.

Local source builds used for acceptance must remain in separate test directories;
do not overwrite or republish the immutable PyPI 0.2.2.6 release.

## mlir validation evidence

Manual migration on the existing data home created private backup
control-migration-1795327798149809282 and started control PID 417011.
A local-only 0.2.2.9001 wheel was submitted through the manager operation queue:
supervisor PID 417010 stayed alive; manager changed to PID 418939; old run
329fc671-dd25-46eb-bc9a-19f1e4225996 stopped and run
40a898e1-8125-43d4-a107-97e22bf17844 resumed using the unchanged execution Python.
The activation operation succeeded only after fresh heartbeat and running state.
A subsequent platform stop succeeded while control heartbeat remained available.
A local-only 0.2.2.9002 wheel intentionally failed at Manager.run startup;
activation rolled back, the operation failed, and 0.2.2.9001 restarted as PID
421465 without starting tasks (the prior session was stopped).

These wheels were not published. Upgrade submission used local IPC with a local
wheel directory, not the production platform upgrade endpoint. Network-outage,
supervisor-interruption and rollback with running NPU work remain unverified.
The server-side execution-upload/control-version fix has not been deployed.

## Platform command acceptance

Candidate server 0.2.2.9003 was deployed after a private platform backup.
On mlir, actual authenticated platform HTTP requests verified:
- start: task PID 423558;
- restart: PID 423558 stopped, PID 424363 started, matching request acknowledged;
- current-version update: succeeded/up_to_date with no installation;
- update to local test build 0.2.2.9004: queued/activating/succeeded observed in
  the server summary, new control PID 426236, task restored automatically;
- a second upgrade request during activation: only one real install/activation;
  the later request received up_to_date after activation;
- final stop: task stopped, control PID 426236 and heartbeat remained alive;
  both server target fields cleared.

The upgrade requests used the production platform HTTP endpoint. For unpublished
candidate artifacts, pip's find-links pointed to the test wheel directory on mlir.
This validates command delivery and activation, not public package publication.
Full regression: 181 tests passed. JavaScript syntax check passed.
