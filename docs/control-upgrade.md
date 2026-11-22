# Control upgrade lifecycle

The resident manager is the replaceable control agent. It owns platform sync,
commands, task admission and release staging. The supervisor owns process
replacement and the activation journal. Execution uses the registered environment
manifest's interpreter; active-client.json never overrides worker Python.

A release is prepared after whole-case draining in a clean control-venv without system site packages.
Preparation failure leaves active-client.json unchanged and resumes the old queues. A successful
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

Client upgrades now fence admission before installation. Already admitted Cases
finish all backends and records; unclaimed entries and the campaign remain in the
same durable queue. The installer waits until task ownership and task descendants
are absent continuously for five seconds. It then stages both control and execution
autofusex packages at the target version. Execution releases inherit the registered
site dependency paths without uninstalling or upgrading Torch, torch_npu or CANN.
Activation changes the execution manifests with the control pointer; rollback
restores both. Resumption uses the existing run and queue rather than enqueuing
the original replay list or resetting the campaign. Worker version evidence is
required before successful activation acknowledgement.

This protocol requires the updated supervisor and scheduler on the source side.
Existing 0.2.2.8 processes cannot acquire new drain semantics merely by downloading
a target wheel; a baseline migration is required before testing this workflow.

## Whole-case upgrade validation

Candidate source was tested on mlir in /mnt/workspace/afx-upgrade-case-check,
separate from the installed client. Management regression: 85 passed. The
updated delivery tests (including old execution with current control version):
3 passed. Targeted activation/drain and real Case tests: 13 passed, including
CPU plus eager-npu on each of npu:0 and npu:1. The test requests draining after
generation begins, checks that both backends finish, leaves the next Case
pending, then resumes and checks exactly four result records.

An isolated execution install probe loaded candidate autofusex 0.2.2.8 from
its new execution directory while loading torch 2.10.0+cpu and torch_npu
2.10.0.post7.dev20261015 from the original torch210_daily site-packages.
This local candidate wheel was not published. Full platform-triggered upgrade
with installation, running-NPU queue restoration, and supervisor crash recovery
as one end-to-end transaction still requires acceptance before deployment.

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
   Verify the new control PID, interpreter and imported package, matching execution
   package version, unchanged Torch/CANN, restored queue and server acknowledgement.
4. Repeat while stopped; upgrade must not start an unwanted task.
5. Fail download/install after draining; old work must resume from the preserved queue.
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

## 0.2.2.9 candidate acceptance

Commit c94f66f4c905baae6148c94901b794607b58137c was tagged v0.2.2.9
and pushed to both GitCode repositories. The isolated mlir workspace is
`/mnt/workspace/afx-0229-e2e`; management and release-tooling tests passed
(102 tests, `tests.log`). The test client used a manually installed candidate
baseline, so this does not establish automatic migration from an unmodified
0.2.2.8 supervisor.

Two full candidate installations succeeded while preserving run
`0ad09e6a-ffe6-40ac-a693-85d2b3809898`. Local IPC operation
`450b9469-51cb-4b0e-88dd-9d2b3fd57391` recorded 5.053 seconds of quiescence.
The production platform then dispatched request
`bd068227-2f39-47e5-9660-f93be7f49c52` to isolated client 18
(`upgrade-0229-e2e-npu0`). Operation
`a90b0867-e0da-472b-827e-5dc392de198d` recorded 5.052 seconds of quiescence
before installation, switched the control daemon to PID 832018, and restored
the same run with execution worker version evidence of 0.2.2.9. The server
recorded `client.updated` and cleared its upgrade target and request ID.

The workload used CPU and eager-NPU backends. Device contention occurred on
some attempts; successful real NPU execution was also observed. After acceptance,
the isolated run was stopped with 56 results and its supervisor was shut down.
The original mlir client was not migrated or stopped.

Both installations used the candidate wheel via `PIP_FIND_LINKS`. They establish
platform delivery, drain, installation, activation, queue restoration and
acknowledgement for the candidate. Public PyPI artifact acceptance remains pending
until the GitCode push mirror delivers the tag to GitHub and the release workflow
publishes it. Supervisor interruption and network-outage recovery retain the
validation boundaries described above.

## 0.2.2.9 publication

After the GitCode mirror was manually synchronized, GitHub received master
commit `4d625d564cd929dcfa56909a70dee474c9b39067` and tag `v0.2.2.9`.
Release workflow run 45021299098 completed successfully: build, wheel smoke
test, metadata/layout validation, GitHub Release upload and PyPI publication
all passed. The public release is available at
https://github.com/Intmanxzx/afx/releases/tag/v0.2.2.9 and
https://pypi.org/project/autofusex/0.2.2.9/.

A clean mlir virtual environment installed `autofusex==0.2.2.9` from PyPI and
reported both package metadata and import version `0.2.2.9`. The downloaded
`autofusex-0.2.2.9-py3-none-any.whl` has SHA256
`7417fe3b07d8dc187d0c054064aee879ef64cd136bb2044df566fd2c440c3664`, matching
the GitHub Release asset and the PyPI JSON metadata. The full platform E2E
above used the same source commit and candidate wheel before publication; the
public artifact itself has additionally passed clean-environment installation
and provenance checks.
