from __future__ import annotations

import time

import pytest

from autofusex.progress import progress_stage


def test_progress_stage_logs_start_and_done(capsys: pytest.CaptureFixture[str]) -> None:
    with progress_stage("unit stage", detail="item=1", heartbeat_seconds=60, stall_seconds=120):
        pass

    stderr = capsys.readouterr().err
    assert "START unit stage item=1" in stderr
    assert "DONE unit stage" in stderr


def test_progress_stage_logs_failure(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match="bad value"):
        with progress_stage("failing stage", heartbeat_seconds=60, stall_seconds=120):
            raise ValueError("bad value")

    stderr = capsys.readouterr().err
    assert "FAILED failing stage" in stderr
    assert "ValueError: bad value" in stderr


def test_progress_stage_warns_when_cpu_does_not_advance(capsys: pytest.CaptureFixture[str]) -> None:
    with progress_stage("idle stage", heartbeat_seconds=0.01, stall_seconds=0.02):
        time.sleep(0.06)

    stderr = capsys.readouterr().err
    assert "WARNING idle stage" in stderr
    assert "may be waiting or stalled" in stderr
