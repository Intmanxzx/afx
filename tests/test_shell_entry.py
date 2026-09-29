from __future__ import annotations

from pathlib import Path

import pytest

from autofusedebugger.shell_entry import (
    apply_env_overrides,
    is_shell_script_invocation,
    load_env_file,
    parse_shell_script,
)


LAUNCH_SH = """#!/bin/bash
export PRE_GRAPH_OPTIMIZER=1

rm -rf /tmp/.npu_kernels_root/
rm -rf /workspace/xingzhixiong/torch_cache/

DEVICE_ID=0
DEVICE=npu
MODELS=("sasrec")
MODE=test_qps
TEST_BATCH_SIZE=(128)
export JOB_ID=10085

if [ $DEVICE == "npu" ]; then
    export ASCEND_DEVICE_ID=${DEVICE_ID}
    export RANK_ID=${DEVICE_ID}
fi

export TORCHINDUCTOR_CACHE_DIR=/workspace/xingzhixiong/torch_cache/
export TORCHINDUCTOR_NPU_BACKEND="ascendc"
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export INDUCTOR_TINY_KERNEL=1
export USE_STORE_IN_CAT=True

echo "use "${DEVICE}:${DEVICE_ID}

for model in ${MODELS[@]}; do
    for batch_size in ${TEST_BATCH_SIZE[@]}; do
        python3 ${model}.py --device ${DEVICE} \
                            --device_id=${DEVICE_ID} \
                            --mode=${MODE} \
                            --batch_size=${batch_size}
    done
done

echo "finished!"
"""


@pytest.fixture
def launch_sh(tmp_path: Path) -> Path:
    path = tmp_path / "launch.sh"
    path.write_text(LAUNCH_SH, encoding="utf-8")
    return path


def test_is_shell_script_invocation() -> None:
    assert is_shell_script_invocation(["bash", "launch.sh"])
    assert is_shell_script_invocation(["sh", "run.sh"])
    assert is_shell_script_invocation(["./launch.sh"])
    assert is_shell_script_invocation(["/abs/path/launch.sh"])
    assert not is_shell_script_invocation(["python3", "model.py"])
    assert not is_shell_script_invocation(["run-script"])
    assert not is_shell_script_invocation([])


def test_parse_shell_script_extracts_env(launch_sh: Path) -> None:
    plan = parse_shell_script(launch_sh)

    assert plan.env["PRE_GRAPH_OPTIMIZER"] == "1"
    assert plan.env["JOB_ID"] == "10085"
    assert plan.env["TORCHINDUCTOR_CACHE_DIR"] == "/workspace/xingzhixiong/torch_cache/"
    assert plan.env["TORCHINDUCTOR_MAX_AUTOTUNE"] == "1"
    assert plan.env["INDUCTOR_TINY_KERNEL"] == "1"
    # if-branch exports are picked up (shell if is not evaluated, both sides kept)
    assert plan.env["ASCEND_DEVICE_ID"] == "0"
    assert plan.env["RANK_ID"] == "0"


def test_parse_shell_script_tool_owned_vars_warned(launch_sh: Path) -> None:
    plan = parse_shell_script(launch_sh)

    assert "TORCHINDUCTOR_NPU_BACKEND" not in plan.env
    assert any("TORCHINDUCTOR_NPU_BACKEND" in warning for warning in plan.warnings)


def test_parse_shell_script_finds_python_entry(launch_sh: Path) -> None:
    plan = parse_shell_script(launch_sh)

    # for-loop var expanded to first element
    assert plan.python_script is not None
    assert plan.python_script.name == "sasrec.py"
    assert plan.python_script.parent == launch_sh.parent
    args = plan.python_tokens[plan.python_tokens.index(str(plan.python_script)) + 1 :]
    assert "--device" in args
    assert "npu" in args
    assert "--batch_size=128" in args


def test_parse_shell_script_rejects_missing_entry(launch_sh: Path) -> None:
    empty = launch_sh.parent / "no_python.sh"
    empty.write_text("#!/bin/bash\necho hi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no python model invocation"):
        parse_shell_script(empty)


def test_parse_shell_script_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_shell_script(tmp_path / "missing.sh")


def test_parse_shell_script_quotes_and_comments(tmp_path: Path) -> None:
    script = tmp_path / "s.sh"
    script.write_text(
        "#!/bin/bash\n"
        "export FOO='hello world' # trailing comment\n"
        "export BAR=\"quoted\"\n"
        "# export IGNORED=1\n"
        "python3 run.py --flag\n",
        encoding="utf-8",
    )
    plan = parse_shell_script(script)

    assert plan.env == {"FOO": "hello world", "BAR": "quoted"}
    assert plan.python_script is not None and plan.python_script.name == "run.py"


def test_parse_shell_script_warns_on_source(tmp_path: Path) -> None:
    script = tmp_path / "s.sh"
    script.write_text(
        "#!/bin/bash\nsource /usr/local/Ascend/set_env.sh\nexport A=1\npython3 run.py\n",
        encoding="utf-8",
    )
    plan = parse_shell_script(script)

    assert plan.env == {"A": 1} or plan.env == {"A": "1"}
    assert any("set_env.sh" in warning for warning in plan.warnings)


def test_apply_env_overrides_env_items(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MY_TEST_VAR", raising=False)
    apply_env_overrides(["MY_TEST_VAR=42", "EMPTY="], None)

    import os

    assert os.environ["MY_TEST_VAR"] == "42"


def test_apply_env_overrides_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        apply_env_overrides(["NO_EQUALS_SIGN"], None)


def test_apply_env_overrides_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "env.sh"
    env_file.write_text(
        "#!/bin/bash\nexport FROM_FILE=yes\nexport ANOTHER=2\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("FROM_FILE", raising=False)

    apply_env_overrides(None, env_file)

    import os

    assert os.environ["FROM_FILE"] == "yes"
    assert os.environ["ANOTHER"] == "2"


def test_load_env_file_rejects_empty(tmp_path: Path) -> None:
    env_file = tmp_path / "empty.sh"
    env_file.write_text("#!/bin/bash\necho nothing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no export"):
        load_env_file(env_file)


def test_variable_expansion_between_exports(tmp_path: Path) -> None:
    script = tmp_path / "s.sh"
    script.write_text(
        "#!/bin/bash\n"
        "export BASE=/opt/cann\n"
        "export LIB_DIR=${BASE}/lib64\n"
        "python3 run.py\n",
        encoding="utf-8",
    )
    plan = parse_shell_script(script)

    assert plan.env["LIB_DIR"] == "/opt/cann/lib64"
