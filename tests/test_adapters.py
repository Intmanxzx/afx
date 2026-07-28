from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from autofusedebugger import script_runner
from autofusedebugger.adapters import (
    CapturedInvocation,
    InductorAutofuseBackend,
    JaxAdapter,
    PyTorchAdapter,
    TensorFlowAdapter,
    get_framework_adapter,
)


def test_pytorch_adapter_captures_model_args_kwargs_and_stops_script(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    script = tmp_path / "model.py"
    script.write_text(
        """
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--value', type=float, required=True)
options = parser.parse_args()

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, *, bias):
        self.calls += 1
        return x + bias

model = Model().eval()
model(torch.full((2, 4), options.value), bias=torch.ones(4))
raise RuntimeError('the adapter should stop at the selected invocation')
""",
        encoding="utf-8",
    )

    captured = PyTorchAdapter().capture(script=script, script_args=("--value", "3"))

    assert captured.framework == "pytorch"
    assert captured.name == "Model"
    assert captured.model.calls == 0
    assert captured.args[0].tolist() == [[3.0] * 4, [3.0] * 4]
    assert captured.kwargs["bias"].tolist() == [1.0] * 4
    assert captured.metadata["capture_call"] == 0
    assert captured.metadata["execution"]["training"] is False
    assert captured.metadata["execution"]["grad_enabled"] is True


def test_pytorch_adapter_selects_module_and_matching_call(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    script = tmp_path / "multiple.py"
    script.write_text(
        """
import torch

class Noise(torch.nn.Module):
    def forward(self, x):
        return x + 1

class Target(torch.nn.Module):
    def forward(self, x):
        return x * 2

x = Noise()(torch.ones(2))
Target()(x)
""",
        encoding="utf-8",
    )

    captured = PyTorchAdapter().capture(script=script, module_pattern="Target", capture_call=0)

    assert captured.name == "Target"
    assert captured.args[0].tolist() == [2.0, 2.0]


def test_reserved_framework_adapters_fail_explicitly() -> None:
    assert get_framework_adapter("auto").name == "pytorch"
    with pytest.raises(NotImplementedError, match="TensorFlowAdapter"):
        TensorFlowAdapter().capture(script="unused.py")
    with pytest.raises(NotImplementedError, match="JaxAdapter"):
        JaxAdapter().capture(script="unused.py")
    assert InductorAutofuseBackend().supports("pytorch")
    assert not InductorAutofuseBackend().supports("tensorflow")


def test_inductor_backend_preserves_captured_execution_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from autofusedebugger import torch_model

    received = {}

    def fake_run_torch_case(*, case, **kwargs):
        received["case"] = case
        received["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(torch_model, "run_torch_case", fake_run_torch_case)
    invocation = CapturedInvocation(
        framework="pytorch",
        model=object(),
        args=(1,),
        kwargs={},
        name="Model",
        metadata={
            "script": "/tmp/model.py",
            "execution": {"training": True, "grad_enabled": False, "autocast": {"npu": {"enabled": True}}},
        },
    )

    result = InductorAutofuseBackend().run(invocation, out_dir="/tmp/out")

    assert result == {"ok": True}
    assert received["case"].fullgraph is False
    assert received["case"].training is True
    assert received["case"].grad_enabled is False
    assert received["case"].autocast == {"npu": {"enabled": True}}
    assert received["case"].rewrite_device_hints is False


def test_run_script_keeps_script_directory_importable_during_backend_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "libs.py").write_text(
        "def add_one(value):\n    return value + 1\n",
        encoding="utf-8",
    )
    script = tmp_path / "model.py"
    script.write_text("# Captured by the fake frontend.\n", encoding="utf-8")

    class LazyModel:
        def __call__(self, value):
            from libs import add_one

            return add_one(value)

    class FakeFrontend:
        def capture(self, **kwargs):
            del kwargs
            return CapturedInvocation(
                framework="pytorch",
                model=LazyModel(),
                args=(0,),
                kwargs={},
                name="LazyModel",
            )

    class FakeBackend:
        def supports(self, framework):
            return framework == "pytorch"

        def run(self, invocation, **kwargs):
            del kwargs
            local_output = invocation.model(*invocation.args)
            child_output = subprocess.check_output(
                [sys.executable, "-c", "from libs import add_one; print(add_one(1))"],
                text=True,
            )
            return {"local_output": local_output, "child_output": int(child_output)}

    monkeypatch.setattr(script_runner, "get_framework_adapter", lambda name: FakeFrontend())
    monkeypatch.setattr(script_runner, "get_backend_adapter", lambda name: FakeBackend())

    result = script_runner.run_framework_script(script=script, out_dir=tmp_path / "out")

    assert result == {"local_output": 1, "child_output": 2}
