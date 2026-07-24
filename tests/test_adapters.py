from __future__ import annotations

from pathlib import Path

import pytest

from autofusedebugger.adapters import (
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
