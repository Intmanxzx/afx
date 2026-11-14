from __future__ import annotations

import pickle
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
    detect_framework,
    get_framework_adapter,
)


@pytest.fixture(autouse=True)
def isolate_fixture_script_imports(tmp_path):
    # Each test creates a different script tree with common names such as libs.
    previous = dict(sys.modules)
    yield
    for name, module in list(sys.modules.items()):
        origin = getattr(module, '__file__', None)
        if origin and tmp_path in Path(origin).resolve().parents:
            if name in previous:
                sys.modules[name] = previous[name]
            else:
                sys.modules.pop(name, None)


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


def test_pytorch_adapter_invokes_cann_ge_case_entry_when_script_only_defines_case(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    root = tmp_path / "cann-ge"
    (root / "libs").mkdir(parents=True)
    (root / "libs" / "__init__.py").write_text("", encoding="utf-8")
    (root / "feature").mkdir()
    script = root / "feature" / "AutoFuse" / "PyTorch" / "Inductor" / "tc_case.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        """
import torch

def check_torch_version_for_inductor():
    return False

class StaticGraphModel(torch.nn.Module):
    def forward(self, x):
        return x + 1

def tc_case(kernel_path, case_path, case_name, soc_version, device_id):
    del kernel_path, case_path, case_name, soc_version, device_id
    if not check_torch_version_for_inductor():
        return
    StaticGraphModel().eval()(torch.ones(2))
""",
        encoding="utf-8",
    )

    captured = PyTorchAdapter().capture(script=script)

    assert captured.name == "StaticGraphModel"
    assert captured.args[0].tolist() == [1.0, 1.0]
    assert captured.metadata["script"] == str(script.resolve())
    pickle.dumps(captured.model)


def test_pytorch_adapter_passes_cann_ge_arguments_through_opaque_decorator(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    root = tmp_path / "cann-ge"
    (root / "libs").mkdir(parents=True)
    (root / "libs" / "__init__.py").write_text("", encoding="utf-8")
    (root / "feature").mkdir()
    script = root / "feature" / "AutoFuse" / "PyTorch" / "Inductor" / "tc_decorated.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        """
import torch

def set_timeout(_seconds):
    def decorate(func):
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped
    return decorate

class StaticGraphModel(torch.nn.Module):
    def forward(self, x):
        return x + 1

@set_timeout(600)
def tc_decorated(kernel_path, case_path, case_name, soc_version, device_id):
    assert kernel_path
    assert case_path
    assert case_name == "tc_decorated"
    assert soc_version
    assert device_id == "0"
    StaticGraphModel().eval()(torch.ones(2))
""",
        encoding="utf-8",
    )

    captured = PyTorchAdapter().capture(script=script)

    assert captured.name == "StaticGraphModel"
    assert captured.args[0].tolist() == [1.0, 1.0]
    pickle.dumps(captured.model)


def test_pytorch_adapter_captures_before_cann_ge_run_af_benchmark(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    root = tmp_path / "cann-ge"
    (root / "libs" / "autofuse_gen").mkdir(parents=True)
    (root / "libs" / "__init__.py").write_text("", encoding="utf-8")
    (root / "libs" / "autofuse_gen" / "__init__.py").write_text("", encoding="utf-8")
    (root / "libs" / "autofuse_gen" / "autofuse_tools.py").write_text(
        """
import numpy as np

def gen_inputs(inputs_param, framework='tensorflow'):
    assert framework == 'pytorch'
    return [
        [np.ones(value['shape'], dtype=value['dtype']) for value in inputs_param.values()],
        [np.full(value['shape'], 2, dtype=value['dtype']) for value in inputs_param.values()],
    ]

def run_af(*args, **kwargs):
    raise AssertionError('the debugger must not enter the Baize benchmark')
""",
        encoding="utf-8",
    )
    (root / "feature").mkdir()
    script = root / "feature" / "tc_run_af.py"
    script.write_text(
        """
from libs.autofuse_gen.autofuse_tools import run_af
import torch

def set_timeout(_seconds):
    def decorate(func):
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped
    return decorate

class StaticGraphModel(torch.nn.Module):
    def forward(self, x):
        return x + 1

@set_timeout(600)
def tc_run_af(kernel_path, case_path, case_name, soc_version, device_id):
    del kernel_path, case_path, case_name, soc_version, device_id
    run_af(
        {'data1': {'shape': (2,), 'dtype': 'float32'}},
        StaticGraphModel().eval(),
        '0',
        '/tmp/case',
        'tc_run_af',
        'Ascend910B1',
        framework='pytorch',
    )
""",
        encoding="utf-8",
    )

    captured = PyTorchAdapter().capture(script=script)

    assert captured.name == "StaticGraphModel"
    assert captured.args[0].tolist() == [2.0, 2.0]
    assert captured.metadata["capture_method"] == "cann_ge_run_af_boundary"
    assert captured.metadata["capture_process"] == "parent"
    assert captured.model(torch.ones(2)).tolist() == [2.0, 2.0]


def test_pytorch_adapter_captures_model_from_cann_ge_fork_worker(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    root = tmp_path / "cann-ge"
    (root / "libs").mkdir(parents=True)
    (root / "libs" / "__init__.py").write_text("", encoding="utf-8")
    (root / "feature").mkdir()
    script = root / "feature" / "AutoFuse" / "PyTorch" / "Inductor" / "tc_forked.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        """
import multiprocessing as mp
import torch

class StaticGraphModel(torch.nn.Module):
    def forward(self, x):
        return x + 1

def worker(model):
    model(torch.ones(2))

def tc_forked(kernel_path, case_path, case_name, soc_version, device_id):
    del kernel_path, case_path, case_name, soc_version, device_id
    model = StaticGraphModel().eval()
    process = mp.get_context("fork").Process(target=worker, args=(model,))
    process.start()
    process.join()
    assert process.exitcode == 0
""",
        encoding="utf-8",
    )

    captured = PyTorchAdapter().capture(script=script)

    assert captured.name == "StaticGraphModel"
    assert captured.args[0].tolist() == [1.0, 1.0]
    assert captured.metadata["capture_process"] == "fork_child"
    pickle.dumps(captured.model)


def test_pytorch_adapter_prefers_cann_ge_entry_matching_script_name(tmp_path: Path) -> None:
    from autofusedebugger.adapters.pytorch import _select_cann_ge_entry

    script = tmp_path / "tc_selected.py"

    def tc_other(*args):
        del args

    def tc_selected(*args):
        del args

    selected = _select_cann_ge_entry(
        {"tc_other": tc_other, "tc_selected": tc_selected}, script
    )

    assert selected is not None
    assert selected[0] == "tc_selected"


def test_cann_ge_run_af_parts_supports_keyword_and_config_calls() -> None:
    from autofusedebugger.adapters.pytorch import _cann_ge_run_af_parts

    def run_af(
        inputs_param=None,
        build_graph=None,
        device_id=None,
        case_path=None,
        case_name=None,
        framework="tensorflow",
        config=None,
    ):
        del inputs_param, build_graph, device_id, case_path, case_name, framework, config

    keyword_parts = _cann_ge_run_af_parts(
        run_af,
        (),
        {
            "inputs_param": {"x": {"shape": (2,), "dtype": "float32"}},
            "build_graph": object(),
            "case_path": "/tmp/case",
            "case_name": "example",
            "framework": "pytorch",
        },
    )
    assert keyword_parts is not None
    assert keyword_parts[2] == "pytorch"
    assert keyword_parts[3] == Path("/tmp/case")
    assert keyword_parts[4] == "example"

    class Config:
        inputs_param = {"x": {"shape": (2,), "dtype": "float32"}}
        build_graph = object()
        framework = "pytorch"
        case_path = "/tmp/config-case"
        case_name = "config-example"

    config_parts = _cann_ge_run_af_parts(run_af, (Config(),), {})
    assert config_parts is not None
    assert config_parts[3] == Path("/tmp/config-case")
    assert config_parts[4] == "config-example"


def test_reserved_framework_adapters_fail_explicitly() -> None:
    assert get_framework_adapter("auto").name == "pytorch"
    with pytest.raises(NotImplementedError, match="TensorFlowAdapter"):
        TensorFlowAdapter().capture(script="unused.py")
    with pytest.raises(NotImplementedError, match="JaxAdapter"):
        JaxAdapter().capture(script="unused.py")
    assert InductorAutofuseBackend().supports("pytorch")
    assert not InductorAutofuseBackend().supports("tensorflow")


def test_detect_framework_from_script_imports(tmp_path: Path) -> None:
    cases = {
        "torch_case.py": ("from torch import nn\n", "pytorch"),
        "tf_case.py": ("import tensorflow as tf\n", "tensorflow"),
        "jax_case.py": ("import jax.numpy as jnp\n", "jax"),
    }
    for name, (source, expected) in cases.items():
        script = tmp_path / name
        script.write_text(source, encoding="utf-8")
        assert detect_framework(script) == expected


def test_detect_framework_rejects_multiple_frameworks(tmp_path: Path) -> None:
    script = tmp_path / "mixed.py"
    script.write_text("import torch\nimport tensorflow\n", encoding="utf-8")

    with pytest.raises(ValueError, match="multiple frameworks"):
        detect_framework(script)


def test_detect_framework_falls_back_to_pytorch_for_indirect_model_import(tmp_path: Path) -> None:
    script = tmp_path / "indirect.py"
    script.write_text("from model_package import Model\n", encoding="utf-8")

    assert detect_framework(script) == "pytorch"


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
    root = tmp_path / "Baize-master"
    (root / "libs").mkdir(parents=True)
    (root / "libs" / "__init__.py").write_text(
        "def add_one(value):\n    return value + 1\n",
        encoding="utf-8",
    )
    script = root / "feature" / "AutoFuse_torch" / "model.py"
    script.parent.mkdir(parents=True)
    (script.parent / "local_helper.py").write_text(
        "def double(value):\n    return value * 2\n",
        encoding="utf-8",
    )
    script.write_text("# Captured by the fake frontend.\n", encoding="utf-8")
    received = {}

    class LazyModel:
        def __call__(self, value):
            from libs import add_one
            from local_helper import double

            return double(add_one(value))

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
            received.update(kwargs)
            local_output = invocation.model(*invocation.args)
            child_output = subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "from libs import add_one; from local_helper import double; print(double(add_one(1)))",
                ],
                text=True,
            )
            return {"local_output": local_output, "child_output": int(child_output)}

    monkeypatch.setattr(script_runner, "get_framework_adapter", lambda name: FakeFrontend())
    monkeypatch.setattr(script_runner, "get_backend_adapter", lambda name: FakeBackend())

    result = script_runner.run_framework_script(
        script=script,
        out_dir=tmp_path / "out",
        save_elementwise_data=True,
    )

    assert result == {"local_output": 2, "child_output": 4}
    assert received["save_elementwise_data"] is True


def test_parse_python_script_invocation_preserves_script_arguments(tmp_path: Path) -> None:
    script = tmp_path / "model.py"
    script.write_text("# test\n", encoding="utf-8")

    parsed = script_runner.parse_python_script_invocation(
        ["python3", "-u", str(script), "--device", "npu", "--batch-size", "2"]
    )

    assert parsed == (script.resolve(), ("--device", "npu", "--batch-size", "2"))


def test_parse_python_script_invocation_rejects_module_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not -c or -m"):
        script_runner.parse_python_script_invocation(["python3", "-m", "some_module"])


def test_default_direct_output_paths_use_current_working_directory(tmp_path: Path) -> None:
    script = tmp_path / "sasrec.py"

    out_dir, html_path = script_runner.default_direct_output_paths(script, tmp_path)

    assert out_dir == tmp_path / "sasrec_autofusedebugger"
    assert html_path == tmp_path / "sasrec_autofusedebugger.html"
