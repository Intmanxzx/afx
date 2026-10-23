"""NNSmith BackendFactory that routes compilation through the autofuse (ascendc) path.

This intentionally mirrors `nnsmith.backends.pt2.PT2` (the upstream reference
backend for `torch.compile`) but pins the inductor NPU backend to "ascendc" and
is kept out of the NNSmith source tree per the "don't touch the NNSmith repo"
decision: `BackendFactory.init()` only dispatches by a hardcoded name table, so
this class is instantiated directly instead of registered into it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.fx
from multipledispatch import dispatch

from nnsmith.backends.factory import BackendCallable, BackendFactory
from nnsmith.materialize.torch import TorchModel, numpify
from nnsmith.materialize.torch.symbolnet import FxTracing

from autofusedebugger.inductor_autofuse import activate_autofuse_backend


class AscendC(BackendFactory):
    """Compiles NNSmith torch models via `torch.compile(..., backend="inductor",
    options={"npu_backend": "ascendc"})`, i.e. the autofuse frontend path.
    """

    def __init__(self, target: str = "npu", optmax: bool = True, **kwargs):
        super().__init__(target, optmax)
        if self.target not in ("npu", "cuda", "cpu"):
            raise ValueError(f"Unknown target: {self.target}. Expected 'npu'.")
        self.device = torch.device(self.target)
        self.mode = kwargs.get("mode")
        # Raises immediately (at construction, not at first compile) if the
        # autofuse-enabled torch_npu / inductor_npu_ext package isn't importable
        # on this environment.
        activate_autofuse_backend()

    @property
    def system_name(self) -> str:
        return "ascendc"

    @property
    def version(self) -> str:
        try:
            import torch_npu

            return torch_npu.__version__
        except Exception:
            return "unknown"

    @property
    def import_libs(self) -> List[str]:
        return ["import torch", "import torch_npu"]

    @dispatch(TorchModel)
    def make_backend(self, model: TorchModel) -> BackendCallable:
        torch_net = model.torch_model.to(self.device)
        param_names = [k for k, _ in model.torch_model.named_parameters()]

        do_grad_check = model.needs_grad_check()
        torch_net = torch_net.train() if do_grad_check else torch_net.eval()

        with torch.no_grad():
            with FxTracing():
                traced = torch.fx.symbolic_trace(torch_net)
                compiled = torch.compile(
                    traced,
                    fullgraph=True,
                    backend="inductor",
                    mode=self.mode,
                    options={"npu_backend": "ascendc"},
                )

        def closure(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            nonlocal do_grad_check
            input_ts = [torch.from_numpy(v).to(self.device) for _, v in inputs.items()]
            if do_grad_check:
                outputs: List[torch.Tensor] = compiled(*input_ts)
                ret = {}
                for name, output in zip(torch_net.output_like.keys(), outputs):
                    ret[name] = numpify(output)
                    if output.requires_grad:
                        out_grad = torch.autograd.grad(
                            outputs=output,
                            inputs=compiled.parameters(),
                            grad_outputs=torch.ones_like(output),
                            retain_graph=True,
                            allow_unused=True,
                        )
                        for k, v in zip(param_names, out_grad):
                            ret[name + "_vjp_" + k] = numpify(v)
            else:
                with torch.no_grad():
                    outputs: Tuple[torch.Tensor] = compiled(*input_ts)
                ret = {k: numpify(v) for k, v in zip(torch_net.output_like, outputs)}
            return ret

        return closure

    def emit_compile(
        self, opt_name: str, mod_name: str, inp_name: Optional[str] = None
    ) -> str:
        mode = f"'{self.mode}'" if self.mode else "None"
        return (
            f"{opt_name} = torch.compile({mod_name}, fullgraph=True, backend='inductor', "
            f"mode={mode}, options={{'npu_backend': 'ascendc'}})"
        )

    def emit_run(self, out_name: str, opt_name: str, inp_name: str) -> str:
        return f"""{out_name} = {opt_name}(*[torch.from_numpy(v).to('{self.device.type}') for v in {inp_name}])
{out_name} = [v.cpu().detach() for v in {out_name}] # torch2numpy
{out_name} = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in {out_name}] # torch2numpy"""
