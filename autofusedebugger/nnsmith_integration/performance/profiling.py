"""Torch-NPU profiling adapter. Compilation and warmup are completed by caller."""
from pathlib import Path

from .schema import ProfileConfig
from .statistics import find_op_statistic, read_op_statistic


def profile_kernels(compiled, inputs, *, device, output_dir: Path, config: ProfileConfig) -> dict:
    import torch
    import torch_npu

    output_dir.mkdir(parents=True, exist_ok=False)
    profiler = torch_npu.profiler
    experimental = profiler._ExperimentalConfig(
        export_type=[profiler.ExportType.Text],
        profiler_level=profiler.ProfilerLevel.Level2,
        msprof_tx=False, aic_metrics=profiler.AiCMetrics.PipeUtilization,
        l2_cache=False, op_attr=False, data_simplification=False,
        record_op_args=False, gc_detect_threshold=None,
    )
    torch.npu.synchronize(device)
    with torch.no_grad(), profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=0, active=config.steps, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(output_dir)),
        record_shapes=True, profile_memory=False, with_stack=False,
        with_modules=False, with_flops=False, experimental_config=experimental,
    ) as prof:
        for step in range(config.steps):
            compiled(*inputs)
            if step == config.steps - 1:
                # Complete the final device step before RECORD_AND_SAVE stops
                # collection and exports the capture at prof.step().
                torch.npu.synchronize(device)
            prof.step()
    return read_op_statistic(find_op_statistic(output_dir), config.steps)
