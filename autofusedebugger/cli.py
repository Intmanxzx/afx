from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from autofusedebugger.agent import AgentConfig
from autofusedebugger.adapters import backend_names, framework_names
from autofusedebugger.autofuse_graph import attach_autofuse_graphs, enable_autofuse_graph_dump
from autofusedebugger.camodel import camodel_info, format_info
from autofusedebugger.compare import compare_manifests, compare_pair, compare_three_manifests, format_summary, save_report
from autofusedebugger.graph import build_manifest
from autofusedebugger.html import write_report_html, write_report_html_from_json
from autofusedebugger.progress import progress_stage
from autofusedebugger.runner import run_camodel_command, run_command, run_debug_cpu_command
from autofusedebugger.script_runner import (
    default_direct_output_paths,
    parse_python_script_invocation,
    run_framework_script,
)
from autofusedebugger.shell_entry import (
    apply_env_overrides,
    apply_plan_env,
    is_shell_script_invocation,
    parse_shell_script,
)
from autofusedebugger.torch_model import format_run_summary as format_torch_run_summary
from autofusedebugger.torch_model import run_torch_model


def _add_agent_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--agent-normalize",
        action="store_true",
        help="call the configured Agent and save a validated model normalization JSON before execution",
    )
    parser.add_argument(
        "--agent-recover",
        action="store_true",
        help="ask the configured Agent for a structured recovery plan when execution fails",
    )
    parser.add_argument("--agent-endpoint", default=None, help="OpenAI-compatible Agent HTTP endpoint")
    parser.add_argument("--agent-transport", choices=("auto", "http", "cannbot"), default=None)
    parser.add_argument("--agent-command", default=None, help="local Agent command, for example /path/to/cannbot")
    parser.add_argument("--agent-model", default=None, help="Agent model name; defaults to environment configuration")
    parser.add_argument("--agent-timeout", type=float, default=None, help="Agent HTTP timeout in seconds")
    parser.add_argument(
        "--agent-output",
        default=None,
        help="normalization JSON path; defaults to OUT_DIR/agent_normalization.json",
    )


def _add_common_thresholds(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rtol", type=float, default=None, help="override dtype-aware relative tolerance")
    parser.add_argument("--atol", type=float, default=None, help="override dtype-aware absolute tolerance")


def _add_elementwise_data_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--save-elementwise-data",
        action="store_true",
        help="persist per-element tensor values and masks for the HTML Tensor viewer",
    )


def _add_with_cpu_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--with-cpu",
        action="store_true",
        help="also capture the Inductor pre-fusion CPU replay and run the three-way comparison; "
        "default compares only NPU eager vs NPU Autofuse",
    )


def cmd_make_manifest(args: argparse.Namespace) -> int:
    manifest = build_manifest(
        side=args.side,
        graph_path=args.graph,
        dump_dir=args.dump_dir,
        out_path=args.out,
        model_name=args.model_name,
    )
    nodes = len(manifest.get("nodes", []))
    fusions = len(manifest.get("fusions", []))
    print(f"wrote {args.out}: nodes={nodes}, fusions={fusions}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    report = compare_manifests(
        args.cpu_manifest,
        args.npu_manifest,
        rtol=args.rtol,
        atol=args.atol,
        save_elementwise_data=args.save_elementwise_data,
    )
    autofuse_dump = (
        Path(args.autofuse_dump)
        if args.autofuse_dump
        else Path(args.npu_manifest).resolve().parent / "autofuse_dump"
    )
    if autofuse_dump.is_dir():
        attach_autofuse_graphs(report, autofuse_dump)
    save_report(report, args.out, args.csv)
    if args.html:
        write_report_html(report, args.html)
    print(format_summary(report))
    return 0 if report["summary"].get("dual_failed", report["summary"]["failed"]) == 0 else 1


def cmd_compare_three(args: argparse.Namespace) -> int:
    with progress_stage("Three-way tensor comparison", detail="existing manifests"):
        report = compare_three_manifests(
            args.cpu_manifest,
            args.npu_eager_manifest,
            args.npu_autofuse_manifest,
            rtol=args.rtol,
            atol=args.atol,
            save_elementwise_data=args.save_elementwise_data,
        )
    autofuse_dump = (
        Path(args.autofuse_dump)
        if args.autofuse_dump
        else Path(args.npu_autofuse_manifest).resolve().parent / "autofuse_dump"
    )
    if autofuse_dump.is_dir():
        with progress_stage("AutoFuse graph collection", detail=f"dump_dir={autofuse_dump}"):
            attach_autofuse_graphs(report, autofuse_dump)
    with progress_stage("Report JSON and CSV write"):
        save_report(report, args.out, args.csv)
    if args.html:
        write_report_html(report, args.html)
    for key in report.get("comparison_order", []):
        comparison = report["comparisons"][key]
        summary = comparison["summary"]
        print(
            f"{comparison['left']} <-> {comparison['right']}: "
            f"strict={summary.get('strict_passed', summary.get('passed', 0))}/{summary.get('compared', 0)} "
            f"cann={summary.get('cann_passed', 0)}/{summary.get('compared', 0)}"
        )
    return 0 if all(
        report["comparisons"][key]["summary"].get("dual_failed", 0) == 0
        for key in report.get("comparison_order", [])
    ) else 1


def cmd_compare_npy(args: argparse.Namespace) -> int:
    from pathlib import Path

    from autofusedebugger.compare import TensorRef

    cpu = TensorRef("golden", 0, Path(args.golden), args.golden_format, "")
    npu = TensorRef("actual", 0, Path(args.actual), args.actual_format, "")
    row = compare_pair(
        cpu,
        npu,
        rtol=args.rtol,
        atol=args.atol,
        save_elementwise_data=args.save_elementwise_data,
    )
    report = {
        "schema": "autofusedebugger-report/1",
        "elementwise_data_saved": args.save_elementwise_data,
        "rtol": args.rtol,
        "atol": args.atol,
        "tolerance_mode": row.get("tolerance_source", "override"),
        "summary": {
            "total": 1,
            "passed": 1 if row.get("passed") else 0,
            "failed": 0 if row.get("passed") else 1,
            "max_abs_error": row.get("max_abs_error", 0.0),
            "max_rel_error": row.get("max_rel_error", 0.0),
        },
        "results": [row],
    }
    save_report(report, args.out, args.csv)
    if args.html:
        write_report_html(report, args.html)
    print(format_summary(report))
    return 0 if row.get("passed") else 1


def cmd_run(args: argparse.Namespace) -> int:
    ret = run_command(args.command, args.out_dir, args.env)
    print(f"command exited with {ret}; logs: {args.out_dir}/run.stdout, {args.out_dir}/run.stderr")
    return ret


def cmd_run_debug_cpu(args: argparse.Namespace) -> int:
    ret = run_debug_cpu_command(args.command, args.out_dir, args.env)
    print(f"debug-cpu command exited with {ret}; logs: {args.out_dir}/run.stdout, {args.out_dir}/run.stderr")
    return ret


def cmd_camodel_info(args: argparse.Namespace) -> int:
    print(format_info(camodel_info(args.cann_root)))
    return 0


def cmd_run_camodel(args: argparse.Namespace) -> int:
    ret = run_camodel_command(args.command, args.out_dir, args.cann_root, args.soc, args.env)
    print(f"camodel command exited with {ret}; logs: {args.out_dir}/run.stdout, {args.out_dir}/run.stderr")
    return ret


def cmd_af_fuzz(args: argparse.Namespace) -> int:
    op_categories = (
        [c.strip() for c in args.op_categories.split(",") if c.strip()]
        if args.op_categories
        else None
    )
    devices = (
        [int(d.strip()) for d in args.devices.split(",") if d.strip()]
        if args.devices
        else None
    )

    if getattr(args, "compare_performance", False):
        from autofusedebugger.nnsmith_integration.performance.cli import run
        return run(args, op_categories=op_categories, devices=devices)

    if getattr(args, "mixed_shapes", False):
        raise ValueError("--mixed-shapes requires --compare-performance")

    if getattr(args, "resume", False) or getattr(args, "keep_perf_artifacts", False):
        raise ValueError("--resume and --keep-perf-artifacts require --compare-performance")

    if args.repro_seed is not None:
        from autofusedebugger.nnsmith_integration.fuzz_loop import run_repro_case

        result = run_repro_case(
            out_dir=args.out_dir,
            repro_seed=args.repro_seed,
            max_nodes=args.max_nodes,
            timeout_ms=args.timeout_ms,
            method=args.method,
            max_elem_per_tensor=args.max_elem_per_tensor,
            tensor_scale=args.tensor_scale,
            target=args.target,
            debug=True if args.debug is None else args.debug,
            soc_tier=args.soc_tier,
            value_dist_mode=args.value_dist_mode,
            dynamic_shapes=args.dynamic_shapes, dynamic_variants=args.dynamic_variants,
            op_categories=op_categories,
        )
        print(
            f"case_id={result.case_id} status={result.status} "
            f"symptom={result.symptom} stage={result.stage}"
        )
        if result.bug_dir:
            print(f"artifacts: {result.bug_dir}")
        return 0 if result.status == "pass" else 1

    # Must be set before fuzz_loop (and with it torch/inductor) is imported:
    # torch._inductor.config reads this env var once at import time. Spawned
    # --jobs workers inherit it. Overrides the package-level default of 4
    # (see autofusedebugger/__init__.py).
    if args.compile_threads is not None:
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(args.compile_threads)

    from autofusedebugger.nnsmith_integration.fuzz_loop import run_fuzz_session_parallel

    summary = run_fuzz_session_parallel(
        out_dir=args.out_dir,
        n_cases=100 if args.n_cases is None else args.n_cases,
        max_nodes=args.max_nodes,
        timeout_ms=args.timeout_ms,
        method=args.method,
        max_elem_per_tensor=args.max_elem_per_tensor,
        tensor_scale=args.tensor_scale,
        seed=args.seed,
        target=args.target,
        jobs=args.jobs,
        devices=devices,
        soc_tier=args.soc_tier,
        value_dist_mode=args.value_dist_mode,
        dynamic_shapes=args.dynamic_shapes, dynamic_variants=args.dynamic_variants,
        op_categories=op_categories,
    )
    print(
        f"total={summary.total} passed={summary.passed} failed={summary.failed} "
        f"gen_errors={summary.gen_errors}; summary: {args.out_dir}/summary.html"
    )
    return 0 if summary.failed == 0 else 1


def cmd_report_html(args: argparse.Namespace) -> int:
    write_report_html_from_json(args.report, args.out, args.title)
    print(f"wrote {args.out}")
    return 0


def cmd_normalize_script(args: argparse.Namespace) -> int:
    script_args = tuple(args.script_args)
    if script_args[:1] == ("--",):
        script_args = script_args[1:]
    script = Path(args.script).resolve()
    output = (
        Path(args.out).resolve()
        if args.out
        else Path.cwd() / f"{script.stem}_autofusedebugger" / "agent_normalization.json"
    )
    from autofusedebugger.normalization import normalize_script

    framework = args.framework
    if framework == "auto":
        framework = detect_framework(script)
    config = AgentConfig.from_env(
        endpoint=args.agent_endpoint,
        model=args.agent_model,
        timeout=args.agent_timeout,
        transport=args.agent_transport,
        command=args.agent_command,
    )
    result = normalize_script(
        script=script,
        framework=framework,
        script_args=script_args,
        config=config,
        output=output,
    )
    print(
        f"Agent normalization written: {output} "
        f"framework={result['framework']} status={result['status']} "
        f"confidence={result['diagnostics'].get('confidence', '-')}",
    )
    return 0 if result["status"] == "ready" else 1


def cmd_run_torch_model(args: argparse.Namespace) -> int:
    enable_autofuse_graph_dump(Path(args.out_dir).resolve() / "autofuse_dump")
    summary = run_torch_model(
        script=args.script,
        entry=args.entry,
        out_dir=args.out_dir,
        html_path=args.html,
        cann_root=args.cann_root,
        device=args.device,
        rtol=args.rtol,
        atol=args.atol,
        save_elementwise_data=args.save_elementwise_data,
        fullgraph=args.fullgraph,
        rewrite_device_hints=args.rewrite_device_hints,
        with_cpu=args.with_cpu,
    )
    print(format_torch_run_summary(summary))
    return 0 if summary["passed"] else 1


def cmd_run_script(args: argparse.Namespace) -> int:
    script_args = tuple(args.script_args)
    if script_args[:1] == ("--",):
        script_args = script_args[1:]
    if getattr(args, "env_file", None) or getattr(args, "env", None):
        apply_env_overrides(getattr(args, "env", None), getattr(args, "env_file", None))
    summary = run_framework_script(
        script=args.script,
        out_dir=args.out_dir,
        framework=args.framework,
        backend=args.backend,
        script_args=script_args,
        capture_call=args.capture_call,
        module_pattern=args.module_pattern,
        name=args.name,
        fullgraph=args.fullgraph,
        dynamic=args.dynamic,
        rewrite_device_hints=args.rewrite_device_hints,
        html_path=args.html,
        cann_root=args.cann_root,
        device=args.device,
        rtol=args.rtol,
        atol=args.atol,
        save_elementwise_data=args.save_elementwise_data,
        with_cpu=args.with_cpu,
        agent_normalize=args.agent_normalize,
        agent_recover=args.agent_recover,
        agent_endpoint=args.agent_endpoint,
        agent_transport=args.agent_transport,
        agent_command=args.agent_command,
        agent_model=args.agent_model,
        agent_timeout=args.agent_timeout,
        agent_output=args.agent_output,
    )
    print(format_torch_run_summary(summary))
    return 0 if summary["passed"] else 1


def cmd_run_direct_shell(argv: list[str]) -> int:
    """Convenience form: ``autofusedebugger bash launch.sh`` or ``autofusedebugger ./launch.sh``."""
    script_index = 0
    if Path(argv[0]).name in {"bash", "sh", "zsh", "dash", "ksh"}:
        if len(argv) < 2:
            raise ValueError("shell launcher mode requires a script path, for example: autofusedebugger bash launch.sh")
        script_index = 1
    plan = parse_shell_script(argv[script_index])
    apply_plan_env(plan)
    python_script = plan.python_script
    if python_script is None:
        raise ValueError(
            f"no .py entry found in the python invocation of {plan.script}; "
            "use 'autofusedebugger python3 model.py' instead"
        )
    script_args = tuple(plan.python_tokens[plan.python_tokens.index(str(python_script)) + 1 :])
    return cmd_run_direct(python_script, script_args)


def cmd_run_direct(script: Path, script_args: tuple[str, ...]) -> int:
    out_dir, html_path = default_direct_output_paths(script)
    print(f"AutofuseDebugger direct mode: script={script}")
    print(f"AutofuseDebugger outputs: out_dir={out_dir} html={html_path}")
    summary = run_framework_script(
        script=script,
        out_dir=out_dir,
        framework="auto",
        backend="inductor_autofuse",
        script_args=script_args,
        html_path=html_path,
        agent_normalize=os.environ.get("AUTOFUSEDEBUGGER_AGENT_NORMALIZE", "0").lower()
        in {"1", "true", "yes"},
        agent_recover=os.environ.get("AUTOFUSEDEBUGGER_AGENT_RECOVER", "0").lower()
        in {"1", "true", "yes"},
    )
    print(format_torch_run_summary(summary))
    return 0 if summary["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autofusedebugger",
        description="CPU/NPU eager/NPU Autofuse 精度对比和图分析工具",
        epilog="快捷模式：autofusedebugger python3 model.py [模型脚本参数...]；"
    "或 autofusedebugger bash launch.sh / autofusedebugger ./launch.sh 解析启动脚本继承环境变量",
    )
    sub = parser.add_subparsers(dest="command_name", required=True)

    make = sub.add_parser("make-manifest", help="build a manifest from graph JSON and an npy dump directory")
    make.add_argument(
        "--side",
        choices=[
            "cpu",
            "npu",
            "npu_eager",
            "npu_autofuse",
            "inductor_prefusion_cpu",
            "inductor_prefusion_npu_eager",
        ],
        required=True,
    )
    make.add_argument("--graph", required=True)
    make.add_argument("--dump-dir", required=True)
    make.add_argument("--out", required=True)
    make.add_argument("--model-name", default=None)
    make.set_defaults(func=cmd_make_manifest)

    cmp_parser = sub.add_parser("compare", help="compare a reference and NPU manifest")
    cmp_parser.add_argument("--cpu-manifest", required=True)
    cmp_parser.add_argument("--npu-manifest", required=True)
    cmp_parser.add_argument("--out", default=None)
    cmp_parser.add_argument("--csv", default=None)
    cmp_parser.add_argument("--html", default=None, help="write HTML visualization")
    cmp_parser.add_argument(
        "--autofuse-dump",
        default=None,
        help="AutoFuse dump root; defaults to NPU manifest sibling directory",
    )
    _add_common_thresholds(cmp_parser)
    _add_elementwise_data_option(cmp_parser)
    cmp_parser.set_defaults(func=cmd_compare)

    cmp_three = sub.add_parser(
        "compare-three", help="compare Inductor pre-fusion CPU, NPU eager, and NPU Autofuse manifests"
    )
    cmp_three.add_argument("--cpu-manifest", required=True)
    cmp_three.add_argument("--npu-eager-manifest", required=True)
    cmp_three.add_argument("--npu-autofuse-manifest", required=True)
    cmp_three.add_argument("--out", default=None)
    cmp_three.add_argument("--csv", default=None)
    cmp_three.add_argument("--html", default=None, help="write HTML visualization")
    cmp_three.add_argument("--autofuse-dump", default=None)
    _add_common_thresholds(cmp_three)
    _add_elementwise_data_option(cmp_three)
    cmp_three.set_defaults(func=cmd_compare_three)

    npy = sub.add_parser("compare-npy", help="compare two npy files directly")
    npy.add_argument("--golden", required=True)
    npy.add_argument("--actual", required=True)
    npy.add_argument("--golden-format", default=None)
    npy.add_argument("--actual-format", default=None)
    npy.add_argument("--out", default=None)
    npy.add_argument("--csv", default=None)
    npy.add_argument("--html", default=None, help="write HTML visualization")
    _add_common_thresholds(npy)
    _add_elementwise_data_option(npy)
    npy.set_defaults(func=cmd_compare_npy)

    html = sub.add_parser("report-html", help="render an existing report JSON with lazy tensor data")
    html.add_argument("--report", required=True)
    html.add_argument("--out", required=True)
    html.add_argument("--title", default="Autofuse Accuracy Report")
    html.set_defaults(func=cmd_report_html)

    normalize = sub.add_parser(
        "normalize-script",
        help="use an Agent to analyze a model script and write a validated normalization JSON",
    )
    normalize.add_argument("--script", required=True, help="model script to analyze")
    normalize.add_argument("--framework", choices=("auto", *framework_names()), default="auto")
    normalize.add_argument("--out", default=None, help="normalization JSON path")
    normalize.add_argument("--agent-endpoint", default=None, help="OpenAI-compatible Agent HTTP endpoint")
    normalize.add_argument("--agent-transport", choices=("auto", "http", "cannbot"), default=None)
    normalize.add_argument("--agent-command", default=None, help="local Agent command, for example /path/to/cannbot")
    normalize.add_argument("--agent-model", default=None)
    normalize.add_argument("--agent-timeout", type=float, default=None)
    normalize.add_argument("script_args", nargs=argparse.REMAINDER, help="arguments used by the model script")

    normalize.set_defaults(func=cmd_normalize_script)

    torch_model = sub.add_parser(
        "run-torch-model", help="compare shared Inductor pre-fusion CPU/NPU eager with native Autofuse execution"
    )
    torch_model.add_argument("--script", required=True, help="python script defining create_case()")
    torch_model.add_argument("--entry", default="create_case", help="case factory function name")
    torch_model.add_argument("--out-dir", required=True)
    torch_model.add_argument("--html", default=None, help="write HTML path; defaults to OUT_DIR/report.html")
    torch_model.add_argument("--cann-root", default=None, help="CANN installation root")
    torch_model.add_argument("--device", default="npu:0")
    torch_graph_mode = torch_model.add_mutually_exclusive_group()
    torch_graph_mode.add_argument("--fullgraph", action="store_const", const=True, help="override create_case and compile one full graph")
    torch_graph_mode.add_argument("--allow-graph-breaks", dest="fullgraph", action="store_const", const=False, help=argparse.SUPPRESS)
    torch_model.set_defaults(fullgraph=None)
    torch_model.add_argument(
        "--rewrite-device-hints",
        action="store_true",
        help="compatibility mode: temporarily rewrite model dev/device attributes for each target device",
    )
    _add_common_thresholds(torch_model)
    _add_elementwise_data_option(torch_model)
    _add_with_cpu_option(torch_model)
    torch_model.set_defaults(func=cmd_run_torch_model)

    script_runner = sub.add_parser(
        "run-script", help="capture a framework model invocation and compare it with an NPU backend"
    )
    script_runner.add_argument("--script", required=True, help="model script that executes the target inference")
    script_runner.add_argument("--out-dir", required=True)
    script_runner.add_argument("--framework", choices=("auto", *framework_names()), default="auto")
    script_runner.add_argument("--backend", choices=backend_names(), default="inductor_autofuse")
    script_runner.add_argument(
        "--capture-call", type=int, default=None, help="zero-based matching top-level model call; defaults to 0"
    )
    script_runner.add_argument("--module-pattern", default=None, help="regex matched against module class name")
    script_runner.add_argument("--name", default=None, help="report model name; defaults to captured module class")
    graph_mode = script_runner.add_mutually_exclusive_group()
    graph_mode.add_argument("--fullgraph", action="store_true", help="compile one full graph instead of allowing graph breaks")
    graph_mode.add_argument("--allow-graph-breaks", dest="fullgraph", action="store_false", help="compatibility alias; graph breaks are now the default")
    script_runner.set_defaults(fullgraph=False)
    script_runner.add_argument("--dynamic", action="store_true")
    script_runner.add_argument(
        "--rewrite-device-hints",
        action="store_true",
        help="compatibility mode: temporarily rewrite model dev/device attributes for each target device",
    )
    script_runner.add_argument("--html", default=None, help="write HTML path; defaults to OUT_DIR/report.html")
    script_runner.add_argument("--cann-root", default=None, help="CANN installation root")
    script_runner.add_argument("--device", default="npu:0")
    script_runner.add_argument(
        "--env-file",
        default=None,
        help="shell file whose export KEY=VALUE lines are applied before execution (not executed)",
    )
    script_runner.add_argument("--env", action="append", default=[], help="extra KEY=VALUE env, repeatable")
    _add_common_thresholds(script_runner)
    _add_elementwise_data_option(script_runner)
    _add_with_cpu_option(script_runner)
    _add_agent_options(script_runner)
    script_runner.add_argument("script_args", nargs=argparse.REMAINDER, help="arguments passed after -- to the script")
    script_runner.set_defaults(func=cmd_run_script)

    run = sub.add_parser("run", help="run an external command and capture logs")
    run.add_argument("--out-dir", required=True)
    run.add_argument("--command", required=True, help="shell command; supports {out_dir} placeholder")
    run.add_argument("--env", action="append", default=[], help="extra KEY=VALUE env, repeatable")
    run.set_defaults(func=cmd_run)

    debug_cpu = sub.add_parser("run-debug-cpu", help="run a command with TORCHINDUCTOR_NPU_EXT_DEBUG=cpu")
    debug_cpu.add_argument("--out-dir", required=True)
    debug_cpu.add_argument("--command", required=True, help="shell command; supports {out_dir} placeholder")
    debug_cpu.add_argument("--env", action="append", default=[], help="extra KEY=VALUE env, repeatable")
    debug_cpu.set_defaults(func=cmd_run_debug_cpu)

    info = sub.add_parser("camodel-info", help="inspect CANN simulator/CAModel directories")
    info.add_argument("--cann-root", required=True, help="CANN root, e.g. /home/xingzhixiong/Ascend/cann-9.1.0")
    info.set_defaults(func=cmd_camodel_info)

    camodel = sub.add_parser("run-camodel", help="run a command with CANN simulator/CAModel environment")
    camodel.add_argument("--cann-root", required=True)
    camodel.add_argument("--soc", default=None, help="simulator SoC directory name, e.g. dav_3510")
    camodel.add_argument("--out-dir", required=True)
    camodel.add_argument("--command", required=True, help="shell command; supports {out_dir} placeholder")
    camodel.add_argument("--env", action="append", default=[], help="extra KEY=VALUE env, repeatable")
    camodel.set_defaults(func=cmd_run_camodel)

    fuzz = sub.add_parser(
        "af-fuzz", help="generate NNSmith cases and verify them end-to-end against the autofuse backend"
    )
    fuzz.add_argument("--out-dir", required=True)
    fuzz.add_argument("--n-cases", type=int, default=None,
                      help="case limit; omitted: continuous in performance mode, 100 in correctness mode; resume uses the saved limit")
    fuzz.add_argument("--max-nodes", type=int, default=5)
    fuzz.add_argument(
        "--timeout-ms",
        type=int,
        default=None,
        help="Per-case generation timeout. Default: None, so --tensor-scale's "
        "profile value takes effect; passing this explicitly overrides the "
        "profile's timeout_ms (see TENSOR_SCALE_PROFILES in fuzz_loop.py).",
    )
    fuzz.add_argument("--dynamic-shapes", action="store_true", help="validate multiple input shapes using strict symbolic dimensions and torch.compile(dynamic=True)")
    fuzz.add_argument("--dynamic-variants", type=int, default=3, help="number of distinct shapes per dynamic case, 2..8 (default 3)")
    fuzz.add_argument("--method", default="symbolic-cinit")
    fuzz.add_argument(
        "--max-elem-per-tensor",
        type=int,
        default=None,
        help="Per-tensor element-count cap. Default: None, so --tensor-scale's "
        "profile value takes effect; passing this explicitly overrides the "
        "profile's max_elem_per_tensor (see TENSOR_SCALE_PROFILES in "
        "fuzz_loop.py).",
    )
    fuzz.add_argument(
        "--tensor-scale",
        # Hardcoded literal, not derived from fuzz_loop.TENSOR_SCALE_PROFILES:
        # build_parser() runs on every CLI invocation regardless of which
        # subcommand is selected, so computing this from fuzz_loop would force
        # fuzz_loop's heavy nnsmith/torch imports (and autofusedebugger's
        # nnsmith patch side effect) on every invocation, not just af-fuzz.
        # Keep in sync with the keys of TENSOR_SCALE_PROFILES in fuzz_loop.py.
        choices=["default", "large", "medium", "small", "16k", "32k", "64k", "128k"],
        default="default",
        help="Controls max_elem_per_tensor + concr_ph_dim_rng + timeout_ms "
        "together (see TENSOR_SCALE_PROFILES in fuzz_loop.py). 'default' "
        "samples per-dim sizes from [1, 128k] with a half-normal "
        "distribution (small shapes common, large shapes rare). 'small' "
        "matches the original hardcoded defaults for regression "
        "compatibility. '16k'/'32k'/'64k'/'128k' push concr_ph_dim_rng's "
        "per-dim upper bound to that many elements (long-context seq_len "
        "style scenarios), well beyond 'large''s 8192.",
    )
    fuzz.add_argument(
        "--value-dist-mode",
        choices=["llm", "log_scale", "recommendation", "uniform"],
        default="uniform",
        help="Input value sampling mode (see "
        "nnsmith.materialize.torch.symbolnet.random_tensor). 'uniform' "
        "(default) matches the original [3,7) behavior; 'log_scale' spans "
        "+-10^-4..10^4 magnitudes; 'llm' mimics LLM activations (mostly "
        "N(0, 0.2^2), with 5%% of elements amplified 100x-10000x as "
        "'massive activations'); 'recommendation' mimics min-max "
        "normalized recommendation-system features (uniform [0,1)). "
        "Applies both to make_oracle's fallback random inputs and to the "
        "inputs model.refine_weights() pre-samples as PracticalHybridSearch's "
        "search starting points, so the cached sat_inputs it may settle on "
        "also reflects this mode instead of silently reverting to uniform. "
        "Does not affect PracticalHybridSearch's internal weight sampling, "
        "which remains fixed at uniform.",
    )
    fuzz.add_argument("--seed", type=int, default=None)
    fuzz.add_argument("--target", default="npu")
    fuzz.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="run N persistent case workers consuming a shared task pool "
        "(each worker compiles on real NPU hardware, which is memory/CPU "
        "heavy -- start low and watch system load before raising this; "
        "jobs=8 has been observed to crash a 40-core/229GB instance). "
        "Default 2: on A5 (32GB cgroup, 1 NPU, cold compile cache, inductor "
        "compile pool 4) jobs=2 ran 20 cases in 81s / 17GB peak vs 211s / "
        "11GB at jobs=1, while jobs=4 only reached 74s at 28GB peak. Raise "
        "it on machines with more memory; peak memory is roughly "
        "jobs * (~2GB + compile-threads * ~1.5GB)",
    )
    fuzz.add_argument(
        "--devices",
        default=None,
        help="comma-separated NPU device indices to spread --jobs workers across "
        "(e.g. '0,1' on a 2-card box): worker i stays on devices[i %% len(devices)]; idle workers claim the next case, "
        "round-robin, so --jobs can exceed the card count. Only affects --target "
        "npu (the default); ignored for cuda/cpu. Default: unset, meaning every "
        "worker runs on plain npu (device 0), identical to single-device behavior.",
    )
    fuzz.add_argument(
        "--compile-threads",
        type=int,
        default=None,
        help="size of each process's inductor parallel-compile worker pool "
        "(sets TORCHINDUCTOR_COMPILE_THREADS). Each worker is a full python "
        "process importing torch (~0.7-1.5GB RSS) and every --jobs process "
        "gets its own pool. Default: the TORCHINDUCTOR_COMPILE_THREADS env "
        "var if set, else 4. Inductor's own default is min(32, host cpu "
        "count), which OOM-kills a 32GB container on a cold compile cache",
    )
    fuzz.add_argument(
        "--soc-tier",
        choices=("A3", "A5"),
        default="A3",
        help="narrow the generated op catalog to ops ascendc registers a fused lowering for on this "
        "NPU generation (see ascendc_opset.py); use A5 once validating on A5 hardware",
    )
    fuzz.add_argument(
        "--op-categories",
        default=None,
        help="comma-separated subset of op categories to generate cases from (e.g. "
        "'elewise,reduce'), replacing the default full op catalog entirely -- not "
        "additionally restricting it. Choices: elewise, reduce, view, gather, norm, "
        "matmul (see OP_CATEGORIES in ascendc_opset.py for which ops belong to each). "
        "Applied after --soc-tier's own hardware-support filtering, so it can only "
        "narrow further, never add back an op the soc_tier doesn't support. Default: "
        "unset, meaning no restriction beyond --soc-tier.",
    )
    fuzz.add_argument(
        "--repro-seed",
        type=int,
        default=None,
        help="re-run exactly one case (its recorded `case_seed`, e.g. from summary.json's case_id "
        "suffix or CaseResult.seed) instead of a batch; implies --debug unless --no-debug is passed",
    )
    fuzz.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=None,
        help="dump torch_compile_debug/ + inductor cache + model/oracle/repro.py for the reproduced "
        "case, pass or fail (only applies with --repro-seed)",
    )
    fuzz.add_argument("--no-debug", dest="debug", action="store_false")
    from autofusedebugger.nnsmith_integration.performance.cli import add_arguments
    add_arguments(fuzz)
    fuzz.set_defaults(func=cmd_af_fuzz, debug=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        direct = parse_python_script_invocation(raw_argv)
        if direct is not None:
            return cmd_run_direct(*direct)
        if raw_argv and is_shell_script_invocation(raw_argv):
            return cmd_run_direct_shell(raw_argv)
        parser = build_parser()
        args = parser.parse_args(raw_argv)
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
