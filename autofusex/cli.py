from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path


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
    from autofusex.execution.graph import build_manifest

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
    from autofusex.workloads.model.autofuse_graph import attach_autofuse_graphs, enable_autofuse_graph_dump
    from autofusex.execution.compare import compare_manifests, compare_pair, compare_three_manifests, format_summary, save_report
    from autofusex.viewer.report import write_report_html, write_report_html_from_json

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
    from autofusex.workloads.model.autofuse_graph import attach_autofuse_graphs, enable_autofuse_graph_dump
    from autofusex.execution.compare import compare_manifests, compare_pair, compare_three_manifests, format_summary, save_report
    from autofusex.viewer.report import write_report_html, write_report_html_from_json
    from autofusex.progress import progress_stage

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
    from autofusex.execution.compare import compare_manifests, compare_pair, compare_three_manifests, format_summary, save_report
    from autofusex.viewer.report import write_report_html, write_report_html_from_json

    from pathlib import Path

    from autofusex.execution.compare import TensorRef

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
        "schema": "autofusex-report/1",
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
    from autofusex.execution.runner import run_camodel_command, run_command, run_debug_cpu_command

    ret = run_command(args.command, args.out_dir, args.env)
    print(f"command exited with {ret}; logs: {args.out_dir}/run.stdout, {args.out_dir}/run.stderr")
    return ret


def cmd_run_debug_cpu(args: argparse.Namespace) -> int:
    from autofusex.execution.runner import run_camodel_command, run_command, run_debug_cpu_command

    ret = run_debug_cpu_command(args.command, args.out_dir, args.env)
    print(f"debug-cpu command exited with {ret}; logs: {args.out_dir}/run.stdout, {args.out_dir}/run.stderr")
    return ret


def cmd_camodel_info(args: argparse.Namespace) -> int:
    from autofusex.execution.camodel import camodel_info, format_info

    print(format_info(camodel_info(args.cann_root)))
    return 0


def cmd_run_camodel(args: argparse.Namespace) -> int:
    from autofusex.execution.runner import run_camodel_command, run_command, run_debug_cpu_command

    ret = run_camodel_command(args.command, args.out_dir, args.cann_root, args.soc, args.env)
    print(f"camodel command exited with {ret}; logs: {args.out_dir}/run.stdout, {args.out_dir}/run.stderr")
    return ret


def cmd_report_html(args: argparse.Namespace) -> int:
    from autofusex.viewer.report import write_report_html, write_report_html_from_json

    write_report_html_from_json(args.report, args.out, args.title)
    print(f"wrote {args.out}")
    return 0


def cmd_normalize_script(args: argparse.Namespace) -> int:
    from autofusex.workloads.model.agent import AgentConfig

    script_args = tuple(args.script_args)
    if script_args[:1] == ("--",):
        script_args = script_args[1:]
    script = Path(args.script).resolve()
    output = (
        Path(args.out).resolve()
        if args.out
        else Path.cwd() / f"{script.stem}_autofusex" / "agent_normalization.json"
    )
    from autofusex.workloads.model.normalization import normalize_script

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
    from autofusex.workloads.model.autofuse_graph import attach_autofuse_graphs, enable_autofuse_graph_dump
    from autofusex.workloads.model.torch_model import format_run_summary as format_torch_run_summary
    from autofusex.workloads.model.torch_model import run_torch_model

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


def cmd_model_run(args: argparse.Namespace) -> int:
    if args.entry is not None:
        if args.script_args:
            raise ValueError("factory entries do not accept script arguments")
        return cmd_run_torch_model(args)
    if args.fullgraph is None:
        args.fullgraph = False
    return cmd_run_script(args)


def cmd_run_script(args: argparse.Namespace) -> int:
    from autofusex.workloads.model.script_runner import (
        default_direct_output_paths,
        parse_python_script_invocation,
        run_framework_script,
    )
    from autofusex.workloads.model.shell_entry import (
        apply_env_overrides,
        apply_plan_env,
        is_shell_script_invocation,
        parse_shell_script,
    )
    from autofusex.workloads.model.torch_model import format_run_summary as format_torch_run_summary

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
    from autofusex.workloads.model.shell_entry import (
        apply_env_overrides,
        apply_plan_env,
        is_shell_script_invocation,
        parse_shell_script,
    )

    """Convenience form: ``afx bash launch.sh`` or ``afx ./launch.sh``."""
    script_index = 0
    if Path(argv[0]).name in {"bash", "sh", "zsh", "dash", "ksh"}:
        if len(argv) < 2:
            raise ValueError("shell launcher mode requires a script path, for example: afx bash launch.sh")
        script_index = 1
    plan = parse_shell_script(argv[script_index])
    apply_plan_env(plan)
    python_script = plan.python_script
    if python_script is None:
        raise ValueError(
            f"no .py entry found in the python invocation of {plan.script}; "
            "use 'afx python3 model.py' instead"
        )
    script_args = tuple(plan.python_tokens[plan.python_tokens.index(str(python_script)) + 1 :])
    return cmd_run_direct(python_script, script_args)


def cmd_run_direct(script: Path, script_args: tuple[str, ...]) -> int:
    from autofusex.workloads.model.script_runner import (
        default_direct_output_paths,
        parse_python_script_invocation,
        run_framework_script,
    )
    from autofusex.workloads.model.torch_model import format_run_summary as format_torch_run_summary

    out_dir, html_path = default_direct_output_paths(script)
    print(f"AutofuseX direct mode: script={script}")
    print(f"AutofuseX outputs: out_dir={out_dir} html={html_path}")
    summary = run_framework_script(
        script=script,
        out_dir=out_dir,
        framework="auto",
        backend="inductor_autofuse",
        script_args=script_args,
        html_path=html_path,
        agent_normalize=os.environ.get("AUTOFUSEX_AGENT_NORMALIZE", "0").lower()
        in {"1", "true", "yes"},
        agent_recover=os.environ.get("AUTOFUSEX_AGENT_RECOVER", "0").lower()
        in {"1", "true", "yes"},
    )
    print(format_torch_run_summary(summary))
    return 0 if summary["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    from autofusex.workloads.model.adapters import backend_names, framework_names

    parser = argparse.ArgumentParser(
        prog="afx",
        description="CPU/NPU eager/NPU Autofuse 精度对比和图分析工具",
        epilog="快捷模式：afx python3 model.py [模型脚本参数...]；"
    "或 afx bash launch.sh / afx ./launch.sh 解析启动脚本继承环境变量",
    )
    sub = parser.add_subparsers(dest="command_name", required=True)

    analyze = sub.add_parser("analyze", help="compare tensors and analyze reports").add_subparsers(dest="analysis", required=True)
    model = sub.add_parser("model", help="run and normalize whole models").add_subparsers(dest="model_command", required=True)
    debug = sub.add_parser("debug", help="capture commands and inspect simulators").add_subparsers(dest="debug_command", required=True)
    simulator = debug.add_parser("camodel").add_subparsers(dest="camodel_command", required=True)
    make = analyze.add_parser("manifest", help="build a manifest from graph JSON and an npy dump directory")
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

    cmp_parser = analyze.add_parser("compare", help="compare a reference and NPU manifest")
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

    cmp_three = analyze.add_parser(
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

    npy = analyze.add_parser("compare-npy", help="compare two npy files directly")
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

    html = analyze.add_parser("report", help="render an existing report JSON with lazy tensor data")
    html.add_argument("--report", required=True)
    html.add_argument("--out", required=True)
    html.add_argument("--title", default="Autofuse Accuracy Report")
    html.set_defaults(func=cmd_report_html)

    normalize = model.add_parser(
        "normalize",
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

    script_runner = model.add_parser(
        "run", help="capture a framework model invocation and compare it with an NPU backend"
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
    script_runner.set_defaults(fullgraph=None)
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
    script_runner.add_argument("--entry", default=None, help="case factory function; omit to capture script inference")
    script_runner.set_defaults(func=cmd_model_run)

    run = debug.add_parser("run", help="run an external command and capture logs")
    run.add_argument("--out-dir", required=True)
    run.add_argument("--command", required=True, help="shell command; supports {out_dir} placeholder")
    run.add_argument("--env", action="append", default=[], help="extra KEY=VALUE env, repeatable")
    run.add_argument("--mode", choices=("normal", "cpu"), default="normal")
    run.set_defaults(func=lambda args: cmd_run_debug_cpu(args) if args.mode == "cpu" else cmd_run(args))

    info = simulator.add_parser("info", help="inspect CANN simulator/CAModel directories")
    info.add_argument("--cann-root", required=True, help="CANN root, e.g. /home/xingzhixiong/Ascend/cann-9.1.0")
    info.set_defaults(func=cmd_camodel_info)

    camodel = simulator.add_parser("run", help="run a command with CANN simulator/CAModel environment")
    camodel.add_argument("--cann-root", required=True)
    camodel.add_argument("--soc", default=None, help="simulator SoC directory name, e.g. dav_3510")
    camodel.add_argument("--out-dir", required=True)
    camodel.add_argument("--command", required=True, help="shell command; supports {out_dir} placeholder")
    camodel.add_argument("--env", action="append", default=[], help="extra KEY=VALUE env, repeatable")
    camodel.set_defaults(func=cmd_run_camodel)

    return parser


def root_help():
    parser = argparse.ArgumentParser(prog='afx', description='AFX unified testing platform')
    parser.add_argument('-v', '--version', action='store_true', help='show installed version')
    sub = parser.add_subparsers(title='commands')
    for name, help_text in (
        ('init', 'initialize a local or managed client'), ('start', 'start testing'),
        ('stop', 'stop testing'), ('restart', 'restart client'), ('status', 'show client status'),
        ('config', 'show, edit, reset, validate or export configuration'),
        ('env', 'manage execution environments'), ('replay', 'replay using historical configuration'),
        ('task', 'submit and control tasks'), ('operation', 'inspect asynchronous operations'),
        ('build', 'build execution environments'), ('update', 'upgrade the client'),
        ('release', 'manage signed releases'), ('device', 'manage device registration'),
        ('fuzz', 'generate and replay fuzz cases'), ('model', 'run and normalize whole models'),
        ('analyze', 'compare tensors and render reports'), ('debug', 'capture commands and inspect simulators'),
        ('results', 'query, compare, snapshot and maintain result databases'),
        ('view', 'open a local read-only viewer'), ('doctor', 'inspect execution capabilities'),
        ('server', 'run and administer the central server'), ('sync', 'synchronize standalone results'),
        ('retest', 'execute platform retests')):
        sub.add_parser(name, help=help_text, add_help=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv or raw_argv in (["-h"], ["--help"]):
        root_help().print_help()
        return 0
    if raw_argv[0] == 'results':
        from autofusex.execution.results_cli import main as results_main
        return results_main(raw_argv[1:])
    if raw_argv[0] == 'view':
        parser = argparse.ArgumentParser(prog='afx view')
        parser.add_argument('database')
        parser.add_argument('--port', type=int, default=0)
        parser.add_argument('--no-browser', action='store_true')
        args = parser.parse_args(raw_argv[1:])
        from autofusex.viewer.server import serve
        try:
            return serve(args.database, port=args.port, open_browser=not args.no_browser)
        except (ValueError, OSError, sqlite3.Error) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    if raw_argv[0] == 'doctor':
        parser = argparse.ArgumentParser(prog='afx doctor')
        parser.add_argument('--workload', choices=('fuzz',), default='fuzz')
        parser.add_argument('--data-dir')
        args = parser.parse_args(raw_argv[1:])
        from autofusex.workloads.fuzz.cli import doctor
        import json
        try:
            print(json.dumps(doctor(args.data_dir), indent=2))
            return 0
        except (ValueError, OSError, sqlite3.Error) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    # Compact client surface. The legacy management tree remains available
    # for compatibility and for administrator/maintenance workflows.
    modern = raw_argv and raw_argv[0] in {"init", "start", "status", "stop", "restart", "replay", "env", "-h", "--help", "-v", "--version"}
    modern = modern or (raw_argv and raw_argv[0] == "config")
    if modern:
        from autofusex.client.modern_cli import main as modern_main
        # Accept the repository-wide convention of placing --data-dir at the
        # end as well as before the subcommand.
        if "--data-dir" in raw_argv[1:]:
            index = raw_argv.index("--data-dir", 1)
            if index + 1 >= len(raw_argv):
                return modern_main(raw_argv)
            raw_argv = ["--data-dir", raw_argv[index + 1], *raw_argv[:index], *raw_argv[index + 2:]]
        return modern_main(raw_argv)
    if raw_argv and raw_argv[0] in {"device", "task", "operation", "build", "release", "update"}:
        from autofusex.client.cli import main as management_main
        if raw_argv[0] == "release":
            raw_argv[0] = "tool"
        elif raw_argv[0] == "update":
            raw_argv = ["client", "update", *raw_argv[1:]]
        return management_main(raw_argv)
    if raw_argv and raw_argv[0] == "fuzz":
        from autofusex.workloads.fuzz.cli import main as fuzz_main
        return fuzz_main(raw_argv[1:])
    if raw_argv and raw_argv[0] in {"sync", "retest"}:
        from autofusex.client.sync_cli import main as sync_main
        return sync_main(raw_argv[1:], retest=raw_argv[0] == "retest")
    if raw_argv and raw_argv[0] == "server":
        from autofusex.server.cli import main as platform_main
        return platform_main(raw_argv[1:])
    from autofusex.workloads.model.script_runner import (
        default_direct_output_paths,
        parse_python_script_invocation,
        run_framework_script,
    )
    from autofusex.workloads.model.shell_entry import (
        apply_env_overrides,
        apply_plan_env,
        is_shell_script_invocation,
        parse_shell_script,
    )

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
