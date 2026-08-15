from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autofusedebugger.adapters import backend_names, framework_names
from autofusedebugger.autofuse_graph import attach_autofuse_graphs, enable_autofuse_graph_dump
from autofusedebugger.camodel import camodel_info, format_info
from autofusedebugger.compare import compare_manifests, compare_pair, compare_three_manifests, format_summary, save_report
from autofusedebugger.graph import build_manifest
from autofusedebugger.html import write_report_html, write_report_html_from_json
from autofusedebugger.progress import progress_stage
from autofusedebugger.runner import run_camodel_command, run_command, run_debug_cpu_command
from autofusedebugger.script_runner import run_framework_script
from autofusedebugger.torch_model import format_run_summary as format_torch_run_summary
from autofusedebugger.torch_model import run_torch_model


def _add_common_thresholds(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rtol", type=float, default=None, help="override dtype-aware relative tolerance")
    parser.add_argument("--atol", type=float, default=None, help="override dtype-aware absolute tolerance")


def _add_elementwise_data_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--save-elementwise-data",
        action="store_true",
        help="persist per-element tensor values and masks for the HTML Tensor viewer",
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


def cmd_report_html(args: argparse.Namespace) -> int:
    write_report_html_from_json(args.report, args.out, args.title)
    print(f"wrote {args.out}")
    return 0


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
    )
    print(format_torch_run_summary(summary))
    return 0 if summary["passed"] else 1


def cmd_run_script(args: argparse.Namespace) -> int:
    script_args = tuple(args.script_args)
    if script_args[:1] == ("--",):
        script_args = script_args[1:]
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
    )
    print(format_torch_run_summary(summary))
    return 0 if summary["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autofusedbg")
    sub = parser.add_subparsers(dest="command_name", required=True)

    make = sub.add_parser("make-manifest", help="build a manifest from graph JSON and an npy dump directory")
    make.add_argument("--side", choices=["cpu", "npu"], required=True)
    make.add_argument("--graph", required=True)
    make.add_argument("--dump-dir", required=True)
    make.add_argument("--out", required=True)
    make.add_argument("--model-name", default=None)
    make.set_defaults(func=cmd_make_manifest)

    cmp_parser = sub.add_parser("compare", help="compare CPU and NPU manifests")
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

    cmp_three = sub.add_parser("compare-three", help="compare CPU, NPU eager, and NPU Autofuse manifests")
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

    torch_model = sub.add_parser(
        "run-torch-model", help="compare CPU export with native Inductor+AscendC Autofuse execution"
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
    torch_model.set_defaults(func=cmd_run_torch_model)

    script_runner = sub.add_parser(
        "run-script", help="capture a framework model invocation and compare it with an NPU backend"
    )
    script_runner.add_argument("--script", required=True, help="model script that executes the target inference")
    script_runner.add_argument("--out-dir", required=True)
    script_runner.add_argument("--framework", choices=("auto", *framework_names()), default="auto")
    script_runner.add_argument("--backend", choices=backend_names(), default="inductor_autofuse")
    script_runner.add_argument("--capture-call", type=int, default=0, help="zero-based matching top-level model call")
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
    _add_common_thresholds(script_runner)
    _add_elementwise_data_option(script_runner)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
