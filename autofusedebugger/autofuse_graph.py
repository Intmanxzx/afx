from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


_NODE_RE = re.compile(r"^\s*\[(\d+)\]\s+(.+?)\s+:\s+ascir\.ops\.([^\s]+)\s*$")
_GRAPH_RE = re.compile(r"^Graph:\s+(.+?)\s*$")
_MEM_RE = re.compile(r"\.mem\s*=\s*([^\[]+)\[([^\]]*)\]")
_EXEC_CONDITION_RE = re.compile(r"\.exec_condition\s*=\s*([A-Za-z0-9_]+)")
_REF_RE = re.compile(r"([A-Za-z0-9_./:-]+)\.y")
_INT_RE = re.compile(r"(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)=([-\d]+)")
_KERNEL_RE = re.compile(r"^autofused_.+_[0-9a-f]{32}$")
_BROADCAST_CACHE_CONDITIONS = {
    "cache_block_split_fused_brc_axis",
    "cache_block_split_origin_brc_axis",
}


def _parse_ints(value: str) -> dict[str, int]:
    return {key: int(raw) for key, raw in _INT_RE.findall(value)}


def _memory_item(place: str, attrs: str) -> dict[str, Any]:
    item: dict[str, Any] = {"place": place.strip()}
    item.update(_parse_ints(attrs))
    return item


def _finish_node(node: dict[str, Any], lines: list[str]) -> dict[str, Any]:
    inputs: list[str] = []
    memories: list[dict[str, Any]] = []
    exec_condition = "no_cache"
    for line in lines:
        if ".x =" in line:
            inputs.extend(_REF_RE.findall(line))
        match = _MEM_RE.search(line)
        if match:
            memories.append(_memory_item(match.group(1), match.group(2)))
        match = _EXEC_CONDITION_RE.search(line)
        if match:
            exec_condition = match.group(1)
    node["inputs"] = inputs
    node["outputs"] = memories
    node["exec_condition"] = exec_condition
    node["broadcast_cache"] = exec_condition in _BROADCAST_CACHE_CONDITIONS
    return node


def _classify_queues(nodes: list[dict[str, Any]]) -> None:
    queue_users = [
        (node, memory)
        for node in nodes
        for memory in node["outputs"]
        if memory.get("que_id", -1) >= 0
    ]
    pair_counts: dict[tuple[int, int], int] = {}
    reuse_sets: dict[int, set[int]] = {}
    for _, memory in queue_users:
        key = (memory["que_id"], memory.get("reuse_id", -1))
        pair_counts[key] = pair_counts.get(key, 0) + 1
        reuse_sets.setdefault(memory["que_id"], set()).add(memory.get("reuse_id", -1))
    node_classes: dict[int, set[str]] = {}
    for node, memory in queue_users:
        que_id = memory["que_id"]
        reuse_id = memory.get("reuse_id", -1)
        if pair_counts[(que_id, reuse_id)] > 1:
            queue_class = "shared"
        elif len(reuse_sets[que_id]) > 1:
            queue_class = "reused"
        else:
            queue_class = "single"
        memory["queue_class"] = queue_class
        node_classes.setdefault(node["id"], set()).add(queue_class)

    priority = ("shared", "reused", "single")
    for node in nodes:
        classes = node_classes.get(node["id"], set())
        if not classes:
            continue
        node["queue_class"] = next(item for item in priority if item in classes)
        if node["type"].lower() == "load":
            node["load_class"] = node["queue_class"]


def _consumer_nodes(edges: list[dict[str, int]]) -> dict[int, list[int]]:
    consumers: dict[int, list[int]] = {}
    for edge in edges:
        consumers.setdefault(edge["src"], []).append(edge["dst"])
    for values in consumers.values():
        values.sort()
    return consumers


def _finish_interval(instance: dict[str, Any], consumers: dict[int, list[int]]) -> None:
    producers = sorted(set(instance.pop("_producer_nodes", [])))
    consumer_nodes = sorted({consumer for producer in producers for consumer in consumers.get(producer, [])})
    related = sorted(set(producers + consumer_nodes))
    instance["producer_nodes"] = producers
    instance["consumer_nodes"] = consumer_nodes
    instance["start"] = min(related) if related else -1
    instance["end"] = max(related) if related else -1


def parse_ascgraph_text(text: str, source: str = "") -> dict[str, Any]:
    graph_name = next((match.group(1) for line in text.splitlines() if (match := _GRAPH_RE.match(line))), "")
    raw_nodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_lines: list[str] = []
    in_structure = False
    for line in text.splitlines():
        if line.strip() == "Nodes:":
            in_structure = True
            continue
        if in_structure and line.strip() == "--------------------------------------------------------------------------------":
            if current is not None:
                raw_nodes.append(_finish_node(current, current_lines))
                current = None
                current_lines = []
            in_structure = False
            continue
        if not in_structure:
            continue
        match = _NODE_RE.match(line)
        if match:
            if current is not None:
                raw_nodes.append(_finish_node(current, current_lines))
            current = {"id": int(match.group(1)), "name": match.group(2), "type": match.group(3)}
            current_lines = []
        elif current is not None:
            current_lines.append(line)
    if current is not None:
        raw_nodes.append(_finish_node(current, current_lines))

    raw_nodes.sort(key=lambda node: node["id"])
    node_ids = {node["name"]: node["id"] for node in raw_nodes}
    edges: list[dict[str, int]] = []
    seen_edges: set[tuple[int, int]] = set()
    for node in raw_nodes:
        for source_name in node["inputs"]:
            source_id = node_ids.get(source_name)
            if source_id is None or (source_id, node["id"]) in seen_edges:
                continue
            seen_edges.add((source_id, node["id"]))
            edges.append({"src": source_id, "dst": node["id"]})

    _classify_queues(raw_nodes)
    consumers = _consumer_nodes(edges)
    queues: dict[int, dict[str, Any]] = {}
    queue_instances: dict[tuple[int, int], dict[str, Any]] = {}
    buffers: dict[int, dict[str, Any]] = {}
    for node in raw_nodes:
        for memory in node["outputs"]:
            que_id = memory.get("que_id", -1)
            if que_id >= 0:
                queue = queues.setdefault(que_id, {"id": que_id, "uses": [], "load_nodes": [], "instances": []})
                if node["id"] not in queue["uses"]:
                    queue["uses"].append(node["id"])
                if node["type"].lower() == "load" and node["id"] not in queue["load_nodes"]:
                    queue["load_nodes"].append(node["id"])
                reuse_id = memory.get("reuse_id", -1)
                instance = queue_instances.setdefault(
                    (que_id, reuse_id),
                    {
                        "reuse_id": reuse_id,
                        "depth": memory.get("depth", -1),
                        "buf_num": memory.get("buf_num", -1),
                        "queue_class": memory.get("queue_class", "single"),
                        "_producer_nodes": [],
                    },
                )
                instance["_producer_nodes"].append(node["id"])
            buf_id = memory.get("buf_id", -1)
            if buf_id >= 0:
                buffer = buffers.setdefault(buf_id, {"id": buf_id, "uses": [], "instances": []})
                if node["id"] not in buffer["uses"]:
                    buffer["uses"].append(node["id"])
                buffer["instances"].append(
                    {"kind": "output", "source_node": node["id"], "_producer_nodes": [node["id"]]}
                )
    for (que_id, _), instance in sorted(queue_instances.items()):
        _finish_interval(instance, consumers)
        queues[que_id]["instances"].append(instance)
    for buffer in buffers.values():
        for instance in buffer["instances"]:
            _finish_interval(instance, consumers)

    for resource in (*queues.values(), *buffers.values()):
        resource["uses"].sort()
    return {
        "name": graph_name or Path(source).stem,
        "source": source,
        "nodes": raw_nodes,
        "edges": sorted(edges, key=lambda edge: (edge["src"], edge["dst"])),
        "queues": [queues[key] for key in sorted(queues)],
        "buffers": [buffers[key] for key in sorted(buffers)],
    }


def collect_autofuse_graphs(root: str | Path) -> list[dict[str, Any]]:
    dump_root = Path(root).resolve()
    graphs = []
    for path in sorted(dump_root.rglob("ascgraph_*AutoFuseAfterOptimize*.txt")):
        try:
            graph = parse_ascgraph_text(path.read_text(encoding="utf-8"), str(path))
            graph["kernel_name"] = next(
                (part for part in reversed(path.parts) if _KERNEL_RE.match(part)),
                "",
            )
            graph["variant_name"] = path.stem
            graph["source_relative"] = str(path.relative_to(dump_root))
            graphs.append(graph)
        except (OSError, UnicodeDecodeError):
            continue
    return graphs


def enable_autofuse_graph_dump(path: str | Path) -> None:
    debug_dir = str(Path(path).resolve())
    existing = [item for item in os.environ.get("AUTOFUSE_DFX_FLAGS", "").split(";") if item]
    existing = [item for item in existing if "codegen_compile_debug" not in item and "debug_dir" not in item]
    os.environ["AUTOFUSE_DFX_FLAGS"] = ";".join(
        [*existing, "--codegen_compile_debug=true", f"--debug_dir={debug_dir}"]
    )


def attach_autofuse_graphs(report: dict[str, Any], root: str | Path) -> None:
    report.setdefault("graphs", {})["autofuse"] = {"graphs": collect_autofuse_graphs(root)}
