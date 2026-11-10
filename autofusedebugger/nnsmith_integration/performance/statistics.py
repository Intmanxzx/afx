"""Parse profiler totals once, then divide by the actual measured step count."""
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re


class ProfileDataError(ValueError):
    pass


def find_op_statistic(root: Path) -> Path:
    paths = sorted(root.rglob("op_statistic.csv"))
    # torch_npu may also keep a raw msprof copy. Prefer its final analysis.
    analyzed = [p for p in paths if p.parent.name == "ASCEND_PROFILER_OUTPUT"]
    paths = analyzed or paths
    if len(paths) != 1:
        raise ProfileDataError(f"Expected one op_statistic.csv in this capture; found {len(paths)} under {root}")
    return paths[0]


def read_op_statistic(path: Path, steps: int) -> dict:
    if steps < 1:
        raise ValueError("Measured steps must be positive")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        duration = None
        factor = None
        for header in headers:
            normalized = re.sub(r"[\s_]+", "", header).lower().replace("μ", "u").replace("µ", "u")
            match = re.fullmatch(r"totaltime\((us|ms|ns|s)\)", normalized)
            if match:
                if duration is not None:
                    raise ProfileDataError("Multiple total-time columns")
                duration = header
                factor = {"us": Decimal(1), "ms": Decimal(1000), "ns": Decimal("0.001"), "s": Decimal(1000000)}[match[1]]
        if duration is None:
            raise ProfileDataError(f"Missing Total Time with an explicit unit: {headers}")
        count_column = next((h for h in headers if re.sub(r"[\s_]+", "", h).lower()
                             in {"count", "occurrences", "taskcount"}), None)
        total_count = 0 if count_column else None
        total = Decimal(0)
        rows = []
        for line, row in enumerate(reader, 2):
            if None in row:
                raise ProfileDataError(f"Malformed CSV row at {path}:{line}")
            if not any(v and str(v).strip() for v in row.values()):
                continue
            try:
                value = Decimal((row.get(duration) or "").strip().replace(",", "")) * factor
            except InvalidOperation as e:
                raise ProfileDataError(f"Invalid total duration at {path}:{line}") from e
            if not value.is_finite() or value < 0:
                raise ProfileDataError(f"Invalid total duration at {path}:{line}")
            if count_column:
                try:
                    count = Decimal((row.get(count_column) or "").strip().replace(",", ""))
                except InvalidOperation as e:
                    raise ProfileDataError(f"Invalid kernel count at {path}:{line}") from e
                if not count.is_finite() or count < 0 or count != count.to_integral_value():
                    raise ProfileDataError(f"Invalid kernel count at {path}:{line}")
                total_count += int(count)
            total += value
            rows.append({"columns": row, "total_us": float(value)})
    if not rows:
        raise ProfileDataError(f"No operator statistics in {path}")
    # Total Time is already aggregated across occurrences. Never multiply by
    # occurrences again or sum per-op averages.
    return {"total_kernel_us": float(total), "mean_kernel_us": float(total / steps),
            "total_kernel_count": total_count,
            "mean_kernel_count": total_count / steps if total_count is not None else None,
            "measured_steps": steps, "csv_path": str(path), "op_statistics": rows}
