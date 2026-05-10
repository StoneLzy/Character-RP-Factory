from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def find_csv_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".csv":
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"CSV directory does not exist: {root}")
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def read_csv_any_encoding(path: Path, encodings: Iterable[str] = ENCODINGS) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not decode {path} with {', '.join(encodings)}: {last_error}",
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_csv_tree(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in find_csv_files(root):
        for index, row in enumerate(read_csv_any_encoding(path), start=1):
            normalized = {str(key).strip(): value for key, value in row.items() if key is not None}
            normalized["source_file"] = path.as_posix()
            normalized["source_row"] = str(index)
            normalized["source_kind"] = _source_kind(root, path)
            rows.append(normalized)
    return rows


def _source_kind(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return ""
    return relative.parts[0] if relative.parts else ""
