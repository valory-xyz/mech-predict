"""
Shared JSONL I/O helpers for benchmark scripts.

Provides load, append, and write functions used across runner.py,
tournament.py, scorer.py, score_tournament.py, fetch_open.py,
fetch_replay.py, and analyze.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all rows from a JSONL file.

    :param path: path to the JSONL file.
    :return: list of parsed dicts.
    """
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_existing_ids(path: Path, *, key: str = "row_id") -> set[str]:
    """Load IDs from a JSONL file for deduplication.

    :param path: path to the JSONL file.
    :param key: JSON key to extract as the ID.
    :return: set of existing IDs, empty if file does not exist.
    """
    if not path.exists():
        return set()
    ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)[key])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    """Append rows to a JSONL file. Returns count of rows written.

    :param path: path to the JSONL file.
    :param rows: list of dicts to append.
    :return: number of rows written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    """Write rows to a JSONL file (overwrites existing content).

    :param path: path to the JSONL file.
    :param rows: list of dicts to write.
    :return: number of rows written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
