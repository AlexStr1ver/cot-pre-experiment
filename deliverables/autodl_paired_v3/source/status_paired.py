#!/usr/bin/env python3
"""Inspect a paired-action run; optionally archive a stopped run without overwriting files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"unavailable": type(exc).__name__}


def process_alive(root, name="runner"):
    """Match PID and process start time, including direct foreground invocations."""
    try:
        pid = int((root / (name + ".pid")).read_text().strip())
        os.kill(pid, 0)
        recorded_start = root / (name + ".start_ticks")
        if recorded_start.exists():
            # Fields after the final ')' start at field 3; field 22 is starttime.
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                return False
            actual_start = fields[19]
            return actual_start == recorded_start.read_text().strip()
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return str(root).encode() in cmdline
    except (FileNotFoundError, ValueError, IndexError, ProcessLookupError, PermissionError):
        return False


def runner_alive(root):
    return process_alive(root, "runner")


def pack_result(root):
    """Write a complete temporary ZIP, then publish with an atomic no-overwrite link."""
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Run path must be a directory")
    target = root.with_name(root.name + "_return.zip")
    if target.exists():
        target = root.with_name(root.name + f"_return_repack_{time.time_ns()}.zip")
    fd, temporary = tempfile.mkstemp(prefix="." + root.name + "_archive_", suffix=".tmp", dir=root.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                # Never follow a symlink out of the result directory.
                if path.is_file() and not path.is_symlink():
                    archive.write(path, path.relative_to(root))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Archive CRC verification failed")
        os.link(temporary, target)  # Fails rather than replacing a concurrent/existing archive.
        return target
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--pack", action="store_true", help="Archive a stopped run, including incomplete results")
    args = parser.parse_args()
    root = args.run_directory.resolve(strict=True)
    live = runner_alive(root)
    worker_live = process_alive(root, "worker")
    exit_file = root / "exit_status.txt"
    exit_status = exit_file.read_text().strip() if exit_file.exists() else None
    summary = read_json(root / "results" / "summary.json")
    if exit_status is not None:
        state = "completed" if exit_status == "0" and summary.get("completed") is True else "failed_or_incomplete"
    else:
        state = "running" if live or worker_live else "interrupted_or_not_started"
    print(json.dumps({"run_directory": str(root), "state": state, "runner_alive": live, "worker_alive": worker_live,
                      "exit_status": exit_status, "summary": summary,
                      "archives": [str(p) for p in sorted(root.parent.glob(root.name + "_return*.zip"))]},
                     ensure_ascii=False, indent=2))
    if args.pack:
        if live or worker_live:
            parser.error("Runner or worker is still active; stop/wait before packing a consistent result")
        print("RETURN_ZIP=" + str(pack_result(root)), flush=True)


if __name__ == "__main__":
    main()
