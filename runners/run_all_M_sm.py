"""Parallel runner for the 30 M+SM main_th_sweep variants.

Usage:
    uv run python runners/run_all_M_sm.py
    uv run python runners/run_all_M_sm.py --retrain
    uv run python runners/run_all_M_sm.py --retrain --throttle 4 --threads 2 --model xgb

Launches `--throttle` subprocesses concurrently. Each subprocess gets
EBM_N_JOBS=`--threads` so 4×2 = 8 threads stays inside the 12 logical core
budget while leaving room for OS + bitcoind. MODEL_KIND=`--model` picks ebm
(default) or xgb to write into preds_cache_ebm/ vs preds_cache_xgb/.

Each variant's stdout/stderr is redirected to logs/<version>_<filename>.log.
The launcher exits non-zero if any variant failed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)


def discover_variants() -> list[tuple[str, Path]]:
    """Return [(version, script_path), ...] for every M+SM sweep entry-point."""
    out: list[tuple[str, Path]] = []
    for v in ("v0", "v3", "v4", "v5", "v5_2", "v6", "v7", "v5_3"):
        vdir = ROOT / "main_th_sweep" / v
        if not vdir.exists():
            continue
        for p in sorted(vdir.glob("main_th_sweep_*.py")):
            if "wsun" in p.name:
                continue
            out.append((v, p))
    return out


def _log_marks_done(log_path: Path) -> bool:
    """A variant is considered done if its log's last non-empty line starts
    with 'Markdown:' — every successful sweep ends by printing the report
    markdown path. Used by --skip-done to make the launcher idempotent after
    a crash/Ctrl-C."""
    if not log_path.exists():
        return False
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            tail = [ln.rstrip("\n") for ln in f.readlines()[-50:]]
    except OSError:
        return False
    for ln in reversed(tail):
        s = ln.strip()
        if s:
            return s.startswith("Markdown:")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--throttle", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--model", choices=("ebm", "xgb"), default="ebm")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip variants whose previous log already ends "
                             "with a 'Markdown:' line (idempotent resume).")
    parser.add_argument("--only-version", action="append", default=None,
                        metavar="V",
                        help="Restrict to one or more version dirs (v0/v3/v4/v5/v6). "
                             "Repeatable: --only-version v5 --only-version v6.")
    args = parser.parse_args()

    variants = discover_variants()
    print(f"Discovered {len(variants)} variants. "
          f"throttle={args.throttle}  threads={args.threads}  model={args.model}",
          flush=True)

    if args.only_version:
        wanted = set(args.only_version)
        before = len(variants)
        variants = [(v, s) for v, s in variants if v in wanted]
        print(f"  --only-version {sorted(wanted)} -> {len(variants)}/{before} variants",
              flush=True)

    if args.skip_done:
        kept: list[tuple[str, Path]] = []
        skipped: list[tuple[str, Path]] = []
        for version, script in variants:
            log_path = LOGS / f"{version}_{script.stem}_{args.model}.log"
            if _log_marks_done(log_path):
                skipped.append((version, script))
            else:
                kept.append((version, script))
        for version, script in skipped:
            print(f"  SKIP (already done) {version}/{script.name}", flush=True)
        print(f"Resuming {len(kept)}/{len(variants)} variants "
              f"({len(skipped)} skipped).", flush=True)
        variants = kept

    pending = list(variants)
    running: list[tuple[str, Path, subprocess.Popen, Path]] = []
    completed: list[tuple[str, Path, int]] = []

    extra_env = {
        "EBM_N_JOBS": str(args.threads),
        "MODEL_KIND": args.model,
        "PYTHONIOENCODING": "utf-8",
    }
    env = {**os.environ, **extra_env}

    def _launch_next() -> None:
        if not pending:
            return
        version, script = pending.pop(0)
        log_path = LOGS / f"{version}_{script.stem}_{args.model}.log"
        cmd = ["uv", "run", "python", str(script)]
        if args.retrain:
            cmd.append("--retrain")
        f = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                                 stdout=f, stderr=subprocess.STDOUT)
        running.append((version, script, proc, log_path))
        print(f"  START [pid={proc.pid}] {version}/{script.name}  -> {log_path.name}",
              flush=True)

    t0 = time.time()
    while pending or running:
        while len(running) < args.throttle and pending:
            _launch_next()
        time.sleep(2.0)
        still: list[tuple[str, Path, subprocess.Popen, Path]] = []
        for version, script, proc, log_path in running:
            rc = proc.poll()
            if rc is None:
                still.append((version, script, proc, log_path))
            else:
                elapsed = time.time() - t0
                tag = "DONE" if rc == 0 else f"FAIL[rc={rc}]"
                print(f"  {tag} {version}/{script.name}  "
                      f"(t={elapsed:.0f}s, log={log_path.name})",
                      flush=True)
                completed.append((version, script, rc))
        running = still

    failures = [c for c in completed if c[2] != 0]
    print(f"\nSummary: completed={len(completed)}, failed={len(failures)}",
          flush=True)
    if failures:
        for version, script, rc in failures:
            print(f"  FAIL rc={rc}  {version}/{script.name}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
