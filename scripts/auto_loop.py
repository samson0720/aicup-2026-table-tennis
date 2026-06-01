"""Autonomous improvement loop for AI Cup 2026 table tennis.

Runs indefinitely:
  1. Checks pending jobs for completion
  2. Gates each completed model against the current production score
  3. Ships if overall > production + floor
  4. Starts next experiment from the priority queue
  5. Logs everything to /tmp/auto_loop.log

Usage:
  conda run -n aicup-tt python3 -m scripts.auto_loop
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

LOG = Path("/tmp/auto_loop.log")
FLOOR = 0.00168
CUDA = "CUDA_VISIBLE_DEVICES=0"
CWD = Path("/home/tom1030507/ai_cup_table/aicup-2026-table-tennis")
OOF_DIR = CWD / "artifacts" / "oof"
SCORES_JSON = CWD / "artifacts" / "final_perrow_scores.json"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def get_current_score() -> float:
    if SCORES_JSON.exists():
        return json.loads(SCORES_JSON.read_text())["overall"]
    return 0.371610  # last known production score


def run_gate(model_name: str, targets: list[str]) -> float:
    """Run SCORE_ONLY gate and return overall score."""
    env = os.environ.copy()
    for t in targets:
        env[f"AICUP_EXTRA_{t.upper()}_BASE"] = model_name
    env["AICUP_SCORE_ONLY"] = "1"
    result = subprocess.run(
        ["conda", "run", "-n", "aicup-tt", "python3", "-m", "scripts.build_final_perrow"],
        capture_output=True, text=True, env=env, cwd=CWD
    )
    for line in result.stdout.splitlines():
        if '"overall"' in line:
            return float(line.split(":")[1].strip().rstrip(","))
    log(f"  GATE FAILED to parse score. stdout={result.stdout[-500:]}")
    return 0.0


def ship_model(model_name: str, targets: list[str]) -> None:
    """Add model to production BASES and rebuild submissions."""
    perrow_path = CWD / "scripts" / "build_final_perrow.py"
    src = perrow_path.read_text()

    # Add to each target's BASES list
    for target in targets:
        old = f'"{target}": ['
        if model_name in src:
            log(f"  {model_name} already in BASES for {target}, skipping")
            continue
        # Find the closing bracket of the target list and insert before it
        start = src.index(old)
        end = src.index("]", start)
        src = src[:end] + f', "{model_name}"' + src[end:]

    perrow_path.write_text(src)
    log(f"  Updated BASES: added {model_name} to {targets}")

    # Rebuild final submission
    subprocess.run(
        ["conda", "run", "-n", "aicup-tt", "python3", "-m", "scripts.build_final_perrow"],
        cwd=CWD, capture_output=True
    )
    subprocess.run(
        ["conda", "run", "-n", "aicup-tt", "python3", "-m", "scripts.build_leakmax_submission"],
        cwd=CWD, capture_output=True
    )
    log(f"  Rebuilt submission_FINAL_leakmax.csv")


def start_cpu_job(module: str) -> subprocess.Popen:
    return subprocess.Popen(
        ["conda", "run", "-n", "aicup-tt", "python3", "-m", module],
        cwd=CWD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def start_gpu_job(module: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return subprocess.Popen(
        ["conda", "run", "-n", "aicup-tt", "python3", "-m", module],
        cwd=CWD, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def oof_exists(model_name: str, target: str) -> bool:
    return (OOF_DIR / f"{model_name}_{target}.parquet").exists()


def all_oof_exist(model_name: str, targets: list[str]) -> bool:
    return all(oof_exists(model_name, t) for t in targets)


# ── Experiment definitions ──────────────────────────────────────────────────
# Each entry:
#   name: model name
#   module: python -m scripts.<module>
#   gpu: bool
#   targets: which prediction targets this model covers
#   depends_on: list of model names that must be in production before starting
#   gate_targets: which BASES lists to test against (may be subset of targets)
ALL_EXPERIMENTS = [
    # Currently running (just check / gate when done):
    {
        "name": "phase_xgb12_extra",
        "module": "scripts.produce_phase_xgb12_extra_oof",
        "gpu": True,
        "targets": ["action", "point", "server"],
        "depends_on": [],
    },
    {
        "name": "phase_xgb8_900_extra",
        "module": "scripts.produce_phase_xgb8_900_extra_oof",
        "gpu": True,
        "targets": ["action", "point", "server"],
        "depends_on": [],
    },
    # Chain action/point augmentation (CPU, independent)
    {
        "name": "chain_action_extra",
        "module": "scripts.produce_chain_action_extra_oof",
        "gpu": False,
        "targets": ["action"],
        "depends_on": [],
    },
    {
        "name": "chain_point_extra",
        "module": "scripts.produce_chain_point_extra_oof",
        "gpu": False,
        "targets": ["point"],
        "depends_on": ["chain_action_extra"],
    },
    # XGBoost depth ladder continuation
    {
        "name": "phase_xgb14_extra",
        "module": "scripts.produce_phase_xgb14_extra_oof",
        "gpu": True,
        "targets": ["action", "point", "server"],
        "depends_on": [],
    },
    # CatBoost with more iterations
    {
        "name": "phase_cat8_800_extra",
        "module": "scripts.produce_phase_cat8_800_extra_oof",
        "gpu": True,
        "targets": ["action", "point", "server"],
        "depends_on": [],
    },
    # XGBoost depth=8 with even more iterations
    {
        "name": "phase_xgb8_1200_extra",
        "module": "scripts.produce_phase_xgb8_1200_extra_oof",
        "gpu": True,
        "targets": ["action", "point", "server"],
        "depends_on": [],
    },
]


def get_production_bases() -> dict[str, list[str]]:
    src = (CWD / "scripts" / "build_final_perrow.py").read_text()
    start = src.index('BASES = {')
    end = src.index('\n}', start) + 2
    snippet = src[start:end]
    bases: dict[str, list[str]] = {}
    for line in snippet.splitlines():
        for target in ("action", "point", "server"):
            if f'"{target}":' in line or f"'{target}':" in line:
                import ast
                lst_start = line.index("[")
                lst_end = line.rindex("]") + 1
                bases[target] = ast.literal_eval(line[lst_start:lst_end])
    return bases


def in_production(model_name: str) -> bool:
    bases = get_production_bases()
    return any(model_name in v for v in bases.values())


def main() -> None:
    log("=" * 60)
    log("AUTO LOOP STARTED")
    log(f"Production score: {get_current_score():.6f}")
    log(f"Gate floor: {FLOOR}")
    log("=" * 60)

    shipped: set[str] = set()
    running: dict[str, subprocess.Popen | None] = {}
    rejected: set[str] = set()

    # Detect already-running jobs
    for exp in ALL_EXPERIMENTS:
        if all_oof_exist(exp["name"], exp["targets"]):
            log(f"OOF already done: {exp['name']}")
        elif in_production(exp["name"]):
            shipped.add(exp["name"])
            log(f"Already in production: {exp['name']}")

    # Add already-running GPU jobs as "running" with PID=None (monitor by file)
    for name in ["phase_xgb12_extra", "phase_xgb8_900_extra"]:
        if not all_oof_exist(name, ["action", "point", "server"]):
            running[name] = None
            log(f"Detecting existing job: {name}")

    iteration = 0
    while True:
        iteration += 1
        log(f"\n── Iteration {iteration} ──")
        prod_score = get_current_score()
        log(f"Current production score: {prod_score:.6f}")

        # 1. Check completed jobs → gate → maybe ship
        for exp in ALL_EXPERIMENTS:
            name = exp["name"]
            if name in shipped or name in rejected:
                continue
            if not all_oof_exist(name, exp["targets"]):
                continue
            if name in running:
                del running[name]
                log(f"Job COMPLETED: {name}")

            log(f"Gating {name}...")
            gate_score = run_gate(name, exp["targets"])
            lift = gate_score - prod_score
            log(f"  gate overall={gate_score:.6f}  lift={lift:+.6f}  floor={FLOOR}")

            if lift >= FLOOR:
                log(f"  ✅ PASS (lift={lift:+.5f}) → SHIPPING {name}")
                ship_model(name, exp["targets"])
                shipped.add(name)
                prod_score = gate_score
            else:
                log(f"  ❌ REJECT (lift={lift:+.5f} < {FLOOR})")
                rejected.add(name)

        # 2. Start new jobs if GPU/CPU slots are free
        gpu_running = sum(1 for n, p in running.items() if p is None or (p is not None and p.poll() is None)
                          for e in ALL_EXPERIMENTS if e["name"] == n and e["gpu"])
        cpu_running = sum(1 for n, p in running.items() if p is None or (p is not None and p.poll() is None)
                          for e in ALL_EXPERIMENTS if e["name"] == n and not e["gpu"])

        for exp in ALL_EXPERIMENTS:
            name = exp["name"]
            if name in shipped or name in rejected or name in running:
                continue
            if all_oof_exist(name, exp["targets"]):
                continue
            # Check dependencies are shipped
            if any(d not in shipped for d in exp["depends_on"]):
                log(f"  Skipping {name}: waiting for deps {exp['depends_on']}")
                continue
            # Check if script exists
            script_path = CWD / "scripts" / (exp["module"].replace("scripts.", "") + ".py")
            if not script_path.exists():
                log(f"  Skipping {name}: script {script_path} not found")
                continue
            # Start job
            if exp["gpu"]:
                if gpu_running >= 2:
                    log(f"  GPU busy ({gpu_running} running), deferring {name}")
                    continue
                p = start_gpu_job(exp["module"])
                log(f"  🚀 Started GPU job: {name} (PID {p.pid})")
            else:
                if cpu_running >= 1:
                    log(f"  CPU busy ({cpu_running} running), deferring {name}")
                    continue
                p = start_cpu_job(exp["module"])
                log(f"  🚀 Started CPU job: {name} (PID {p.pid})")
            running[name] = p

        # 3. Check if loop is done
        pending = [
            e["name"] for e in ALL_EXPERIMENTS
            if e["name"] not in shipped and e["name"] not in rejected
            and not all_oof_exist(e["name"], e["targets"])
        ]
        if not pending and not running:
            log("\nAll experiments exhausted. Loop complete.")
            break

        log(f"  Pending: {pending}")
        log(f"  Running: {list(running.keys())}")
        time.sleep(60)

    log("AUTO LOOP DONE")


if __name__ == "__main__":
    os.chdir(CWD)
    main()
