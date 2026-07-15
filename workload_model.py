#!/usr/bin/env python3
"""
workload_model.py
─────────────────
Realistic "user-on-a-PC" workload generator for thermal stress-testing a
phase-change-material (PCM) cooler + fan controller.

Press Run (or `python workload_model.py`) and it starts emitting a
*non-uniform, sub-maximal* stream of real work by driving PARSEC benchmarks
the way an actual desktop session loads a CPU: mostly light/medium activity,
occasional heavy bursts, idle stretches in between. It runs until you stop
it (Ctrl+C).

The point: push the PCM through charge/discharge cycles so you can watch
whether your fan algorithm keeps it inside the thermal-headroom band where
it can still absorb a spike — rather than pegging the CPU at 100% the whole
time (which only ever tests the saturated case).

Key idea: a single PARSEC run is too short (<2 s) to move thermal state, so
each activity is a "session" — benchmarks fired back-to-back for a target
DWELL (tens of seconds to minutes), then a cool-down GAP. Thread count and
intra-session micro-gaps set the *intensity*; dwell + gaps set the *thermal
timescale*.

It writes nothing to disk itself; align with your separate temperature
logger by wall-clock time.
"""

import argparse
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Paths (resolved from this file, so "Run" works regardless of CWD) ────────
SCRIPT_DIR  = Path(__file__).resolve().parent
PARSEC_ROOT = SCRIPT_DIR / "parsec"
PARSECMGMT  = PARSEC_ROOT / "bin" / "parsecmgmt"

# ── Machine ──────────────────────────────────────────────────────────────────
NCORES = os.cpu_count() or 4
HALF   = max(2, NCORES // 2)

# ── Benchmark pools ───────────────────────────────────────────────────────────
# Proven single-process PARSEC workloads that run cleanly headless in WSL
# (the set scheduler.py already uses). LIGHT = quick, happy at 1 thread.
LIGHT_BENCH   = ["blackscholes", "swaptions"]
GENERAL_BENCH = ["blackscholes", "swaptions", "canneal",
                 "dedup", "streamcluster", "freqmine", "bodytrack"]

# ── Usage model ────────────────────────────────────────────────────────────────
# Each activity "state" is one session of work. Tune all of this freely.
#   cores  : (lo, hi) threads — pinned for the whole session (≈ how wide it runs)
#   inputs : PARSEC input sizes drawn per run
#   dwell  : (lo, hi) seconds the session lasts (sustained heating window)
#   micro  : (lo, hi) seconds idle BETWEEN runs inside a session (keeps it sub-max)
#   gap    : (lo, hi) seconds cool-down AFTER the session (PCM discharge window)
STATES = {
    "idle":   dict(cores=(0, 0),         inputs=[],                         dwell=(10, 40),  micro=(0.0, 0.0), gap=(0, 0)),
    "light":  dict(cores=(1, 1),         inputs=["test", "simsmall"],       dwell=(5, 20),   micro=(0.5, 3.0), gap=(5, 15)),
    "medium": dict(cores=(2, HALF),      inputs=["simsmall", "simmedium"],  dwell=(20, 60),  micro=(0.2, 1.5), gap=(5, 15)),
    "heavy":  dict(cores=(HALF, NCORES), inputs=["simmedium"],              dwell=(30, 120), micro=(0.0, 0.4), gap=(3, 10)),
}

# Sticky Markov transitions → realistic "sessions" that linger in a mode and
# only occasionally spike to heavy. Each row matches ORDER and sums to 1.0.
ORDER = ["idle", "light", "medium", "heavy"]
TRANSITIONS = {
    #            idle  light  medium  heavy
    "idle":   [0.10, 0.50, 0.30, 0.10],   # don't linger idle — an active user gets back to it
    "light":  [0.15, 0.45, 0.30, 0.10],
    "medium": [0.10, 0.30, 0.45, 0.15],
    "heavy":  [0.08, 0.22, 0.40, 0.30],
}
START_STATE = "light"


def installed_benchmarks(config):
    """Apps that actually have a built binary for the given PARSEC config."""
    found = set()
    for inst in PARSEC_ROOT.glob(f"pkgs/*/*/inst/*{config}*"):
        app = inst.parent.parent.name          # pkgs/<group>/<app>/inst/<config>
        bindir = inst / "bin"
        if bindir.is_dir() and any(bindir.iterdir()):
            found.add(app)
    return found


def stamp():
    return datetime.now().strftime("%H:%M:%S")


def run_benchmark(app, threads, input_size, config):
    """Drive one PARSEC run to completion. Returns (duration_s, returncode)."""
    cmd = [str(PARSECMGMT), "-a", "run", "-p", app, "-c", config,
           "-i", input_size, "-n", str(threads)]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=str(PARSEC_ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return time.perf_counter() - t0, proc.returncode


def main():
    ap = argparse.ArgumentParser(
        description="Realistic desktop-usage workload generator (PARSEC) for PCM/fan testing.")
    ap.add_argument("--config", default="gcc", help="PARSEC build config (default: gcc)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Fix the RNG for a repeatable session "
                         "(compare fan-algorithm versions against the same workload trace)")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Let the Stop button (SIGTERM) end as cleanly as Ctrl+C (SIGINT).
    def _stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)

    if not PARSECMGMT.exists():
        sys.exit(f"[FATAL] parsecmgmt not found at {PARSECMGMT}\n"
                 f"        Build PARSEC first (see README), then re-run.")

    built   = installed_benchmarks(args.config)
    light   = [b for b in LIGHT_BENCH if b in built]
    general = [b for b in GENERAL_BENCH if b in built]
    if not general:
        sys.exit(f"[FATAL] None of the expected benchmarks are built for config '{args.config}'.\n"
                 f"        Built apps found: {sorted(built) or 'none'}\n"
                 f"        Build some with: parsecmgmt -a build -p <app> -c {args.config}")
    if not light:
        light = general  # keep the light state alive even if the short benches aren't built

    pools = {"light": light, "medium": general, "heavy": general}

    print("─" * 66)
    print(" Realistic workload model — PCM / fan thermal stress test")
    print("─" * 66)
    print(f"  cores detected : {NCORES}   (medium 2–{HALF}, heavy {HALF}–{NCORES})")
    print(f"  benchmarks     : {', '.join(general)}")
    print(f"  build config   : {args.config}")
    if args.seed is not None:
        print(f"  rng seed       : {args.seed}")
    print(f"  model          : sticky idle/light/medium/heavy — sustained sessions + cool-downs")
    print(f"  stop           : Ctrl+C  (prints a duty-cycle summary)")
    print("─" * 66, flush=True)

    state    = START_STATE
    runs     = 0
    active_s = 0.0
    idle_s   = 0.0
    load_core_s = 0.0   # Σ run_duration × threads → mean-utilization (heat) proxy
    t_start  = time.time()

    try:
        while True:
            spec = STATES[state]

            if state == "idle":
                dwell = random.uniform(*spec["dwell"])
                print(f"[{stamp()}]  · IDLE                                  away {dwell:5.1f}s",
                      flush=True)
                time.sleep(dwell)
                idle_s += dwell

            else:
                threads = min(max(random.randint(*spec["cores"]), 1), NCORES)
                bench   = random.choice(pools[state])
                target  = random.uniform(*spec["dwell"])
                print(f"[{stamp()}]  ┌ {state.upper():<6} {bench:<13} n={threads:<2} "
                      f"(~{target:.0f}s session)", flush=True)

                sess_t0, sess_runs, sess_active = time.time(), 0, 0.0
                while time.time() - sess_t0 < target:
                    inp = random.choice(spec["inputs"])
                    dur, rc = run_benchmark(bench, threads, inp, args.config)
                    sess_runs  += 1
                    sess_active += dur
                    active_s   += dur
                    load_core_s += dur * threads
                    runs       += 1
                    if rc != 0:
                        print(f"[{stamp()}]  │   {bench} exited rc={rc} — ending session early",
                              flush=True)
                        break
                    if dur > 2.0:   # surface long runs; stay quiet for sub-2s spikes
                        print(f"[{stamp()}]  │   {bench} {inp} {dur:4.1f}s", flush=True)
                    micro = random.uniform(*spec["micro"])
                    if micro > 0:
                        time.sleep(micro)
                        idle_s += micro

                sess_dur = time.time() - sess_t0
                gap = random.uniform(*spec["gap"])
                print(f"[{stamp()}]  └ {state.upper():<6} {sess_runs} runs · "
                      f"{sess_active:.0f}s busy / {sess_dur:.0f}s   → cool {gap:4.1f}s", flush=True)
                time.sleep(gap)
                idle_s += gap

            state = random.choices(ORDER, weights=TRANSITIONS[state])[0]

    except KeyboardInterrupt:
        elapsed = time.time() - t_start
        duty = 100.0 * active_s / elapsed if elapsed > 0 else 0.0
        print("\n" + "─" * 66)
        print(f"  stopped after {elapsed/60:.1f} min   runs={runs}   "
              f"busy={active_s/60:.1f} min   idle={idle_s/60:.1f} min")
        load = 100.0 * load_core_s / (elapsed * NCORES) if elapsed > 0 else 0.0
        print(f"  duty cycle   ≈ {duty:3.0f}%  (wall-time with a workload running)")
        print(f"  avg CPU load ≈ {load:3.0f}%  of {NCORES} cores — thread-weighted heat proxy "
              f"(sub-100% = PCM had room to discharge)")
        print("─" * 66)


if __name__ == "__main__":
    main()
