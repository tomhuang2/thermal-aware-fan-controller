#!/usr/bin/env python3
"""
WSL-safe PARSEC Active Learning Orchestrator

Changes vs original:
- No sudo / PQoS / cpufreq / LLC control
- No external shell-template dependency
- Runs PARSEC directly with parsecmgmt
- Samples software-visible knobs only:
    * thread count
    * PARSEC input size
    * benchmark app
- Measures wall-clock runtime from Python
- Optionally parses ROI timing from PARSEC stdout if present
"""

import itertools
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler


CONFIG = {
    "PARSEC_ROOT": Path("./parsec"),   # submodule path inside your larger repo
    "RESULTS_BASE_DIR": Path("./sampling_results_wsl"),
    "RESULTS_JSON_FILE": "final_wsl_sampling_results.json",

    # Pick benchmarks you already built
    "BENCHMARK_APPS": [
        "blackscholes",
        "bodytrack",
        "canneal",
        "dedup",
        "freqmine",
        "streamcluster",
        "swaptions",
    ],

    # Only software knobs that work in WSL
    "PARAMETER_SPACE": {
        "threads": [1, 2, 4, 8],
        "input_size": ["test", "simsmall", "simmedium"],
    },

    "PARSEC_BUILD_CONFIG": "gcc",
    "BUILD_IF_MISSING": False,   # set True if you want to auto-build
    "RUN_TIMEOUT_SEC": 3600,

    "LHS_SAMPLES": 12,
    "ACTIVE_LEARNING_STEPS": 20,
    "SIGMA_WEIGHT_ALPHA": 0.5,

    # Maximize throughput-like metric, minimize runtime
    "TARGET_METRICS": ["score", "runtime_sec"],
    "SCORE_PENALTY_VALUE": 0.0,
    "RUNTIME_PENALTY_VALUE": 1e9,
}

print("SCRIPT STARTED")

def get_full_search_space(param_space: dict) -> pd.DataFrame:
    keys = list(param_space.keys())
    values = list(param_space.values())
    combos = list(itertools.product(*values))
    df = pd.DataFrame(combos, columns=keys)
    df["status"] = "untested"
    return df


def load_results() -> dict:
    path = CONFIG["RESULTS_BASE_DIR"] / CONFIG["RESULTS_JSON_FILE"]
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def save_results(results_data: dict):
    CONFIG["RESULTS_BASE_DIR"].mkdir(parents=True, exist_ok=True)
    path = CONFIG["RESULTS_BASE_DIR"] / CONFIG["RESULTS_JSON_FILE"]
    with open(path, "w") as f:
        json.dump(results_data, f, indent=2)


def maybe_build_benchmark(app_name: str):
    if not CONFIG["BUILD_IF_MISSING"]:
        return

    parsec_root = CONFIG["PARSEC_ROOT"].resolve()
    env = {"PATH": f"{parsec_root / 'bin'}:{Path().resolve()}"}
    cmd = [
        str(parsec_root / "bin" / "parsecmgmt"),
        "-a", "build",
        "-p", app_name,
        "-c", CONFIG["PARSEC_BUILD_CONFIG"],
    ]
    subprocess_run(cmd, cwd=parsec_root, extra_env=env, timeout=CONFIG["RUN_TIMEOUT_SEC"])


def subprocess_run(cmd, cwd=None, extra_env=None, timeout=None):
    import os
    import subprocess

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def parse_parsec_stdout(stdout: str) -> dict:
    """
    Try to extract a useful timing signal from PARSEC output.
    Falls back to Python wall-clock time if nothing is found.
    """
    metrics = {"roi_time_sec": None}

    roi_patterns = [
        r"ROI time.*?([0-9]+(?:\.[0-9]+)?)",
        r"real\s+([0-9]+(?:\.[0-9]+)?)",
        r"Total time.*?([0-9]+(?:\.[0-9]+)?)",
    ]

    for pat in roi_patterns:
        m = re.search(pat, stdout, flags=re.IGNORECASE)
        if m:
            metrics["roi_time_sec"] = float(m.group(1))
            break

    return metrics


def run_single_experiment(config: dict, app_name: str, run_type: str) -> dict:
    run_id = f"{app_name}_{run_type}_{int(time.time())}"
    output_dir = CONFIG["RESULTS_BASE_DIR"] / app_name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    parsec_root = CONFIG["PARSEC_ROOT"].resolve()
    parsecmgmt = parsec_root / "bin" / "parsecmgmt"

    print(f"\n--- Running {app_name} [{run_type}] ---")
    print(f"  threads={config['threads']} input={config['input_size']}")
    print(f"  output_dir={output_dir}")

    if not parsecmgmt.exists():
        print(f"[ERROR] parsecmgmt not found at {parsecmgmt}")
        return {
            "score": CONFIG["SCORE_PENALTY_VALUE"],
            "runtime_sec": CONFIG["RUNTIME_PENALTY_VALUE"],
            "roi_time_sec": None,
            "exit_code": -1,
        }

    maybe_build_benchmark(app_name)

    cmd = [
        str(parsecmgmt),
        "-a", "run",
        "-p", app_name,
        "-c", CONFIG["PARSEC_BUILD_CONFIG"],
        "-i", config["input_size"],
        "-n", str(config["threads"]),
    ]

    t0 = time.perf_counter()
    result = subprocess_run(
        cmd,
        cwd=parsec_root,
        timeout=CONFIG["RUN_TIMEOUT_SEC"],
    )
    runtime_sec = time.perf_counter() - t0

    (output_dir / "stdout.txt").write_text(result.stdout or "")
    (output_dir / "stderr.txt").write_text(result.stderr or "")

    parsed = parse_parsec_stdout(result.stdout or "")
    roi_time = parsed["roi_time_sec"]

    if result.returncode != 0:
        print(f"  [ERROR] run failed with code {result.returncode}")
        return {
            "score": CONFIG["SCORE_PENALTY_VALUE"],
            "runtime_sec": CONFIG["RUNTIME_PENALTY_VALUE"],
            "roi_time_sec": roi_time,
            "exit_code": result.returncode,
        }

    # Higher score is better. Use inverse runtime.
    # Prefer ROI time if available, else wall time.
    effective_time = roi_time if roi_time and roi_time > 0 else runtime_sec
    score = 1.0 / max(effective_time, 1e-9)

    print(f"  [RESULT] runtime_sec={runtime_sec:.4f} score={score:.6f} roi_time={roi_time}")

    return {
        "score": score,
        "runtime_sec": runtime_sec,
        "roi_time_sec": roi_time,
        "exit_code": 0,
    }


def get_next_sample_gpr(collected_df: pd.DataFrame, X_full_untested: pd.DataFrame) -> dict:
    param_keys = list(CONFIG["PARAMETER_SPACE"].keys())

    X_train = collected_df[param_keys].copy()

    # Encode categorical input_size
    X_train = pd.get_dummies(X_train, columns=["input_size"], dtype=float)
    X_untested = X_full_untested[param_keys].copy()
    X_untested = pd.get_dummies(X_untested, columns=["input_size"], dtype=float)

    X_train, X_untested = X_train.align(X_untested, join="left", axis=1, fill_value=0.0)

    y_score = collected_df["score"].values
    y_runtime = collected_df["runtime_sec"].values

    if len(X_train) < 2:
        return X_full_untested.sample(1).iloc[0].to_dict()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.values)
    X_untested_scaled = scaler.transform(X_untested.values)

    y_score_mean, y_score_std = y_score.mean(), max(y_score.std(), 1e-6)
    y_runtime_mean, y_runtime_std = y_runtime.mean(), max(y_runtime.std(), 1e-6)

    y_score_scaled = (y_score - y_score_mean) / y_score_std
    y_runtime_scaled = (y_runtime - y_runtime_mean) / y_runtime_std

    kernel = (
        ConstantKernel(1.0)
        * Matern(length_scale=[1.0] * X_scaled.shape[1], nu=2.5)
        + WhiteKernel(noise_level=0.1)
    )

    gpr_score = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=0)
    gpr_runtime = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=1)

    try:
        gpr_score.fit(X_scaled, y_score_scaled)
        gpr_runtime.fit(X_scaled, y_runtime_scaled)
    except Exception as e:
        print(f"[WARNING] GPR training failed: {e}. Falling back to random.")
        return X_full_untested.sample(1).iloc[0].to_dict()

    _, std_score_scaled = gpr_score.predict(X_untested_scaled, return_std=True)
    _, std_runtime_scaled = gpr_runtime.predict(X_untested_scaled, return_std=True)

    std_score = std_score_scaled * y_score_std
    std_runtime = std_runtime_scaled * y_runtime_std

    alpha = CONFIG["SIGMA_WEIGHT_ALPHA"]
    combined_uncertainty = alpha * std_score + (1 - alpha) * std_runtime

    idx = int(np.argmax(combined_uncertainty))
    return X_full_untested.iloc[idx].to_dict()


def run_independent_sampling():
    all_results = load_results()
    param_keys = list(CONFIG["PARAMETER_SPACE"].keys())

    for app in CONFIG["BENCHMARK_APPS"]:
        print(f"\n{'=' * 60}\nSTARTING: {app}\n{'=' * 60}")

        full_df = get_full_search_space(CONFIG["PARAMETER_SPACE"])
        app_data = all_results.get(app, [])
        collected_df = pd.DataFrame(app_data)

        if not collected_df.empty:
            for _, row in collected_df.iterrows():
                cond = np.ones(len(full_df), dtype=bool)
                for k in param_keys:
                    cond &= full_df[k] == row[k]
                full_df.loc[cond, "status"] = row.get("run_type", "tested")

        lhs_tested_count = 0 if collected_df.empty else (collected_df["run_type"] == "lhs").sum()
        lhs_needed = CONFIG["LHS_SAMPLES"] - int(lhs_tested_count)

        if lhs_needed > 0:
            untested = full_df[full_df["status"] == "untested"]
            lhs_samples = untested.sample(n=min(lhs_needed, len(untested)), random_state=42)

            for _, config in lhs_samples.iterrows():
                cfg = config[param_keys].to_dict()
                metrics = run_single_experiment(cfg, app, "lhs")

                row = cfg.copy()
                row.update(metrics)
                row["run_type"] = "lhs"
                app_data.append(row)

                collected_df = pd.DataFrame(app_data)
                all_results[app] = app_data
                save_results(all_results)

                cond = np.ones(len(full_df), dtype=bool)
                for k in param_keys:
                    cond &= full_df[k] == cfg[k]
                full_df.loc[cond, "status"] = "lhs"

        for step in range(1, CONFIG["ACTIVE_LEARNING_STEPS"] + 1):
            untested = full_df[full_df["status"] == "untested"]
            if untested.empty or len(collected_df) < 2:
                break

            print(f"\n--- Active Learning Step {step}/{CONFIG['ACTIVE_LEARNING_STEPS']} ---")
            next_cfg = get_next_sample_gpr(collected_df, untested)
            metrics = run_single_experiment(next_cfg, app, f"al_step_{step}")

            row = next_cfg.copy()
            row.update(metrics)
            row["run_type"] = f"al_step_{step}"
            app_data.append(row)

            collected_df = pd.DataFrame(app_data)
            all_results[app] = app_data
            save_results(all_results)

            cond = np.ones(len(full_df), dtype=bool)
            for k in param_keys:
                cond &= full_df[k] == next_cfg[k]
            full_df.loc[cond, "status"] = f"al_step_{step}"

        print(f"[SUMMARY] {app}: total samples = {len(collected_df)}")


def main():
    print("Starting WSL-safe PARSEC Active Learning")
    print("=" * 60)
    run_independent_sampling()


if __name__ == "__main__":
    main()