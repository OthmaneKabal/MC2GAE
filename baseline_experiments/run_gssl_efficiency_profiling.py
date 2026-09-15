"""Profile GSSL architectures before choosing production parallelism.

Run ``--suite standard`` in the regular environment and ``--suite cugraph``
in the cuGraph environment, on separate GPU nodes. Each configuration runs in
isolation so its timing and memory measurements remain attributable to it.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = REPO_ROOT / "baseline_experiments" / "run_gssl_dual_graph_baselines.py"
DEFAULT_ROOT = REPO_ROOT / "baseline_experiments" / "gssl_efficiency_profiling"

GRAPHS = ["biomed_clean", "biomed_noisy", "dbpedia_clean", "dbpedia_noisy"]
TASKS = ["recons_r", "recons_x_symmetric", "recons_x_mlp", "contrastive"]
STANDARD_ENCODERS = [
    "RotatEGCN_attn",
    "RotatEGCN_conv",
    "TransGCN_attn",
    "TransGCN_conv",
    "GCN",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("standard", "cugraph"), required=True)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--linear-probe-batch-size", type=int, default=1024)
    parser.add_argument("--profile-sample-seconds", type=float, default=1.0)
    parser.add_argument(
        "--scheduler-memory-budget-gb",
        type=float,
        default=72.0,
        help="Usable GPU memory when deriving safe worker counts on an 80 GB GPU.",
    )
    parser.add_argument(
        "--scheduler-memory-margin",
        type=float,
        default=1.15,
        help="Safety multiplier applied to each measured peak.",
    )
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def phase_specs(suite: str):
    if suite == "standard":
        return [{
            "name": "standard",
            "graphs": GRAPHS,
            "tasks": TASKS,
            "encoders": STANDARD_ENCODERS,
            "rgcn_bases": None,
        }]
    return [
        {
            "name": "cugraph_gat",
            "graphs": GRAPHS,
            "tasks": TASKS,
            "encoders": ["CuGraphGAT"],
            "rgcn_bases": None,
        },
        {
            "name": "cugraph_rgcn",
            "graphs": ["biomed_clean", "biomed_noisy", "dbpedia_clean"],
            "tasks": ["recons_r", "recons_x_mlp", "contrastive"],
            "encoders": ["CuGraphRGCN"],
            "rgcn_bases": [10],
        },
    ]


def build_command(args, phase: dict, phase_root: Path):
    command = [
        sys.executable,
        "-u",
        str(BASE_RUNNER),
        "--graphs",
        *phase["graphs"],
        "--tasks",
        *phase["tasks"],
        "--encoders",
        *phase["encoders"],
        "--hidden-sizes",
        "512",
        "--seeds",
        "0",
        "--num-epochs",
        str(args.num_epochs),
        "--batch-size",
        str(args.batch_size),
        "--linear-probe-batch-size",
        str(args.linear_probe_batch_size),
        "--graph-aware-num-neighbors",
        "--dropout",
        "0.2",
        "--max-parallel",
        "1",
        "--profile-efficiency",
        "--profile-sample-seconds",
        str(args.profile_sample_seconds),
        "--wandb-mode",
        "disabled",
        "--out-root",
        str(phase_root / "runs"),
        "--db",
        str(phase_root / "runs.jsonl"),
        "--summary",
        str(phase_root / "summary.csv"),
        "--plan",
        str(phase_root / "plan.json"),
    ]
    if phase["rgcn_bases"]:
        command.extend(["--rgcn-bases", *(str(value) for value in phase["rgcn_bases"])])
    if args.stream_output:
        command.append("--stream-output")
    if args.status_only:
        command.append("--status-only")
    if args.rerun_completed:
        command.append("--rerun-completed")
    return command


def _flatten_profile(profile: dict, profile_path: Path):
    row = {
        key: value
        for key, value in profile.items()
        if not isinstance(value, (dict, list))
    }
    row["profile_file"] = str(profile_path)
    for key, value in profile.items():
        if isinstance(value, (dict, list)):
            row[key] = json.dumps(value, ensure_ascii=True)
    row.update({
        f"metric_{key}": value
        for key, value in profile.get("result_metrics", {}).items()
        if not isinstance(value, (dict, list))
    })
    return row


def _profile_peak_gb(profile: dict):
    values = [
        profile.get("peak_nvml_process_memory_gb"),
        profile.get("peak_torch_reserved_gb"),
        profile.get("peak_train_nvml_process_memory_gb"),
        profile.get("peak_train_torch_reserved_gb"),
    ]
    values = [float(value) for value in values if value is not None]
    return max(values) if values else None


def aggregate_profiles(root: Path, memory_budget_gb: float, memory_margin: float):
    profiles = []
    for profile_path in sorted(root.rglob("efficiency_profile.json")):
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping invalid profile {profile_path}: {exc}", flush=True)
            continue
        profile["profile_file"] = str(profile_path)
        profiles.append(profile)

    jsonl_path = root / "efficiency_profiles.jsonl"
    csv_path = root / "efficiency_profiles.csv"
    scheduler_path = root / "scheduler_memory_profile.json"
    root.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for profile in profiles:
            handle.write(json.dumps(profile, ensure_ascii=True) + "\n")
    if profiles:
        pd.DataFrame(
            [_flatten_profile(profile, Path(profile["profile_file"])) for profile in profiles]
        ).to_csv(csv_path, index=False)

    scheduler = {
        "memory_budget_gb": memory_budget_gb,
        "safety_margin": memory_margin,
        "profiles": {},
    }
    for profile in profiles:
        if profile.get("status") != "completed":
            continue
        peak_gb = _profile_peak_gb(profile)
        if peak_gb is None:
            continue
        safe_peak_gb = peak_gb * memory_margin
        recommended_workers = max(1, math.floor(memory_budget_gb / safe_peak_gb))
        scheduler["profiles"][profile["run_id"]] = {
            "graph": profile.get("graph"),
            "task": profile.get("task"),
            "encoder": profile.get("encoder"),
            "decoder": profile.get("decoder"),
            "num_bases": profile.get("num_bases"),
            "hidden_size": profile.get("hidden_size"),
            "num_neighbors": profile.get("num_neighbors"),
            "measured_peak_gb": peak_gb,
            "safe_peak_gb": safe_peak_gb,
            "recommended_workers_if_homogeneous": recommended_workers,
            "steady_train_mean_seconds": profile.get("steady_train_mean_seconds"),
            "steady_epoch_mean_seconds": profile.get("steady_epoch_mean_seconds"),
        }
    scheduler_path.write_text(
        json.dumps(scheduler, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return profiles, csv_path, scheduler_path


def main():
    args = parse_args()
    if args.num_epochs < 1:
        raise ValueError("--num-epochs must be >= 1")
    if args.batch_size < 1 or args.linear_probe_batch_size < 1:
        raise ValueError("Batch sizes must be >= 1")
    if args.scheduler_memory_budget_gb <= 0:
        raise ValueError("--scheduler-memory-budget-gb must be > 0")
    if args.scheduler_memory_margin < 1:
        raise ValueError("--scheduler-memory-margin must be >= 1")

    root = (args.out_root or (DEFAULT_ROOT / args.suite)).resolve()
    phases = phase_specs(args.suite)
    expected = 80 if args.suite == "standard" else 25
    print(
        f"Profiling suite={args.suite}: expected configurations={expected}, "
        f"epochs={args.num_epochs}, max_parallel=1",
        flush=True,
    )

    return_code = 0
    for phase in phases:
        phase_root = root / phase["name"]
        command = build_command(args, phase, phase_root)
        print(f"\n=== Profiling phase: {phase['name']} ===", flush=True)
        print(" ".join(command), flush=True)
        result = subprocess.run(command, cwd=REPO_ROOT)
        return_code = max(return_code, result.returncode)
        aggregate_profiles(
            root,
            args.scheduler_memory_budget_gb,
            args.scheduler_memory_margin,
        )

    profiles, csv_path, scheduler_path = aggregate_profiles(
        root,
        args.scheduler_memory_budget_gb,
        args.scheduler_memory_margin,
    )
    completed = sum(profile.get("status") == "completed" for profile in profiles)
    failed = sum(profile.get("status") != "completed" for profile in profiles)
    print(
        f"\nProfiles collected: {len(profiles)} (completed={completed}, failed={failed})\n"
        f"CSV: {csv_path}\nScheduler profile: {scheduler_path}",
        flush=True,
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
