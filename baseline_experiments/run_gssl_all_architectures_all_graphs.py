"""Run the complete GSSL architecture matrix on all graph variants.

The underlying runner handles resumable JSONL/CSV state, W&B logging,
per-run logs, and failure isolation. This file only defines the experiment
grid and its execution defaults.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = REPO_ROOT / "baseline_experiments" / "run_gssl_dual_graph_baselines.py"
DEFAULT_ROOT = (
    REPO_ROOT
    / "baseline_experiments"
    / "gssl_dual_graph_baselines"
    / "all_architectures_all_graphs"
)

ARCHITECTURES = [
    "RotatEGCN_attn",
    "RotatEGCN_conv",
    "TransGCN_attn",
    "TransGCN_conv",
    "GCN",
    "CuGraphGAT",
]

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=7,
        help="Hard upper bound on concurrent workers; memory scheduling chooses the actual count.",
    )
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.90)
    parser.add_argument("--gpu-reserve-gb", type=float, default=4.0)
    parser.add_argument("--memory-poll-seconds", type=float, default=2.0)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--num-neighbors",
        nargs=2,
        type=int,
        default=[-1, -1],
        metavar=("HOP1", "HOP2"),
        help="Neighbors per hop; -1 -1 keeps all neighbors.",
    )
    parser.add_argument(
        "--wandb-project",
        default="GSSL_All_Architectures_All_Graphs",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT / "runs")
    parser.add_argument("--db", type=Path, default=DEFAULT_ROOT / "runs.jsonl")
    parser.add_argument("--summary", type=Path, default=DEFAULT_ROOT / "summary.csv")
    parser.add_argument("--plan", type=Path, default=DEFAULT_ROOT / "plan.json")
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def build_command(args):
    command = [
        sys.executable,
        "-u",
        str(BASE_RUNNER),
        "--graphs",
        "biomed_clean",
        "biomed_noisy",
        "dbpedia_clean",
        "dbpedia_noisy",
        "--tasks",
        "recons_r",
        "recons_x_symmetric",
        "recons_x_mlp",
        "contrastive",
        "--encoders",
        *ARCHITECTURES,
        "--hidden-sizes",
        "256",
        "384",
        "512",
        "--seeds",
        "0",
        "42",
        "123",
        "789",
        "2024",
        "--rgcn-bases",
        "5",
        "10",
        "--num-epochs",
        str(args.num_epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-neighbors",
        *(str(value) for value in args.num_neighbors),
        "--dropout",
        "0.2",
        "--wandb-project",
        args.wandb_project,
        "--wandb-mode",
        args.wandb_mode,
        "--max-parallel",
        str(args.max_parallel),
        "--memory-aware",
        "--gpu-memory-fraction",
        str(args.gpu_memory_fraction),
        "--gpu-reserve-gb",
        str(args.gpu_reserve_gb),
        "--memory-poll-seconds",
        str(args.memory_poll_seconds),
        "--out-root",
        str(args.out_root),
        "--db",
        str(args.db),
        "--summary",
        str(args.summary),
        "--plan",
        str(args.plan),
    ]
    if args.stream_output:
        command.append("--stream-output")
    if args.status_only:
        command.append("--status-only")
    if args.rerun_completed:
        command.append("--rerun-completed")
    return command


def main() -> int:
    args = parse_args()
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")

    command = build_command(args)
    print("Launching memory-aware architecture scheduler:", flush=True)
    print(" ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
