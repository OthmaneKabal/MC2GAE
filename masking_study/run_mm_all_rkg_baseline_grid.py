import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SINGLE_SCRIPT = REPO_ROOT / "masking_study" / "run_biomedical_masking_study.py"

GRAPH = "MM_mapped_nci_All_R_KG"
DEFAULT_PROJECT = "MM_All_R_KG_Baselines_CommonNodes"
DEFAULT_OUT_ROOT = REPO_ROOT / "masking_study" / GRAPH / "baseline_grid"
DEFAULT_DB = DEFAULT_OUT_ROOT / "runs.jsonl"
DEFAULT_SUMMARY = DEFAULT_OUT_ROOT / "summary.csv"
DEFAULT_PLAN = DEFAULT_OUT_ROOT / "plan.json"

BASELINE_MODES = [
    "whole_graph",
    "random_static",
    "random_dynamic",
    "recons_x_whole_graph",
    "recons_x_static",
    "graphmae_dynamic_x",
    "struct_node_pagerank_masking",
    "struct_node_degree_masking",
    "struct_node_learnable_masking",
    "edge_curriculum_dynamic",
    "balanced_static",
    "balanced_dynamic",
]
NO_MASK_RATE_MODES = {"whole_graph", "recons_x_whole_graph"}
BALANCED_MODES = {"balanced_static", "balanced_dynamic"}


def parse_channels(value):
    parts = [int(part) for part in str(value).replace(",", "-").split("-") if part]
    if len(parts) == 1:
        return [parts[0], parts[0]]
    if len(parts) == 2:
        return parts
    raise ValueError(f"Invalid channel value: {value}")


def mask_rate_tag(mask_rate):
    if mask_rate is None:
        return "nomask"
    return f"m{str(mask_rate).replace('.', 'p')}"


def job_key(record):
    return (
        record.get("graph"),
        record.get("mode"),
        tuple(record.get("channels", [])),
        int(record.get("seed")),
        record.get("mask_rate"),
        record.get("encoder"),
        record.get("decoder"),
        record.get("dropout"),
        record.get("linear_probe"),
    )


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def latest_finish_records(db_path):
    latest = {}
    if not db_path.exists():
        return latest
    with open(db_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "finish":
                latest[job_key(record)] = record
    return latest


def extract_metrics(out_dir):
    result_files = sorted(out_dir.rglob("results_seed_*.xlsx"), key=lambda path: path.stat().st_mtime)
    if not result_files:
        return {"metrics_found": False}
    result_file = result_files[-1]
    df = pd.read_excel(result_file)
    if df.empty:
        return {"metrics_found": False, "results_file": str(result_file)}
    row = df.iloc[-1].to_dict()
    metrics = {"metrics_found": True, "results_file": str(result_file)}
    for key, value in row.items():
        if pd.isna(value):
            continue
        if hasattr(value, "item"):
            value = value.item()
        metrics[str(key)] = value
    return metrics


def update_summary_csv(summary_path, record):
    metrics = record.get("metrics", {})
    row = {
        "time": record.get("time"),
        "status": record.get("status"),
        "graph": record.get("graph"),
        "mode": record.get("mode"),
        "channels": "-".join(str(value) for value in record.get("channels", [])),
        "seed": record.get("seed"),
        "mask_rate": record.get("mask_rate"),
        "encoder": record.get("encoder"),
        "decoder": record.get("decoder"),
        "dropout": record.get("dropout"),
        "linear_probe": record.get("linear_probe"),
        "out_dir": record.get("out_dir"),
        "metrics_found": metrics.get("metrics_found", False),
    }
    for key, value in metrics.items():
        key = str(key)
        if key in ("accuracy", "f1_score", "precision", "recall", "R_accuracy", "R_f1", "best_epoch"):
            row[f"metric_{key}"] = value
        elif key.startswith("best_loss_unsup_") or key.startswith("linear_probe_"):
            row[f"metric_{key}"] = value

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(summary_path, index=False)


def balanced_last_modes(modes):
    requested = list(modes)
    return [mode for mode in requested if mode not in BALANCED_MODES] + [mode for mode in requested if mode in BALANCED_MODES]


def build_jobs(args):
    jobs = []
    channels_grid = [parse_channels(value) for value in args.channels_grid]
    for mode in balanced_last_modes(args.modes):
        rates = [None] if mode in NO_MASK_RATE_MODES else args.mask_rates
        for channels in channels_grid:
            for seed in args.seeds:
                for mask_rate in rates:
                    jobs.append({
                        "graph": GRAPH,
                        "mode": mode,
                        "channels": list(channels),
                        "seed": int(seed),
                        "mask_rate": mask_rate,
                        "encoder": args.encoder,
                        "decoder": args.decoder,
                        "dropout": float(args.dropout),
                        "linear_probe": bool(args.linear_probe),
                    })
    return jobs


def build_run_dir(out_root, job, index):
    channels_tag = "-".join(str(value) for value in job["channels"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return out_root / (
        f"mm__{job['mode']}__c{channels_tag}__s{job['seed']}__"
        f"{mask_rate_tag(job['mask_rate'])}__j{index:04d}__{stamp}"
    )


def build_command(args, job, out_dir):
    mask_rate = 0.0 if job["mask_rate"] is None else job["mask_rate"]
    command = [
        sys.executable,
        str(RUN_SINGLE_SCRIPT),
        "--single",
        "--graph", GRAPH,
        "--mode", job["mode"],
        "--channels", *[str(value) for value in job["channels"]],
        "--seed", str(job["seed"]),
        "--out", str(out_dir),
        "--db", str(out_dir / "run_events.jsonl"),
        "--summary", str(out_dir / "summary.csv"),
        "--encoder", args.encoder,
        "--decoder", args.decoder,
        "--dropout", str(args.dropout),
        "--mask-rate", str(mask_rate),
        "--wandb-project", args.wandb_project,
        "--num-epochs", str(args.num_epochs),
        "--short-run-dirs",
    ]
    if args.linear_probe:
        command.extend([
            "--linear-probe",
            "--linear-probe-gs-path", args.linear_probe_gs_path,
            "--linear-probe-splits-dir", args.linear_probe_splits_dir,
            "--linear-probe-split-seeds", *[str(value) for value in args.linear_probe_split_seeds],
            "--linear-probe-epochs", str(args.linear_probe_epochs),
            "--linear-probe-lr", str(args.linear_probe_lr),
            "--linear-probe-weight-decay", str(args.linear_probe_weight_decay),
            "--linear-probe-patience", str(args.linear_probe_patience),
        ])
    if args.wandb_mode:
        command.extend(["--wandb-mode", args.wandb_mode])
    return command


def print_status(jobs, latest):
    pending = []
    print("\n=== MM All_R_KG baseline grid status ===\n")
    for job in jobs:
        finish = latest.get(job_key(job))
        label = (
            f"{job['mode']:<32} ch{'-'.join(str(v) for v in job['channels']):<8} "
            f"seed {job['seed']:<4} mask {job['mask_rate']}"
        )
        if finish is not None and finish.get("status") in ("completed", "completed_no_metrics"):
            metrics = finish.get("metrics") or {}
            acc = metrics.get("best_loss_unsup_accuracy", metrics.get("accuracy"))
            f1 = metrics.get("best_loss_unsup_f1_score", metrics.get("f1_score"))
            if acc is None or f1 is None:
                print(f"COMPLETED {label}")
            else:
                print(f"COMPLETED {label} accuracy={acc:.6f} f1={f1:.6f}")
        else:
            print(f"PENDING   {label}")
            pending.append(job)
    print(f"\nCompleted: {len(jobs) - len(pending)}")
    print(f"To run:    {len(pending)}\n")
    return pending


def run_suite(args):
    out_root = Path(args.out_root).resolve()
    db_path = Path(args.db).resolve()
    summary_path = Path(args.summary).resolve()
    jobs = build_jobs(args)
    latest = latest_finish_records(db_path)
    pending = print_status(jobs, latest)

    write_json(Path(args.plan).resolve(), {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "graph": GRAPH,
        "gold_standard": "data/UMLS/common_nodes.xlsx",
        "wandb_project": args.wandb_project,
        "max_parallel": args.max_parallel,
        "num_epochs": args.num_epochs,
        "modes": args.modes,
        "no_mask_rate_modes": sorted(NO_MASK_RATE_MODES),
        "mask_rates": args.mask_rates,
        "channels_grid": [parse_channels(value) for value in args.channels_grid],
        "seeds": args.seeds,
        "encoder": args.encoder,
        "decoder": args.decoder,
        "dropout": args.dropout,
        "linear_probe": args.linear_probe,
        "linear_probe_split_seeds": args.linear_probe_split_seeds,
        "total_jobs": len(jobs),
        "pending_jobs": len(pending),
    })

    if args.status_only:
        print(f"Plan:    {Path(args.plan).resolve()}")
        print(f"DB:      {db_path}")
        print(f"Summary: {summary_path}")
        print(f"Runs:    {out_root}\n")
        return

    append_jsonl(db_path, {
        "event": "suite_start",
        "status": "running",
        "time": datetime.now().isoformat(timespec="seconds"),
        "total_jobs": len(jobs),
        "pending_jobs": len(pending),
        "max_parallel": args.max_parallel,
    })

    running = []
    next_index = 0
    try:
        while next_index < len(pending) or running:
            while next_index < len(pending) and len(running) < args.max_parallel:
                job = pending[next_index]
                run_dir = build_run_dir(out_root, job, next_index)
                command = build_command(args, job, run_dir)
                start_record = {
                    "event": "start",
                    "status": "running",
                    "time": datetime.now().isoformat(timespec="seconds"),
                    **job,
                    "out_dir": str(run_dir),
                    "command": command,
                }
                append_jsonl(db_path, start_record)
                print(
                    f"\n=== Launch {next_index + 1}/{len(pending)} | {job['mode']} | "
                    f"ch{'-'.join(str(v) for v in job['channels'])} | seed {job['seed']} | "
                    f"mask {job['mask_rate']} ===\n"
                )
                process = subprocess.Popen(command, cwd=str(REPO_ROOT))
                running.append({"process": process, "job": job, "run_dir": run_dir, "start": start_record})
                next_index += 1

            time.sleep(args.poll_seconds)
            still_running = []
            for item in running:
                returncode = item["process"].poll()
                if returncode is None:
                    still_running.append(item)
                    continue

                metrics = extract_metrics(item["run_dir"])
                status = "completed" if returncode == 0 and metrics.get("metrics_found") else "failed"
                if returncode == 0 and not metrics.get("metrics_found"):
                    status = "completed_no_metrics"
                finish_record = {
                    "event": "finish",
                    "status": status,
                    "time": datetime.now().isoformat(timespec="seconds"),
                    **item["job"],
                    "out_dir": str(item["run_dir"]),
                    "returncode": returncode,
                    "metrics": metrics,
                }
                append_jsonl(db_path, finish_record)
                update_summary_csv(summary_path, finish_record)
                print(
                    f"\n=== Finish | {item['job']['mode']} | "
                    f"seed {item['job']['seed']} | mask {item['job']['mask_rate']} | {status} ===\n"
                )
            running = still_running
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating child runs...")
        for item in running:
            item["process"].terminate()
        raise

    append_jsonl(db_path, {
        "event": "suite_finish",
        "status": "completed",
        "time": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"\nSuite finished. DB: {db_path}")
    print(f"Summary CSV: {summary_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="*", default=BASELINE_MODES)
    parser.add_argument("--channels-grid", nargs="*", default=["256-256", "384-384", "512-512"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 42, 123, 789, 2024])
    parser.add_argument("--mask-rates", nargs="*", type=float, default=[0.2, 0.4, 0.5, 0.7, 0.8])
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--encoder", default="RotatEGCN_attn")
    parser.add_argument("--decoder", default="MLP")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--linear-probe", action="store_true", default=True)
    parser.add_argument("--no-linear-probe", dest="linear_probe", action="store_false")
    parser.add_argument("--linear-probe-gs-path", default="../../data/UMLS/common_nodes.xlsx")
    parser.add_argument("--linear-probe-splits-dir", default="../../data/UMLS/splits/umls_kg_splits")
    parser.add_argument("--linear-probe-split-seeds", nargs="*", type=int, default=[42, 123, 456, 789, 2024])
    parser.add_argument("--linear-probe-epochs", type=int, default=300)
    parser.add_argument("--linear-probe-lr", type=float, default=0.01)
    parser.add_argument("--linear-probe-weight-decay", type=float, default=0.0)
    parser.add_argument("--linear-probe-patience", type=int, default=50)
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT / "runs"))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()

    invalid_modes = sorted(set(args.modes) - set(BASELINE_MODES))
    if invalid_modes:
        raise ValueError(f"Invalid MM baseline modes: {invalid_modes}. Valid modes: {BASELINE_MODES}")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")
    run_suite(args)


if __name__ == "__main__":
    main()
