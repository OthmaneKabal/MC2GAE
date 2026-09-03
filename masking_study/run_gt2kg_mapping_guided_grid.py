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

GRAPH = "GT2KG_mapped_and_old_rel_norm"
DEFAULT_PROJECT = "GT2KG_Mapping_Guided_Masking_CommonNodes"
DEFAULT_OUT_ROOT = REPO_ROOT / "masking_study" / GRAPH / "mapping_guided_grid"
DEFAULT_DB = DEFAULT_OUT_ROOT / "runs.jsonl"
DEFAULT_SUMMARY = DEFAULT_OUT_ROOT / "summary.csv"
DEFAULT_PLAN = DEFAULT_OUT_ROOT / "plan.json"

NO_RATE_MODES = [
    "canonicalized_whole_graph",
    "canonicalized_mapped_only",
    "canonicalized_mapped_visible",
    "mapped_selector_old_predicate",
]
GLOBAL_MASK_RATE_MODES = [
    "canonicalized_random_dynamic",
    "mapped_biased_dynamic",
]
MAPPED_ONLY_RATE_MODES = [
    "mapped_only_dynamic_random",
    "mapped_selector_dynamic_random",
]
MIX_POOL_RATE_MODES = [
    "mapped_mix_dynamic_random",
]
ALL_MAPPED_PLUS_MODES = [
    "all_mapped_plus_random_dynamic",
]
MAPPED_CONTEXT_NON_MAPPED_MODES = [
    "mapped_context_non_mapped_dynamic_random",
]
BALANCED_MODES = [
    "canonicalized_balanced_dynamic",
    "mapped_only_dynamic_balanced",
    "mapped_selector_dynamic_balanced",
    "mapped_mix_dynamic_balanced",
    "all_mapped_plus_balanced_dynamic",
    "mapped_context_non_mapped_dynamic_balanced",
]
LEGACY_GLOBAL_BUDGET_MODES = {
    "mapped_random_dynamic_15_15": 0.3,
    "mapped_random_dynamic_20_10": 0.3,
    "mapped_random_dynamic_20_30": 0.5,
}
ALL_MODES = (
    NO_RATE_MODES
    + GLOBAL_MASK_RATE_MODES
    + MAPPED_ONLY_RATE_MODES
    + MIX_POOL_RATE_MODES
    + ALL_MAPPED_PLUS_MODES
    + MAPPED_CONTEXT_NON_MAPPED_MODES
    + list(LEGACY_GLOBAL_BUDGET_MODES)
    + BALANCED_MODES
)


def parse_channels(value):
    parts = [int(part) for part in str(value).replace(",", "-").split("-") if part]
    if len(parts) == 1:
        return [parts[0], parts[0]]
    if len(parts) == 2:
        return parts
    raise ValueError(f"Invalid channel value: {value}")


def parse_pair(value):
    parts = [float(part) for part in str(value).replace(",", "-").split("-") if part]
    if len(parts) != 2:
        raise ValueError(f"Invalid rate pair: {value}. Expected like 0.5-0.1")
    return parts[0], parts[1]


def rate_tag(value):
    if value is None:
        return "none"
    return str(value).replace(".", "p")


def job_key(record):
    return (
        record.get("graph"),
        record.get("mode"),
        tuple(record.get("channels", [])),
        int(record.get("seed")),
        record.get("mask_rate"),
        record.get("mapped_only_dynamic_rate"),
        record.get("mapped_mix_mapped_rate"),
        record.get("mapped_mix_non_mapped_rate"),
        record.get("all_mapped_plus_non_mapped_rate"),
        record.get("mapped_context_non_mapped_rate"),
        record.get("mapped_biased_beta"),
        record.get("edge_curriculum_split_ratio"),
        record.get("edge_curriculum_initial_rate"),
        record.get("edge_curriculum_schedule"),
        record.get("lambda_onto"),
        record.get("track_kg_negative_sampling"),
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
        "mapped_only_dynamic_rate": record.get("mapped_only_dynamic_rate"),
        "mapped_mix_mapped_rate": record.get("mapped_mix_mapped_rate"),
        "mapped_mix_non_mapped_rate": record.get("mapped_mix_non_mapped_rate"),
        "all_mapped_plus_non_mapped_rate": record.get("all_mapped_plus_non_mapped_rate"),
        "mapped_context_non_mapped_rate": record.get("mapped_context_non_mapped_rate"),
        "mapped_biased_beta": record.get("mapped_biased_beta"),
        "edge_curriculum_split_ratio": record.get("edge_curriculum_split_ratio"),
        "edge_curriculum_initial_rate": record.get("edge_curriculum_initial_rate"),
        "edge_curriculum_schedule": record.get("edge_curriculum_schedule"),
        "lambda_onto": record.get("lambda_onto"),
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
    balanced = set(BALANCED_MODES)
    return [mode for mode in requested if mode not in balanced] + [mode for mode in requested if mode in balanced]


def build_jobs(args):
    jobs = []
    channels_grid = [parse_channels(value) for value in args.channels_grid]
    mix_pairs = [parse_pair(value) for value in args.mix_pool_rates]

    for mode in balanced_last_modes(args.modes):
        variants = []
        if mode in NO_RATE_MODES:
            variants = [{}]
        elif mode == "mapped_biased_dynamic":
            variants = [
                {"mask_rate": float(mask_rate), "mapped_biased_beta": float(beta)}
                for mask_rate in args.mask_rates
                for beta in args.mapped_biased_betas
            ]
        elif mode in GLOBAL_MASK_RATE_MODES or mode == "canonicalized_balanced_dynamic":
            variants = [{"mask_rate": float(rate)} for rate in args.mask_rates]
        elif mode in MAPPED_ONLY_RATE_MODES or mode in ("mapped_only_dynamic_balanced", "mapped_selector_dynamic_balanced"):
            variants = [{"mapped_only_dynamic_rate": float(rate)} for rate in args.mapped_only_rates]
        elif mode in MIX_POOL_RATE_MODES or mode == "mapped_mix_dynamic_balanced":
            variants = [
                {"mapped_mix_mapped_rate": float(mapped_rate), "mapped_mix_non_mapped_rate": float(other_rate)}
                for mapped_rate, other_rate in mix_pairs
            ]
        elif mode in ALL_MAPPED_PLUS_MODES or mode == "all_mapped_plus_balanced_dynamic":
            variants = [
                {"all_mapped_plus_non_mapped_rate": float(rate)}
                for rate in args.all_mapped_plus_non_mapped_rates
            ]
        elif mode in MAPPED_CONTEXT_NON_MAPPED_MODES or mode == "mapped_context_non_mapped_dynamic_balanced":
            variants = [
                {"mapped_context_non_mapped_rate": float(rate)}
                for rate in args.mapped_context_non_mapped_rates
            ]
        elif mode in LEGACY_GLOBAL_BUDGET_MODES:
            variants = [{"mask_rate": LEGACY_GLOBAL_BUDGET_MODES[mode]}]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        for channels in channels_grid:
            for seed in args.seeds:
                for variant in variants:
                    jobs.append({
                        "graph": GRAPH,
                        "mode": mode,
                        "channels": list(channels),
                        "seed": int(seed),
                        "mask_rate": variant.get("mask_rate"),
                        "mapped_only_dynamic_rate": variant.get("mapped_only_dynamic_rate"),
                        "mapped_mix_mapped_rate": variant.get("mapped_mix_mapped_rate"),
                        "mapped_mix_non_mapped_rate": variant.get("mapped_mix_non_mapped_rate"),
                        "all_mapped_plus_non_mapped_rate": variant.get("all_mapped_plus_non_mapped_rate"),
                        "mapped_context_non_mapped_rate": variant.get("mapped_context_non_mapped_rate"),
                        "mapped_biased_beta": variant.get("mapped_biased_beta"),
                        "edge_curriculum_split_ratio": args.edge_curriculum_split_ratio,
                        "edge_curriculum_initial_rate": args.edge_curriculum_initial_rate,
                        "edge_curriculum_schedule": args.edge_curriculum_schedule,
                        "lambda_onto": float(args.lambda_onto),
                        "track_kg_negative_sampling": bool(args.track_kg_negative_sampling),
                        "encoder": args.encoder,
                        "decoder": args.decoder,
                        "dropout": float(args.dropout),
                        "linear_probe": bool(args.linear_probe),
                    })
    return jobs


def build_run_dir(out_root, job, index):
    channels_tag = "-".join(str(value) for value in job["channels"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tags = [
        f"mask{rate_tag(job['mask_rate'])}",
        f"mod{rate_tag(job['mapped_only_dynamic_rate'])}",
        f"mix{rate_tag(job['mapped_mix_mapped_rate'])}-{rate_tag(job['mapped_mix_non_mapped_rate'])}",
        f"amp{rate_tag(job['all_mapped_plus_non_mapped_rate'])}",
        f"mctx{rate_tag(job['mapped_context_non_mapped_rate'])}",
        f"beta{rate_tag(job['mapped_biased_beta'])}",
    ]
    rate_part = "__".join(tag for tag in tags if not tag.endswith("none") and "-none" not in tag)
    if not rate_part:
        rate_part = "norate"
    return out_root / (
        f"gt2kg__{job['mode']}__c{channels_tag}__s{job['seed']}__"
        f"{rate_part}__j{index:04d}__{stamp}"
    )


def build_command(args, job, out_dir):
    mask_rate = 0.0 if job["mask_rate"] is None else job["mask_rate"]
    mapped_only_rate = 0.5 if job["mapped_only_dynamic_rate"] is None else job["mapped_only_dynamic_rate"]
    mix_mapped_rate = 0.5 if job["mapped_mix_mapped_rate"] is None else job["mapped_mix_mapped_rate"]
    mix_other_rate = 0.5 if job["mapped_mix_non_mapped_rate"] is None else job["mapped_mix_non_mapped_rate"]
    all_mapped_plus_rate = (
        0.1
        if job["all_mapped_plus_non_mapped_rate"] is None
        else job["all_mapped_plus_non_mapped_rate"]
    )
    mapped_context_non_mapped_rate = (
        1.0
        if job["mapped_context_non_mapped_rate"] is None
        else job["mapped_context_non_mapped_rate"]
    )
    mapped_biased_beta = 1.0 if job["mapped_biased_beta"] is None else job["mapped_biased_beta"]
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
        "--mapped-only-dynamic-rate", str(mapped_only_rate),
        "--mapped-mix-mapped-rate", str(mix_mapped_rate),
        "--mapped-mix-non-mapped-rate", str(mix_other_rate),
        "--all-mapped-plus-non-mapped-rate", str(all_mapped_plus_rate),
        "--mapped-context-non-mapped-rate", str(mapped_context_non_mapped_rate),
        "--mapped-biased-beta", str(mapped_biased_beta),
        "--wandb-project", args.wandb_project,
        "--lambda-onto", str(args.lambda_onto),
        "--num-epochs", str(args.num_epochs),
        "--edge-curriculum-split-ratio", str(args.edge_curriculum_split_ratio),
        "--edge-curriculum-initial-rate", str(args.edge_curriculum_initial_rate),
        "--edge-curriculum-schedule", args.edge_curriculum_schedule,
        "--short-run-dirs",
    ]
    if args.track_kg_negative_sampling:
        command.append("--track-kg-negative-sampling")
        if args.kg_negative_tracking_max_examples is not None:
            command.extend(["--kg-negative-tracking-max-examples", str(args.kg_negative_tracking_max_examples)])
    if args.debug_negative_sampling_epochs:
        command.extend([
            "--debug-negative-sampling-epochs",
            *[str(value) for value in args.debug_negative_sampling_epochs],
            "--debug-negative-sampling-batches-per-epoch",
            str(args.debug_negative_sampling_batches_per_epoch),
        ])
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
    print("\n=== GT2KG mapping-guided grid status ===\n")
    for job in jobs:
        finish = latest.get(job_key(job))
        label = (
            f"{job['mode']:<38} ch{'-'.join(str(v) for v in job['channels']):<8} "
            f"seed {job['seed']:<4} mask {job['mask_rate']} "
            f"mapped {job['mapped_only_dynamic_rate']} "
            f"mix {job['mapped_mix_mapped_rate']}/{job['mapped_mix_non_mapped_rate']} "
            f"extra {job['all_mapped_plus_non_mapped_rate']} "
            f"ctx_non_mapped {job['mapped_context_non_mapped_rate']} "
            f"beta {job['mapped_biased_beta']} "
            f"lambda_onto {job['lambda_onto']}"
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
        "mask_rates": args.mask_rates,
        "mapped_only_rates": args.mapped_only_rates,
        "mix_pool_rates": args.mix_pool_rates,
        "all_mapped_plus_non_mapped_rates": args.all_mapped_plus_non_mapped_rates,
        "mapped_context_non_mapped_rates": args.mapped_context_non_mapped_rates,
        "mapped_biased_betas": args.mapped_biased_betas,
        "lambda_onto": args.lambda_onto,
        "track_kg_negative_sampling": args.track_kg_negative_sampling,
        "kg_negative_tracking_max_examples": args.kg_negative_tracking_max_examples,
        "debug_negative_sampling_epochs": args.debug_negative_sampling_epochs,
        "debug_negative_sampling_batches_per_epoch": args.debug_negative_sampling_batches_per_epoch,
        "edge_curriculum_split_ratio": args.edge_curriculum_split_ratio,
        "edge_curriculum_initial_rate": args.edge_curriculum_initial_rate,
        "edge_curriculum_schedule": args.edge_curriculum_schedule,
        "legacy_global_budget_modes": LEGACY_GLOBAL_BUDGET_MODES,
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
                    f"ch{'-'.join(str(v) for v in job['channels'])} | seed {job['seed']} ===\n"
                )
                process = subprocess.Popen(command, cwd=str(REPO_ROOT))
                running.append({"process": process, "job": job, "run_dir": run_dir})
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
                print(f"\n=== Finish | {item['job']['mode']} | seed {item['job']['seed']} | {status} ===\n")
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
    parser.add_argument("--modes", nargs="*", default=ALL_MODES)
    parser.add_argument("--channels-grid", nargs="*", default=["256-256", "384-384", "512-512"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 42, 123, 789, 2024])
    parser.add_argument("--mask-rates", nargs="*", type=float, default=[0.2, 0.4, 0.5, 0.7, 0.8])
    parser.add_argument("--mapped-only-rates", nargs="*", type=float, default=[0.2, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument(
        "--mix-pool-rates",
        nargs="*",
        default=["0.2-0.1", "0.5-0.1", "0.5-0.3", "0.5-0.5", "0.7-0.3", "0.9-0.1"],
    )
    parser.add_argument("--all-mapped-plus-non-mapped-rates", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--mapped-context-non-mapped-rates", nargs="*", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--mapped-biased-betas", nargs="*", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--lambda-onto", type=float, default=0.3)
    parser.add_argument("--track-kg-negative-sampling", action="store_true")
    parser.add_argument("--kg-negative-tracking-max-examples", type=int, default=None)
    parser.add_argument("--debug-negative-sampling-epochs", nargs="*", type=int, default=[])
    parser.add_argument("--debug-negative-sampling-batches-per-epoch", type=int, default=0)
    parser.add_argument("--edge-curriculum-split-ratio", type=float, default=0.5)
    parser.add_argument("--edge-curriculum-initial-rate", type=float, default=0.05)
    parser.add_argument("--edge-curriculum-schedule", choices=["linear", "root", "geometric", "constant", "none"], default="linear")
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

    invalid_modes = sorted(set(args.modes) - set(ALL_MODES))
    if invalid_modes:
        raise ValueError(f"Invalid GT2KG modes: {invalid_modes}. Valid modes: {ALL_MODES}")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")
    run_suite(args)


if __name__ == "__main__":
    main()
