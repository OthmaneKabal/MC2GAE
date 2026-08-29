import argparse
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_MODEL = REPO_ROOT / "src" / "model"
DEFAULT_DB = REPO_ROOT / "masking_study" / "biomedical_masking_study.jsonl"
DEFAULT_SUMMARY = REPO_ROOT / "masking_study" / "biomedical_masking_study.csv"
DEFAULT_RUNS_DIR = REPO_ROOT / "masking_study" / "runs"
WANDB_PROJECT = "Masking_Study_BioMedical_Recons_R"

SEEDS = [0, 42, 123, 100, 2026]
CHANNELS = [[384, 384], [256, 256], [512, 512]]

GRAPHS = {
    "MM_mapped_nci_All_R_KG": {
        "kg_path": "../../data/UMLS/noisy/org/MM_mapped_nci_All_R_KG.json",
        "entities_path": "outputs/MM_mapped_nci_All_R_KG/sentence-transformers_all-MiniLM-L6-v2_entities.pickle",
        "edges_path": "outputs/MM_mapped_nci_All_R_KG/sentence-transformers_all-MiniLM-L6-v2_predicates.pickle",
        "gs_path": "../../data/UMLS/MM_mapped_nci_GS.xlsx",
        "group": "baseline",
    },
    "GT2KG_mapped_and_old_rel_norm": {
        "kg_path": "../../data/UMLS/noisy/org/GT2KG_mapped_and_old_rel_norm.json",
        "entities_path": "outputs/GT2KG_mapped_and_old_rel_norm/sentence-transformers_all-MiniLM-L6-v2_entities.pickle",
        "edges_path": "outputs/GT2KG_mapped_and_old_rel_norm/sentence-transformers_all-MiniLM-L6-v2_predicates.pickle",
        "gs_path": "../../data/UMLS/MM_mapped_nci_GS.xlsx",
        "group": "mapped_graph",
    },
}

BASELINE_MODES = [
    {
        "name": "whole_graph",
        "recons_r_training_mode": "all_batch_edges",
        "target_relation_field": "predicate",
        "description": "Reconstruct whole graph without masking.",
    },
    {
        "name": "random_static",
        "recons_r_training_mode": "random_static_masked_only",
        "target_relation_field": "predicate",
        "description": "Random static masking, reconstruct only masked edges.",
    },
    {
        "name": "balanced_static",
        "recons_r_training_mode": "balanced_static_masked_only",
        "target_relation_field": "predicate",
        "description": "Type-balanced static masking, reconstruct only masked edges.",
    },
    {
        "name": "random_dynamic",
        "recons_r_training_mode": "random_dynamic_masked_only",
        "target_relation_field": "predicate",
        "description": "Random dynamic masking, new mask each epoch.",
    },
    {
        "name": "balanced_dynamic",
        "recons_r_training_mode": "balanced_dynamic_masked_only",
        "target_relation_field": "predicate",
        "description": "Type-balanced dynamic masking, new mask each epoch.",
    },
]

MAPPING_GUIDED_MODES = [
    {
        "name": "mapping_guided_mapped_predicate",
        "recons_r_training_mode": "mapped_only",
        "target_relation_field": "predicate",
        "description": "Mask mapped edges and reconstruct mapped predicates.",
    },
    {
        "name": "mapping_guided_old_predicate",
        "recons_r_training_mode": "mapped_only",
        "target_relation_field": "old_predicate",
        "description": "Mask mapped edges but reconstruct old predicates.",
    },
]


def resolve_cli_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (REPO_ROOT / path).resolve()


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def capture_rng_state():
    import numpy as np
    import torch

    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def save_rng_state(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(capture_rng_state(), file)


def set_all_seeds(seed):
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def experiment_key(record):
    return (
        record.get("graph"),
        record.get("mode"),
        record.get("recons_r_training_mode"),
        record.get("target_relation_field"),
        tuple(record.get("channels", [])),
        int(record.get("seed")),
    )


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
                latest[experiment_key(record)] = record
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


def update_summary_csv(summary_path, finish_record):
    metrics = finish_record.get("metrics", {})
    row = {
        "time": finish_record.get("time"),
        "status": finish_record.get("status"),
        "graph": finish_record.get("graph"),
        "group": finish_record.get("group"),
        "mode": finish_record.get("mode"),
        "recons_r_training_mode": finish_record.get("recons_r_training_mode"),
        "target_relation_field": finish_record.get("target_relation_field"),
        "channels": "-".join(str(value) for value in finish_record.get("channels", [])),
        "seed": finish_record.get("seed"),
        "negative_corruption_mode": finish_record.get("negative_corruption_mode"),
        "negative_entity_sampling_scope": finish_record.get("negative_entity_sampling_scope"),
        "out_dir": finish_record.get("out_dir"),
        "metrics_found": metrics.get("metrics_found", False),
    }
    for key in (
        "accuracy", "f1_score", "precision", "recall",
        "R_accuracy", "R_precision", "R_recall", "R_f1",
        "best_epoch", "status", "exp_name",
    ):
        if key in metrics:
            row[f"metric_{key}"] = metrics[key]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(summary_path, index=False)


def install_wandb_naming(model_main, project, run_name, extra_config):
    wandb = model_main.wandb
    original_init = wandb.init

    def study_init(*args, **kwargs):
        kwargs["project"] = project
        kwargs["name"] = run_name
        run_config = kwargs.get("config")
        if isinstance(run_config, dict):
            kwargs["config"] = {**run_config, **extra_config}
        else:
            kwargs["config"] = dict(extra_config)
        return original_init(*args, **kwargs)

    wandb.init = study_init
    if hasattr(model_main, "_ensure_seeded_wandb_init"):
        model_main._ensure_seeded_wandb_init = lambda: None


def install_dropout_zero_hook(model_main):
    original_encoder = model_main.TransGCNEncoder

    def no_dropout_encoder(*args, **kwargs):
        kwargs["dropout"] = 0.0
        encoder = original_encoder(*args, **kwargs)
        if hasattr(encoder, "dropout"):
            encoder.dropout.p = 0.0
        print("\nMasking study: TransGCNEncoder dropout forced to 0.0\n")
        return encoder

    model_main.TransGCNEncoder = no_dropout_encoder


def modes_for_graph(graph_name):
    modes = list(BASELINE_MODES)
    if graph_name == "GT2KG_mapped_and_old_rel_norm":
        modes.extend(MAPPING_GUIDED_MODES)
    return modes


def build_experiments(graphs, modes, channels_list, seeds):
    experiments = []
    for graph_name in graphs:
        available_modes = {mode["name"]: mode for mode in modes_for_graph(graph_name)}
        selected_modes = list(available_modes) if modes is None else modes
        for mode_name in selected_modes:
            mode = available_modes[mode_name]
            for channels in channels_list:
                for seed in seeds:
                    experiments.append({
                        "graph": graph_name,
                        "graph_config": GRAPHS[graph_name],
                        "mode": mode["name"],
                        "mode_config": mode,
                        "channels": list(channels),
                        "seed": int(seed),
                    })
    return experiments


def configure_model(config, experiment, run_dir, num_epochs=None):
    graph_cfg = experiment["graph_config"]
    mode_cfg = experiment["mode_config"]
    config.update({
        "seed": int(experiment["seed"]),
        "active_seed": int(experiment["seed"]),
        "dataset": experiment["graph"],
        "KG_path": graph_cfg["kg_path"],
        "Entities_path": graph_cfg["entities_path"],
        "Edges_path": graph_cfg["edges_path"],
        "Gs_path_no_other": graph_cfg["gs_path"],
        "training_task": ["Recons_R"],
        "recons_r_training_mode": mode_cfg["recons_r_training_mode"],
        "recons_r_target_relation_field": mode_cfg["target_relation_field"],
        "negative_sampling_mode": "uniform",
        "negative_corruption_mode": "entity_only",
        "negative_entity_sampling_scope": "batch",
        "kg_negative_sampling_seed": None,
        "track_kg_negative_sampling": False,
        "track_onto_negative_sampling": False,
        "replay_kg_negative_sampling": False,
        "replay_onto_negative_sampling": False,
        "num_neighbors": [-1, -1],
        "batch_size": 1024,
        "test_batch_size": 1024,
        "shuffle": False,
        "num_epochs": int(num_epochs) if num_epochs is not None else 50,
        "num_steps": None,
        "lambda_onto": 0,
        "lambda_align": 0,
        "lambda_core_contrastive": 0,
        "lambda_core_align": 0,
        "lambda_domain_range": 0,
        "lambda_onto_hierarchy": 0,
        "hyperparams_grid": {"num_bases": [5, 10], "out_channels": [experiment["channels"]]},
        "encoders": ["RotatEGCN_attn"],
        "decoders": ["MLP"],
        "message_sens": ["source_to_target"],
        "root_save_dir": str(run_dir / "checkpoints"),
        "wandb_project_name": WANDB_PROJECT,
        "wandb_mode": os.environ.get("WANDB_MODE"),
    })


def run_single(args):
    graph_cfg = GRAPHS[args.graph]
    mode_cfg = {mode["name"]: mode for mode in modes_for_graph(args.graph)}[args.mode]
    experiment = {
        "graph": args.graph,
        "graph_config": graph_cfg,
        "mode": args.mode,
        "mode_config": mode_cfg,
        "channels": args.channels,
        "seed": int(args.seed),
    }
    out_dir = resolve_cli_path(args.out)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    elif out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(int(args.seed))
    save_rng_state(out_dir / "rng_state_before.pkl")

    os.chdir(SRC_MODEL)
    sys.path.insert(0, str(SRC_MODEL))
    import config as model_config
    import main as model_main

    configure_model(model_config.config, experiment, out_dir, num_epochs=args.num_epochs)
    model_main.config.update(model_config.config)
    model_main.seed = int(args.seed)
    install_dropout_zero_hook(model_main)

    run_name = (
        f"{args.graph}__{args.mode}__"
        f"ch{'-'.join(str(value) for value in args.channels)}__seed{args.seed}"
    )
    install_wandb_naming(
        model_main,
        WANDB_PROJECT,
        run_name,
        {
            "study": "biomedical_recons_r_masking",
            "graph": args.graph,
            "group": graph_cfg["group"],
            "mode": args.mode,
            "description": mode_cfg["description"],
            "channels": args.channels,
            "global_seed": int(args.seed),
            "dropout": 0,
            "ontology_reconstruction": False,
            "negative_corruption_mode": "entity_only",
            "negative_entity_sampling_scope": "batch",
            "output_dir": str(out_dir),
        },
    )

    write_json(out_dir / "run_config.json", model_config.config)
    write_json(out_dir / "experiment.json", {
        "graph": args.graph,
        "group": graph_cfg["group"],
        "mode": args.mode,
        "mode_config": mode_cfg,
        "channels": args.channels,
        "seed": int(args.seed),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })

    if hasattr(model_main, "_set_all_seeds"):
        model_main._set_all_seeds(int(args.seed))
    model_main.main()

    save_rng_state(out_dir / "rng_state_after.pkl")
    metrics = extract_metrics(out_dir)
    finish_record = {
        "event": "finish",
        "status": "completed" if metrics.get("metrics_found") else "completed_no_metrics",
        "time": datetime.now().isoformat(timespec="seconds"),
        "graph": args.graph,
        "group": graph_cfg["group"],
        "mode": args.mode,
        "description": mode_cfg["description"],
        "recons_r_training_mode": mode_cfg["recons_r_training_mode"],
        "target_relation_field": mode_cfg["target_relation_field"],
        "channels": args.channels,
        "seed": int(args.seed),
        "negative_corruption_mode": "entity_only",
        "negative_entity_sampling_scope": "batch",
        "out_dir": str(out_dir),
        "metrics": metrics,
    }
    append_jsonl(resolve_cli_path(args.db), finish_record)
    update_summary_csv(resolve_cli_path(args.summary), finish_record)


def print_status(experiments, db_path):
    latest = latest_finish_records(db_path)
    pending = []
    print("\n=== BioMedical masking study status ===\n")
    for exp in experiments:
        mode_cfg = exp["mode_config"]
        record = {
            "graph": exp["graph"],
            "mode": exp["mode"],
            "recons_r_training_mode": mode_cfg["recons_r_training_mode"],
            "target_relation_field": mode_cfg["target_relation_field"],
            "channels": exp["channels"],
            "seed": exp["seed"],
        }
        key = experiment_key(record)
        finish = latest.get(key)
        label = (
            f"{exp['graph']:<31} {exp['mode']:<32} "
            f"ch{'-'.join(str(v) for v in exp['channels']):<8} seed {exp['seed']:<4}"
        )
        if finish is None or finish.get("status") not in ("completed", "completed_no_metrics"):
            print(f"PENDING   {label}")
            pending.append(exp)
            continue
        metrics = finish.get("metrics") or {}
        accuracy = metrics.get("accuracy")
        f1_score = metrics.get("f1_score")
        if accuracy is None or f1_score is None:
            print(f"COMPLETED {label}")
        else:
            print(f"COMPLETED {label} accuracy={accuracy:.6f} f1={f1_score:.6f}")
    print(f"\nCompleted: {len(experiments) - len(pending)}")
    print(f"To run:    {len(pending)}\n")
    return pending


def run_suite(args):
    if len(args.graphs) == 1:
        graph_dir = REPO_ROOT / "masking_study" / args.graphs[0]
        if args.db == str(DEFAULT_DB):
            args.db = str(graph_dir / f"{args.graphs[0]}_masking_study.jsonl")
        if args.summary == str(DEFAULT_SUMMARY):
            args.summary = str(graph_dir / f"{args.graphs[0]}_masking_study.csv")
        if args.runs_dir == str(DEFAULT_RUNS_DIR):
            args.runs_dir = str(graph_dir / "runs")

    db_path = resolve_cli_path(args.db)
    summary_path = resolve_cli_path(args.summary)
    runs_dir = resolve_cli_path(args.runs_dir)
    channels_list = [parse_channels(value) for value in args.channels_grid]
    experiments = build_experiments(args.graphs, args.modes, channels_list, args.seeds)
    latest = latest_finish_records(db_path)
    pending = print_status(experiments, db_path)
    plan_path = db_path.with_name(f"{db_path.stem}_plan.json")
    write_json(plan_path, {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "wandb_project": WANDB_PROJECT,
        "common": {
            "domain": "BioMedical",
            "encoder": "RotatEGCN_attn",
            "decoder": "DistMult",
            "negative_corruption_mode": "entity_only",
            "negative_entity_sampling_scope": "batch",
            "batch_size": 1024,
            "dropout": 0,
            "ontology_reconstruction": False,
            "num_neighbors": [-1, -1],
            "shuffle": False,
        },
        "total_experiments": len(experiments),
        "experiments": [
            {
                "graph": exp["graph"],
                "group": exp["graph_config"]["group"],
                "mode": exp["mode"],
                "description": exp["mode_config"]["description"],
                "recons_r_training_mode": exp["mode_config"]["recons_r_training_mode"],
                "target_relation_field": exp["mode_config"]["target_relation_field"],
                "channels": exp["channels"],
                "seed": exp["seed"],
            }
            for exp in experiments
        ],
    })
    if args.status_only:
        print(f"Plan:    {plan_path}")
        print(f"DB:      {db_path}")
        print(f"Summary: {summary_path}")
        print(f"Runs:    {runs_dir}\n")
        return

    append_jsonl(db_path, {
        "event": "suite_start",
        "status": "running",
        "time": datetime.now().isoformat(timespec="seconds"),
        "total_experiments": len(experiments),
        "pending_experiments": len(pending),
        "wandb_project": WANDB_PROJECT,
    })

    for exp in experiments:
        mode_cfg = exp["mode_config"]
        key_record = {
            "graph": exp["graph"],
            "mode": exp["mode"],
            "recons_r_training_mode": mode_cfg["recons_r_training_mode"],
            "target_relation_field": mode_cfg["target_relation_field"],
            "channels": exp["channels"],
            "seed": exp["seed"],
        }
        finish = latest.get(experiment_key(key_record))
        if not args.rerun_completed and finish is not None and finish.get("status") in ("completed", "completed_no_metrics"):
            print(f"Skipping completed {exp['graph']} {exp['mode']} {exp['channels']} seed {exp['seed']}")
            continue

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        channel_tag = "-".join(str(value) for value in exp["channels"])
        run_dir = runs_dir / f"{exp['graph']}__{exp['mode']}__ch{channel_tag}__seed{exp['seed']}__{stamp}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single",
            "--graph", exp["graph"],
            "--mode", exp["mode"],
            "--channels", *[str(value) for value in exp["channels"]],
            "--seed", str(exp["seed"]),
            "--out", str(run_dir),
            "--db", str(db_path),
            "--summary", str(summary_path),
        ]
        if args.num_epochs is not None:
            command.extend(["--num-epochs", str(args.num_epochs)])

        start_record = {
            "event": "start",
            "status": "running",
            "time": datetime.now().isoformat(timespec="seconds"),
            "graph": exp["graph"],
            "group": exp["graph_config"]["group"],
            "mode": exp["mode"],
            "description": mode_cfg["description"],
            "recons_r_training_mode": mode_cfg["recons_r_training_mode"],
            "target_relation_field": mode_cfg["target_relation_field"],
            "channels": exp["channels"],
            "seed": exp["seed"],
            "negative_corruption_mode": "entity_only",
            "negative_entity_sampling_scope": "batch",
            "out_dir": str(run_dir),
            "command": command,
        }
        append_jsonl(db_path, start_record)
        print(f"\n=== Running {exp['graph']} | {exp['mode']} | ch{channel_tag} | seed {exp['seed']} ===\n")
        process = subprocess.run(command, cwd=str(REPO_ROOT), text=True)
        if process.returncode != 0:
            fail_record = {
                **start_record,
                "event": "finish",
                "status": "failed",
                "time": datetime.now().isoformat(timespec="seconds"),
                "returncode": process.returncode,
                "metrics": {"metrics_found": False},
            }
            append_jsonl(db_path, fail_record)
            update_summary_csv(summary_path, fail_record)
            print(f"FAILED: {exp['graph']} {exp['mode']} {exp['channels']} seed {exp['seed']}")
            continue
        latest[experiment_key(key_record)] = {"status": "completed"}

    append_jsonl(db_path, {
        "event": "suite_finish",
        "status": "completed",
        "time": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"\nMasking study suite finished. DB: {db_path}")
    print(f"Summary CSV: {summary_path}\n")


def parse_channels(value):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item) for item in str(value).replace(",", "-").split("-") if item]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--graph", choices=sorted(GRAPHS), default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--channels", nargs=2, type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--graphs", nargs="*", choices=sorted(GRAPHS), default=list(GRAPHS))
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--channels-grid", nargs="*", default=["384-384", "256-256", "512-512"])
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.single:
        if args.graph is None or args.mode is None or args.channels is None or args.seed is None or args.out is None:
            raise ValueError("--single requires --graph, --mode, --channels, --seed and --out")
        valid_modes = {mode["name"] for mode in modes_for_graph(args.graph)}
        if args.mode not in valid_modes:
            raise ValueError(f"Mode {args.mode} is not valid for graph {args.graph}. Valid: {sorted(valid_modes)}")
        run_single(args)
    else:
        for graph in args.graphs:
            valid_modes = {mode["name"] for mode in modes_for_graph(graph)}
            if args.modes is not None:
                invalid = set(args.modes) - valid_modes
                if invalid:
                    raise ValueError(f"Modes {sorted(invalid)} are not valid for graph {graph}.")
        run_suite(args)


if __name__ == "__main__":
    main()
