"""Run the four-graph GSSL baseline matrix with resumable parallel execution.

The runner creates one subprocess per configuration.  Each subprocess writes its
own log and result workbook; the parent process appends status and metrics to a
JSONL database and a CSV summary after every completed run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_MODEL = REPO_ROOT / "src" / "model"
DEFAULT_ROOT = REPO_ROOT / "baseline_experiments" / "gssl_dual_graph_baselines"
DEFAULT_DB = DEFAULT_ROOT / "runs.jsonl"
DEFAULT_SUMMARY = DEFAULT_ROOT / "summary.csv"
DEFAULT_PLAN = DEFAULT_ROOT / "plan.json"

SEEDS = [0, 42, 123, 789, 2024]
HIDDEN_SIZES = [256, 384, 512]
ENCODERS = [
    "RotatEGCN_attn",
    "RotatEGCN_conv",
    "TransGCN_attn",
    "TransGCN_conv",
    "RGCN",
    "CuGraphRGCN",
    "CuGraphGAT",
    "GAT",
    "GCN",
]
RGCN_BASES = [5, 10, 30]

BIOMEDICAL_CORE_CONCEPTS = [
    "Body Part, Organ, or Organ Component",
    "Disease or Syndrome",
    "Finding",
    "Intellectual Product",
    "Laboratory Procedure",
    "Organic Chemical",
    "Pharmacologic Substance",
    "Therapeutic or Preventive Procedure",
]

GRAPHS = {
    "biomed_clean": {
        "dataset": "biomed_cg",
        "kg_path": "data/UMLS/clean/biomed_cg.json",
        "gs_path": "data/UMLS/common_nodes.xlsx",
        "embedding_dir": "biomed_cg",
        "core_concepts": BIOMEDICAL_CORE_CONCEPTS,
        "splits_dir": "data/UMLS/splits/umls_kg_splits",
    },
    "biomed_noisy": {
        "dataset": "biomed_tdg",
        "kg_path": "data/UMLS/noisy/org/biomed_tdg.json",
        "gs_path": "data/UMLS/common_nodes.xlsx",
        "embedding_dir": "biomed_tdg",
        "core_concepts": BIOMEDICAL_CORE_CONCEPTS,
        "splits_dir": "data/UMLS/splits/umls_kg_splits",
    },
    "dbpedia_clean": {
        "dataset": "DBpedia_174_clean_kg",
        "kg_path": "data/dbpedia_174/DBpedia_174_clean_kg.json",
        "gs_path": "data/dbpedia_174/GS_dbpedia_174.xlsx",
        "embedding_dir": "DBpedia_174_clean_kg",
        "splits_dir": "data/UMLS/splits/dbpedia_174_splits",
    },
    "dbpedia_noisy": {
        "dataset": "dbpedia_174_kg_tdg",
        "kg_path": "data/dbpedia_174/dbpedia_174_kg_tdg.json",
        "gs_path": "data/dbpedia_174/GS_dbpedia_174.xlsx",
        "embedding_dir": "dbpedia_174_kg_tdg",
        "splits_dir": "data/UMLS/splits/dbpedia_174_splits",
    },
}

TASKS = ("recons_r", "recons_x_symmetric", "recons_x_mlp", "contrastive")
JSONL_LOCK = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", choices=sorted(GRAPHS), default=sorted(GRAPHS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--encoders", nargs="+", choices=ENCODERS, default=list(ENCODERS))
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=HIDDEN_SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--rgcn-bases", nargs="+", type=int, default=RGCN_BASES)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--num-neighbors", nargs=2, type=int, default=[-1, -1],
        metavar=("HOP1", "HOP2"),
        help="Maximum sampled neighbors per GNN hop; use -1 -1 for all neighbors.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--plm-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-tag", default="sentence-transformers_all-MiniLM-L6-v2")
    parser.add_argument("--wandb-project", default="GSSL_Dual_Graph_Baselines")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT / "runs")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Show worker output directly in the terminal instead of run.log.",
    )
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--worker-config", type=Path)
    return parser.parse_args()


def absolute_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def append_jsonl(path: Path, record: dict):
    path = absolute_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with JSONL_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def load_finished(path: Path) -> dict[str, dict]:
    path = absolute_path(path)
    finished = {}
    if not path.exists():
        return finished
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # An interrupted parallel write from an older runner must not
                # prevent the suite from resuming the valid records.
                continue
            if record.get("event") == "finish":
                finished[record["run_id"]] = record
    return finished


def safe_tag(value) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))


def graph_core_concepts(graph_cfg: dict) -> list[str]:
    if graph_cfg.get("core_concepts") is not None:
        return list(graph_cfg["core_concepts"])
    frame = pd.read_excel(absolute_path(graph_cfg["gs_path"]), sheet_name=0)
    column = "label" if "label" in frame.columns else frame.columns[-1]
    return sorted(frame[column].dropna().astype(str).unique().tolist())


def experiment_id(graph: str, task: str, encoder: str, decoder: str | None,
                  channels: int, seed: int, bases: int | None) -> str:
    parts = [graph, task, encoder]
    if decoder:
        parts.append(f"dec-{decoder}")
    if bases is not None:
        parts.append(f"bases-{bases}")
    parts.extend([f"ch{channels}-{channels}", f"seed{seed}"])
    return "__".join(safe_tag(part) for part in parts)


def build_experiments(args) -> list[dict]:
    experiments = []
    for graph in args.graphs:
        for task in args.tasks:
            for encoder in args.encoders:
                bases_values = args.rgcn_bases if encoder in ("RGCN", "CuGraphRGCN") else [None]
                for bases in bases_values:
                    decoders = [None]
                    if task == "recons_x_symmetric":
                        decoders = [encoder]
                    elif task == "recons_x_mlp":
                        decoders = ["MLP"]
                    for decoder in decoders:
                        for channels in args.hidden_sizes:
                            for seed in args.seeds:
                                run_id = experiment_id(
                                    graph, task, encoder, decoder, channels, seed, bases
                                )
                                experiments.append({
                                    "run_id": run_id,
                                    "graph": graph,
                                    "task": task,
                                    "encoder": encoder,
                                    "decoder": decoder,
                                    "num_bases": bases,
                                    "channels": [channels, channels],
                                    "seed": seed,
                                })
    return experiments


def write_plan(path: Path, experiments: list[dict], args):
    path = absolute_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_experiments": len(experiments),
        "common": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "num_neighbors": args.num_neighbors,
            "negative_corruption_mode": "entity_only",
            "ontology": False,
            "save_checkpoints": False,
            "plm_model": args.plm_model,
        },
        "experiments": experiments,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_metrics(run_dir: Path) -> dict:
    files = sorted(run_dir.rglob("results_seed_*.xlsx"), key=lambda item: item.stat().st_mtime)
    if not files:
        return {"metrics_found": False}
    frame = pd.read_excel(files[-1])
    if frame.empty:
        return {"metrics_found": False, "results_file": str(files[-1])}
    row = frame.iloc[-1].to_dict()
    metrics = {"metrics_found": True, "results_file": str(files[-1])}
    for key, value in row.items():
        if pd.isna(value):
            continue
        if hasattr(value, "item"):
            value = value.item()
        metrics[str(key)] = value
    return metrics


def write_summary(path: Path, records: list[dict]):
    path = absolute_path(path)
    rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key not in {"metrics", "command"}}
        row.update({f"metric_{key}": value for key, value in record.get("metrics", {}).items()})
        rows.append(row)
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)


def worker_config(experiment: dict, args, run_dir: Path) -> dict:
    graph_cfg = GRAPHS[experiment["graph"]]
    kg_path = absolute_path(graph_cfg["kg_path"])
    gs_path = absolute_path(graph_cfg["gs_path"])
    embedding_root = absolute_path("outputs") / graph_cfg["embedding_dir"]
    tag = args.embedding_tag
    return {
        "run_id": experiment["run_id"],
        "graph": experiment["graph"],
        "task": experiment["task"],
        "encoder": experiment["encoder"],
        "decoder": experiment["decoder"],
        "num_bases": experiment["num_bases"],
        "channels": experiment["channels"],
        "seed": experiment["seed"],
        "kg_path": str(kg_path),
        "gs_path": str(gs_path),
        "entities_path": str(embedding_root / f"{tag}_entities.pickle"),
        "edges_path": str(embedding_root / f"{tag}_predicates.pickle"),
        "core_concepts": graph_core_concepts(graph_cfg),
        "splits_dir": str(absolute_path(graph_cfg["splits_dir"])),
        "run_dir": str(absolute_path(run_dir)),
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "num_neighbors": args.num_neighbors,
        "dropout": args.dropout,
        "plm_model": args.plm_model,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
    }


def run_worker(path: Path) -> int:
    spec = json.loads(absolute_path(path).read_text(encoding="utf-8"))
    run_dir = Path(spec["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    os.environ["WANDB_MODE"] = spec["wandb_mode"]
    os.environ["WANDB_SILENT"] = "true"
    os.environ["PYTHONHASHSEED"] = str(spec["seed"])
    random.seed(spec["seed"])
    os.chdir(SRC_MODEL)
    # trans_gcn_layer imports the package as ``src.layers...``.  The worker
    # starts from baseline_experiments and then changes directory, so expose
    # both the repository root and src/model explicitly.
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(SRC_MODEL))

    import numpy as np
    import torch

    np.random.seed(spec["seed"])
    torch.manual_seed(spec["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec["seed"])

    import config as model_config

    cfg = model_config.config
    cfg.update({
        "seed": spec["seed"],
        "active_seed": spec["seed"],
        "dataset": spec["graph"],
        "KG_path": spec["kg_path"],
        "Entities_path": spec["entities_path"],
        "Edges_path": spec["edges_path"],
        "Gs_path_no_other": spec["gs_path"],
        "core_concepts": spec["core_concepts"],
        "plm_embedding_model": spec["plm_model"],
        "training_task": ["Recons_R" if spec["task"] == "recons_r" else
                          "Recons_X" if spec["task"].startswith("recons_x") else "Contrastive"],
        "recons_r_training_mode": "all_batch_edges",
        "recons_r_target_relation_field": "predicate",
        "negative_corruption_mode": "entity_only",
        "negative_entity_sampling_scope": "batch",
        "num_neighbors": spec["num_neighbors"],
        "batch_size": spec["batch_size"],
        "test_batch_size": spec["batch_size"],
        "num_epochs": spec["num_epochs"],
        "dropout": spec["dropout"],
        "lambda_onto": 0,
        "lambda_align": 0,
        "lambda_core_contrastive": 0,
        "lambda_core_align": 0,
        "lambda_domain_range": 0,
        "lambda_domain_range_embedding": 0,
        "lambda_onto_hierarchy": 0,
        "onto_KG_path": None,
        "onto_entities_path": None,
        "onto_edges_path": None,
        "domain_range_constraints_path": None,
        "run_linear_probe_on_best_loss": True,
        "linear_probe_gs_path": spec["gs_path"],
        "linear_probe_splits_dir": spec["splits_dir"],
        "num_steps": None,
        "shuffle": False,
        "save_checkpoints": False,
        "root_save_dir": str(run_dir / "checkpoints"),
        "wandb_project_name": spec["wandb_project"],
        # main.py keeps the legacy RGCN branch; for the cuGraph variant we
        # route that branch to CuGraphRGCNEncoder below.
        "encoders": [
            "RGCN" if spec["encoder"] == "CuGraphRGCN" else
            "GAT" if spec["encoder"] == "CuGraphGAT" else spec["encoder"]
        ],
        "decoders": ([
            "RGCN" if spec["encoder"] == "CuGraphRGCN" else
            "GAT" if spec["encoder"] == "CuGraphGAT" else spec["encoder"]
        ]
                      if spec["task"] == "recons_x_symmetric" else
                     ["MLP"] if spec["task"] == "recons_x_mlp" else []),
        "message_sens": ["source_to_target"],
        "hyperparams_grid": {
            "num_bases": [spec["num_bases"]] if spec["num_bases"] is not None else [],
            "out_channels": [spec["channels"]],
        },
    })

    import train_optimize_parms
    import main as model_main

    if spec["encoder"] == "CuGraphRGCN":
        model_main.RGCNEncoder = model_main.CuGraphRGCNEncoder
    elif spec["encoder"] == "CuGraphGAT":
        model_main.GATEncoder = model_main.CuGraphGATEncoder
        model_main.GATDecoder = model_main.CuGraphGATDecoder

    # Some legacy constructor calls in main.py rely on their class defaults.
    # Wrap them for this study so every GNN encoder uses the requested dropout.
    for class_name in ("GCNEncoder", "RGCNEncoder", "GATEncoder"):
        original_encoder = getattr(model_main, class_name)

        class DropoutConfiguredEncoder(original_encoder):
            def __init__(self, *encoder_args, **encoder_kwargs):
                encoder_kwargs.setdefault("dropout", spec["dropout"])
                super().__init__(*encoder_args, **encoder_kwargs)

        DropoutConfiguredEncoder.__name__ = class_name
        setattr(model_main, class_name, DropoutConfiguredEncoder)

    # Newer W&B versions reject the legacy start_method argument used by main.py.
    original_settings = model_main.wandb.Settings
    def compatible_settings(*args, **kwargs):
        kwargs.pop("start_method", None)
        return original_settings(*args, **kwargs)
    model_main.wandb.Settings = compatible_settings

    original_init = model_main.wandb.init
    def named_init(*args, **kwargs):
        kwargs["project"] = spec["wandb_project"]
        kwargs["name"] = spec["run_id"]
        extra = {
            "run_id": spec["run_id"],
            "graph": spec["graph"],
            "task": spec["task"],
            "encoder": spec["encoder"],
            "decoder": spec["decoder"],
            "num_bases": spec["num_bases"],
            "channels": spec["channels"],
            "seed": spec["seed"],
            "dropout": spec["dropout"],
            "ontology": False,
        }
        current = kwargs.get("config") or {}
        kwargs["config"] = {**current, **extra}
        return original_init(*args, **kwargs)
    model_main.wandb.init = named_init

    # Keep best states in memory for evaluation, but do not write checkpoints.
    original_save_model = train_optimize_parms.save_model
    def save_model_if_enabled(*args, **kwargs):
        if cfg.get("save_checkpoints", True):
            return original_save_model(*args, **kwargs)
        return None
    train_optimize_parms.save_model = save_model_if_enabled

    config_path = run_dir / "run_config.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    try:
        model_main.config.update(cfg)
        model_main.main()
        metrics = extract_metrics(run_dir)
        (run_dir / "worker_status.json").write_text(
            json.dumps({"status": "completed", "metrics": metrics}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        (run_dir / "worker_status.json").write_text(
            json.dumps({"status": "failed", "error": repr(exc)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\nWORKER ERROR: {exc!r}\n")
        return 1


def run_one(experiment: dict, args, db_path: Path, out_root: Path) -> dict:
    run_dir = absolute_path(out_root) / experiment["run_id"]
    spec_path = run_dir / "worker_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        json.dumps(worker_config(experiment, args, run_dir), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_path = run_dir / "run.log"
    # Keep worker diagnostics visible immediately in the per-run log.  Without
    # -u, output can remain buffered for several minutes during graph loading.
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-config", str(spec_path)]
    start = {
        "event": "start",
        "status": "running",
        "time": datetime.now().isoformat(timespec="seconds"),
        **experiment,
        "out_dir": str(run_dir),
        "command": command,
    }
    append_jsonl(db_path, start)
    if args.stream_output:
        process = subprocess.run(command, cwd=REPO_ROOT)
    else:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
            )
    metrics = extract_metrics(run_dir)
    finish = {
        "event": "finish",
        "status": "completed" if process.returncode == 0 and metrics.get("metrics_found") else "failed",
        "returncode": process.returncode,
        "time": datetime.now().isoformat(timespec="seconds"),
        **experiment,
        "out_dir": str(run_dir),
        "metrics": metrics,
    }
    append_jsonl(db_path, finish)
    return finish


def main():
    args = parse_args()
    if args.worker_config:
        raise SystemExit(run_worker(args.worker_config))

    db_path = absolute_path(args.db)
    summary_path = absolute_path(args.summary)
    experiments = build_experiments(args)
    write_plan(args.plan, experiments, args)
    finished = load_finished(db_path)
    pending = [
        experiment for experiment in experiments
        if args.rerun_completed or finished.get(experiment["run_id"], {}).get("status") != "completed"
    ]
    print(f"Total experiments: {len(experiments)}")
    print(f"Completed: {len(experiments) - len(pending)}")
    print(f"To run: {len(pending)}")
    print(f"Plan: {absolute_path(args.plan)}")
    print(f"DB: {db_path}")
    print(f"Summary: {summary_path}")
    if args.status_only:
        return

    completed_records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as pool:
        futures = {
            pool.submit(run_one, experiment, args, db_path, args.out_root): experiment
            for experiment in pending
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            record = future.result()
            completed_records.append(record)
            print(
                f"[{index}/{len(pending)}] {record['status']} {record['run_id']}",
                flush=True,
            )
            all_finished = list(load_finished(db_path).values())
            write_summary(summary_path, all_finished)

    write_summary(summary_path, list(load_finished(db_path).values()))
    print("GSSL baseline suite finished.")


if __name__ == "__main__":
    main()
