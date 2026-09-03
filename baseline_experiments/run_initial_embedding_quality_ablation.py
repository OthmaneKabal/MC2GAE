import argparse
import copy
import json
import os
import pickle
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_MODEL = REPO_ROOT / "src" / "model"

DEFAULT_OUT_ROOT = REPO_ROOT / "baseline_experiments" / "initial_embedding_quality_ablation"
DEFAULT_DB = DEFAULT_OUT_ROOT / "runs.jsonl"
DEFAULT_SUMMARY = DEFAULT_OUT_ROOT / "summary.csv"

UMLS_CORE_CONCEPTS = [
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
    "biomed": {
        "domain": "biomedical",
        "dataset": "MM_mapped_nci_All_R_KG",
        "kg_path": "../../data/UMLS/noisy/org/MM_mapped_nci_All_R_KG.json",
        "gs_path": "../../data/UMLS/common_nodes.xlsx",
        "core_concepts": UMLS_CORE_CONCEPTS,
        "embedding_output_name": "MM_mapped_nci_All_R_KG",
    },
    "dbpedia_clean": {
        "domain": "dbpedia",
        "dataset": "DBpedia_174_clean_kg",
        "kg_path": "../../data/dbpedia_174/DBpedia_174_clean_kg.json",
        "gs_path": "../../data/dbpedia_174/GS_dbpedia_174.xlsx",
        "core_concepts": None,
        "embedding_output_name": "DBpedia_174_clean_kg",
    },
    "dbpedia_tdg": {
        "domain": "dbpedia",
        "dataset": "dbpedia_174_kg_tdg",
        "kg_path": "../../data/dbpedia_174/dbpedia_174_kg_tdg.json",
        "gs_path": "../../data/dbpedia_174/GS_dbpedia_174.xlsx",
        "core_concepts": None,
        "embedding_output_name": "dbpedia_174_kg_tdg",
    },
}

EMBEDDINGS = {
    "sentencebert": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "tag": "sentence-transformers_all-MiniLM-L6-v2",
    },
    "biolinkbert": {
        "model": "michiyasunaga/BioLinkBERT-base",
        "tag": "michiyasunaga_BioLinkBERT-base",
        "domains": {"biomedical"},
    },
    "roberta": {
        "model": "roberta-base",
        "tag": "roberta-base",
        "domains": {"dbpedia"},
    },
}

TASKS = {
    "recons_r": {
        "training_task": "Recons_R",
        "recons_r_training_mode": "random_static_masked_only",
        "recons_r_target_relation_field": "predicate",
        "relation_mask_rate": 0.2,
    },
    "recons_x": {
        "training_task": "Recons_X",
        "recons_r_training_mode": "none",
        "recons_r_target_relation_field": "none",
        "attribute_mask_rate": 0.3,
        "recons_x_feature_masking": True,
    },
}


def resolve_path(path_value):
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


def safe_tag(value):
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in str(value))


def embedding_paths(graph_cfg, embedding_cfg):
    output_name = graph_cfg["embedding_output_name"]
    tag = embedding_cfg["tag"]
    return (
        f"outputs/{output_name}/{tag}_entities.pickle",
        f"outputs/{output_name}/{tag}_predicates.pickle",
    )


def dbpedia_core_concepts(gs_path):
    gs_full_path = resolve_path(gs_path)
    df = pd.read_excel(gs_full_path)
    return sorted(df["label"].astype(str).unique())


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


def save_rng_state(path):
    import numpy as np
    import torch

    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(state, file)


def experiment_key(record):
    return (
        record.get("graph"),
        record.get("domain"),
        record.get("embedding"),
        record.get("task"),
        tuple(record.get("channels", [])),
        int(record.get("seed")),
        record.get("attribute_mask_rate"),
        record.get("relation_mask_rate"),
        record.get("encoder"),
        record.get("decoder"),
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
        "domain": finish_record.get("domain"),
        "embedding": finish_record.get("embedding"),
        "plm_model": finish_record.get("plm_model"),
        "task": finish_record.get("task"),
        "training_task": finish_record.get("training_task"),
        "channels": "-".join(str(value) for value in finish_record.get("channels", [])),
        "seed": finish_record.get("seed"),
        "attribute_mask_rate": finish_record.get("attribute_mask_rate"),
        "relation_mask_rate": finish_record.get("relation_mask_rate"),
        "encoder": finish_record.get("encoder"),
        "decoder": finish_record.get("decoder"),
        "out_dir": finish_record.get("out_dir"),
        "metrics_found": metrics.get("metrics_found", False),
    }
    for key, value in metrics.items():
        if pd.isna(value) if isinstance(value, float) else False:
            continue
        row[f"metric_{key}"] = value
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(summary_path, index=False)


def install_wandb_patch(model_main, project, run_name, wandb_config):
    wandb = model_main.wandb
    if not hasattr(wandb, "_mc2gae_initial_ablation_original_init"):
        wandb._mc2gae_initial_ablation_original_init = wandb.init
    if not hasattr(wandb, "_mc2gae_initial_ablation_original_settings"):
        wandb._mc2gae_initial_ablation_original_settings = getattr(wandb, "Settings", None)
    original_init = wandb._mc2gae_initial_ablation_original_init
    original_settings = wandb._mc2gae_initial_ablation_original_settings

    def patched_init(*args, **kwargs):
        kwargs["project"] = project
        kwargs["name"] = run_name
        kwargs["config"] = dict(wandb_config)
        return original_init(*args, **kwargs)

    def patched_settings(*args, **kwargs):
        kwargs.pop("start_method", None)
        if original_settings is None:
            return None
        return original_settings(*args, **kwargs)

    wandb.init = patched_init
    if original_settings is not None:
        wandb.Settings = patched_settings
    if hasattr(model_main, "_ensure_seeded_wandb_init"):
        model_main._ensure_seeded_wandb_init = lambda: None


def install_dropout_hook(model_main, dropout):
    if not hasattr(model_main, "_mc2gae_initial_ablation_original_transgcn"):
        model_main._mc2gae_initial_ablation_original_transgcn = model_main.TransGCNEncoder
    original_encoder = model_main._mc2gae_initial_ablation_original_transgcn

    def dropout_encoder(*args, **kwargs):
        kwargs["dropout"] = float(dropout)
        encoder = original_encoder(*args, **kwargs)
        if hasattr(encoder, "dropout"):
            encoder.dropout.p = float(dropout)
        print(f"\nInitial embedding ablation: TransGCNEncoder dropout forced to {float(dropout)}\n")
        return encoder

    model_main.TransGCNEncoder = dropout_encoder


def configure_model(config, experiment, run_dir, args):
    graph_cfg = GRAPHS[experiment["graph"]]
    embedding_cfg = EMBEDDINGS[experiment["embedding"]]
    task_cfg = TASKS[experiment["task"]]
    entities_path, edges_path = embedding_paths(graph_cfg, embedding_cfg)
    core_concepts = graph_cfg["core_concepts"] or dbpedia_core_concepts(graph_cfg["gs_path"])
    mask_rate = task_cfg.get("attribute_mask_rate", task_cfg.get("relation_mask_rate"))

    config.update({
        "seed": int(experiment["seed"]),
        "active_seed": int(experiment["seed"]),
        "dataset": graph_cfg["dataset"],
        "KG_path": graph_cfg["kg_path"],
        "Entities_path": entities_path,
        "Edges_path": edges_path,
        "plm_embedding_model": embedding_cfg["model"],
        "Gs_path_no_other": graph_cfg["gs_path"],
        "core_concepts": core_concepts,
        "training_task": [task_cfg["training_task"]],
        "recons_r_training_mode": task_cfg["recons_r_training_mode"],
        "recons_r_target_relation_field": task_cfg["recons_r_target_relation_field"],
        "recons_x_feature_masking": task_cfg.get("recons_x_feature_masking", True),
        "max_masking_percentage": float(mask_rate),
        "total_drop_rate": float(mask_rate),
        "negative_sampling_mode": "uniform",
        "negative_corruption_mode": "entity_only",
        "negative_entity_sampling_scope": "batch",
        "kg_negative_sampling_seed": None,
        "track_kg_negative_sampling": False,
        "track_onto_negative_sampling": False,
        "replay_kg_negative_sampling": False,
        "replay_onto_negative_sampling": False,
        "num_neighbors": [-1, -1],
        "batch_size": int(args.batch_size),
        "test_batch_size": int(args.batch_size),
        "shuffle": False,
        "num_epochs": int(args.num_epochs),
        "num_steps": None,
        "lambda_onto": 0,
        "lambda_align": 0,
        "lambda_core_contrastive": 0,
        "lambda_core_align": 0,
        "lambda_domain_range": 0,
        "lambda_domain_range_embedding": 0,
        "lambda_onto_hierarchy": 0,
        "run_linear_probe_on_best_loss": False,
        "hyperparams_grid": {"num_bases": [5, 10], "out_channels": [experiment["channels"]]},
        "encoders": [args.encoder],
        "decoders": [args.decoder],
        "message_sens": ["source_to_target"],
        "root_save_dir": str(run_dir / "checkpoints"),
        "wandb_project_name": args.wandb_project,
        "wandb_mode": args.wandb_mode,
    })


def build_experiments(args):
    experiments = []
    for graph_name in args.graphs:
        domain = GRAPHS[graph_name]["domain"]
        for embedding_name in args.embeddings:
            embedding_domains = EMBEDDINGS[embedding_name].get("domains")
            if embedding_domains is not None and domain not in embedding_domains:
                continue
            for task_name in args.tasks:
                experiments.append({
                    "graph": graph_name,
                    "domain": domain,
                    "embedding": embedding_name,
                    "task": task_name,
                    "channels": args.channels,
                    "seed": int(args.seed),
                })
    return experiments


def build_run_dir_name(experiment):
    channel_tag = "-".join(str(value) for value in experiment["channels"])
    return (
        f"{experiment['graph']}__{experiment['task']}__{experiment['embedding']}__"
        f"ch{channel_tag}__seed{experiment['seed']}"
    )


def run_one(experiment, args):
    out_root = resolve_path(args.out_root)
    db_path = resolve_path(args.db)
    summary_path = resolve_path(args.summary)
    run_dir = out_root / "runs" / build_run_dir_name(experiment)
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    elif run_dir.exists():
        raise FileExistsError(f"Output directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(int(experiment["seed"]))
    save_rng_state(run_dir / "rng_state_before.pkl")

    os.environ.setdefault("WANDB_SILENT", "true")
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    os.chdir(SRC_MODEL)
    sys.path.insert(0, str(SRC_MODEL))

    import config as model_config
    import main as model_main

    base_config = copy.deepcopy(model_config.config)
    configure_model(model_config.config, experiment, run_dir, args)
    model_main.config.update(model_config.config)
    model_main.seed = int(experiment["seed"])
    install_dropout_hook(model_main, args.dropout)

    graph_cfg = GRAPHS[experiment["graph"]]
    embedding_cfg = EMBEDDINGS[experiment["embedding"]]
    task_cfg = TASKS[experiment["task"]]
    channel_tag = "-".join(str(value) for value in experiment["channels"])
    run_name = (
        f"{experiment['graph']}__{experiment['task']}__{experiment['embedding']}__"
        f"ch{channel_tag}__seed{experiment['seed']}"
    )
    wandb_config = {
        "study": "initial_embedding_quality_ablation",
        "graph": experiment["graph"],
        "domain": graph_cfg["domain"],
        "embedding": experiment["embedding"],
        "plm_model": embedding_cfg["model"],
        "task": experiment["task"],
        "training_task": task_cfg["training_task"],
        "channels": experiment["channels"],
        "seed": int(experiment["seed"]),
        "encoder": args.encoder,
        "decoder": args.decoder,
        "dropout": args.dropout,
        "attribute_mask_rate": task_cfg.get("attribute_mask_rate"),
        "relation_mask_rate": task_cfg.get("relation_mask_rate"),
        "linear_probe": False,
        "output_dir": str(run_dir),
    }
    install_wandb_patch(model_main, args.wandb_project, run_name, wandb_config)

    start_record = {
        "event": "start",
        "status": "running",
        "time": datetime.now().isoformat(timespec="seconds"),
        **wandb_config,
        "out_dir": str(run_dir),
    }
    append_jsonl(db_path, start_record)
    write_json(run_dir / "run_config.json", model_config.config)
    write_json(run_dir / "experiment.json", start_record)

    try:
        if hasattr(model_main, "_set_all_seeds"):
            model_main._set_all_seeds(int(experiment["seed"]))
        model_main.main()
        save_rng_state(run_dir / "rng_state_after.pkl")
        metrics = extract_metrics(run_dir)
        finish_record = {
            **start_record,
            "event": "finish",
            "status": "completed" if metrics.get("metrics_found") else "completed_no_metrics",
            "time": datetime.now().isoformat(timespec="seconds"),
            "metrics": metrics,
        }
    except Exception as exc:
        finish_record = {
            **start_record,
            "event": "finish",
            "status": "failed",
            "time": datetime.now().isoformat(timespec="seconds"),
            "error": repr(exc),
            "metrics": {"metrics_found": False},
        }
        raise
    finally:
        append_jsonl(db_path, finish_record)
        update_summary_csv(summary_path, finish_record)
        model_config.config.clear()
        model_config.config.update(base_config)


def print_status(experiments, latest):
    print("\n=== Initial Embedding Quality Ablation Status ===\n")
    to_run = 0
    for experiment in experiments:
        task_cfg = TASKS[experiment["task"]]
        record = {
            **experiment,
            "attribute_mask_rate": task_cfg.get("attribute_mask_rate"),
            "relation_mask_rate": task_cfg.get("relation_mask_rate"),
            "encoder": "RotatEGCN_attn",
            "decoder": "MLP",
        }
        done = latest.get(experiment_key(record), {}).get("status") in ("completed", "completed_no_metrics")
        if not done:
            to_run += 1
        print(
            f"{'DONE' if done else 'PENDING':8s} "
            f"{experiment['graph']:14s} {experiment['task']:8s} {experiment['embedding']:12s} "
            f"ch{'-'.join(map(str, experiment['channels']))} seed {experiment['seed']}"
        )
    print(f"\nCompleted: {len(experiments) - to_run}")
    print(f"To run:    {to_run}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", nargs="*", choices=sorted(GRAPHS), default=["biomed", "dbpedia_clean", "dbpedia_tdg"])
    parser.add_argument("--embeddings", nargs="*", choices=sorted(EMBEDDINGS), default=["sentencebert", "biolinkbert", "roberta"])
    parser.add_argument("--tasks", nargs="*", choices=sorted(TASKS), default=["recons_r", "recons_x"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", nargs=2, type=int, default=[384, 384])
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--encoder", default="RotatEGCN_attn")
    parser.add_argument("--decoder", default="MLP")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--wandb-project", default="Initial_Embedding_Quality_Ablation")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE"))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    experiments = build_experiments(args)
    db_path = resolve_path(args.db)
    latest = latest_finish_records(db_path)
    print_status(experiments, latest)
    if args.status_only:
        return

    for idx, experiment in enumerate(experiments, start=1):
        task_cfg = TASKS[experiment["task"]]
        key_record = {
            **experiment,
            "attribute_mask_rate": task_cfg.get("attribute_mask_rate"),
            "relation_mask_rate": task_cfg.get("relation_mask_rate"),
            "encoder": args.encoder,
            "decoder": args.decoder,
        }
        done = latest.get(experiment_key(key_record), {}).get("status") in ("completed", "completed_no_metrics")
        if done and not args.rerun_completed:
            continue
        print(
            f"\n=== Launch {idx}/{len(experiments)} | {experiment['graph']} | "
            f"{experiment['task']} | {experiment['embedding']} ===\n"
        )
        run_one(experiment, args)

    print(f"\nFinished. DB: {resolve_path(args.db)}")
    print(f"Summary: {resolve_path(args.summary)}")


if __name__ == "__main__":
    main()
