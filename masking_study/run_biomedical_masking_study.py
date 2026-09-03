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
WANDB_PROJECT = os.environ.get("MASKING_STUDY_WANDB_PROJECT", "Masking_Study_BioMedical_Recons_R")
RELATION_MASK_RATE = float(os.environ.get("MASKING_STUDY_RELATION_MASK_RATE", "0.3"))
DEFAULT_PLM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PLM_MODEL = os.environ.get("MASKING_STUDY_PLM_MODEL", DEFAULT_PLM_MODEL)
TRAINING_TASK = os.environ.get("MASKING_STUDY_TASK", "Recons_R")
LAMBDA_ONTO = float(os.environ.get("MASKING_STUDY_LAMBDA_ONTO", "0"))
ENCODER_NAME = os.environ.get("MASKING_STUDY_ENCODER", "RotatEGCN_attn")
DECODER_NAME = os.environ.get("MASKING_STUDY_DECODER", "MLP")

SEEDS = [0, 42, 123, 100, 2026]
CHANNELS = [[384, 384], [256, 256], [512, 512]]
SHORT_GRAPH_NAMES = {
    "GT2KG_mapped_and_old_rel_norm": "gt2kg",
    "MM_mapped_nci_All_R_KG": "mm",
}
SHORT_MODE_NAMES = {
    "canonicalized_whole_graph": "c_whole",
    "canonicalized_random_dynamic": "c_rdyn",
    "canonicalized_balanced_dynamic": "c_bdyn",
    "canonicalized_mapped_only": "c_map",
    "canonicalized_mapped_visible": "c_mvis",
    "mapped_only_dynamic_random": "modr",
    "mapped_only_dynamic_balanced": "modb",
    "mapped_selector_old_predicate": "ms_old",
    "mapped_selector_dynamic_random": "msdr",
    "mapped_selector_dynamic_balanced": "msdb",
    "mapped_mix_dynamic_random": "mixr",
    "mapped_mix_dynamic_balanced": "mixb",
    "all_mapped_plus_random_dynamic": "amprd",
    "all_mapped_plus_balanced_dynamic": "ampbd",
    "mapped_context_non_mapped_dynamic_random": "mcnmr",
    "mapped_context_non_mapped_dynamic_balanced": "mcnmb",
    "mapped_biased_dynamic": "mbdyn",
    "mapped_random_dynamic_15_15": "mrd15_15",
    "mapped_random_dynamic_20_10": "mrd20_10",
    "mapped_random_dynamic_20_30": "mrd20_30",
    "mapping_guided_mapped_predicate": "mg_pred",
    "mapping_guided_old_predicate": "mg_old",
    "whole_graph": "whole",
    "random_static": "rstat",
    "balanced_static": "bstat",
    "random_dynamic": "rdyn",
    "balanced_dynamic": "bdyn",
    "recons_x_whole_graph": "rx_whole",
    "recons_x_static": "rx",
    "graphmae_dynamic_x": "gmae_x",
    "struct_node_pagerank_masking": "s_pgr",
    "struct_node_degree_masking": "s_deg",
    "struct_node_learnable_masking": "s_lrn",
    "edge_curriculum_dynamic": "e_cur",
}
NO_MASK_RATE_MODES = {
    "whole_graph",
    "recons_x_whole_graph",
    "canonicalized_whole_graph",
    "canonicalized_mapped_only",
    "canonicalized_mapped_visible",
    "mapped_selector_old_predicate",
    "mapped_only_dynamic_random",
    "mapped_only_dynamic_balanced",
    "mapped_selector_dynamic_random",
    "mapped_selector_dynamic_balanced",
    "mapped_mix_dynamic_random",
    "mapped_mix_dynamic_balanced",
    "all_mapped_plus_random_dynamic",
    "all_mapped_plus_balanced_dynamic",
    "mapped_context_non_mapped_dynamic_random",
    "mapped_context_non_mapped_dynamic_balanced",
}

GRAPHS = {
    "MM_mapped_nci_All_R_KG": {
        "kg_path": "../../data/UMLS/noisy/org/MM_mapped_nci_All_R_KG.json",
        "entities_path": "outputs/MM_mapped_nci_All_R_KG/sentence-transformers_all-MiniLM-L6-v2_entities.pickle",
        "edges_path": "outputs/MM_mapped_nci_All_R_KG/sentence-transformers_all-MiniLM-L6-v2_predicates.pickle",
        "gs_path": "../../data/UMLS/common_nodes.xlsx",
        "group": "baseline",
    },
    "GT2KG_mapped_and_old_rel_norm": {
        "kg_path": "../../data/UMLS/noisy/org/GT2KG_mapped_and_old_rel_norm.json",
        "entities_path": "outputs/GT2KG_mapped_and_old_rel_norm/sentence-transformers_all-MiniLM-L6-v2_entities.pickle",
        "edges_path": "outputs/GT2KG_mapped_and_old_rel_norm/sentence-transformers_all-MiniLM-L6-v2_predicates.pickle",
        "gs_path": "../../data/UMLS/common_nodes.xlsx",
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
    {
        "name": "edge_curriculum_dynamic",
        "recons_r_training_mode": "edge_curriculum_dynamic",
        "target_relation_field": "predicate",
        "description": "Cur-MGAE-style dynamic edge masking guided by current DistMult confidence.",
    },
    {
        "name": "recons_x_whole_graph",
        "training_task": "Recons_X",
        "recons_r_training_mode": "none",
        "target_relation_field": "none",
        "recons_x_feature_masking": False,
        "description": "Whole graph feature reconstruction without feature masking.",
    },
    {
        "name": "recons_x_static",
        "training_task": "Recons_X",
        "recons_r_training_mode": "none",
        "target_relation_field": "none",
        "recons_x_feature_masking": True,
        "description": "Original Recons_X baseline with static feature masking.",
    },
    {
        "name": "graphmae_dynamic_x",
        "training_task": "GraphMAE_Recons_X",
        "recons_r_training_mode": "none",
        "target_relation_field": "none",
        "description": "GraphMAE-style dynamic feature masking with a learned mask token.",
    },
    {
        "name": "struct_node_pagerank_masking",
        "training_task": "GraphMAE_Recons_X",
        "recons_r_training_mode": "none",
        "target_relation_field": "none",
        "graphmae_structure_masking": "pagerank",
        "description": "StructMAE-style dynamic feature masking guided by PageRank node scores.",
    },
    {
        "name": "struct_node_degree_masking",
        "training_task": "GraphMAE_Recons_X",
        "recons_r_training_mode": "none",
        "target_relation_field": "none",
        "graphmae_structure_masking": "degree",
        "description": "StructMAE-style dynamic feature masking guided by node degree scores.",
    },
    {
        "name": "struct_node_learnable_masking",
        "training_task": "GraphMAE_Recons_X",
        "recons_r_training_mode": "none",
        "target_relation_field": "none",
        "graphmae_structure_masking": "learnable",
        "description": "StructMAE-style dynamic feature masking guided by a learnable node scorer.",
    },
]

MAPPING_GUIDED_MODES = [
    {
        "name": "canonicalized_whole_graph",
        "recons_r_training_mode": "all_batch_edges",
        "target_relation_field": "predicate",
        "description": "Canonicalized graph: reconstruct all canonical predicates without masking.",
    },
    {
        "name": "canonicalized_random_dynamic",
        "recons_r_training_mode": "random_dynamic_masked_only",
        "target_relation_field": "predicate",
        "description": "Canonicalized graph: dynamic random masking, reconstruct canonical predicates.",
    },
    {
        "name": "canonicalized_balanced_dynamic",
        "recons_r_training_mode": "balanced_dynamic_masked_only",
        "target_relation_field": "predicate",
        "description": "Canonicalized graph: dynamic balanced masking, reconstruct canonical predicates.",
    },
    {
        "name": "canonicalized_mapped_only",
        "recons_r_training_mode": "mapped_only",
        "target_relation_field": "predicate",
        "description": "Canonicalized graph: mask all mapped edges and reconstruct canonical predicates.",
    },
    {
        "name": "canonicalized_mapped_visible",
        "recons_r_training_mode": "mapped_visible",
        "target_relation_field": "predicate",
        "description": "Canonicalized graph: reconstruct all mapped edges while keeping them visible in message passing.",
    },
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
    {
        "name": "mapped_selector_old_predicate",
        "recons_r_training_mode": "mapped_only",
        "target_relation_field": "old_predicate",
        "description": "Canonicalized graph: use is_mapped only as mask selector, reconstruct original old predicates.",
    },
    {
        "name": "mapped_selector_dynamic_random",
        "recons_r_training_mode": "mapped_selector_dynamic_random",
        "target_relation_field": "old_predicate",
        "description": "Canonicalized graph: dynamically sample mapped edges at mapped_only_dynamic_rate and reconstruct old predicates.",
    },
    {
        "name": "mapped_selector_dynamic_balanced",
        "recons_r_training_mode": "mapped_selector_dynamic_balanced",
        "target_relation_field": "old_predicate",
        "description": "Canonicalized graph: dynamically balanced-sample mapped edges and reconstruct old predicates.",
    },
    {
        "name": "mapped_only_dynamic_random",
        "recons_r_training_mode": "mapped_only_dynamic_random",
        "target_relation_field": "predicate",
        "description": "Canonicalized graph: dynamically mask a fraction of mapped edges, leaving the rest visible.",
    },
    {
        "name": "mapped_only_dynamic_balanced",
        "recons_r_training_mode": "mapped_only_dynamic_balanced",
        "target_relation_field": "predicate",
        "description": "Canonicalized graph: dynamically balanced-mask a fraction of mapped edges, leaving the rest visible.",
    },
    {
        "name": "mapped_mix_dynamic_random",
        "recons_r_training_mode": "mapped_mix_dynamic_random",
        "target_relation_field": "predicate",
        "description": "Canonicalized graph: dynamically mask mapped_rate of mapped edges and non_mapped_rate of other edges, random in both pools.",
    },
    {
        "name": "mapped_mix_dynamic_balanced",
        "recons_r_training_mode": "mapped_mix_dynamic_balanced",
        "target_relation_field": "predicate",
        "description": "Canonicalized graph: dynamically mask mapped_rate of mapped edges and non_mapped_rate of other edges, balanced in both pools.",
    },
    {
        "name": "all_mapped_plus_random_dynamic",
        "recons_r_training_mode": "all_mapped_plus_random_dynamic",
        "target_relation_field": "predicate",
        "description": "Canonicalized graph: mask all mapped edges plus a dynamic random fraction of non-mapped edges.",
    },
    {
        "name": "all_mapped_plus_balanced_dynamic",
        "recons_r_training_mode": "all_mapped_plus_balanced_dynamic",
        "target_relation_field": "predicate",
        "description": "Canonicalized graph: mask all mapped edges plus a dynamic balanced fraction of non-mapped edges.",
    },
    {
        "name": "mapped_context_non_mapped_dynamic_random",
        "recons_r_training_mode": "mapped_context_non_mapped_dynamic_random",
        "target_relation_field": "predicate",
        "description": "Canonicalized graph: keep mapped edges visible and dynamically mask a random fraction of non-mapped edges.",
    },
    {
        "name": "mapped_context_non_mapped_dynamic_balanced",
        "recons_r_training_mode": "mapped_context_non_mapped_dynamic_balanced",
        "target_relation_field": "predicate",
        "description": "Canonicalized graph: keep mapped edges visible and dynamically mask a balanced fraction of non-mapped edges.",
    },
    {
        "name": "mapped_biased_dynamic",
        "recons_r_training_mode": "mapped_biased_dynamic",
        "target_relation_field": "predicate",
        "description": "Canonicalized graph: weighted global dynamic masking with gamma=U(0,1)+beta*is_mapped.",
    },
    {
        "name": "mapped_random_dynamic_15_15",
        "recons_r_training_mode": "mapped_random_dynamic",
        "target_relation_field": "predicate",
        "mapped_random_dynamic_mapped_fraction": 0.5,
        "description": "Dynamic masking with 15% mapped edges and 15% other edges when total_drop_rate=0.3.",
    },
    {
        "name": "mapped_random_dynamic_20_10",
        "recons_r_training_mode": "mapped_random_dynamic",
        "target_relation_field": "predicate",
        "mapped_random_dynamic_mapped_fraction": 2 / 3,
        "description": "Dynamic masking with 20% mapped edges and 10% other edges when total_drop_rate=0.3.",
    },
    {
        "name": "mapped_random_dynamic_20_30",
        "recons_r_training_mode": "mapped_random_dynamic",
        "target_relation_field": "predicate",
        "mapped_random_dynamic_mapped_fraction": 0.4,
        "description": "Dynamic masking with 20% mapped edges and 30% other edges when total_drop_rate=0.5.",
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


def safe_model_tag(model_name):
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in model_name)


def graph_embedding_paths(graph_name, graph_cfg):
    plm_model = getattr(graph_embedding_paths, "plm_model", PLM_MODEL)
    embedding_tag = getattr(graph_embedding_paths, "embedding_tag", os.environ.get("MASKING_STUDY_EMBEDDING_TAG"))
    if plm_model == DEFAULT_PLM_MODEL and not embedding_tag:
        return graph_cfg["entities_path"], graph_cfg["edges_path"]
    tag = embedding_tag or safe_model_tag(plm_model)
    return (
        f"outputs/{graph_name}/{tag}_entities.pickle",
        f"outputs/{graph_name}/{tag}_predicates.pickle",
    )


def build_run_dir_name(exp, stamp, short_run_dirs=None):
    channel_tag = "-".join(str(value) for value in exp["channels"])
    if short_run_dirs is None:
        short_run_dirs = os.environ.get("MASKING_STUDY_SHORT_RUN_DIRS", "").lower() in ("1", "true", "yes")
    if short_run_dirs:
        graph = SHORT_GRAPH_NAMES.get(exp["graph"], exp["graph"])
        mode = SHORT_MODE_NAMES.get(exp["mode"], exp["mode"])
        return f"{graph}__{mode}__c{channel_tag}__s{exp['seed']}__{stamp}"
    return f"{exp['graph']}__{exp['mode']}__ch{channel_tag}__seed{exp['seed']}__{stamp}"


def mode_uses_mask_rate(mode_name):
    return mode_name not in NO_MASK_RATE_MODES


def effective_mask_rate(mode_name, mask_rate):
    return float(mask_rate) if mode_uses_mask_rate(mode_name) else None


def mode_parameter_tag(mode_name, mode_cfg, args):
    def fmt(value):
        return str(value).replace(".", "p")

    if mode_name in (
        "mapped_only_dynamic_random",
        "mapped_only_dynamic_balanced",
        "mapped_selector_dynamic_random",
        "mapped_selector_dynamic_balanced",
    ):
        value = mode_cfg.get("mapped_only_dynamic_rate", args.mapped_only_dynamic_rate)
        return f"__mod{fmt(value)}"
    if mode_name in ("mapped_mix_dynamic_random", "mapped_mix_dynamic_balanced"):
        mapped = mode_cfg.get("mapped_mix_mapped_rate", args.mapped_mix_mapped_rate)
        other = mode_cfg.get("mapped_mix_non_mapped_rate", args.mapped_mix_non_mapped_rate)
        return f"__mix{fmt(mapped)}-{fmt(other)}"
    if mode_name in ("all_mapped_plus_random_dynamic", "all_mapped_plus_balanced_dynamic"):
        value = mode_cfg.get("all_mapped_plus_non_mapped_rate", args.all_mapped_plus_non_mapped_rate)
        return f"__amp{fmt(value)}"
    if mode_name in ("mapped_context_non_mapped_dynamic_random", "mapped_context_non_mapped_dynamic_balanced"):
        value = mode_cfg.get("mapped_context_non_mapped_rate", args.mapped_context_non_mapped_rate)
        return f"__mctx{fmt(value)}"
    if mode_name == "mapped_biased_dynamic":
        value = mode_cfg.get("mapped_biased_beta", args.mapped_biased_beta)
        return f"__beta{fmt(value)}"
    return ""


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
        record.get("training_task"),
        record.get("encoder"),
        record.get("decoder"),
        record.get("dropout"),
        record.get("recons_r_training_mode"),
        record.get("target_relation_field"),
        record.get("mapped_random_dynamic_mapped_fraction"),
        record.get("mapped_only_dynamic_rate"),
        record.get("mapped_mix_mapped_rate"),
        record.get("mapped_mix_non_mapped_rate"),
        record.get("all_mapped_plus_non_mapped_rate"),
        record.get("mapped_context_non_mapped_rate"),
        record.get("mapped_biased_beta"),
        record.get("lambda_domain_range_embedding"),
        record.get("domain_range_embedding_temperature"),
        record.get("edge_curriculum_split_ratio"),
        record.get("edge_curriculum_initial_rate"),
        record.get("edge_curriculum_schedule"),
        record.get("recons_x_feature_masking"),
        record.get("graphmae_structure_masking"),
        record.get("graphmae_structure_alpha"),
        record.get("graphmae_structure_schedule"),
        record.get("run_linear_probe_on_best_loss"),
        record.get("mask_rate"),
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
        "mapped_random_dynamic_mapped_fraction": finish_record.get("mapped_random_dynamic_mapped_fraction"),
        "mapped_only_dynamic_rate": finish_record.get("mapped_only_dynamic_rate"),
        "mapped_mix_mapped_rate": finish_record.get("mapped_mix_mapped_rate"),
        "mapped_mix_non_mapped_rate": finish_record.get("mapped_mix_non_mapped_rate"),
        "all_mapped_plus_non_mapped_rate": finish_record.get("all_mapped_plus_non_mapped_rate"),
        "mapped_context_non_mapped_rate": finish_record.get("mapped_context_non_mapped_rate"),
        "mapped_biased_beta": finish_record.get("mapped_biased_beta"),
        "lambda_domain_range_embedding": finish_record.get("lambda_domain_range_embedding"),
        "domain_range_embedding_temperature": finish_record.get("domain_range_embedding_temperature"),
        "edge_curriculum_split_ratio": finish_record.get("edge_curriculum_split_ratio"),
        "edge_curriculum_initial_rate": finish_record.get("edge_curriculum_initial_rate"),
        "edge_curriculum_schedule": finish_record.get("edge_curriculum_schedule"),
        "recons_x_feature_masking": finish_record.get("recons_x_feature_masking"),
        "graphmae_structure_masking": finish_record.get("graphmae_structure_masking"),
        "graphmae_structure_alpha": finish_record.get("graphmae_structure_alpha"),
        "graphmae_structure_schedule": finish_record.get("graphmae_structure_schedule"),
        "training_task": finish_record.get("training_task"),
        "encoder": finish_record.get("encoder"),
        "decoder": finish_record.get("decoder"),
        "dropout": finish_record.get("dropout"),
        "run_linear_probe_on_best_loss": finish_record.get("run_linear_probe_on_best_loss"),
        "mask_rate": finish_record.get("mask_rate"),
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
    for key, value in metrics.items():
        key = str(key)
        if key.startswith("best_loss_unsup_") or key.startswith("linear_probe_"):
            row[f"metric_{key}"] = value
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
        kwargs["config"] = dict(extra_config)
        return original_init(*args, **kwargs)

    wandb.init = study_init
    if hasattr(model_main, "_ensure_seeded_wandb_init"):
        model_main._ensure_seeded_wandb_init = lambda: None


def install_dropout_hook(model_main, dropout):
    original_encoder = model_main.TransGCNEncoder

    def dropout_encoder(*args, **kwargs):
        kwargs["dropout"] = float(dropout)
        encoder = original_encoder(*args, **kwargs)
        if hasattr(encoder, "dropout"):
            encoder.dropout.p = float(dropout)
        print(f"\nMasking study: TransGCNEncoder dropout forced to {float(dropout)}\n")
        return encoder

    model_main.TransGCNEncoder = dropout_encoder


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


def configure_model(config, experiment, run_dir, args, num_epochs=None):
    graph_cfg = experiment["graph_config"]
    mode_cfg = experiment["mode_config"]
    entities_path, edges_path = graph_embedding_paths(experiment["graph"], graph_cfg)
    training_task = mode_cfg.get("training_task", args.training_task)
    config.update({
        "seed": int(experiment["seed"]),
        "active_seed": int(experiment["seed"]),
        "dataset": experiment["graph"],
        "KG_path": graph_cfg["kg_path"],
        "Entities_path": entities_path,
        "Edges_path": edges_path,
        "plm_embedding_model": args.plm_model,
        "Gs_path_no_other": graph_cfg["gs_path"],
        "training_task": [training_task],
        "recons_r_training_mode": mode_cfg["recons_r_training_mode"],
        "recons_r_target_relation_field": mode_cfg["target_relation_field"],
        "mapped_random_dynamic_mapped_fraction": mode_cfg.get("mapped_random_dynamic_mapped_fraction", 0.5),
        "mapped_only_dynamic_rate": mode_cfg.get("mapped_only_dynamic_rate", args.mapped_only_dynamic_rate),
        "mapped_mix_mapped_rate": mode_cfg.get("mapped_mix_mapped_rate", args.mapped_mix_mapped_rate),
        "mapped_mix_non_mapped_rate": mode_cfg.get("mapped_mix_non_mapped_rate", args.mapped_mix_non_mapped_rate),
        "all_mapped_plus_non_mapped_rate": mode_cfg.get("all_mapped_plus_non_mapped_rate", args.all_mapped_plus_non_mapped_rate),
        "mapped_context_non_mapped_rate": mode_cfg.get("mapped_context_non_mapped_rate", args.mapped_context_non_mapped_rate),
        "mapped_biased_beta": mode_cfg.get("mapped_biased_beta", args.mapped_biased_beta),
        "edge_curriculum_split_ratio": args.edge_curriculum_split_ratio,
        "edge_curriculum_initial_rate": args.edge_curriculum_initial_rate,
        "edge_curriculum_schedule": args.edge_curriculum_schedule,
        "recons_x_feature_masking": mode_cfg.get("recons_x_feature_masking", True),
        "graphmae_mask_rate": args.mask_rate,
        "graphmae_replace_rate": args.graphmae_replace_rate,
        "graphmae_loss_fn": args.graphmae_loss_fn,
        "graphmae_sce_alpha": args.graphmae_sce_alpha,
        "graphmae_decoder_remask": args.graphmae_decoder_remask,
        "graphmae_structure_masking": mode_cfg.get("graphmae_structure_masking", args.graphmae_structure_masking),
        "graphmae_structure_alpha": args.graphmae_structure_alpha,
        "graphmae_structure_schedule": args.graphmae_structure_schedule,
        "graphmae_learnable_scorer_hidden": args.graphmae_learnable_scorer_hidden,
        "run_linear_probe_on_best_loss": args.linear_probe,
        "linear_probe_gs_path": args.linear_probe_gs_path,
        "linear_probe_splits_dir": args.linear_probe_splits_dir,
        "linear_probe_split_seeds": args.linear_probe_split_seeds,
        "linear_probe_epochs": args.linear_probe_epochs,
        "linear_probe_lr": args.linear_probe_lr,
        "linear_probe_weight_decay": args.linear_probe_weight_decay,
        "linear_probe_patience": args.linear_probe_patience,
        "negative_sampling_mode": "uniform",
        "negative_corruption_mode": "entity_only",
        "negative_entity_sampling_scope": "batch",
        "total_drop_rate": args.mask_rate,
        "max_masking_percentage": args.mask_rate,
        "kg_negative_sampling_seed": None,
        "track_kg_negative_sampling": args.track_kg_negative_sampling,
        "kg_negative_tracking_dir": str(run_dir / "negative_sampling"),
        "kg_negative_tracking_max_examples": args.kg_negative_tracking_max_examples,
        "debug_negative_sampling_epochs": args.debug_negative_sampling_epochs,
        "debug_negative_sampling_batches_per_epoch": args.debug_negative_sampling_batches_per_epoch,
        "debug_negative_sampling_path": str(run_dir / "kg_negative_debug_batches.json"),
        "track_onto_negative_sampling": False,
        "replay_kg_negative_sampling": False,
        "replay_onto_negative_sampling": False,
        "num_neighbors": [-1, -1],
        "batch_size": 1024,
        "test_batch_size": 1024,
        "shuffle": False,
        "num_epochs": int(num_epochs) if num_epochs is not None else 50,
        "num_steps": None,
        "lambda_onto": args.lambda_onto,
        "lambda_align": 0,
        "lambda_core_contrastive": 0,
        "lambda_core_align": 0,
        "lambda_domain_range": 0,
        "lambda_domain_range_embedding": args.lambda_domain_range_embedding,
        "domain_range_embedding_temperature": args.domain_range_embedding_temperature,
        "lambda_onto_hierarchy": 0,
        "hyperparams_grid": {"num_bases": [5, 10], "out_channels": [experiment["channels"]]},
        "encoders": [args.encoder],
        "decoders": [args.decoder],
        "message_sens": ["source_to_target"],
        "root_save_dir": str(run_dir / "checkpoints"),
        "wandb_project_name": args.wandb_project,
        "wandb_mode": args.wandb_mode,
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

    configure_model(model_config.config, experiment, out_dir, args, num_epochs=args.num_epochs)
    model_main.config.update(model_config.config)
    model_main.seed = int(args.seed)
    install_dropout_hook(model_main, args.dropout)

    uses_mask_rate = mode_uses_mask_rate(args.mode)
    record_mask_rate = effective_mask_rate(args.mode, args.mask_rate)
    mask_tag = f"__mask{args.mask_rate:g}" if uses_mask_rate else ""
    param_tag = mode_parameter_tag(args.mode, mode_cfg, args)
    run_name = (
        f"{args.graph}__{args.mode}__"
        f"ch{'-'.join(str(value) for value in args.channels)}__seed{args.seed}{mask_tag}{param_tag}"
    )
    wandb_config = {
        "study": "biomedical_recons_r_masking",
        "graph": args.graph,
        "group": graph_cfg["group"],
        "mode": args.mode,
        "description": mode_cfg["description"],
        "channels": args.channels,
        "global_seed": int(args.seed),
        "dropout": args.dropout,
        "encoder": args.encoder,
        "decoder": args.decoder,
        "training_task": mode_cfg.get("training_task", args.training_task),
        "plm_embedding_model": args.plm_model,
        "negative_corruption_mode": "entity_only",
        "negative_entity_sampling_scope": "batch",
        "output_dir": str(out_dir),
    }
    if uses_mask_rate:
        wandb_config["mask_rate"] = args.mask_rate
    if args.mode in ("mapped_only_dynamic_random", "mapped_only_dynamic_balanced",
                     "mapped_selector_dynamic_random", "mapped_selector_dynamic_balanced"):
        wandb_config["mapped_only_dynamic_rate"] = mode_cfg.get("mapped_only_dynamic_rate", args.mapped_only_dynamic_rate)
    if args.mode in ("mapped_mix_dynamic_random", "mapped_mix_dynamic_balanced"):
        wandb_config["mapped_mix_mapped_rate"] = mode_cfg.get("mapped_mix_mapped_rate", args.mapped_mix_mapped_rate)
        wandb_config["mapped_mix_non_mapped_rate"] = mode_cfg.get("mapped_mix_non_mapped_rate", args.mapped_mix_non_mapped_rate)
    if args.mode in ("all_mapped_plus_random_dynamic", "all_mapped_plus_balanced_dynamic"):
        wandb_config["all_mapped_plus_non_mapped_rate"] = mode_cfg.get(
            "all_mapped_plus_non_mapped_rate",
            args.all_mapped_plus_non_mapped_rate,
        )
    if args.mode in ("mapped_context_non_mapped_dynamic_random", "mapped_context_non_mapped_dynamic_balanced"):
        wandb_config["mapped_context_non_mapped_rate"] = mode_cfg.get(
            "mapped_context_non_mapped_rate",
            args.mapped_context_non_mapped_rate,
        )
    if args.mode == "mapped_biased_dynamic":
        wandb_config["mapped_biased_beta"] = mode_cfg.get("mapped_biased_beta", args.mapped_biased_beta)
    if mode_cfg.get("training_task", args.training_task) == "Recons_R_with_onto":
        wandb_config["ontology_reconstruction"] = True
        wandb_config["lambda_onto"] = args.lambda_onto
    if args.lambda_domain_range_embedding != 0:
        wandb_config["lambda_domain_range_embedding"] = args.lambda_domain_range_embedding
        wandb_config["domain_range_embedding_temperature"] = args.domain_range_embedding_temperature
    if mode_cfg.get("training_task") == "Recons_X":
        wandb_config["recons_x_feature_masking"] = mode_cfg.get("recons_x_feature_masking", True)
    if mode_cfg.get("training_task") == "GraphMAE_Recons_X":
        wandb_config.update({
            "graphmae_replace_rate": args.graphmae_replace_rate,
            "graphmae_loss_fn": args.graphmae_loss_fn,
            "graphmae_sce_alpha": args.graphmae_sce_alpha,
            "graphmae_decoder_remask": args.graphmae_decoder_remask,
            "graphmae_structure_masking": mode_cfg.get("graphmae_structure_masking", args.graphmae_structure_masking),
            "graphmae_structure_alpha": args.graphmae_structure_alpha,
            "graphmae_structure_schedule": args.graphmae_structure_schedule,
        })
    if args.mode == "edge_curriculum_dynamic":
        wandb_config.update({
            "edge_curriculum_split_ratio": args.edge_curriculum_split_ratio,
            "edge_curriculum_initial_rate": args.edge_curriculum_initial_rate,
            "edge_curriculum_schedule": args.edge_curriculum_schedule,
        })
    if args.linear_probe:
        wandb_config.update({
            "run_linear_probe_on_best_loss": True,
            "linear_probe_gs_path": args.linear_probe_gs_path,
            "linear_probe_splits_dir": args.linear_probe_splits_dir,
            "linear_probe_split_seeds": args.linear_probe_split_seeds,
            "linear_probe_epochs": args.linear_probe_epochs,
            "linear_probe_lr": args.linear_probe_lr,
            "linear_probe_weight_decay": args.linear_probe_weight_decay,
            "linear_probe_patience": args.linear_probe_patience,
        })
    install_wandb_naming(model_main, args.wandb_project, run_name, wandb_config)

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
        "training_task": mode_cfg.get("training_task", args.training_task),
        "encoder": args.encoder,
        "decoder": args.decoder,
        "description": mode_cfg["description"],
        "recons_r_training_mode": mode_cfg["recons_r_training_mode"],
        "target_relation_field": mode_cfg["target_relation_field"],
        "mapped_only_dynamic_rate": mode_cfg.get("mapped_only_dynamic_rate", args.mapped_only_dynamic_rate),
        "mapped_mix_mapped_rate": mode_cfg.get("mapped_mix_mapped_rate", args.mapped_mix_mapped_rate),
        "mapped_mix_non_mapped_rate": mode_cfg.get("mapped_mix_non_mapped_rate", args.mapped_mix_non_mapped_rate),
        "all_mapped_plus_non_mapped_rate": mode_cfg.get("all_mapped_plus_non_mapped_rate", args.all_mapped_plus_non_mapped_rate),
        "mapped_context_non_mapped_rate": mode_cfg.get("mapped_context_non_mapped_rate", args.mapped_context_non_mapped_rate),
        "mapped_biased_beta": mode_cfg.get("mapped_biased_beta", args.mapped_biased_beta),
        "lambda_domain_range_embedding": args.lambda_domain_range_embedding,
        "domain_range_embedding_temperature": args.domain_range_embedding_temperature,
        "graphmae_structure_masking": mode_cfg.get("graphmae_structure_masking", args.graphmae_structure_masking),
        "graphmae_structure_alpha": args.graphmae_structure_alpha,
        "graphmae_structure_schedule": args.graphmae_structure_schedule,
        "edge_curriculum_split_ratio": args.edge_curriculum_split_ratio,
        "edge_curriculum_initial_rate": args.edge_curriculum_initial_rate,
        "edge_curriculum_schedule": args.edge_curriculum_schedule,
        "recons_x_feature_masking": mode_cfg.get("recons_x_feature_masking", True),
        "run_linear_probe_on_best_loss": args.linear_probe,
        "mask_rate": record_mask_rate,
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
            "training_task": mode_cfg.get("training_task", getattr(print_status, "training_task", None)),
            "encoder": getattr(print_status, "encoder", None),
            "decoder": getattr(print_status, "decoder", None),
            "dropout": getattr(print_status, "dropout", None),
            "recons_r_training_mode": mode_cfg["recons_r_training_mode"],
            "target_relation_field": mode_cfg["target_relation_field"],
            "mapped_only_dynamic_rate": mode_cfg.get("mapped_only_dynamic_rate", getattr(print_status, "mapped_only_dynamic_rate", 0.5)),
            "mapped_mix_mapped_rate": mode_cfg.get("mapped_mix_mapped_rate", getattr(print_status, "mapped_mix_mapped_rate", 0.5)),
            "mapped_mix_non_mapped_rate": mode_cfg.get("mapped_mix_non_mapped_rate", getattr(print_status, "mapped_mix_non_mapped_rate", 0.5)),
            "all_mapped_plus_non_mapped_rate": mode_cfg.get("all_mapped_plus_non_mapped_rate", getattr(print_status, "all_mapped_plus_non_mapped_rate", 0.1)),
            "mapped_context_non_mapped_rate": mode_cfg.get("mapped_context_non_mapped_rate", getattr(print_status, "mapped_context_non_mapped_rate", 1.0)),
            "mapped_biased_beta": mode_cfg.get("mapped_biased_beta", getattr(print_status, "mapped_biased_beta", 1.0)),
            "lambda_domain_range_embedding": getattr(print_status, "lambda_domain_range_embedding", 0.0),
            "domain_range_embedding_temperature": getattr(print_status, "domain_range_embedding_temperature", 0.1),
            "edge_curriculum_split_ratio": getattr(print_status, "edge_curriculum_split_ratio", 0.5),
            "edge_curriculum_initial_rate": getattr(print_status, "edge_curriculum_initial_rate", 0.05),
            "edge_curriculum_schedule": getattr(print_status, "edge_curriculum_schedule", "linear"),
            "recons_x_feature_masking": mode_cfg.get("recons_x_feature_masking", True),
            "graphmae_structure_masking": mode_cfg.get("graphmae_structure_masking", getattr(print_status, "graphmae_structure_masking", "random")),
            "graphmae_structure_alpha": getattr(print_status, "graphmae_structure_alpha", 1.0),
            "graphmae_structure_schedule": getattr(print_status, "graphmae_structure_schedule", "linear"),
            "run_linear_probe_on_best_loss": getattr(print_status, "linear_probe", False),
            "mask_rate": effective_mask_rate(exp["mode"], getattr(print_status, "mask_rate", RELATION_MASK_RATE)),
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
    print_status.encoder = args.encoder
    print_status.decoder = args.decoder
    print_status.dropout = args.dropout
    print_status.training_task = args.training_task
    print_status.graphmae_structure_masking = args.graphmae_structure_masking
    print_status.graphmae_structure_alpha = args.graphmae_structure_alpha
    print_status.graphmae_structure_schedule = args.graphmae_structure_schedule
    print_status.edge_curriculum_split_ratio = args.edge_curriculum_split_ratio
    print_status.edge_curriculum_initial_rate = args.edge_curriculum_initial_rate
    print_status.edge_curriculum_schedule = args.edge_curriculum_schedule
    print_status.linear_probe = args.linear_probe
    print_status.mask_rate = args.mask_rate
    print_status.mapped_only_dynamic_rate = args.mapped_only_dynamic_rate
    print_status.mapped_mix_mapped_rate = args.mapped_mix_mapped_rate
    print_status.mapped_mix_non_mapped_rate = args.mapped_mix_non_mapped_rate
    print_status.all_mapped_plus_non_mapped_rate = args.all_mapped_plus_non_mapped_rate
    print_status.mapped_context_non_mapped_rate = args.mapped_context_non_mapped_rate
    print_status.mapped_biased_beta = args.mapped_biased_beta
    print_status.lambda_domain_range_embedding = args.lambda_domain_range_embedding
    print_status.domain_range_embedding_temperature = args.domain_range_embedding_temperature
    pending = print_status(experiments, db_path)
    plan_path = db_path.with_name(f"{db_path.stem}_plan.json")
    write_json(plan_path, {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "wandb_project": args.wandb_project,
        "common": {
            "domain": "BioMedical",
            "encoder": args.encoder,
            "decoder": args.decoder,
            "negative_corruption_mode": "entity_only",
            "negative_entity_sampling_scope": "batch",
            "batch_size": 1024,
            "training_task": args.training_task,
            "ontology_reconstruction": args.training_task == "Recons_R_with_onto",
            "lambda_onto": args.lambda_onto,
            "plm_embedding_model": args.plm_model,
            "graphmae_mask_rate": args.mask_rate,
            "default_mask_rate": args.mask_rate,
            "graphmae_replace_rate": args.graphmae_replace_rate,
            "graphmae_structure_alpha": args.graphmae_structure_alpha,
            "graphmae_structure_schedule": args.graphmae_structure_schedule,
            "edge_curriculum_split_ratio": args.edge_curriculum_split_ratio,
            "edge_curriculum_initial_rate": args.edge_curriculum_initial_rate,
            "edge_curriculum_schedule": args.edge_curriculum_schedule,
            "mapped_only_dynamic_rate": args.mapped_only_dynamic_rate,
            "mapped_mix_mapped_rate": args.mapped_mix_mapped_rate,
            "mapped_mix_non_mapped_rate": args.mapped_mix_non_mapped_rate,
            "all_mapped_plus_non_mapped_rate": args.all_mapped_plus_non_mapped_rate,
            "mapped_context_non_mapped_rate": args.mapped_context_non_mapped_rate,
            "mapped_biased_beta": args.mapped_biased_beta,
            "lambda_domain_range_embedding": args.lambda_domain_range_embedding,
            "domain_range_embedding_temperature": args.domain_range_embedding_temperature,
            "num_neighbors": [-1, -1],
            "shuffle": False,
            "dropout": args.dropout,
            "run_linear_probe_on_best_loss": args.linear_probe,
            "linear_probe_gs_path": args.linear_probe_gs_path,
            "linear_probe_splits_dir": args.linear_probe_splits_dir,
            "linear_probe_split_seeds": args.linear_probe_split_seeds,
            "linear_probe_epochs": args.linear_probe_epochs,
            "linear_probe_lr": args.linear_probe_lr,
            "linear_probe_weight_decay": args.linear_probe_weight_decay,
            "linear_probe_patience": args.linear_probe_patience,
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
                "mapped_only_dynamic_rate": exp["mode_config"].get("mapped_only_dynamic_rate", args.mapped_only_dynamic_rate),
                "mapped_mix_mapped_rate": exp["mode_config"].get("mapped_mix_mapped_rate", args.mapped_mix_mapped_rate),
                "mapped_mix_non_mapped_rate": exp["mode_config"].get("mapped_mix_non_mapped_rate", args.mapped_mix_non_mapped_rate),
                "all_mapped_plus_non_mapped_rate": exp["mode_config"].get("all_mapped_plus_non_mapped_rate", args.all_mapped_plus_non_mapped_rate),
                "mapped_context_non_mapped_rate": exp["mode_config"].get("mapped_context_non_mapped_rate", args.mapped_context_non_mapped_rate),
                "mapped_biased_beta": exp["mode_config"].get("mapped_biased_beta", args.mapped_biased_beta),
                "lambda_domain_range_embedding": args.lambda_domain_range_embedding,
                "domain_range_embedding_temperature": args.domain_range_embedding_temperature,
                "edge_curriculum_split_ratio": args.edge_curriculum_split_ratio,
                "edge_curriculum_initial_rate": args.edge_curriculum_initial_rate,
                "edge_curriculum_schedule": args.edge_curriculum_schedule,
                "recons_x_feature_masking": exp["mode_config"].get("recons_x_feature_masking", True),
            "graphmae_structure_masking": exp["mode_config"].get("graphmae_structure_masking", args.graphmae_structure_masking),
            "channels": exp["channels"],
            "seed": exp["seed"],
            "run_linear_probe_on_best_loss": args.linear_probe,
            "mask_rate": effective_mask_rate(exp["mode"], args.mask_rate),
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
        "wandb_project": args.wandb_project,
    })

    for exp in experiments:
        mode_cfg = exp["mode_config"]
        key_record = {
            "graph": exp["graph"],
            "mode": exp["mode"],
            "training_task": mode_cfg.get("training_task", args.training_task),
            "encoder": args.encoder,
            "decoder": args.decoder,
            "dropout": args.dropout,
            "recons_r_training_mode": mode_cfg["recons_r_training_mode"],
            "target_relation_field": mode_cfg["target_relation_field"],
            "mapped_only_dynamic_rate": mode_cfg.get("mapped_only_dynamic_rate", args.mapped_only_dynamic_rate),
            "mapped_mix_mapped_rate": mode_cfg.get("mapped_mix_mapped_rate", args.mapped_mix_mapped_rate),
            "mapped_mix_non_mapped_rate": mode_cfg.get("mapped_mix_non_mapped_rate", args.mapped_mix_non_mapped_rate),
            "all_mapped_plus_non_mapped_rate": mode_cfg.get("all_mapped_plus_non_mapped_rate", args.all_mapped_plus_non_mapped_rate),
            "mapped_context_non_mapped_rate": mode_cfg.get("mapped_context_non_mapped_rate", args.mapped_context_non_mapped_rate),
            "mapped_biased_beta": mode_cfg.get("mapped_biased_beta", args.mapped_biased_beta),
            "edge_curriculum_split_ratio": args.edge_curriculum_split_ratio,
            "edge_curriculum_initial_rate": args.edge_curriculum_initial_rate,
            "edge_curriculum_schedule": args.edge_curriculum_schedule,
            "recons_x_feature_masking": mode_cfg.get("recons_x_feature_masking", True),
            "graphmae_structure_masking": mode_cfg.get("graphmae_structure_masking", args.graphmae_structure_masking),
            "graphmae_structure_alpha": args.graphmae_structure_alpha,
            "graphmae_structure_schedule": args.graphmae_structure_schedule,
            "run_linear_probe_on_best_loss": args.linear_probe,
            "mask_rate": effective_mask_rate(exp["mode"], args.mask_rate),
            "channels": exp["channels"],
            "seed": exp["seed"],
        }
        finish = latest.get(experiment_key(key_record))
        if not args.rerun_completed and finish is not None and finish.get("status") in ("completed", "completed_no_metrics"):
            print(f"Skipping completed {exp['graph']} {exp['mode']} {exp['channels']} seed {exp['seed']}")
            continue

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        channel_tag = "-".join(str(value) for value in exp["channels"])
        run_dir = runs_dir / build_run_dir_name(exp, stamp, short_run_dirs=args.short_run_dirs)
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
            "--encoder", args.encoder,
            "--decoder", args.decoder,
            "--dropout", str(args.dropout),
            "--mask-rate", str(args.mask_rate),
            "--wandb-project", args.wandb_project,
            "--training-task", args.training_task,
            "--lambda-onto", str(args.lambda_onto),
            "--plm-model", args.plm_model,
            "--graphmae-replace-rate", str(args.graphmae_replace_rate),
            "--graphmae-loss-fn", args.graphmae_loss_fn,
            "--graphmae-sce-alpha", str(args.graphmae_sce_alpha),
            "--graphmae-structure-masking", mode_cfg.get("graphmae_structure_masking", args.graphmae_structure_masking),
            "--graphmae-structure-alpha", str(args.graphmae_structure_alpha),
            "--graphmae-structure-schedule", args.graphmae_structure_schedule,
            "--edge-curriculum-split-ratio", str(args.edge_curriculum_split_ratio),
            "--edge-curriculum-initial-rate", str(args.edge_curriculum_initial_rate),
            "--edge-curriculum-schedule", args.edge_curriculum_schedule,
            "--mapped-only-dynamic-rate", str(mode_cfg.get("mapped_only_dynamic_rate", args.mapped_only_dynamic_rate)),
            "--mapped-mix-mapped-rate", str(mode_cfg.get("mapped_mix_mapped_rate", args.mapped_mix_mapped_rate)),
            "--mapped-mix-non-mapped-rate", str(mode_cfg.get("mapped_mix_non_mapped_rate", args.mapped_mix_non_mapped_rate)),
            "--all-mapped-plus-non-mapped-rate", str(mode_cfg.get("all_mapped_plus_non_mapped_rate", args.all_mapped_plus_non_mapped_rate)),
            "--mapped-context-non-mapped-rate", str(mode_cfg.get("mapped_context_non_mapped_rate", args.mapped_context_non_mapped_rate)),
            "--mapped-biased-beta", str(mode_cfg.get("mapped_biased_beta", args.mapped_biased_beta)),
            "--lambda-domain-range-embedding", str(args.lambda_domain_range_embedding),
            "--domain-range-embedding-temperature", str(args.domain_range_embedding_temperature),
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
        if args.graphmae_learnable_scorer_hidden is not None:
            command.extend(["--graphmae-learnable-scorer-hidden", str(args.graphmae_learnable_scorer_hidden)])
        if args.graphmae_decoder_remask:
            command.append("--graphmae-decoder-remask")
        else:
            command.append("--no-graphmae-decoder-remask")
        if args.short_run_dirs:
            command.append("--short-run-dirs")
        if args.wandb_mode:
            command.extend(["--wandb-mode", args.wandb_mode])
        if args.embedding_tag:
            command.extend(["--embedding-tag", args.embedding_tag])
        if args.num_epochs is not None:
            command.extend(["--num-epochs", str(args.num_epochs)])

        start_record = {
            "event": "start",
            "status": "running",
            "time": datetime.now().isoformat(timespec="seconds"),
            "graph": exp["graph"],
            "group": exp["graph_config"]["group"],
            "mode": exp["mode"],
            "training_task": mode_cfg.get("training_task", args.training_task),
            "encoder": args.encoder,
            "decoder": args.decoder,
            "dropout": args.dropout,
            "description": mode_cfg["description"],
            "recons_r_training_mode": mode_cfg["recons_r_training_mode"],
            "target_relation_field": mode_cfg["target_relation_field"],
            "mapped_only_dynamic_rate": mode_cfg.get("mapped_only_dynamic_rate", args.mapped_only_dynamic_rate),
            "mapped_mix_mapped_rate": mode_cfg.get("mapped_mix_mapped_rate", args.mapped_mix_mapped_rate),
            "mapped_mix_non_mapped_rate": mode_cfg.get("mapped_mix_non_mapped_rate", args.mapped_mix_non_mapped_rate),
            "all_mapped_plus_non_mapped_rate": mode_cfg.get("all_mapped_plus_non_mapped_rate", args.all_mapped_plus_non_mapped_rate),
            "mapped_context_non_mapped_rate": mode_cfg.get("mapped_context_non_mapped_rate", args.mapped_context_non_mapped_rate),
            "mapped_biased_beta": mode_cfg.get("mapped_biased_beta", args.mapped_biased_beta),
            "edge_curriculum_split_ratio": args.edge_curriculum_split_ratio,
            "edge_curriculum_initial_rate": args.edge_curriculum_initial_rate,
            "edge_curriculum_schedule": args.edge_curriculum_schedule,
            "recons_x_feature_masking": mode_cfg.get("recons_x_feature_masking", True),
            "graphmae_structure_masking": mode_cfg.get("graphmae_structure_masking", args.graphmae_structure_masking),
            "graphmae_structure_alpha": args.graphmae_structure_alpha,
            "graphmae_structure_schedule": args.graphmae_structure_schedule,
            "run_linear_probe_on_best_loss": args.linear_probe,
            "mask_rate": effective_mask_rate(exp["mode"], args.mask_rate),
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
    parser.add_argument("--encoder", default=ENCODER_NAME)
    parser.add_argument("--decoder", default=DECODER_NAME)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mask-rate", type=float, default=RELATION_MASK_RATE)
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE"))
    parser.add_argument("--training-task", default=TRAINING_TASK)
    parser.add_argument("--lambda-onto", type=float, default=LAMBDA_ONTO)
    parser.add_argument("--plm-model", default=PLM_MODEL)
    parser.add_argument("--embedding-tag", default=os.environ.get("MASKING_STUDY_EMBEDDING_TAG"))
    parser.add_argument("--graphmae-replace-rate", type=float, default=float(os.environ.get("MASKING_STUDY_GRAPHMAE_REPLACE_RATE", "0")))
    parser.add_argument("--graphmae-loss-fn", default=os.environ.get("MASKING_STUDY_GRAPHMAE_LOSS_FN", "SCE"))
    parser.add_argument("--graphmae-sce-alpha", type=float, default=float(os.environ.get("MASKING_STUDY_GRAPHMAE_SCE_ALPHA", "3")))
    parser.add_argument("--graphmae-structure-masking", choices=["random", "pagerank", "degree", "learnable"], default=os.environ.get("MASKING_STUDY_GRAPHMAE_STRUCTURE_MASKING", "random"))
    parser.add_argument("--graphmae-structure-alpha", type=float, default=float(os.environ.get("MASKING_STUDY_GRAPHMAE_STRUCTURE_ALPHA", "1.0")))
    parser.add_argument("--graphmae-structure-schedule", choices=["linear", "root", "geometric", "constant", "none"], default=os.environ.get("MASKING_STUDY_GRAPHMAE_STRUCTURE_SCHEDULE", "linear"))
    parser.add_argument("--graphmae-learnable-scorer-hidden", type=int, default=None)
    parser.add_argument("--edge-curriculum-split-ratio", type=float, default=float(os.environ.get("MASKING_STUDY_EDGE_CURRICULUM_SPLIT_RATIO", "0.5")))
    parser.add_argument("--edge-curriculum-initial-rate", type=float, default=float(os.environ.get("MASKING_STUDY_EDGE_CURRICULUM_INITIAL_RATE", "0.05")))
    parser.add_argument("--edge-curriculum-schedule", choices=["linear", "root", "geometric", "constant", "none"], default=os.environ.get("MASKING_STUDY_EDGE_CURRICULUM_SCHEDULE", "linear"))
    parser.add_argument("--mapped-only-dynamic-rate", type=float, default=0.5)
    parser.add_argument("--mapped-mix-mapped-rate", type=float, default=0.5)
    parser.add_argument("--mapped-mix-non-mapped-rate", type=float, default=0.5)
    parser.add_argument("--all-mapped-plus-non-mapped-rate", type=float, default=0.1)
    parser.add_argument("--mapped-context-non-mapped-rate", type=float, default=1.0)
    parser.add_argument("--mapped-biased-beta", type=float, default=1.0)
    parser.add_argument("--lambda-domain-range-embedding", type=float, default=0.0)
    parser.add_argument("--domain-range-embedding-temperature", type=float, default=0.1)
    parser.add_argument("--track-kg-negative-sampling", action="store_true")
    parser.add_argument("--kg-negative-tracking-max-examples", type=int, default=None)
    parser.add_argument("--debug-negative-sampling-epochs", nargs="*", type=int, default=[])
    parser.add_argument("--debug-negative-sampling-batches-per-epoch", type=int, default=0)
    parser.add_argument("--linear-probe", action="store_true")
    parser.add_argument("--linear-probe-gs-path", default="../../data/UMLS/common_nodes.xlsx")
    parser.add_argument("--linear-probe-splits-dir", default="../../data/UMLS/splits/umls_kg_splits")
    parser.add_argument("--linear-probe-split-seeds", nargs="*", type=int, default=[42, 123, 456, 789, 2024])
    parser.add_argument("--linear-probe-epochs", type=int, default=300)
    parser.add_argument("--linear-probe-lr", type=float, default=0.01)
    parser.add_argument("--linear-probe-weight-decay", type=float, default=0.0)
    parser.add_argument("--linear-probe-patience", type=int, default=50)
    parser.add_argument("--graphmae-decoder-remask", dest="graphmae_decoder_remask", action="store_true", default=True)
    parser.add_argument("--no-graphmae-decoder-remask", dest="graphmae_decoder_remask", action="store_false")
    parser.add_argument(
        "--short-run-dirs",
        action="store_true",
        default=os.name == "nt" or os.environ.get("MASKING_STUDY_SHORT_RUN_DIRS", "").lower() in ("1", "true", "yes"),
    )
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    graph_embedding_paths.plm_model = args.plm_model
    graph_embedding_paths.embedding_tag = args.embedding_tag

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
