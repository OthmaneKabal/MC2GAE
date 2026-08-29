import sys
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
from datetime import datetime
from collections import Counter

from networkx.algorithms.connectivity import edge_augmentation

from loss_func import recon_r_loss, sce_loss_fnc, similarity_pair_loss, mse_loss_fnc, contrastive_loss, \
    contrastive_loss_exclude_is, calculate_cluster_assignments, inter_cluster_loss, intra_cluster_loss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'utils')))
from TransGCNDecoder import TransGCNDecoder
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch_geometric.loader import NeighborLoader
from evaluate import evaluate
from utils.ConvENegativeSampling import create_triplet_lookup, generate_negatives, get_positives
from utils.ConvEDataLoader import create_data_loader
from utils.utils import generate_relation_embeddings_tensor, removed_edges_train_test_split, \
    save_model_with_hyperparams, set_seed, save_model, calculate_metrics
from visualization_utils import run_recons_r_with_onto_visualizations
from data_augmentation import relation_based_edge_dropping_balanced, random_edge_dropping
from data_augmentation import view_partial_features_masking
from GraphDataLoader import GraphDataLoader
import torch.nn.functional as F
from TransGCNEncoder import TransGCNEncoder

import pandas as pd
from config import config
from MRGAE import  MRGAE
from tqdm import tqdm
import wandb
import torch
import random
import numpy as np
import torch
import copy
def _first_seed(seed_config):
    if isinstance(seed_config, (list, tuple)):
        return seed_config[0]
    return seed_config


def _resolve_seed(seed_value=None):
    if seed_value is not None:
        return seed_value
    return config.get("active_seed", _first_seed(config["seed"]))


seed = _first_seed(config["seed"])
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
set_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
import torch_geometric.transforms as T


def _split_encoder_output(encoder_output):
    if isinstance(encoder_output, tuple):
        node_embeddings = encoder_output[0]
        relation_embeddings = encoder_output[1] if len(encoder_output) > 1 else None
        return node_embeddings, relation_embeddings
    return encoder_output, None


def _encode(model, data):
    return _split_encoder_output(model.encode(data))


def _encode_nodes(model, data):
    node_embeddings, _ = _encode(model, data)
    return node_embeddings


def _filter_edges(data, edge_mask):
    data.edge_index = data.edge_index[:, edge_mask]
    if hasattr(data, "edge_type") and data.edge_type is not None:
        data.edge_type = data.edge_type[edge_mask]
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data.edge_attr = data.edge_attr[edge_mask]
    if hasattr(data, "edge_is_mapped") and data.edge_is_mapped is not None:
        data.edge_is_mapped = data.edge_is_mapped[edge_mask]
    if hasattr(data, "edge_old_type") and data.edge_old_type is not None:
        data.edge_old_type = data.edge_old_type[edge_mask]
    if hasattr(data, "e_id") and data.e_id is not None:
        data.e_id = data.e_id[edge_mask]
    return data


def _relation_target_attr():
    target = config.get("recons_r_target_relation_field", "predicate")
    if target not in ("predicate", "old_predicate"):
        raise ValueError("recons_r_target_relation_field must be one of: predicate, old_predicate")
    return "edge_old_type" if target == "old_predicate" else "edge_type"


def _apply_relation_target(batch, target_attr, num_relations=None):
    if target_attr == "edge_old_type":
        if not hasattr(batch, "edge_old_type") or batch.edge_old_type is None:
            raise ValueError("old_predicate reconstruction requires data.edge_old_type.")
        batch.edge_type = batch.edge_old_type
    if num_relations is not None:
        batch.num_relations = int(num_relations)
    return batch


def _target_relation_count(data, target_attr):
    if target_attr == "edge_old_type":
        if hasattr(data, "num_old_edge_types"):
            return int(data.num_old_edge_types)
        return int(data.edge_old_type.max().item()) + 1
    if hasattr(data, "num_edge_types"):
        return int(data.num_edge_types)
    return int(data.edge_type.max().item()) + 1


def _target_triplet_lookup(data, target_attr):
    if target_attr == "edge_type":
        return create_triplet_lookup(data)
    target_data = copy.copy(data)
    target_data.edge_type = data.edge_old_type
    return create_triplet_lookup(target_data)


def _full_graph_batch(data):
    batch = copy.copy(data)
    batch.e_id = torch.arange(data.edge_index.size(1), dtype=torch.long, device=data.edge_index.device)
    batch.input_id = torch.arange(data.x.size(0), dtype=torch.long, device=data.x.device)
    batch.n_id = torch.arange(data.x.size(0), dtype=torch.long, device=data.x.device)
    if hasattr(data, "num_edge_types"):
        batch.num_relations = data.num_edge_types
    elif hasattr(data, "edge_type") and data.edge_type is not None:
        batch.num_relations = int(data.edge_type.max().item()) + 1
    return batch


def _sample_recons_r_mask(data, mode, seed_value, device):
    if mode in ("random_static_masked_only", "random_dynamic_masked_only"):
        print("\nUsing random edge masking.\n")
        _, removed_edge_indices, removed_edge_types = random_edge_dropping(
            data, config["total_drop_rate"], random_seed=seed_value
        )
    else:
        print("\nUsing type-balanced edge masking.\n")
        _, removed_edge_indices, removed_edge_types = relation_based_edge_dropping_balanced(
            data, config["total_drop_rate"], max_drop_fraction_per_node=0.3, random_seed=seed_value
        )
    return removed_edge_indices.to(device=device, dtype=torch.long), removed_edge_types.to(device)


def _mapped_edge_indices(data, device):
    if not hasattr(data, "edge_is_mapped") or data.edge_is_mapped is None:
        raise ValueError("mapped_only requires edge_is_mapped loaded from the KG JSON field 'is_mapped'.")
    mapped_mask = data.edge_is_mapped.bool().to(device)
    mapped_indices = torch.where(mapped_mask)[0]
    if mapped_indices.numel() == 0:
        raise ValueError("mapped_only found 0 mapped edges. Check the JSON field 'is_mapped'.")
    return mapped_indices


def _distmult_scores(decoder, z, triplets):
    edge_index = torch.stack((triplets[:, 0], triplets[:, 2]))
    edge_type = triplets[:, 1]
    return decoder.forward(z, edge_index, edge_type)


def _distmult_bce_loss(decoder, z, positive_triplets, negative_triplets):
    pos_scores = _distmult_scores(decoder, z, positive_triplets)
    neg_scores = _distmult_scores(decoder, z, negative_triplets)
    loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
           F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
    return loss, pos_scores, neg_scores


def _relation_alignment_loss(kg_r_decoder, onto_r_decoder, kg_relation_ids, onto_relation_ids, relation_projector,
                             loss_type="cosine"):
    kg_relation_ids = kg_relation_ids.to(kg_r_decoder.relation_embedding.device)
    onto_relation_ids = onto_relation_ids.to(onto_r_decoder.relation_embedding.device)
    kg_rel_embeddings = kg_r_decoder.relation_embedding[kg_relation_ids]
    onto_rel_embeddings = onto_r_decoder.relation_embedding[onto_relation_ids]
    projected_kg_rel_embeddings = relation_projector(kg_rel_embeddings)
    if loss_type == "mse":
        projected_kg_rel_embeddings = F.normalize(projected_kg_rel_embeddings, p=2, dim=1)
        onto_rel_embeddings = F.normalize(onto_rel_embeddings, p=2, dim=1)
        return F.mse_loss(projected_kg_rel_embeddings, onto_rel_embeddings)
    if loss_type == "cosine":
        return 1 - F.cosine_similarity(projected_kg_rel_embeddings, onto_rel_embeddings, dim=1).mean()
    raise ValueError(f"Unknown relation_alignment_loss: {loss_type}")


def _paired_core_contrastive_loss(kg_embeddings, onto_embeddings, temperature=0.2):
    if kg_embeddings.size(0) < 2:
        return torch.tensor(0.0, device=kg_embeddings.device)
    kg_embeddings = F.normalize(kg_embeddings, p=2, dim=1)
    onto_embeddings = F.normalize(onto_embeddings, p=2, dim=1)
    logits = kg_embeddings @ onto_embeddings.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    kg_to_onto = F.cross_entropy(logits, labels)
    onto_to_kg = F.cross_entropy(logits.t(), labels)
    return 0.5 * (kg_to_onto + onto_to_kg)


def _paired_core_alignment_loss(kg_embeddings, onto_embeddings, core_projector=None, loss_type="mse"):
    if core_projector is not None:
        kg_embeddings = core_projector(kg_embeddings)
    kg_embeddings = F.normalize(kg_embeddings, p=2, dim=1)
    onto_embeddings = F.normalize(onto_embeddings, p=2, dim=1)
    if loss_type == "mse":
        return F.mse_loss(kg_embeddings, onto_embeddings)
    if loss_type == "cosine":
        return 1 - F.cosine_similarity(kg_embeddings, onto_embeddings, dim=1).mean()
    raise ValueError(f"Unknown core_alignment_loss: {loss_type}")


def _ontology_hierarchy_similarity_loss(onto_embeddings, child_ids, parent_ids):
    if child_ids is None or parent_ids is None or child_ids.numel() == 0:
        return torch.tensor(0.0, device=onto_embeddings.device)
    child_embeddings = onto_embeddings[child_ids]
    parent_embeddings = onto_embeddings[parent_ids]
    return 1 - F.cosine_similarity(child_embeddings, parent_embeddings, dim=1).mean()


def _domain_range_constraint_loss(node_embeddings, positive_triplets, type_embeddings,
                                  domain_mask_by_relation, range_mask_by_relation,
                                  temperature=0.2, eps=1e-8):
    if positive_triplets.numel() == 0 or type_embeddings is None:
        return torch.tensor(0.0, device=node_embeddings.device)

    relation_ids = positive_triplets[:, 1]
    valid_relation_mask = relation_ids < domain_mask_by_relation.size(0)
    if valid_relation_mask.sum() == 0:
        return torch.tensor(0.0, device=node_embeddings.device)

    positive_triplets = positive_triplets[valid_relation_mask]
    relation_ids = positive_triplets[:, 1]
    domain_masks = domain_mask_by_relation[relation_ids]
    range_masks = range_mask_by_relation[relation_ids]
    constrained_mask = (domain_masks.sum(dim=1) > 0) & (range_masks.sum(dim=1) > 0)
    if constrained_mask.sum() == 0:
        return torch.tensor(0.0, device=node_embeddings.device)

    positive_triplets = positive_triplets[constrained_mask]
    domain_masks = domain_masks[constrained_mask]
    range_masks = range_masks[constrained_mask]

    normalized_nodes = F.normalize(node_embeddings, p=2, dim=1)
    normalized_types = F.normalize(type_embeddings, p=2, dim=1)
    logits = normalized_nodes @ normalized_types.t() / max(temperature, eps)
    type_probs = F.softmax(logits, dim=1)

    head_probs = type_probs[positive_triplets[:, 0]]
    tail_probs = type_probs[positive_triplets[:, 2]]
    domain_scores = (head_probs * domain_masks).sum(dim=1)
    range_scores = (tail_probs * range_masks).sum(dim=1)
    compatibility_scores = domain_scores * range_scores
    return -torch.log(compatibility_scores + eps).mean()


def _extract_node_embeddings_by_global_id(node_embeddings, batch_n_id, global_ids):
    selected = []
    for global_id in global_ids:
        matches = torch.where(batch_n_id == global_id)[0]
        if matches.numel() == 0:
            return None
        selected.append(node_embeddings[matches[0]])
    return torch.stack(selected)


def _calculate_relation_micro_metrics(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="micro", zero_division=0)
    recall = recall_score(true_labels, predictions, average="micro", zero_division=0)
    f1 = f1_score(true_labels, predictions, average="micro", zero_division=0)
    return accuracy, precision, recall, f1


def _max_steps():
    return config.get("num_steps")


def _step_limit_reached(global_step):
    num_steps = _max_steps()
    return num_steps is not None and global_step >= num_steps


def _wandb_classification_metrics(metrics):
    logged_metrics = {
        "accuracy": metrics["accuracy"],
        "f1-score": metrics["f1_score"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
    }
    for average in ["macro", "micro", "weighted"]:
        logged_metrics[f"f1-{average}"] = metrics.get(f"f1_{average}")
        logged_metrics[f"precision-{average}"] = metrics.get(f"precision_{average}")
        logged_metrics[f"recall-{average}"] = metrics.get(f"recall_{average}")
    logged_metrics["f1-average"] = metrics.get("f1_weighted")
    logged_metrics["precision-average"] = metrics.get("precision_weighted")
    logged_metrics["recall-average"] = metrics.get("recall_weighted")
    return {key: value for key, value in logged_metrics.items() if value is not None}


def _results_file_path():
    seed = config.get("active_seed", config.get("seed"))
    return os.path.join(config["root_save_dir"], f"results_seed_{seed}.xlsx")


def _excel_safe_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, dict)):
        return str(value)
    return value


def _upsert_result_row(row):
    results_path = _results_file_path()
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    row = {key: _excel_safe_value(value) for key, value in row.items()}

    if os.path.exists(results_path):
        try:
            df = pd.read_excel(results_path)
        except Exception as exc:
            print(f"Could not read existing results file {results_path}: {exc}")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if not df.empty and {"exp_name", "seed"}.issubset(df.columns):
        mask = (df["exp_name"] == row["exp_name"]) & (df["seed"] == row["seed"])
        df = df.loc[~mask]

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    try:
        df.to_excel(results_path, index=False)
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = results_path.replace(".xlsx", f"_autosave_{timestamp}.xlsx")
        df.to_excel(backup_path, index=False)
        print(f"Results file is locked, autosaved to: {backup_path}")


def _record_best_result(metrics, save_file, epoch, status="abnormal"):
    seed = config.get("active_seed", config.get("seed"))
    row = dict(metrics)
    row.update({
        "exp_name": save_file,
        "seed": seed,
        "status": status,
        "exp_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_epoch": epoch + 1,
        "num_epochs": config.get("num_epochs"),
        "num_steps": config.get("num_steps"),
    })
    _upsert_result_row(row)
    return row


def _finalize_best_result(best_metrics, save_file):
    if not best_metrics:
        return best_metrics
    row = dict(best_metrics)
    row["exp_name"] = save_file
    row["status"] = "normal"
    row["exp_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _upsert_result_row(row)
    return row


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _evaluate_preserving_train_rng(model, data, gs_path, core_concepts, gdp, config):
    rng_state = _capture_rng_state()
    try:
        return evaluate(model, data, gs_path, core_concepts, gdp, config)
    finally:
        _restore_rng_state(rng_state)


def _new_negative_tracking_state():
    return {
        "num_positive": 0,
        "num_negative": 0,
        "head_corruptions": 0,
        "tail_corruptions": 0,
        "relation_corruptions": 0,
        "positive_relation_counts": Counter(),
        "negative_relation_counts": Counter(),
        "corrupted_to_relation_counts": Counter(),
        "examples": [],
        "negative_triplets": [],
    }


def _relation_name_map(gdp):
    return {idx: name for name, idx in getattr(gdp, "predicate_to_id", {}).items()}


def _node_name_map(gdp):
    if hasattr(gdp, "decode_indexes"):
        return gdp.decode_indexes()
    return {idx: name for name, idx in getattr(gdp, "nodes_index", {}).items()}


def _global_node_id(local_id, node_ids):
    if node_ids is None:
        return int(local_id)
    return int(node_ids[int(local_id)].item())


def _triplet_to_record(triplet, node_ids, node_names, relation_names):
    h, r, t = [int(value) for value in triplet]
    global_h = _global_node_id(h, node_ids)
    global_t = _global_node_id(t, node_ids)
    return {
        "h_id": global_h,
        "h": node_names.get(global_h, str(global_h)),
        "r_id": r,
        "r": relation_names.get(r, str(r)),
        "t_id": global_t,
        "t": node_names.get(global_t, str(global_t)),
    }


def _update_negative_tracking_state(state, positives, negatives, batch, gdp, max_examples=50):
    if positives is None or negatives is None or positives.numel() == 0 or negatives.numel() == 0:
        return

    positives_cpu = positives.detach().cpu()
    negatives_cpu = negatives.detach().cpu()
    pair_count = min(positives_cpu.size(0), negatives_cpu.size(0))
    positives_cpu = positives_cpu[:pair_count]
    negatives_cpu = negatives_cpu[:pair_count]
    node_ids = batch.n_id.detach().cpu() if hasattr(batch, "n_id") else None
    relation_names = _relation_name_map(gdp)
    node_names = _node_name_map(gdp)

    state["num_positive"] += int(pair_count)
    state["num_negative"] += int(pair_count)
    state["head_corruptions"] += int((positives_cpu[:, 0] != negatives_cpu[:, 0]).sum().item())
    state["relation_corruptions"] += int((positives_cpu[:, 1] != negatives_cpu[:, 1]).sum().item())
    state["tail_corruptions"] += int((positives_cpu[:, 2] != negatives_cpu[:, 2]).sum().item())
    state["positive_relation_counts"].update(int(r) for r in positives_cpu[:, 1].tolist())
    state["negative_relation_counts"].update(int(r) for r in negatives_cpu[:, 1].tolist())

    relation_changed = positives_cpu[:, 1] != negatives_cpu[:, 1]
    state["corrupted_to_relation_counts"].update(
        int(r) for r in negatives_cpu[relation_changed, 1].tolist()
    )

    remaining_examples = pair_count if max_examples is None else max(0, int(max_examples) - len(state["examples"]))
    if remaining_examples > 0:
        for idx in range(min(pair_count, remaining_examples)):
            pos = positives_cpu[idx].tolist()
            neg = negatives_cpu[idx].tolist()
            changed = []
            if pos[0] != neg[0]:
                changed.append("head")
            if pos[1] != neg[1]:
                changed.append("relation")
            if pos[2] != neg[2]:
                changed.append("tail")
            state["examples"].append({
                "changed": changed,
                "positive": _triplet_to_record(pos, node_ids, node_names, relation_names),
                "negative": _triplet_to_record(neg, node_ids, node_names, relation_names),
            })

    for idx in range(pair_count):
        state["negative_triplets"].append(
            _triplet_to_record(negatives_cpu[idx].tolist(), node_ids, node_names, relation_names)
        )


def _counter_to_named_top(counter, name_map, top_k=15):
    return [
        {"id": int(key), "name": name_map.get(int(key), str(key)), "count": int(value)}
        for key, value in counter.most_common(top_k)
    ]


def _write_negative_tracking_epoch(state, output_path, epoch, metrics, gdp):
    record = {
        "epoch": int(epoch),
        "accuracy": float(metrics.get("accuracy", 0.0)) if metrics else None,
        "f1_macro": float(metrics.get("f1_score", 0.0)) if metrics else None,
        "examples": state["examples"],
        "negative_triplets": state["negative_triplets"],
    }
    with open(output_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_negative_replay(path, source_name="KG"):
    if not path:
        raise ValueError(f"replay_{source_name.lower()}_negative_sampling=True but the replay path is empty.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{source_name} negative replay file not found: {path}")

    replay_by_epoch = {}
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            epoch = int(record["epoch"])
            negatives = record.get("negative_triplets", [])
            if not negatives:
                raise ValueError(f"No negative_triplets found at line {line_number} in {path}")
            replay_by_epoch[epoch] = negatives
    if not replay_by_epoch:
        raise ValueError(f"No replay epochs found in {path}")
    return replay_by_epoch


def _replay_negative_triplets(records, cursor, count, batch, device, epoch, source_name="KG"):
    end = cursor + int(count)
    if end > len(records):
        raise ValueError(
            f"{source_name} negative replay epoch {epoch} has only {len(records)} negatives, "
            f"but the training loop requested {end}."
        )

    if hasattr(batch, "n_id"):
        node_ids = batch.n_id.detach().cpu().tolist()
        local_by_global = {int(global_id): local_id for local_id, global_id in enumerate(node_ids)}
    else:
        local_by_global = None

    replayed = []
    for record in records[cursor:end]:
        global_h = int(record["h_id"])
        global_t = int(record["t_id"])
        relation = int(record["r_id"])
        if local_by_global is None:
            local_h = global_h
            local_t = global_t
        else:
            if global_h not in local_by_global or global_t not in local_by_global:
                raise ValueError(
                    f"{source_name} negative replay mismatch at epoch {epoch}: "
                    f"negative ({record.get('h')}, {record.get('r')}, {record.get('t')}) "
                    "contains a node that is not present in the current batch. "
                    "Use the same seed, batch_size, shuffle, num_neighbors, and graph/masking settings as the tracked run."
                )
            local_h = local_by_global[global_h]
            local_t = local_by_global[global_t]
        replayed.append((local_h, relation, local_t))
    return torch.tensor(replayed, dtype=torch.long, device=device), end






def evaluate_ConvE(model, data, data_loader, test_removed_index, device, relation_embeddings):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Evaluation", unit="batch") as eval_pbar:
            for batch in data_loader:
                batch = batch.to(device)
                removed_batch = copy.copy(batch)

                # Masking the edges based on test_removed_index
                test_removed_index = test_removed_index.to(device)
                mask = torch.isin(batch.e_id, test_removed_index)
                _filter_edges(removed_batch, mask)

                # Encoding the batch
                H_2 = _encode_nodes(model, batch)

                # Generating negative and positive triplets for evaluation
                negative_triplets = generate_negatives(data, removed_batch, negative_ratio=1)
                positive_triplets = get_positives(removed_batch)

                # DataLoader for ConvE evaluation
                eval_loader = create_data_loader(
                    positive_triplets,
                    negative_triplets,
                    H_2,
                    relation_embeddings,
                    batch_size=config["batch_size"] * 3,
                    shuffle=False,
                    seed=_resolve_seed()
                )

                convE_loss = 0
                convE_batches_processed = 0

                for eval_batch in eval_loader:
                    preds = model.r_decoder(eval_batch[0], eval_batch[1], eval_batch[2])
                    loss = model.recon_r_loss(preds, eval_batch[3].to(device))
                    convE_loss += loss.item()
                    predicted_labels = (preds > 0.5).long().detach()
                    all_preds.extend(predicted_labels.cpu().numpy())
                    all_labels.extend(eval_batch[3].cpu().numpy())
                    convE_batches_processed += 1

                total_loss += convE_loss / convE_batches_processed
                eval_pbar.update(1)

    avg_loss = total_loss / len(data_loader)
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
    }
    for average in ["macro", "micro", "weighted"]:
        metrics[f"recall_{average}"] = recall_score(all_labels, all_preds, average=average, zero_division=0)
        metrics[f"precision_{average}"] = precision_score(all_labels, all_preds, average=average, zero_division=0)
        metrics[f"f1_{average}"] = f1_score(all_labels, all_preds, average=average, zero_division=0)
    metrics["recall"] = metrics["recall_macro"]
    metrics["precision"] = metrics["precision_macro"]
    metrics["f1_score"] = metrics["f1_macro"]

    print(
        f"\nEvaluation Results - Loss: {avg_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}, F1: {metrics['f1_score']:.4f}")

    return avg_loss, metrics

def train_GAE(model, data, optimizer, num_epochs, gdp,save_file,
                           save_dir="GAE", device = "cuda", wandb = None, seed = 42):
    seed = _resolve_seed(seed)

    best_loss = float('inf')
    best_accuracy = 0
    best_metrics = {}
    set_seed(seed)
    G_data_loader = GraphDataLoader(data, num_neighbors=config["num_neighbors"],
                                    batch_size=config["batch_size"], shuffle=config["shuffle"], seed=seed).get_loader()
    G_data_loader.edge_attr = data.edge_attr
    total_loss = 0
    transform_directed_without_split = T.Compose([
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=False,
                          split_labels=True, add_negative_train_samples=True),
    ])
    print("Negative_sampling ....\n")
    train_data_directed_without_split, val_data_directed_without_split, test_data_directed_without_split = transform_directed_without_split(
        data)
    global_step = 0

    for epoch in range(num_epochs):
        if _step_limit_reached(global_step):
            break
        model.train()

        with tqdm(total=len(G_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
            z = _encode_nodes(model, data)
            loss = model.recon_loss(z, train_data_directed_without_split.pos_edge_label_index)
            loss.backward()
            optimizer.step()
            global_step += 1
            total_loss += loss.item()
            batch_pbar.set_postfix(batch_loss=loss.item())
            batch_pbar.update(1)
            avg_loss = total_loss / len(G_data_loader)
            print("\nEvaluation\n")
            print(data)
            # metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
            metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp, config)

            print("\n")
            print(metrics)
            print("\n")
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(model, optimizer, epoch, save_dir=save_dir, file_name = save_file, is_best_acc=False)


                print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                save_model(model, optimizer, epoch, save_dir=save_dir, file_name=save_file, is_best_acc=False)

                print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                best_metrics = _record_best_result(metrics, save_file, epoch, status="abnormal")
            wandb.log({"epoch": epoch + 1, "step": global_step, "loss": avg_loss,
                       **_wandb_classification_metrics(metrics)})
    return _finalize_best_result(best_metrics, save_file)




def train_DisMult(model, data, optimizer,num_epochs,gdp, save_file,device,
                           save_dir="train_R_reconstruction", wandb = None, seed = None,
                           removed_edge_indices=None, removed_edge_types=None,
                           onto_data=None, onto_r_decoder=None, lambda_onto=1.0,
                           relation_projector=None, kg_relation_align_ids=None,
                           onto_relation_align_ids=None, lambda_align=0.0,
                           relation_alignment_loss="cosine",
                           onto_gdp=None, shared_relations=None, visualizations_dir=None,
                           kg_core_ids=None, onto_core_ids=None,
                           lambda_core_contrastive=0.0, core_contrastive_temperature=0.2,
                           lambda_core_align=0.0, core_alignment_loss="mse",
                           core_projector=None,
                           domain_range_type_ids=None, domain_mask_by_relation=None,
                           range_mask_by_relation=None, lambda_domain_range=0.0,
                           domain_range_temperature=0.2,
                           onto_hierarchy_child_ids=None, onto_hierarchy_parent_ids=None,
                           lambda_onto_hierarchy=0.0,
                           negative_sampling_mode="uniform", soft_type_candidates=None,
                           soft_type_negative_ratio=0.7, negative_corruption_mode="mixed"):
    seed = _resolve_seed(seed)
    best_loss = float('inf')
    best_F1 = 0
    best_accuracy = 0
    best_metrics = {}
    best_epoch = 0
    best_visual_state = None
    recons_r_training_mode = config.get("recons_r_training_mode", "all_batch_edges")
    relation_target_attr = _relation_target_attr()
    target_num_relations = _target_relation_count(data, relation_target_attr)
    track_kg_negative_sampling = bool(config.get("track_kg_negative_sampling", False))
    negative_tracking_max_examples = config.get("kg_negative_tracking_max_examples", 50)
    if negative_tracking_max_examples is not None:
        negative_tracking_max_examples = int(negative_tracking_max_examples)
    track_onto_negative_sampling = bool(config.get("track_onto_negative_sampling", False))
    onto_negative_tracking_max_examples = config.get("onto_negative_tracking_max_examples", 50)
    if onto_negative_tracking_max_examples is not None:
        onto_negative_tracking_max_examples = int(onto_negative_tracking_max_examples)
    replay_kg_negative_sampling = bool(config.get("replay_kg_negative_sampling", False))
    kg_negative_replay_path = config.get("kg_negative_replay_path")
    replay_onto_negative_sampling = bool(config.get("replay_onto_negative_sampling", False))
    onto_negative_replay_path = config.get("onto_negative_replay_path")
    kg_negative_sampling_seed = config.get("kg_negative_sampling_seed")
    kg_negative_rng = random.Random(kg_negative_sampling_seed) if kg_negative_sampling_seed is not None else None
    negative_entity_sampling_scope = config.get("negative_entity_sampling_scope", "batch")
    if negative_entity_sampling_scope not in ("batch", "global"):
        raise ValueError("negative_entity_sampling_scope must be one of: batch, global")
    negative_tracking_path = None
    onto_negative_tracking_path = None
    negative_replay_by_epoch = None
    onto_negative_replay_by_epoch = None
    if replay_kg_negative_sampling:
        negative_replay_by_epoch = _load_negative_replay(kg_negative_replay_path, source_name="KG")
        print(
            f"\nReplaying KG negative sampling from: {kg_negative_replay_path} "
            f"({len(negative_replay_by_epoch)} epochs loaded).\n"
        )
    if replay_onto_negative_sampling:
        onto_negative_replay_by_epoch = _load_negative_replay(onto_negative_replay_path, source_name="ontology")
        print(
            f"\nReplaying ontology negative sampling from: {onto_negative_replay_path} "
            f"({len(onto_negative_replay_by_epoch)} epochs loaded).\n"
        )
    if track_kg_negative_sampling:
        tracking_dir = config.get("kg_negative_tracking_dir", "analysis/negative_sampling")
        os.makedirs(tracking_dir, exist_ok=True)
        safe_save_file = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in save_file)
        neg_seed_suffix = f"_kgneg{kg_negative_sampling_seed}" if kg_negative_sampling_seed is not None else ""
        negative_tracking_path = os.path.join(
            tracking_dir,
            f"{safe_save_file}_seed{seed}_{negative_corruption_mode}{neg_seed_suffix}.jsonl",
        )
        if os.path.exists(negative_tracking_path):
            os.remove(negative_tracking_path)
        print(f"\nTracking KG negative sampling per epoch: {negative_tracking_path}\n")
    if track_onto_negative_sampling:
        tracking_dir = config.get("onto_negative_tracking_dir", "analysis/negative_sampling")
        os.makedirs(tracking_dir, exist_ok=True)
        safe_save_file = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in save_file)
        onto_negative_tracking_path = os.path.join(
            tracking_dir,
            f"{safe_save_file}_seed{seed}_onto_mixed.jsonl",
        )
        if os.path.exists(onto_negative_tracking_path):
            os.remove(onto_negative_tracking_path)
        print(f"\nTracking ontology negative sampling per epoch: {onto_negative_tracking_path}\n")
    if kg_negative_rng is not None:
        print(f"\nUsing separate KG negative sampling seed: {kg_negative_sampling_seed}\n")
    print(f"\nUsing KG negative corruption mode: {negative_corruption_mode}\n")
    print(f"\nUsing KG negative entity sampling scope: {negative_entity_sampling_scope}\n")
    print(f"\nUsing relation reconstruction target: {config.get('recons_r_target_relation_field', 'predicate')}\n")
    if recons_r_training_mode in ("mapped_only", "mapped_visible"):
        print("\nUsing mapped relation mask from JSON field 'is_mapped'...\n")
        removed_edge_indices = _mapped_edge_indices(data, device)
        removed_edge_types = data.edge_type[removed_edge_indices].to(device)
    elif removed_edge_indices is None:
        if recons_r_training_mode in ("all_batch_edges",):
            print("\nReconstructing whole graph without masking.\n")
            removed_edge_indices = torch.empty(0, dtype=torch.long, device=device)
            removed_edge_types = torch.empty(0, dtype=torch.long, device=device)
        elif recons_r_training_mode in ("removed_only", "random_masked_only", "balanced_static_masked_only"):
            removed_edge_indices, removed_edge_types = _sample_recons_r_mask(data, "balanced_static_masked_only", seed, device)
        elif recons_r_training_mode == "random_static_masked_only":
            removed_edge_indices, removed_edge_types = _sample_recons_r_mask(data, recons_r_training_mode, seed, device)
        elif recons_r_training_mode in ("random_dynamic_masked_only", "balanced_dynamic_masked_only"):
            removed_edge_indices, removed_edge_types = _sample_recons_r_mask(data, recons_r_training_mode, seed, device)
        else:
            raise ValueError(f"Unknown recons_r_training_mode: {recons_r_training_mode}")
    else:
        print("\nUsing precomputed relation mask...\n")
    removed_edge_indices = removed_edge_indices.to(device)
    if removed_edge_types is not None:
        removed_edge_types = removed_edge_types.to(device)

    if relation_target_attr == "edge_old_type":
        print("\nReconstructing old_predicate labels while keeping graph message passing predicates unchanged.\n")
    triplet_set = _target_triplet_lookup(data, relation_target_attr)
    if negative_sampling_mode == "soft_type_aware" and soft_type_candidates is not None:
        print(
            f"\nUsing soft type-aware negative sampling "
            f"(ratio={soft_type_negative_ratio}).\n"
        )
    use_onto = (
        onto_data is not None and onto_r_decoder is not None and
        (lambda_onto != 0 or lambda_align != 0 or lambda_core_contrastive != 0 or
         lambda_core_align != 0 or lambda_domain_range != 0 or lambda_onto_hierarchy != 0)
    )
    if replay_onto_negative_sampling and not use_onto:
        raise ValueError("replay_onto_negative_sampling=True but ontology training is disabled.")
    if use_onto:
        onto_data = onto_data.to(device)
        onto_r_decoder = onto_r_decoder.to(device)
        onto_triplet_set = create_triplet_lookup(onto_data)
        onto_positive_triplets = get_positives(onto_data)
    use_relation_alignment = (
        use_onto and relation_projector is not None and
        kg_relation_align_ids is not None and onto_relation_align_ids is not None and
        lambda_align != 0
    )
    if use_relation_alignment:
        relation_projector = relation_projector.to(device)
        kg_relation_align_ids = kg_relation_align_ids.to(device)
        onto_relation_align_ids = onto_relation_align_ids.to(device)
        print(f"\nAligning {kg_relation_align_ids.numel()} shared relation embeddings KG -> ontology.\n")
    use_core_contrastive = (
        use_onto and kg_core_ids is not None and onto_core_ids is not None and
        lambda_core_contrastive != 0
    )
    use_core_alignment = (
        use_onto and kg_core_ids is not None and onto_core_ids is not None and
        lambda_core_align != 0
    )
    if use_core_contrastive or use_core_alignment:
        kg_core_ids = kg_core_ids.to(device)
        onto_core_ids = onto_core_ids.to(device)
        if core_projector is not None:
            core_projector = core_projector.to(device)
        core_batch_size = int(kg_core_ids.numel())
        core_data_loader = GraphDataLoader(
            data,
            num_neighbors=config["num_neighbors"],
            batch_size=core_batch_size,
            shuffle=False,
            seed=seed,
            input_nodes=kg_core_ids.detach().cpu(),
        ).get_loader()
        core_data_iter = iter(core_data_loader)
        if use_core_contrastive:
            print(f"\nUsing core contrastive loss on {core_batch_size} KG/Onto type pairs.\n")
        if use_core_alignment:
            print(f"\nUsing core alignment loss on {core_batch_size} KG/Onto type pairs.\n")
    use_domain_range_constraints = (
        use_onto and domain_range_type_ids is not None and domain_mask_by_relation is not None and
        range_mask_by_relation is not None and lambda_domain_range != 0
    )
    if use_domain_range_constraints:
        domain_range_type_ids = domain_range_type_ids.to(device)
        domain_mask_by_relation = domain_mask_by_relation.to(device)
        range_mask_by_relation = range_mask_by_relation.to(device)
        print(f"\nUsing domain/range constraint loss with {domain_range_type_ids.numel()} ontology type prototypes.\n")
    use_onto_hierarchy = (
        use_onto and onto_hierarchy_child_ids is not None and onto_hierarchy_parent_ids is not None and
        lambda_onto_hierarchy != 0
    )
    if use_onto_hierarchy:
        onto_hierarchy_child_ids = onto_hierarchy_child_ids.to(device)
        onto_hierarchy_parent_ids = onto_hierarchy_parent_ids.to(device)
        print(f"\nUsing ontology hierarchy similarity loss on {onto_hierarchy_child_ids.numel()} isa edges.\n")

    set_seed(seed)
    if negative_entity_sampling_scope == "global":
        print("\nUsing full-graph batches so entity corruption is sampled from the whole graph.\n")
        G_data_loader = None
    else:
        G_data_loader = GraphDataLoader(data, num_neighbors=config["num_neighbors"],
                                        batch_size=config["batch_size"], shuffle=config["shuffle"], seed=seed).get_loader()
        G_data_loader.edge_attr = data.edge_attr
    set_seed(seed)
    global_step = 0

    for epoch in range(num_epochs):
        if _step_limit_reached(global_step):
            break
        if recons_r_training_mode in ("random_dynamic_masked_only", "balanced_dynamic_masked_only"):
            dynamic_seed = int(seed) + epoch
            removed_edge_indices, removed_edge_types = _sample_recons_r_mask(
                data, recons_r_training_mode, dynamic_seed, device
            )
        model.train()
        total_loss = 0
        all_preds = []
        all_true_labels = []
        steps_this_epoch = 0
        total_kg_loss = 0
        total_onto_loss = 0
        total_align_loss = 0
        total_core_contrastive_loss = 0
        total_core_align_loss = 0
        total_domain_range_loss = 0
        total_onto_hierarchy_loss = 0
        negative_tracking_state = _new_negative_tracking_state() if track_kg_negative_sampling else None
        onto_negative_tracking_state = _new_negative_tracking_state() if track_onto_negative_sampling and use_onto else None
        negative_replay_records = None
        negative_replay_cursor = 0
        onto_negative_replay_records = None
        onto_negative_replay_cursor = 0
        if negative_replay_by_epoch is not None:
            epoch_key = epoch + 1
            if epoch_key not in negative_replay_by_epoch:
                raise ValueError(f"KG negative replay file has no negatives for epoch {epoch_key}.")
            negative_replay_records = negative_replay_by_epoch[epoch_key]
        if onto_negative_replay_by_epoch is not None:
            epoch_key = epoch + 1
            if epoch_key not in onto_negative_replay_by_epoch:
                raise ValueError(f"Ontology negative replay file has no negatives for epoch {epoch_key}.")
            onto_negative_replay_records = onto_negative_replay_by_epoch[epoch_key]
        epoch_loader = [_full_graph_batch(data)] if negative_entity_sampling_scope == "global" else G_data_loader
        with tqdm(total=len(epoch_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as main_pbar:

            for G2_batch in epoch_loader:

                G2_batch = G2_batch.to(device)
                removed_batch = copy.copy(G2_batch)

                removed_edge_indices = removed_edge_indices.to(device=device, dtype=torch.long)
                mask = torch.isin(removed_edge_indices, G2_batch.e_id)
                intersections = removed_edge_indices[mask]
                # Obtenir les nœuds cibles des arêtes intersectantes avec le batch graph
                intersection_targets = data.edge_index[1][intersections]
                # Trouver les intersections qui vérifient la condition
                # (les nœuds cibles sont dans input_id)
                matching_mask = torch.isin(intersection_targets, G2_batch.input_id)
                # Récupérer les e_id correspondants
                batch_matching_e_ids = intersections[matching_mask]
                edges_mask = torch.isin(G2_batch.e_id,batch_matching_e_ids) ## mask pour les edges à supprimer dans le batch

                ## the final masked batch
                if recons_r_training_mode != "mapped_visible":
                    _filter_edges(G2_batch, ~edges_mask)
                _filter_edges(removed_batch, edges_mask)
                _apply_relation_target(G2_batch, relation_target_attr, target_num_relations)
                _apply_relation_target(removed_batch, relation_target_attr, target_num_relations)
                optimizer.zero_grad()

                H_2 = _encode_nodes(model, G2_batch)

                if removed_batch.edge_index.size(1) == 0 and recons_r_training_mode != "all_batch_edges":
                    main_pbar.update(1)
                    continue

                if recons_r_training_mode in (
                    "removed_only", "random_masked_only", "balanced_static_masked_only",
                    "random_static_masked_only", "random_dynamic_masked_only",
                    "balanced_dynamic_masked_only", "mapped_only", "mapped_visible"
                ):
                    all_positive_triplets = get_positives(removed_batch)
                    if negative_replay_records is not None:
                        all_negative_triplets, negative_replay_cursor = _replay_negative_triplets(
                            negative_replay_records,
                            negative_replay_cursor,
                            all_positive_triplets.size(0),
                            G2_batch,
                            device,
                            epoch + 1,
                        )
                    else:
                        all_negative_triplets = generate_negatives(
                            data, removed_batch, negative_ratio=1, triplet_set=triplet_set,
                            negative_sampling_mode=negative_sampling_mode,
                            soft_type_candidates=soft_type_candidates,
                            soft_type_negative_ratio=soft_type_negative_ratio,
                            negative_corruption_mode=negative_corruption_mode,
                            negative_entity_sampling_scope=negative_entity_sampling_scope,
                            rng=kg_negative_rng,
                        )
                elif recons_r_training_mode == "all_batch_edges":
                    positive_triplets = get_positives(G2_batch)
                    if removed_batch.edge_index.size(1) == 0:
                        positive_triplets_removed = None
                        all_positive_triplets = positive_triplets
                    else:
                        positive_triplets_removed = get_positives(removed_batch)
                        all_positive_triplets = torch.cat((positive_triplets, positive_triplets_removed), dim=0)
                    if negative_replay_records is not None:
                        all_negative_triplets, negative_replay_cursor = _replay_negative_triplets(
                            negative_replay_records,
                            negative_replay_cursor,
                            all_positive_triplets.size(0),
                            G2_batch,
                            device,
                            epoch + 1,
                        )
                    else:
                        negative_triplets = generate_negatives(
                            data, G2_batch, negative_ratio=1, triplet_set=triplet_set,
                            negative_sampling_mode=negative_sampling_mode,
                            soft_type_candidates=soft_type_candidates,
                            soft_type_negative_ratio=soft_type_negative_ratio,
                            negative_corruption_mode=negative_corruption_mode,
                            negative_entity_sampling_scope=negative_entity_sampling_scope,
                            rng=kg_negative_rng,
                        )
                        if removed_batch.edge_index.size(1) == 0:
                            all_negative_triplets = negative_triplets
                        else:
                            negative_triplets_removed = generate_negatives(
                                data, removed_batch, negative_ratio=1, triplet_set=triplet_set,
                                negative_sampling_mode=negative_sampling_mode,
                                soft_type_candidates=soft_type_candidates,
                                soft_type_negative_ratio=soft_type_negative_ratio,
                                negative_corruption_mode=negative_corruption_mode,
                                negative_entity_sampling_scope=negative_entity_sampling_scope,
                                rng=kg_negative_rng,
                            )
                            all_negative_triplets = torch.cat((negative_triplets, negative_triplets_removed), dim=0)
                else:
                    raise ValueError(f"Unknown recons_r_training_mode: {recons_r_training_mode}")
                if negative_tracking_state is not None:
                    _update_negative_tracking_state(
                        negative_tracking_state,
                        all_positive_triplets,
                        all_negative_triplets,
                        G2_batch,
                        gdp,
                        max_examples=negative_tracking_max_examples,
                    )
                kg_loss, pos_scores, neg_scores = _distmult_bce_loss(
                    model.r_decoder, H_2, all_positive_triplets, all_negative_triplets
                )
                if use_onto:
                    H_onto = _encode_nodes(model, onto_data)
                    if onto_negative_replay_records is not None:
                        onto_negative_triplets, onto_negative_replay_cursor = _replay_negative_triplets(
                            onto_negative_replay_records,
                            onto_negative_replay_cursor,
                            onto_positive_triplets.size(0),
                            onto_data,
                            device,
                            epoch + 1,
                            source_name="ontology",
                        )
                    else:
                        onto_negative_triplets = generate_negatives(
                            onto_data, onto_data, negative_ratio=1, triplet_set=onto_triplet_set,
                            negative_entity_sampling_scope=negative_entity_sampling_scope,
                        )
                    if onto_negative_tracking_state is not None:
                        _update_negative_tracking_state(
                            onto_negative_tracking_state,
                            onto_positive_triplets,
                            onto_negative_triplets,
                            onto_data,
                            onto_gdp,
                            max_examples=onto_negative_tracking_max_examples,
                        )
                    onto_loss, _, _ = _distmult_bce_loss(
                        onto_r_decoder, H_onto, onto_positive_triplets, onto_negative_triplets
                    )
                else:
                    onto_loss = torch.tensor(0.0, device=device)
                if use_relation_alignment:
                    align_loss = _relation_alignment_loss(
                        model.r_decoder, onto_r_decoder,
                        kg_relation_align_ids, onto_relation_align_ids, relation_projector,
                        loss_type=relation_alignment_loss
                    )
                else:
                    align_loss = torch.tensor(0.0, device=device)
                if use_core_contrastive or use_core_alignment:
                    try:
                        core_batch = next(core_data_iter)
                    except StopIteration:
                        core_data_iter = iter(core_data_loader)
                        core_batch = next(core_data_iter)
                    core_batch = core_batch.to(device)
                    H_kg_core_batch = _encode_nodes(model, core_batch)
                    kg_core_embeddings = _extract_node_embeddings_by_global_id(
                        H_kg_core_batch, core_batch.n_id, kg_core_ids
                    )
                    if kg_core_embeddings is None:
                        core_contrastive_loss = torch.tensor(0.0, device=device)
                        core_align_loss = torch.tensor(0.0, device=device)
                    else:
                        onto_core_embeddings = H_onto[onto_core_ids]
                        if use_core_contrastive:
                            core_contrastive_loss = _paired_core_contrastive_loss(
                                kg_core_embeddings, onto_core_embeddings,
                                temperature=core_contrastive_temperature
                            )
                        else:
                            core_contrastive_loss = torch.tensor(0.0, device=device)
                        if use_core_alignment:
                            core_align_loss = _paired_core_alignment_loss(
                                kg_core_embeddings, onto_core_embeddings,
                                core_projector=core_projector,
                                loss_type=core_alignment_loss
                            )
                        else:
                            core_align_loss = torch.tensor(0.0, device=device)
                else:
                    core_contrastive_loss = torch.tensor(0.0, device=device)
                    core_align_loss = torch.tensor(0.0, device=device)
                if use_domain_range_constraints:
                    domain_range_loss = _domain_range_constraint_loss(
                        H_2,
                        all_positive_triplets,
                        H_onto[domain_range_type_ids],
                        domain_mask_by_relation,
                        range_mask_by_relation,
                        temperature=domain_range_temperature,
                    )
                else:
                    domain_range_loss = torch.tensor(0.0, device=device)
                if use_onto_hierarchy:
                    onto_hierarchy_loss = _ontology_hierarchy_similarity_loss(
                        H_onto,
                        onto_hierarchy_child_ids,
                        onto_hierarchy_parent_ids,
                    )
                else:
                    onto_hierarchy_loss = torch.tensor(0.0, device=device)
                pos_preds = (torch.sigmoid(pos_scores) > 0.55).int()
                neg_preds = (torch.sigmoid(neg_scores) > 0.55).int()

                # True labels
                pos_labels = torch.ones_like(pos_preds)
                neg_labels = torch.zeros_like(neg_preds)

                # Collect predictions and true labels
                all_preds.extend(pos_preds.cpu().numpy())
                all_preds.extend(neg_preds.cpu().numpy())
                all_true_labels.extend(pos_labels.cpu().numpy())
                all_true_labels.extend(neg_labels.cpu().numpy())
                loss = kg_loss + lambda_onto * onto_loss + lambda_align * align_loss + \
                       lambda_core_contrastive * core_contrastive_loss + \
                       lambda_core_align * core_align_loss + \
                       lambda_domain_range * domain_range_loss + \
                       lambda_onto_hierarchy * onto_hierarchy_loss
                loss.backward()
                optimizer.step()
                global_step += 1
                steps_this_epoch += 1
                total_loss += loss.item()
                total_kg_loss += kg_loss.item()
                total_onto_loss += onto_loss.item()
                total_align_loss += align_loss.item()
                total_core_contrastive_loss += core_contrastive_loss.item()
                total_core_align_loss += core_align_loss.item()
                total_domain_range_loss += domain_range_loss.item()
                total_onto_hierarchy_loss += onto_hierarchy_loss.item()
                main_pbar.update(1)
                if _step_limit_reached(global_step):
                    break

            if negative_replay_records is not None and not _step_limit_reached(global_step):
                if negative_replay_cursor != len(negative_replay_records):
                    raise ValueError(
                        f"KG negative replay epoch {epoch + 1} consumed {negative_replay_cursor} negatives "
                        f"but the replay file contains {len(negative_replay_records)}. "
                        "Use the exact same training/masking/batching settings as the tracked run."
                    )
                print(f"KG negative replay epoch {epoch + 1}: reused {negative_replay_cursor} negatives.")
            if onto_negative_replay_records is not None and not _step_limit_reached(global_step):
                if onto_negative_replay_cursor != len(onto_negative_replay_records):
                    raise ValueError(
                        f"Ontology negative replay epoch {epoch + 1} consumed {onto_negative_replay_cursor} negatives "
                        f"but the replay file contains {len(onto_negative_replay_records)}. "
                        "Use the exact same ontology graph and KG batching settings as the tracked run."
                    )
                print(
                    f"Ontology negative replay epoch {epoch + 1}: "
                    f"reused {onto_negative_replay_cursor} negatives."
                )

            avg_loss = total_loss / max(steps_this_epoch, 1)
            avg_kg_loss = total_kg_loss / max(steps_this_epoch, 1)
            avg_onto_loss = total_onto_loss / max(steps_this_epoch, 1)
            avg_align_loss = total_align_loss / max(steps_this_epoch, 1)
            avg_core_contrastive_loss = total_core_contrastive_loss / max(steps_this_epoch, 1)
            avg_core_align_loss = total_core_align_loss / max(steps_this_epoch, 1)
            avg_domain_range_loss = total_domain_range_loss / max(steps_this_epoch, 1)
            avg_onto_hierarchy_loss = total_onto_hierarchy_loss / max(steps_this_epoch, 1)
            print("Evaluation\n")
            # metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
            metrics = _evaluate_preserving_train_rng(
                model, data, config["Gs_path_no_other"], config["core_concepts"], gdp, config
            )

            print("\n")
            print(metrics)

            R_accuracy, R_precision, R_recall, R_f1 = _calculate_relation_micro_metrics(all_preds, all_true_labels)
            print(f"R_accuracy: {R_accuracy}, R_precision: {R_precision}, R_recall: {R_recall},R_f1: {R_f1}")
            if negative_tracking_state is not None and negative_tracking_path is not None:
                _write_negative_tracking_epoch(
                    negative_tracking_state,
                    negative_tracking_path,
                    epoch + 1,
                    metrics,
                    gdp,
                )
                print(f"KG negative sampling tracking updated: {negative_tracking_path}")
            if onto_negative_tracking_state is not None and onto_negative_tracking_path is not None:
                _write_negative_tracking_epoch(
                    onto_negative_tracking_state,
                    onto_negative_tracking_path,
                    epoch + 1,
                    metrics,
                    onto_gdp,
                )
                print(f"Ontology negative sampling tracking updated: {onto_negative_tracking_path}")
            if use_onto:
                print(f"KG_loss: {avg_kg_loss}, Onto_loss: {avg_onto_loss}, lambda_onto: {lambda_onto}")
            if use_relation_alignment:
                print(f"Align_loss: {avg_align_loss}, lambda_align: {lambda_align}, type: {relation_alignment_loss}")
            if use_core_contrastive:
                print(
                    f"Core_contrastive_loss: {avg_core_contrastive_loss}, "
                    f"lambda_core_contrastive: {lambda_core_contrastive}, "
                    f"temperature: {core_contrastive_temperature}"
                )
            if use_core_alignment:
                print(
                    f"Core_align_loss: {avg_core_align_loss}, "
                    f"lambda_core_align: {lambda_core_align}, "
                    f"type: {core_alignment_loss}"
                )
            if use_domain_range_constraints:
                print(
                    f"Domain_range_loss: {avg_domain_range_loss}, "
                    f"lambda_domain_range: {lambda_domain_range}, "
                    f"temperature: {domain_range_temperature}"
                )
            if use_onto_hierarchy:
                print(
                    f"Onto_hierarchy_loss: {avg_onto_hierarchy_loss}, "
                    f"lambda_onto_hierarchy: {lambda_onto_hierarchy}"
                )

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(model, optimizer, epoch, save_dir = save_dir, file_name = save_file, is_best_acc=False)
                print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_visual_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "onto_r_decoder": copy.deepcopy(onto_r_decoder.state_dict()) if onto_r_decoder is not None else None,
                    "relation_projector": copy.deepcopy(relation_projector.state_dict()) if relation_projector is not None else None,
                    "core_projector": copy.deepcopy(core_projector.state_dict()) if core_projector is not None else None,
                }
                best_metrics = _record_best_result({
                    **metrics,
                    "R_accuracy": R_accuracy,
                    "R_precision": R_precision,
                    "R_recall": R_recall,
                    "R_f1": R_f1,
                }, save_file, epoch, status="abnormal")
                save_model(model, optimizer, epoch, save_dir=save_dir,file_name= save_file , is_best_acc=True)
                print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
            wandb.log({"epoch": epoch + 1, "step": global_step, "loss": avg_loss,
                       "kg_loss": avg_kg_loss, "onto_loss": avg_onto_loss,
                       "align_loss": avg_align_loss,
                       "core_contrastive_loss": avg_core_contrastive_loss,
                       "core_align_loss": avg_core_align_loss,
                       "domain_range_loss": avg_domain_range_loss,
                       "onto_hierarchy_loss": avg_onto_hierarchy_loss,
                       **_wandb_classification_metrics(metrics),
                       "R_accuracy": R_accuracy, "R_precision": R_precision,
                       "R_recall": R_recall, "R_f1": R_f1,})

    if use_onto and visualizations_dir is not None:
        if best_visual_state is not None:
            model.load_state_dict(best_visual_state["model"])
            if onto_r_decoder is not None and best_visual_state["onto_r_decoder"] is not None:
                onto_r_decoder.load_state_dict(best_visual_state["onto_r_decoder"])
            if relation_projector is not None and best_visual_state["relation_projector"] is not None:
                relation_projector.load_state_dict(best_visual_state["relation_projector"])
            if core_projector is not None and best_visual_state["core_projector"] is not None:
                core_projector.load_state_dict(best_visual_state["core_projector"])

        run_recons_r_with_onto_visualizations(
            model=model,
            data=data,
            gdp=gdp,
            onto_data=onto_data,
            onto_gdp=onto_gdp,
            onto_r_decoder=onto_r_decoder,
            relation_projector=relation_projector,
            kg_relation_align_ids=kg_relation_align_ids,
            onto_relation_align_ids=onto_relation_align_ids,
            shared_relations=shared_relations or [],
            core_concepts=config["core_concepts"],
            output_dir=visualizations_dir,
            prefix=save_file,
        )

    return _finalize_best_result(best_metrics, save_file)


def train_DisMult_with_onto(model, data, onto_data, onto_r_decoder, optimizer, num_epochs, gdp, save_file, device,
                            save_dir="train_R_reconstruction_with_onto", wandb=None, seed=None,
                            removed_edge_indices=None, removed_edge_types=None, lambda_onto=1.0,
                            relation_projector=None, kg_relation_align_ids=None,
                            onto_relation_align_ids=None, lambda_align=0.0,
                            relation_alignment_loss="cosine",
                            onto_gdp=None, shared_relations=None, visualizations_dir=None,
                            kg_core_ids=None, onto_core_ids=None,
                            lambda_core_contrastive=0.0, core_contrastive_temperature=0.2,
                            lambda_core_align=0.0, core_alignment_loss="mse",
                            core_projector=None,
                            domain_range_type_ids=None, domain_mask_by_relation=None,
                            range_mask_by_relation=None, lambda_domain_range=0.0,
                            domain_range_temperature=0.2,
                            onto_hierarchy_child_ids=None, onto_hierarchy_parent_ids=None,
                            lambda_onto_hierarchy=0.0,
                            negative_sampling_mode="uniform", soft_type_candidates=None,
                            soft_type_negative_ratio=0.7, negative_corruption_mode="mixed"):
    return train_DisMult(
        model, data, optimizer, num_epochs, gdp, save_file, device,
        save_dir=save_dir, wandb=wandb, seed=seed,
        removed_edge_indices=removed_edge_indices, removed_edge_types=removed_edge_types,
        onto_data=onto_data, onto_r_decoder=onto_r_decoder, lambda_onto=lambda_onto,
        relation_projector=relation_projector,
        kg_relation_align_ids=kg_relation_align_ids,
        onto_relation_align_ids=onto_relation_align_ids,
        lambda_align=lambda_align,
        relation_alignment_loss=relation_alignment_loss,
        onto_gdp=onto_gdp,
        shared_relations=shared_relations,
        visualizations_dir=visualizations_dir,
        kg_core_ids=kg_core_ids,
        onto_core_ids=onto_core_ids,
        lambda_core_contrastive=lambda_core_contrastive,
        core_contrastive_temperature=core_contrastive_temperature,
        lambda_core_align=lambda_core_align,
        core_alignment_loss=core_alignment_loss,
        core_projector=core_projector,
        domain_range_type_ids=domain_range_type_ids,
        domain_mask_by_relation=domain_mask_by_relation,
        range_mask_by_relation=range_mask_by_relation,
        lambda_domain_range=lambda_domain_range,
        domain_range_temperature=domain_range_temperature,
        onto_hierarchy_child_ids=onto_hierarchy_child_ids,
        onto_hierarchy_parent_ids=onto_hierarchy_parent_ids,
        lambda_onto_hierarchy=lambda_onto_hierarchy,
        negative_sampling_mode=negative_sampling_mode,
        soft_type_candidates=soft_type_candidates,
        soft_type_negative_ratio=soft_type_negative_ratio,
        negative_corruption_mode=negative_corruption_mode,
    )



def train_X_reconstruction(model, data ,optimizer, num_epochs, gdp, save_file,device, config,loss_fct = ["MSE"],
                           save_dir="train_X_reconstruction", wandb = None, seed = None):
    seed = _resolve_seed(seed)


    best_loss = float('inf')
    best_F1 = 0
    best_accuracy = 0
    best_metrics = {}
    best_epoch = 0
    print("\nmask_features...\n")
    masked_features_data = view_partial_features_masking(data, max_masking_percentage=config["max_masking_percentage"],
                                                         random_seed=seed)
    set_seed(seed)
    G1_data_loader = GraphDataLoader(masked_features_data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"], seed=seed).get_loader()
    G1_data_loader.edge_attr = masked_features_data.edge_attr
    set_seed(seed)
    global_step = 0
    for epoch in range(num_epochs):
        if _step_limit_reached(global_step):
            break
        model.train()
        total_loss = 0
        if "MSE" in loss_fct:
            total_mse_loss = 0
        if "PCSE" in loss_fct:
            total_cos_loss = 0
        if "SCE" in loss_fct:
            total_sce_loss = 0
        steps_this_epoch = 0

        print("\nMSE_Recons_X\n")
        with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}",
                  unit="batch") as batch_pbar:
            for batch in G1_data_loader:

                batch = batch.to(device)
                n_id = batch.n_id  ## The global node index for every sampled node
                mask = torch.isin(n_id, batch.input_id)  ## mask to get only the embedding of input_id nodes
                optimizer.zero_grad()
                embeddings, r_embd = _encode(model, batch)

                # embeddings = model.encode(batch)
                if isinstance(model.x_decoder, TransGCNDecoder):

                    reconstructed_x = model.decode_x(batch, embeddings, r_embd)
                else:
                    reconstructed_x = model.decode_x(batch, embeddings)

                reconstructed_x = reconstructed_x[mask]

                total_loss = 0.0

                # Vérifier chaque terme et ajouter le loss correspondant au total
                if "MSE" in loss_fct:
                    mse_loss = mse_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    total_loss += mse_loss

                if "PCSE" in loss_fct:
                    pcse_loss = similarity_pair_loss(data.x[n_id[mask]], reconstructed_x, embeddings[mask])
                    total_loss += pcse_loss

                if "SCE" in loss_fct:
                    sce_loss = sce_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    total_loss += sce_loss

                loss = total_loss
                loss.backward()
                optimizer.step()
                global_step += 1
                steps_this_epoch += 1
                total_loss += loss.item()
                batch_pbar.set_postfix(batch_loss=loss.item())
                batch_pbar.update(1)
                if _step_limit_reached(global_step):
                    break
            avg_loss = total_loss / max(steps_this_epoch, 1)

            print("Evaluation\n")
            metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp, config)
            print("\n")
            print(metrics)
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(model, optimizer, epoch, save_dir = save_dir, file_name = save_file, is_best_acc=False)

                print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_metrics = _record_best_result(metrics, save_file, epoch, status="abnormal")
                # best_epoch = epoch
                # save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                #                             is_best_acc=True)
                save_model(model, optimizer, epoch, save_dir=save_dir,file_name= save_file , is_best_acc=True)
                print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
            wandb.log({"epoch": epoch + 1, "step": global_step, "mce loss": avg_loss,
                       **_wandb_classification_metrics(metrics)})

    return _finalize_best_result(best_metrics, save_file)


def train_Contrastive(model, data, optimizer, num_epochs, gdp, save_file,
                      masked_features_data, removed_edge_indices,
                      device="cuda", save_dir="contrastive_training",
                      wandb=None, seed=None):
    import copy
    seed = _resolve_seed(seed)
    set_seed(seed)
    best_loss = float('inf')
    best_accuracy = 0
    best_metrics = {}

    #print("\n--- Preparing views for contrastive learning ---\n")
    #masked_features_data = view_partial_features_masking(data, max_masking_percentage=config["max_masking_percentage"])
    #masked_edges_data, removed_edge_indices, _ = relation_based_edge_dropping_balanced(
    #    data, config["total_drop_rate"], max_drop_fraction_per_node=0.3, random_seed=42
    #)

    removed_edge_indices = removed_edge_indices.to(device)

    G_data_loader = GraphDataLoader(data, num_neighbors=config["num_neighbors"],
                                    batch_size=config["batch_size"], shuffle=config["shuffle"], seed=seed).get_loader()
    global_step = 0

    for epoch in range(num_epochs):
        if _step_limit_reached(global_step):
            break
        model.train()
        total_loss = 0
        steps_this_epoch = 0

        with tqdm(total=len(G_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for batch in G_data_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                # View 1: masking features
                view_1 = copy.deepcopy(batch)
                view_1.x = masked_features_data.x[view_1.n_id]

                # View 2: masking edges
                view_2 = copy.deepcopy(batch)
                edge_mask = ~torch.isin(view_2.e_id, removed_edge_indices)
                _filter_edges(view_2, edge_mask)

                # Masquer les input_id
                mask_nodes = torch.isin(batch.n_id, batch.input_id)

                # Encodage + projection
                H1 = _encode_nodes(model, view_1)
                H2 = _encode_nodes(model, view_2)
                ##
                if not isinstance(mask_nodes, torch.Tensor):
                    mask_nodes = torch.tensor(mask_nodes)

                # Si mask_nodes est un masque booléen
                if mask_nodes.dtype == torch.bool:
                    mask_nodes = mask_nodes.to(H1.device)

                # Sinon on suppose que c'est une liste d'indices
                else:
                    mask_nodes = mask_nodes.long().to(H1.device)

                ###

                # Appliquer les projecteurs
                z1 = model.projector_fc1(H1[mask_nodes])
                z2 = model.projector_fc2(H2[mask_nodes])

                # Calcul de la perte contrastive standard
                c_loss = contrastive_loss(z1, z2)

                c_loss.backward()
                optimizer.step()
                global_step += 1
                steps_this_epoch += 1
                total_loss += c_loss.item()

                pbar.set_postfix(loss=c_loss.item())
                pbar.update(1)
                if _step_limit_reached(global_step):
                    break

        avg_loss = total_loss / max(steps_this_epoch, 1)

        print("\n--- Evaluation ---")
        metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp, config)
        print(metrics)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, optimizer, epoch, save_dir=save_dir, file_name=save_file, is_best_acc=False)
            print(f"Model saved with lowest contrastive loss: {best_loss:.4f}")

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_metrics = _record_best_result(metrics, save_file, epoch, status="abnormal")
            save_model(model, optimizer, epoch, save_dir=save_dir, file_name=save_file, is_best_acc=True)
            print(f"Model saved with best accuracy: {best_accuracy:.4f}")

        if wandb is not None:
            wandb.log({
                "epoch": epoch + 1,
                "step": global_step,
                "contrastive_loss": avg_loss,
                **_wandb_classification_metrics(metrics)
            })

    return _finalize_best_result(best_metrics, save_file)


def train_Double_Reconstruction(model, data, optimizer,num_epochs,gdp, save_file,device, loss_fct = ["MSE"],
                           save_dir="train_R_reconstruction", wandb = None, seed = None,
                           masked_features_data=None, removed_edge_indices=None, removed_edge_types=None):
    seed = _resolve_seed(seed)
    best_loss = float('inf')
    best_F1 = 0
    best_accuracy = 0
    best_metrics = {}
    best_epoch = 0
    if masked_features_data is None:
        print("\nmask_features...\n")
        masked_features_data = view_partial_features_masking(data, max_masking_percentage=config["max_masking_percentage"],
                                                             random_seed=seed)
    else:
        print("\nUsing precomputed feature mask...\n")

    if removed_edge_indices is None:
        print("\nRelations_dripping (Masking)...\n")
        _, removed_edge_indices, removed_edge_types = relation_based_edge_dropping_balanced(data, config[
            "total_drop_rate"], max_drop_fraction_per_node=0.3, random_seed=seed)
    else:
        print("\nUsing precomputed relation mask...\n")
    removed_edge_indices = removed_edge_indices.to(device)
    if removed_edge_types is not None:
        removed_edge_types = removed_edge_types.to(device)

    set_seed(seed)
    G_data_loader = GraphDataLoader(masked_features_data, num_neighbors=config["num_neighbors"],
                                    batch_size=config["batch_size"], shuffle=config["shuffle"], seed=seed).get_loader()
    G_data_loader.data.edge_attr = data.edge_attr
    global_step = 0
    for epoch in range(num_epochs):
        if _step_limit_reached(global_step):
            break
        model.train()
        total_loss = 0
        total_Recons_X_loss = 0
        total_R_loss = 0
        all_preds = []
        all_true_labels = []
        steps_this_epoch = 0
        with tqdm(total=len(G_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as main_pbar:

            for G2_batch in G_data_loader:

                G2_batch = G2_batch.to(device)
                removed_batch = copy.copy(G2_batch)

                removed_edge_indices = removed_edge_indices.to(device)
                mask = torch.isin(removed_edge_indices, G2_batch.e_id)
                intersections = removed_edge_indices[mask]
                # Obtenir les nœuds cibles des arêtes intersectantes avec le batch graph
                intersection_targets = data.edge_index[1][intersections]
                # Trouver les intersections qui vérifient la condition
                # (les nœuds cibles sont dans input_id)
                matching_mask = torch.isin(intersection_targets, G2_batch.input_id)
                # Récupérer les e_id correspondants
                batch_matching_e_ids = intersections[matching_mask]
                edges_mask = torch.isin(G2_batch.e_id,batch_matching_e_ids) ## mask pour les edges à supprimer dans le batch

                ## the final masked batch
                _filter_edges(G2_batch, ~edges_mask)
                _filter_edges(removed_batch, edges_mask)

                #### Features reconstruction
                n_id_fm = G2_batch.n_id  ## The global node index for every sampled node
                mask_fm = torch.isin(n_id_fm, G2_batch.input_id)  ## mask to get only the embedding of input_id nodes
                optimizer.zero_grad()
                H_2, r_embd = _encode(model, G2_batch)
                if isinstance(model.x_decoder, TransGCNDecoder):
                    reconstructed_x = model.decode_x(G2_batch, H_2, r_embd)
                else:
                    reconstructed_x = model.decode_x(G2_batch, H_2)
                reconstructed_x = reconstructed_x[mask_fm]
                ##############################
                Recons_X_loss = 0.0

                # Vérifier chaque terme et ajouter le loss correspondant au total
                if "MSE" in loss_fct:
                    mse_loss = mse_loss_fnc(data.x[G2_batch.n_id[mask_fm]], reconstructed_x)
                    Recons_X_loss += mse_loss

                if "PCSE" in loss_fct:
                    pcse_loss = similarity_pair_loss(data.x[G2_batch.n_id[mask_fm]], reconstructed_x)
                    Recons_X_loss += pcse_loss

                if "SCE" in loss_fct:
                    sce_loss = sce_loss_fnc(data.x[G2_batch.n_id[mask_fm]], reconstructed_x)
                    Recons_X_loss += sce_loss




                # Générer les triplets négatifs et positifs
                negative_triplets = generate_negatives(data, G2_batch, negative_ratio=1)
                positive_triplets = get_positives(G2_batch)
                ## Generate negative examples from removed edges:
                negative_triplets_removed = generate_negatives(data, removed_batch, negative_ratio=1)
                positive_triplets_removed = get_positives(removed_batch)

                all_positive_triplets = torch.cat((positive_triplets, positive_triplets_removed), dim=0)
                all_negative_triplets = torch.cat((negative_triplets, negative_triplets_removed), dim=0)
                pos_edge_index = torch.stack((all_positive_triplets[:, 0], all_positive_triplets[:, 2]))  # (2, num_edges)
                pos_edge_type = all_positive_triplets[:, 1]
                neg_edge_index = torch.stack((all_negative_triplets[:, 0], all_negative_triplets[:, 2]))  # (2, num_edges)
                neg_edge_type = all_negative_triplets[:, 1]
                # Scores pour les triplets positifs et négatifs
                pos_scores = model.recon_r_(H_2, pos_edge_index, pos_edge_type)
                neg_scores = model.recon_r_(H_2, neg_edge_index, neg_edge_type)
                # Fonction de perte : Binary Cross Entropy

                pos_preds = (torch.sigmoid(pos_scores) > 0.55).int()
                neg_preds = (torch.sigmoid(neg_scores) > 0.55).int()

                # True labels
                pos_labels = torch.ones_like(pos_preds)
                neg_labels = torch.zeros_like(neg_preds)

                # Collect predictions and true labels
                all_preds.extend(pos_preds.cpu().numpy())
                all_preds.extend(neg_preds.cpu().numpy())
                all_true_labels.extend(pos_labels.cpu().numpy())
                all_true_labels.extend(neg_labels.cpu().numpy())
                loss_R = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
                       F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))

                loss = Recons_X_loss + loss_R

                loss.backward()
                optimizer.step()
                global_step += 1
                steps_this_epoch += 1
                total_loss += loss.item()
                total_R_loss += loss_R.item()
                total_Recons_X_loss += Recons_X_loss
                main_pbar.update(1)
                if _step_limit_reached(global_step):
                    break

            avg_R_loss = total_R_loss / max(steps_this_epoch, 1)
            avg_Recons_X_loss = total_Recons_X_loss / max(steps_this_epoch, 1)
            avg_loss = total_loss / max(steps_this_epoch, 1)
            print("Evaluation\n")
            # metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
            metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp, config)

            print("\n")
            print(metrics)

            R_accuracy, R_precision, R_recall, R_f1 = _calculate_relation_micro_metrics(all_preds, all_true_labels)
            print(f"R_accuracy: {R_accuracy}, R_precision: {R_precision}, R_recall: {R_recall},R_f1: {R_f1}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(model, optimizer, epoch, save_dir = save_dir, file_name = save_file, is_best_acc=False)
                print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_metrics = _record_best_result({
                    **metrics,
                    "R_accuracy": R_accuracy,
                    "R_precision": R_precision,
                    "R_recall": R_recall,
                    "R_f1": R_f1,
                    "R_loss": avg_R_loss,
                    "X_loss": avg_Recons_X_loss,
                }, save_file, epoch, status="abnormal")
                save_model(model, optimizer, epoch, save_dir=save_dir,file_name= save_file , is_best_acc=True)
                print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
            wandb.log({"epoch": epoch + 1, "step": global_step, "total_loss": avg_loss,
                       **_wandb_classification_metrics(metrics),
                       "R_accuracy": R_accuracy, "R_precision": R_precision,
                       "R_recall": R_recall, "R_f1": R_f1, "R_loss": avg_R_loss, "X_loss" : avg_Recons_X_loss})

    return _finalize_best_result(best_metrics, save_file)












# Fonction d'entraînement avec suivi de la perte dans wandb
def train_model(model, data, optimizer, num_epochs, num_bases, out_channels, gdp,
                           save_dir="ckpt_", training_options = "Reconstruct_X_MSE", device = "cuda", wandb = None, split = False, seed = 42):
    seed = _resolve_seed(seed)

    unique_relations = sorted({i.item() for i in data.edge_type})
    relation_embeddings = generate_relation_embeddings_tensor(unique_relations, out_channels[-1], device,
                                                              seed=seed)

    best_loss = float('inf')
    best_F1 = 0
    best_accuracy = 0
    # Application du masque de features
    print("\nmask_features...\n")
    masked_features_data= view_partial_features_masking(data, max_masking_percentage = config["max_masking_percentage"],
                                                        random_seed=seed)
    print("\nRelations_dripping (Masking)...\n")
    masked_edges_data, removed_edge_indices, removed_edge_types  = relation_based_edge_dropping_balanced(data, config["total_drop_rate"], max_drop_fraction_per_node=0.3, random_seed=seed)
    removed_edge_indices = removed_edge_indices.to(device)
    removed_edge_types = removed_edge_types.to(device)

    if split:
        train_removed_edges_indices, test_removed_edges_indices, train_relations, test_relations = removed_edges_train_test_split(removed_edge_indices, removed_edge_types)
    # print(len(removed_edge_indices),"---")
    # removed_edge_indices = train_removed_edges_indices.to(device)
    # test_removed_edges_indices = test_removed_edges_indices.to(device)


    set_seed(seed)
    G1_data_loader = GraphDataLoader(masked_features_data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"], seed=seed).get_loader()

    set_seed(seed)
    G_data_loader = GraphDataLoader(data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"], seed=seed).get_loader()
    set_seed(seed)
    G2_data_loader = GraphDataLoader(masked_edges_data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"], seed=seed).get_loader()
    set_seed(seed)
    cc_indexes = gdp.get_list_indexes(config["core_concepts"])
    cc_data_loader = GraphDataLoader(data, num_neighbors=config["num_neighbors"], batch_size=len(cc_indexes),
                                     shuffle=config["shuffle"], seed=seed, input_nodes = cc_indexes).get_loader()
    gs_terms = pd.read_excel(config["Gs_path_no_other"], sheet_name='Sheet1')
    gs_terms_indexes = gdp.get_list_indexes(list(gs_terms['term']))

    # cc_data_loader_2 = GraphDataLoader(data, num_neighbors=config["num_neighbors"],batch_size = len(cc_indexes), shuffle=config["shuffle"]).get_loader()
    #
    # cc_graph = next(iter(cc_data_loader))
    # cc_graph_2 = next(iter(cc_data_loader_2))
    #
    # cc_clusters = calculate_cluster_assignments(cc_graph.x[cc_graph.input_id], cc_graph_2.x[cc_graph_2.input_id])
    # print(cc_clusters)
    # inter_cluster_loss(cc_graph.x[range(9,18)], cc_clusters,cc_graph_2.x[cc_graph_2.input_id] )
    # exit(55)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_mse_loss = 0
        total_cos_loss = 0
        total_sce_loss = 0
        total_inter_cluster_loss_ = 0
        total_intra_cluster_loss_ = 0
        if "contrastive" in training_options and len(training_options)==1:
            print("\nContrastive\n")
            with tqdm(total=len(G_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
                for g_batch in G_data_loader:
                    g_batch.to(device)
                    g1_batch = copy.deepcopy(g_batch)
                    g2_batch = copy.deepcopy(g_batch)
                    g1_batch.x = masked_features_data.x[g_batch.n_id]
                    edges_mask = ~torch.isin(g2_batch.e_id, removed_edge_indices.to(device))
                    _filter_edges(g2_batch, edges_mask)


                    ################################## Verification ###############################
                    # mask2 = torch.isin(removed_edge_indices, g2_batch.e_id)
                    # # print(removed_edge_types.shape)
                    # # print(mask2.shape, removed_edge_indices.shape)
                    # print(removed_edge_types[mask2].shape)
                    # sorted_g2_batch, i_batch = torch.sort(g2_batch.e_id)
                    # sorted_g2_, i = torch.sort(removed_edge_indices[mask2])
                    # print(g2_batch.edge_type[i_batch] == removed_edge_types[mask2][i])
                    # print(sorted_g2_batch == sorted_g2_)
                    #################################################################################
                    nodes_mask = torch.isin(g_batch.n_id, g_batch.input_id)
                    h1_batch = _encode_nodes(model, g1_batch)
                    h2_batch = _encode_nodes(model, g2_batch)
                    # mask = torch.isin(n_id, batch.input_id) ## select only input_nodes
                    # h1_projected = model.projector_fc1(h1_batch)[mask]
                    # h2_projected = model.projector_fc2(h1_batch)[mask]

                    mask_is = g1_batch.edge_type == gdp.predicate_to_id["is"]
                    c_loss = contrastive_loss_exclude_is(h1_batch,h2_batch,g1_batch.edge_index,mask_is)


                    # exit(12354)

                    # n_id = g1_batch.n_id  ## The global node index for every sampled node
                    # mask_1 = torch.isin(n_id, g1_batch.input_id)
                    # c_loss = contrastive_loss(h1_batch, h2_batch)
                    # reconstructed_x = model.decode_x(g1_batch, h1_batch)
                    # reconstructed_x = reconstructed_x[mask_1]
                    # mce_loss = mse_loss_fnc(data.x[n_id[mask_1]], reconstructed_x)
                    loss = c_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)
                avg_loss = total_loss / len(G1_data_loader)
                print("\nEvaluation\n")
                print(data)
                metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
                print("\n")
                print(metrics)
                print("\n")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=True)
                    print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                wandb.log({"epoch": epoch + 1, "contrastive loss": avg_loss,
                           **_wandb_classification_metrics(metrics)})


                    # G1_batch.to(device)
                    # G2_batch.to(device)
                    # n_id_1 = G1_batch.n_id  ## The global node index for every sampled node
                    # mask_G1 = torch.isin(n_id_1, G1_batch.input_id) ## mask to get only the embedding of input_id nodes
                    # n_id_2 = G2_batch.n_id
                    # mask_G2 = torch.isin(n_id_2, G2_batch.input_id)
                    # H1_batch = model.encode(G1_batch)
                    # H2_batch = model.encode(G2_batch)
                    # H1_projected = model.projector_fc1(H1_batch)[mask_G1]
                    # H2_projected = model.projector_fc2(H2_batch)[mask_G2]
                    # # Compute contrastive loss
                    # loss = model.contrastive_loss(H1_projected, H2_projected)
                    # loss.backward()
                    # optimizer.step()
                    # total_loss += loss.item()
                    # batch_pbar.set_postfix(batch_loss=loss.item())
                    # batch_pbar.update(1)


        elif "reconstruct_r" in training_options and len(training_options)==1:
            print("Reconstruct R only ....")
            with tqdm(total=len(G_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as main_pbar:
                nb_intersections = 0
                matching_e_ids = []
                all_preds = []
                all_labels = []

                for G2_batch in G_data_loader:

                    G2_batch = G2_batch.to(device)
                    removed_batch = copy.copy(G2_batch)

                    removed_edge_indices = removed_edge_indices.to(device)
                    mask = torch.isin(removed_edge_indices, G2_batch.e_id)
                    intersections = removed_edge_indices[mask]
                    # Obtenir les nœuds cibles des arêtes intersectantes avec le batch graph
                    intersection_targets = data.edge_index[1][intersections]
                    # Trouver les intersections qui vérifient la condition
                    # (les nœuds cibles sont dans input_id)
                    matching_mask = torch.isin(intersection_targets, G2_batch.input_id)
                    # Récupérer les e_id correspondants
                    batch_matching_e_ids = intersections[matching_mask]
                    edges_mask = torch.isin(G2_batch.e_id,batch_matching_e_ids) ## mask pour les edges à supprimer dans le batch

                    ## the final masked batch
                    _filter_edges(G2_batch, ~edges_mask)
                    _filter_edges(removed_batch, edges_mask)
                    optimizer.zero_grad()
                    H_2 = _encode_nodes(model, G2_batch)

                    # Générer les triplets négatifs et positifs
                    negative_triplets = generate_negatives(data, G2_batch, negative_ratio=1)
                    positive_triplets = get_positives(G2_batch)
                    ## Generate negative examples from removed edges:
                    negative_triplets_removed = generate_negatives(data, removed_batch, negative_ratio=1)
                    positive_triplets_removed = get_positives(removed_batch)

                    all_positive_triplets = torch.cat((positive_triplets, positive_triplets_removed), dim=0)
                    all_negative_triplets = torch.cat((negative_triplets, negative_triplets_removed), dim=0)

                #     # Créer le DataLoader pour les batchs ConvE
                    convE_loader = create_data_loader(all_positive_triplets, all_negative_triplets, H_2, relation_embeddings,
                                                      config["batch_size"]*3, shuffle=True, seed=seed)

                    convE_loss = 0
                    convE_batches_processed = 0
                    avg_convE_loss = 0
                    for convE_batch in convE_loader:
                        # Prédictions et calcul de la perte
                        preds = model.r_decoder(convE_batch[0], convE_batch[1], convE_batch[2])
                        loss = recon_r_loss(preds, convE_batch[3].to(device))
                        # Backpropagation avec accumulation des gradients
                        predicted_labels = (preds > 0.5).long().detach()  # Seuil pour convertir les scores en 0/1
                        all_preds.extend(predicted_labels.cpu().numpy())
                        all_labels.extend(convE_batch[3].cpu().numpy())
                        loss.backward(retain_graph=True)
                        # Accumuler la perte totale pour ConvE
                        convE_loss += loss.item()
                        convE_batches_processed += 1
                        # Mise à jour de la barre principale avec les détails du batch ConvE
                        main_pbar.set_postfix(
                            convE_batches=f"{convE_batches_processed}/{len(convE_loader)}"
                        )
                    # Optimisation après accumulation
                    optimizer.step()
                    avg_convE_loss += convE_loss/convE_batches_processed
                    total_loss += avg_convE_loss
                    # Mise à jour de la barre principale pour chaque batch du graphe
                    main_pbar.update(1)
                    main_pbar.set_postfix(
                        convE_batches=f"{convE_batches_processed}/{len(convE_loader)}",
                        total_loss=f"{convE_loss:.4f}"
                    )


                avg_loss = total_loss / len(G2_data_loader)
                accuracy_train = accuracy_score(all_labels, all_preds)
                f1_train = f1_score(all_labels, all_preds, average="macro")
                print(f"\nEpoch {epoch + 1}: train_accuracy = {accuracy_train:.4f}, f1-score_train = {f1_train:.4f}")


                test_data_loader = NeighborLoader(
                    data,
                    input_nodes=data.edge_index[1][removed_edge_indices],  # Les nœuds que tu veux embeder
                    num_neighbors=config["num_neighbors"],  # Nombre de voisins à échantillonner par couche
                    batch_size=config["batch_size"],
                    shuffle=False
                )

                test_avg_loss, test_metrics = evaluate_ConvE(model, data, test_data_loader, test_removed_edges_indices, device, relation_embeddings)
                test_f1 = test_metrics["f1_score"]
                test_accuracy = test_metrics["accuracy"]


                if "reconstruct_r" in training_options and len(training_options) == 1:
                    wandb.log({"epoch": epoch + 1, "train_loss": avg_loss, "train_accuracy": accuracy_train, "f1_train": f1_train, "test_loss": test_avg_loss,
                               **{f"test_{key}": value for key, value in _wandb_classification_metrics(test_metrics).items()}})
                if test_f1 > best_F1:
                    best_F1 = test_f1
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best=True)
                    print(f'\nModel saved with Avg Loss: {avg_loss:.4f} , test_F1-Score: {test_f1:.4f}, test_accuracy = {test_accuracy:.4f}')


        elif "Reconstruct_X_MSE_Pairs_similarity" in training_options and len(training_options) == 1:
            print("\nReconstruct_X_MSE_Pairs_similarity only...\n")
            with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
                for batch in G1_data_loader:
                    batch = batch.to(device)
                    n_id = batch.n_id ## The global node index for every sampled node
                    mask = torch.isin(n_id, batch.input_id) ## mask to get only the embedding of input_id nodes
                    optimizer.zero_grad()
                    embeddings = _encode_nodes(model, batch)
                    reconstructed_x = model.decode_x(batch, embeddings)
                    reconstructed_x = reconstructed_x[mask]
                    cos_loss = similarity_pair_loss(data.x[n_id[mask]], reconstructed_x, embeddings[mask])
                    mse_loss = mse_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    loss = mse_loss +  cos_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    total_mse_loss += mse_loss.item()
                    total_cos_loss += cos_loss.item()
                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)

                avg_loss = total_loss / len(G1_data_loader)
                avg_mse_loss = total_mse_loss / len(G1_data_loader)
                avg_cos_loss = total_cos_loss / len(G1_data_loader)
                metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
            # Sauvegarde du modèle si la perte est la plus faible
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,is_best_acc = False)
                    print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                wandb.log(
                    {"epoch": epoch + 1, "global loss": avg_loss, "mse_loss": avg_mse_loss, "cos_loss": avg_cos_loss,
                     **_wandb_classification_metrics(metrics),
                     })

        elif "SCE_Recons_X" in training_options and len(training_options) == 1:
            print("SCE_Recons_X\n")
            with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
                for batch in G1_data_loader:
                    batch = batch.to(device)
                    n_id = batch.n_id  ## The global node index for every sampled node
                    mask = torch.isin(n_id, batch.input_id)  ## mask to get only the embedding of input_id nodes
                    optimizer.zero_grad()
                    embeddings = _encode_nodes(model, batch)
                    # print(batch)
                    reconstructed_x = model.decode_x(batch, embeddings)
                    reconstructed_x = reconstructed_x[mask]
                    sce_loss = sce_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    loss = sce_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)
                avg_loss = total_loss / len(G1_data_loader)
                print("Evaluation\n")
                metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
                print("\n")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=True)
                    print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                wandb.log({"epoch": epoch + 1, "sce loss": avg_loss,
                           **_wandb_classification_metrics(metrics)})

        elif "MSE_Recons_X" in training_options and len(training_options) == 1:
            print("\nMSE_Recons_X\n")
            with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}",
                      unit="batch") as batch_pbar:
                for batch in G1_data_loader:
                    batch = batch.to(device)
                    n_id = batch.n_id  ## The global node index for every sampled node
                    mask = torch.isin(n_id, batch.input_id)  ## mask to get only the embedding of input_id nodes
                    optimizer.zero_grad()
                    embeddings = _encode_nodes(model, batch)
                    reconstructed_x = model.decode_x(batch, embeddings)
                    reconstructed_x = reconstructed_x[mask]
                    mse_loss = mse_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    loss = mse_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)
                avg_loss = total_loss / len(G1_data_loader)

                print("Evaluation\n")
                metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
                print("\n")
                print(metrics)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=True)
                    print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                wandb.log({"epoch": epoch + 1, "mce loss": avg_loss,
                           **_wandb_classification_metrics(metrics)})


        elif "MSE_Recons_X"  in training_options and "SCE_Recons_X" in training_options:
            print("\nMSE_Recons_X + SCE_Recons_X\n")
            with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
                for batch in G1_data_loader:
                    batch = batch.to(device)
                    n_id = batch.n_id  ## The global node index for every sampled node
                    mask = torch.isin(n_id, batch.input_id)  ## mask to get only the embedding of input_id nodes
                    optimizer.zero_grad()
                    embeddings = _encode_nodes(model, batch)
                    reconstructed_x = model.decode_x(batch, embeddings)
                    reconstructed_x = reconstructed_x[mask]
                    sce_loss = sce_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    mse_loss = mse_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    loss = mse_loss + sce_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    total_mse_loss += mse_loss.item()
                    total_sce_loss += sce_loss.item()
                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)

                avg_loss = total_loss / len(G1_data_loader)
                avg_mse_loss = total_mse_loss / len(G1_data_loader)
                avg_sce_loss = total_sce_loss / len(G1_data_loader)
                metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
                # Sauvegarde du modèle si la perte est la plus faible
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=True)
                    print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                wandb.log(
                    {"epoch": epoch + 1, "global loss": avg_loss, "mse_loss": avg_mse_loss, "sce_loss": total_sce_loss,
                     **_wandb_classification_metrics(metrics),
                     })

        elif "MSE_Recons_X"  in training_options and "clustering_obj" in training_options:
            warm_up_epochs = 15
            gradual_introduction_epochs = 50
            print("\nMSE_Recons_X + clustering_obj\n")
            intra_drap = True
            with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
                for batch in G1_data_loader:
                    batch = batch.to(device)
                    cc_graph = next(iter(cc_data_loader)).to(device)
                    n_id = batch.n_id  ## The global node index for every sampled node
                    mask = torch.isin(n_id, batch.input_id) ## mask to get only the embedding of input_id nodes
                    optimizer.zero_grad()
                    cc_embeddings = _encode_nodes(model, cc_graph)[cc_graph.input_id]
                    embeddings = _encode_nodes(model, batch)
                    reconstructed_x = model.decode_x(batch, embeddings)
                    reconstructed_x = reconstructed_x[mask]
                    mse_loss = mse_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    ####### clustering loss ######"
                    if epoch < warm_up_epochs:
                        # Pendant la période de warm-up, seule la reconstruction MSE est optimisée
                        loss = mse_loss
                        total_inter_cluster_loss_ = 0
                        total_intra_cluster_loss_ = 0

                    else:
                        # Calcul du coefficient d'introduction progressive
                        clustering_weight = min(1.0, (epoch - warm_up_epochs) / gradual_introduction_epochs)
                        # Calcul des losses de clustering
                        mse_loss = torch.tensor(0)
                        gs_mask = torch.isin(batch.input_id, torch.tensor(gs_terms_indexes).to(device))
                        if sum(gs_mask) == 0:
                            inter_cluster_loss_ = torch.tensor(0)
                        else:
                            gs_batch_indexes = batch.input_id[gs_mask]
                            gs_mask_embd = torch.isin(batch.n_id, gs_batch_indexes)
                            cluster_assignments = calculate_cluster_assignments(embeddings[gs_mask_embd], cc_embeddings)

                            inter_cluster_loss_ = inter_cluster_loss(embeddings[gs_mask_embd], cluster_assignments, cc_embeddings)
                        if intra_drap:
                            intra_cluster_loss_ = intra_cluster_loss(cc_embeddings)
                            intra_drap = False
                        else:
                            intra_cluster_loss_ = torch.tensor(0.0, device=device, requires_grad=True)
                        # Combinaison des losses avec un poids progressif
                        loss = mse_loss + clustering_weight * (intra_cluster_loss_ + inter_cluster_loss_)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_mse_loss += mse_loss.item()

                    # Gestion des pertes de clustering (uniquement après la période de warm-up)
                    if epoch >= warm_up_epochs:
                        total_inter_cluster_loss_ += inter_cluster_loss_.item()
                        total_intra_cluster_loss_ += intra_cluster_loss_.item()

                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)

                avg_loss = total_loss / len(G1_data_loader)

                avg_mse_loss = total_mse_loss / len(G1_data_loader)
                avg_inter_cluster_loss = total_inter_cluster_loss_ / len(G1_data_loader)
                avg_intra_cluster_loss = total_intra_cluster_loss_ / len(G1_data_loader)
                print("\nEvaluation:")
                metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
                print("\n", metrics,"\n")
                print(f"\n Loss: total:{avg_loss}, mse:{avg_mse_loss},inter:{avg_inter_cluster_loss},intra:{avg_intra_cluster_loss} \n")
                # Sauvegarde du modèle si la perte est la plus faible
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=True)
                    print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                wandb.log(
                    {"epoch": epoch + 1, "global loss": avg_loss, "mse_loss": avg_mse_loss, "intra_cluster_loss": avg_intra_cluster_loss,
                     "inter_cluster_loss": avg_inter_cluster_loss, **_wandb_classification_metrics(metrics),
                     })



