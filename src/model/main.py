
import sys
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import random
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch_geometric.nn import GAE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

# Imports locaux (sans le préfixe src.)
from train_optimize_parms import train_GAE, train_Contrastive, train_X_reconstruction, train_GraphMAE_X_reconstruction, train_DisMult, train_DisMult_with_onto, train_Double_Reconstruction, train_Contrastive
from Dismult import DistMultDecoder
from GCNDecoder import GCNDecoder
from GCNEncoder import GCNEncoder
from GATDecoder import GATDecoder
from GATEncoder import GATEncoder
from MLPDecoder import MLPDecoder
from TransGCNEncoder import TransGCNEncoder
from TransGCNDecoder import TransGCNDecoder
from RGCNEncoder import RGCNEncoder
from RGCNDecoder import RGCNDecoder
from GraphDataPreparation import GraphDataPreparation
from MRGAE import MRGAE
from config import config
from utils.utils import set_seed
import copy
from data_augmentation import relation_based_edge_dropping_balanced, random_edge_dropping, view_partial_features_masking

# Initialisation des seeds pour reproductibilité
def _seed_values(seed_config):
    if isinstance(seed_config, (list, tuple)):
        return list(seed_config)
    return [seed_config]


def _set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


seed = _seed_values(config["seed"])[0]
_set_all_seeds(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initialisation wandb
import wandb
_wandb_mode = os.environ.get("WANDB_MODE") or config.get("wandb_mode")
_wandb_disabled = os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes")
if _wandb_mode:
    os.environ["WANDB_MODE"] = str(_wandb_mode)
if not _wandb_disabled:
    if hasattr(wandb, "require"):
        try:
            wandb.require("legacy-service")
        except Exception as exc:
            print(f"wandb.require('legacy-service') skipped: {exc}")
    if os.environ.get("WANDB_MODE", "").lower() != "offline":
        wandb.login(key="c278e62d2025b60ff8b984a40f7b62b697f9b4fd", relogin=True)


def _prepare_domain_range_tensors(constraints_path, kg_gdp, onto_gdp, device):
    if not constraints_path or not os.path.exists(constraints_path):
        print(f"Domain/range constraints file not found: {constraints_path}")
        return None, None, None

    with open(constraints_path, "r", encoding="utf-8") as file:
        constraints = json.load(file)

    relation_constraints = constraints.get("relations", {})
    type_names = set()
    matched_relations = 0
    skipped_relations = 0

    for relation_name, values in relation_constraints.items():
        if relation_name not in kg_gdp.predicate_to_id:
            skipped_relations += 1
            continue
        domain_types = [type_name for type_name in values.get("domain", []) if type_name in onto_gdp.nodes_index]
        range_types = [type_name for type_name in values.get("range", []) if type_name in onto_gdp.nodes_index]
        if not domain_types or not range_types:
            skipped_relations += 1
            continue
        matched_relations += 1
        type_names.update(domain_types)
        type_names.update(range_types)

    if not type_names or matched_relations == 0:
        print("No usable domain/range constraints matched the KG relation/type indexes.")
        return None, None, None

    sorted_type_names = sorted(type_names)
    type_name_to_position = {type_name: idx for idx, type_name in enumerate(sorted_type_names)}
    type_ids = torch.tensor(
        [onto_gdp.nodes_index[type_name] for type_name in sorted_type_names],
        dtype=torch.long,
        device=device,
    )

    num_relations = max(kg_gdp.predicate_to_id.values()) + 1
    domain_mask = torch.zeros((num_relations, len(sorted_type_names)), dtype=torch.float32, device=device)
    range_mask = torch.zeros((num_relations, len(sorted_type_names)), dtype=torch.float32, device=device)

    for relation_name, values in relation_constraints.items():
        if relation_name not in kg_gdp.predicate_to_id:
            continue
        relation_id = kg_gdp.predicate_to_id[relation_name]
        for type_name in values.get("domain", []):
            if type_name in type_name_to_position:
                domain_mask[relation_id, type_name_to_position[type_name]] = 1.0
        for type_name in values.get("range", []):
            if type_name in type_name_to_position:
                range_mask[relation_id, type_name_to_position[type_name]] = 1.0

    print(
        f"Domain/range constraints matched: {matched_relations} relations, "
        f"{len(sorted_type_names)} ontology type prototypes, skipped {skipped_relations} relations."
    )
    return type_ids, domain_mask, range_mask


def _prepare_domain_range_embedding_tensors(constraints_path, kg_gdp, onto_gdp, device, core_concepts=None):
    if not constraints_path or not os.path.exists(constraints_path):
        print(f"Domain/range embedding constraints file not found: {constraints_path}")
        return None, None, None, None, None

    with open(constraints_path, "r", encoding="utf-8") as file:
        constraints = json.load(file)

    relation_constraints = constraints.get("relations", {})
    allowed_type_names = set(core_concepts or [])
    type_names = set()
    matched_relations = 0
    skipped_relations = 0

    for relation_name, values in relation_constraints.items():
        if relation_name not in kg_gdp.predicate_to_id or relation_name == "isa":
            skipped_relations += 1
            continue
        direct_domain = [
            type_name for type_name in values.get("direct_domain", [])
            if type_name in onto_gdp.nodes_index and (not allowed_type_names or type_name in allowed_type_names)
        ]
        direct_range = [
            type_name for type_name in values.get("direct_range", [])
            if type_name in onto_gdp.nodes_index and (not allowed_type_names or type_name in allowed_type_names)
        ]
        allowed_domain = [
            type_name for type_name in values.get("domain", direct_domain)
            if type_name in onto_gdp.nodes_index and (not allowed_type_names or type_name in allowed_type_names)
        ]
        allowed_range = [
            type_name for type_name in values.get("range", direct_range)
            if type_name in onto_gdp.nodes_index and (not allowed_type_names or type_name in allowed_type_names)
        ]
        if not direct_domain or not direct_range:
            skipped_relations += 1
            continue
        matched_relations += 1
        type_names.update(direct_domain)
        type_names.update(direct_range)
        type_names.update(allowed_domain)
        type_names.update(allowed_range)

    if not type_names or matched_relations == 0:
        print("No usable domain/range embedding constraints matched the KG relation/type indexes.")
        return None, None, None, None, None

    sorted_type_names = sorted(type_names)
    type_name_to_position = {type_name: idx for idx, type_name in enumerate(sorted_type_names)}
    type_ids = torch.tensor(
        [onto_gdp.nodes_index[type_name] for type_name in sorted_type_names],
        dtype=torch.long,
        device=device,
    )

    num_relations = max(kg_gdp.predicate_to_id.values()) + 1
    domain_explicit_mask = torch.zeros((num_relations, len(sorted_type_names)), dtype=torch.float32, device=device)
    range_explicit_mask = torch.zeros((num_relations, len(sorted_type_names)), dtype=torch.float32, device=device)
    domain_allowed_mask = torch.zeros((num_relations, len(sorted_type_names)), dtype=torch.float32, device=device)
    range_allowed_mask = torch.zeros((num_relations, len(sorted_type_names)), dtype=torch.float32, device=device)

    for relation_name, values in relation_constraints.items():
        if relation_name not in kg_gdp.predicate_to_id or relation_name == "isa":
            continue
        relation_id = kg_gdp.predicate_to_id[relation_name]
        for type_name in values.get("direct_domain", []):
            if type_name in type_name_to_position:
                domain_explicit_mask[relation_id, type_name_to_position[type_name]] = 1.0
        for type_name in values.get("direct_range", []):
            if type_name in type_name_to_position:
                range_explicit_mask[relation_id, type_name_to_position[type_name]] = 1.0
        for type_name in values.get("domain", values.get("direct_domain", [])):
            if type_name in type_name_to_position:
                domain_allowed_mask[relation_id, type_name_to_position[type_name]] = 1.0
        for type_name in values.get("range", values.get("direct_range", [])):
            if type_name in type_name_to_position:
                range_allowed_mask[relation_id, type_name_to_position[type_name]] = 1.0

    print(
        f"Domain/range embedding constraints matched: {matched_relations} relations, "
        f"{len(sorted_type_names)} ontology type prototypes, skipped {skipped_relations} relations."
    )
    return type_ids, domain_explicit_mask, range_explicit_mask, domain_allowed_mask, range_allowed_mask


def _prepare_onto_hierarchy_pairs(onto_data, onto_gdp, device):
    if "isa" not in onto_gdp.predicate_to_id:
        print("No 'isa' relation found in ontology predicates; hierarchy loss disabled.")
        return None, None

    isa_relation_id = onto_gdp.predicate_to_id["isa"]
    isa_mask = onto_data.edge_type == isa_relation_id
    if isa_mask.sum() == 0:
        print("No ontology isa edges found; hierarchy loss disabled.")
        return None, None

    child_ids = onto_data.edge_index[0, isa_mask].to(device)
    parent_ids = onto_data.edge_index[1, isa_mask].to(device)
    print(f"Ontology hierarchy pairs: {child_ids.numel()} child-parent isa edges.")
    return child_ids, parent_ids


def _prepare_soft_type_negative_candidates(data, onto_data, domain_range_type_ids,
                                           domain_mask_by_relation, range_mask_by_relation,
                                           top_k=5000, temperature=0.1):
    if domain_range_type_ids is None or domain_mask_by_relation is None or range_mask_by_relation is None:
        print("Soft type-aware negative sampling disabled: missing domain/range tensors.")
        return None

    with torch.no_grad():
        node_embeddings = torch.nn.functional.normalize(data.x, p=2, dim=1)
        type_embeddings = torch.nn.functional.normalize(onto_data.x[domain_range_type_ids], p=2, dim=1)
        logits = node_embeddings @ type_embeddings.t()
        type_probs = torch.softmax(logits / max(temperature, 1e-8), dim=1)

        num_nodes = data.x.size(0)
        top_k = min(int(top_k), num_nodes)
        domain_candidates = {}
        range_candidates = {}
        matched_relations = 0

        for relation_id in range(domain_mask_by_relation.size(0)):
            domain_mask = domain_mask_by_relation[relation_id]
            range_mask = range_mask_by_relation[relation_id]
            has_domain = domain_mask.sum() > 0
            has_range = range_mask.sum() > 0
            if not has_domain and not has_range:
                continue
            matched_relations += 1
            if has_domain:
                domain_scores = (type_probs * domain_mask).sum(dim=1)
                domain_candidates[relation_id] = torch.topk(domain_scores, k=top_k).indices.detach().cpu().tolist()
            if has_range:
                range_scores = (type_probs * range_mask).sum(dim=1)
                range_candidates[relation_id] = torch.topk(range_scores, k=top_k).indices.detach().cpu().tolist()

    print(
        f"Soft type-aware negative candidates prepared for {matched_relations} relations "
        f"with top_k={top_k}, temperature={temperature}."
    )
    return {"domain": domain_candidates, "range": range_candidates}


def _ensure_seeded_wandb_init():
    if getattr(wandb, "_mc2gae_seed_wrapped", False):
        return

    original_init = wandb.init

    def seeded_init(*args, **kwargs):
        active_seed = config.get("active_seed")
        run_name = kwargs.get("name")
        if active_seed is not None and run_name and f"seed_{active_seed}" not in run_name:
            kwargs["name"] = f"{run_name}_seed_{active_seed}"

        run_config = kwargs.get("config")
        if isinstance(run_config, dict):
            kwargs["config"] = {**run_config, "seed": active_seed, "num_steps": config.get("num_steps")}

        return original_init(*args, **kwargs)

    wandb.init = seeded_init
    wandb._mc2gae_seed_wrapped = True


def main():
    seed_config = config["seed"]
    if isinstance(seed_config, (list, tuple)):
        base_root_save_dir = config["root_save_dir"]
        for active_seed in seed_config:
            config["seed"] = active_seed
            config["active_seed"] = active_seed
            config["root_save_dir"] = os.path.join(base_root_save_dir, f"seed_{active_seed}")
            main()
        config["seed"] = seed_config
        config["active_seed"] = None
        config["root_save_dir"] = base_root_save_dir
        return

    active_seed = config["seed"]
    config["active_seed"] = active_seed
    _set_all_seeds(active_seed)
    _ensure_seeded_wandb_init()

    results = []
    wandb_project_name = config["wandb_project_name"]
    # save_dir = config["save_dir"]
    Entities_path = config["Entities_path"]
    KG_path = config["KG_path"]
    Edges_path = config["Edges_path"]
    plm_embedding_model = config["plm_embedding_model"]
    device = config["device"]
    gdp = GraphDataPreparation(
        Entities_path,
        KG_path,
        edges_embd_path=Edges_path,
        is_directed=True,
        model_name_init=plm_embedding_model,
    )
    data = gdp.prepare_graph_with_type()
    print(data)
    data = data.to(device)
    if config.get("recons_r_target_relation_field", "predicate") == "old_predicate":
        data.num_recons_edge_types = int(data.num_old_edge_types)
    else:
        data.num_recons_edge_types = int(data.num_edge_types)
    onto_data = None
    onto_gdp = None
    kg_relation_align_ids = None
    onto_relation_align_ids = None
    shared_relations = []
    kg_core_ids = None
    onto_core_ids = None
    domain_range_type_ids = None
    domain_mask_by_relation = None
    range_mask_by_relation = None
    domain_range_embedding_type_ids = None
    domain_explicit_mask_by_relation = None
    range_explicit_mask_by_relation = None
    domain_allowed_mask_by_relation = None
    range_allowed_mask_by_relation = None
    onto_hierarchy_child_ids = None
    onto_hierarchy_parent_ids = None
    soft_type_negative_candidates = None
    if "Recons_R_with_onto" in config["training_task"]:
        print("\n--- Preparing ontology graph from semantic network ---\n")
        onto_gdp = GraphDataPreparation(
            config["onto_entities_path"],
            config["onto_KG_path"],
            edges_embd_path=config["onto_edges_path"],
            is_directed=True,
            model_name_init=plm_embedding_model,
        )
        onto_data = onto_gdp.prepare_graph_with_type().to(device)
        print(onto_data)
        shared_relations = sorted(set(gdp.predicate_to_id) & set(onto_gdp.predicate_to_id))
        print(f"Shared KG/ontology relations for alignment: {len(shared_relations)}")
        if shared_relations:
            kg_relation_align_ids = torch.tensor(
                [gdp.predicate_to_id[relation] for relation in shared_relations],
                dtype=torch.long,
                device=device,
            )
            onto_relation_align_ids = torch.tensor(
                [onto_gdp.predicate_to_id[relation] for relation in shared_relations],
                dtype=torch.long,
                device=device,
            )
        shared_core_concepts = [
            concept for concept in config["core_concepts"]
            if concept in gdp.nodes_index and concept in onto_gdp.nodes_index
        ]
        print(f"Shared KG/ontology core concepts for contrastive loss: {len(shared_core_concepts)}")
        if shared_core_concepts:
            kg_core_ids = torch.tensor(
                [gdp.nodes_index[concept] for concept in shared_core_concepts],
                dtype=torch.long,
                device=device,
            )
            onto_core_ids = torch.tensor(
                [onto_gdp.nodes_index[concept] for concept in shared_core_concepts],
                dtype=torch.long,
                device=device,
            )
        if (config.get("lambda_domain_range", 0) != 0 or
                config.get("lambda_domain_range_embedding", 0) != 0 or
                config.get("negative_sampling_mode") == "soft_type_aware"):
            domain_range_type_ids, domain_mask_by_relation, range_mask_by_relation = _prepare_domain_range_tensors(
                config.get("domain_range_constraints_path"),
                gdp,
                onto_gdp,
                device,
            )
        if config.get("lambda_domain_range_embedding", 0) != 0:
            (
                domain_range_embedding_type_ids,
                domain_explicit_mask_by_relation,
                range_explicit_mask_by_relation,
                domain_allowed_mask_by_relation,
                range_allowed_mask_by_relation,
            ) = _prepare_domain_range_embedding_tensors(
                config.get("domain_range_constraints_path"),
                gdp,
                onto_gdp,
                device,
                core_concepts=config["core_concepts"],
            )
        if config.get("negative_sampling_mode") == "soft_type_aware":
            soft_type_negative_candidates = _prepare_soft_type_negative_candidates(
                data,
                onto_data,
                domain_range_type_ids,
                domain_mask_by_relation,
                range_mask_by_relation,
                top_k=config.get("soft_type_top_k", 5000),
                temperature=config.get("soft_type_temperature", 0.1),
            )
        if config.get("lambda_onto_hierarchy", 0) != 0:
            onto_hierarchy_child_ids, onto_hierarchy_parent_ids = _prepare_onto_hierarchy_pairs(
                onto_data,
                onto_gdp,
                device,
            )
    masked_features_data = None
    removed_edge_types = None
    removed_edge_indices = None
    # data = Data(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type).to(device)
    for task in config["training_task"]:
        save_dir = config["root_save_dir"] + f"/{task}"
        msg_sens = config["message_sens"][0]
        if task in ("Recons_X", "GraphMAE_Recons_X"):

            for out_channels in config["hyperparams_grid"]["out_channels"]:
                for encoder_ in config["encoders"]:
                    print(encoder_)
                    for decoder_ in config["decoders"]:
                        if (encoder_ in ["RGCN", "GCN", "GAT"]) and (decoder_ in ["TransGCN_conv", "TransGCN_attn", "RotatEGCN_conv", "RotatEGCN_attn"]):
                            print(f"Skipping invalid combination: enc={encoder_}, dec={decoder_}")
                            continue
                        if (encoder_ in ["RGCN", "GAT"]) and (decoder_ in ["GAT","RGCN"]):
                            print(f"Skipping invalid combination: enc={encoder_}, dec={decoder_}")
                            continue
                        use_num_bases = (encoder_ == "RGCN") or (decoder_ == "RGCN")
                        if use_num_bases:
                            for num_bases in config["hyperparams_grid"]["num_bases"]:
                                if encoder_ == "GCN":
                                    encoder = GCNEncoder(data, out_channels, config["num_layers"],
                                                         message_sens=msg_sens).to(device)
                                elif encoder_ == "RGCN":
                                    encoder = RGCNEncoder(data, out_channels, config["num_layers"], num_bases,
                                                          message_sens=msg_sens).to(device)
                                elif encoder_ == "TransGCN_conv":
                                    encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                              kg_score_fn = 'TransE',variant = 'conv',
                                                              use_edges_info = config["use_edges_info"], activation = 'relu',
                                                              bias = False ).to(device)

                                elif encoder_ == "TransGCN_attn":
                                    encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                              kg_score_fn = 'TransE',variant = 'attn',
                                                              use_edges_info = config["use_edges_info"], activation = 'relu',
                                                              bias = False ).to(device)

                                elif encoder_ == "RotatEGCN_conv":
                                    encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                              kg_score_fn = 'RotatE',variant = 'conv',
                                                              use_edges_info = config["use_edges_info"], activation = 'relu',
                                                              bias = False ).to(device)

                                elif encoder_ == "RotatEGCN_attn":
                                    encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                              kg_score_fn = 'RotatE',variant = 'attn',
                                                              use_edges_info = config["use_edges_info"], activation = 'relu',
                                                              bias = False ).to(device)


                                elif encoder_ == "GAT":
                                    encoder = GATEncoder(data, out_channels, config["num_layers"])
                                    # (self, data: Data, out_channels, num_layers=2, heads=4, dropout=0.5)
                                    # print(encoder)
                                else:
                                    print("invalid encoder type ! ")
                                    raise ValueError("Invalid encoder type!")


                                if decoder_ == "GCN":
                                    decoder = GCNDecoder(encoder, data, config["alpha"], message_sens=msg_sens).to(device)
                                elif decoder_ == "RGCN":
                                    decoder = RGCNDecoder(encoder, data, num_bases, config["alpha"],
                                                          message_sens=msg_sens).to(device)
                                elif decoder_ == "MLP":
                                    decoder = MLPDecoder(encoder, data, config["alpha"]).to(device)

                                elif decoder_ == "TransGCN_conv":
                                    decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3, kg_score_fn = 'TransE',
                                                              variant='conv',
                                                              use_edges_info = config["use_edges_info"]).to(device)

                                elif decoder_ == "TransGCN_attn":
                                    decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                                              kg_score_fn='TransE',variant='attn',
                                                              use_edges_info=config["use_edges_info"]).to(device)
                                elif decoder_ == "RotatEGCN_conv":
                                    decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                                              kg_score_fn='RotatE',variant='conv',
                                                              use_edges_info=config["use_edges_info"]).to(device)

                                elif decoder_ == "RotatEGCN_attn":
                                    decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                                              kg_score_fn='RotatE',variant='attn',
                                                              use_edges_info=config["use_edges_info"]).to(device)


                                elif decoder_ == "GAT":
                                    decoder = GATDecoder(encoder, data, heads=4, alpha=0.01, dropout=0.3)

                                else:
                                    print('invalid decoder !')
                                    raise ValueError("Invalid encoder type!")



                                run_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_dec-{decoder_}"
                                file_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_dec-{decoder_}"
                                run_config = {
                                    "device": config["device"],
                                    "num_layers": 2,
                                    "alpha": config["alpha"],
                                    "max_masking_percentage": config["max_masking_percentage"],
                                    "total_drop_rate": config["total_drop_rate"],
                                    "learning_rate": config["learning_rate"],
                                    "batch_size": config["batch_size"],
                                    "num_neighbors": [500, 500],
                                    "num_epochs": 100,
                                    "bases": num_bases,
                                    "out_channels": out_channels,
                                    "training_task": config["training_task"],
                                    "graphmae_mask_rate": config.get("graphmae_mask_rate"),
                                    "graphmae_replace_rate": config.get("graphmae_replace_rate"),
                                    "graphmae_loss_fn": config.get("graphmae_loss_fn"),
                                    "graphmae_sce_alpha": config.get("graphmae_sce_alpha"),
                                    "graphmae_decoder_remask": config.get("graphmae_decoder_remask"),
                                    "graphmae_structure_masking": config.get("graphmae_structure_masking"),
                                    "graphmae_structure_alpha": config.get("graphmae_structure_alpha"),
                                    "graphmae_structure_schedule": config.get("graphmae_structure_schedule"),
                                    "encoders": encoder_,
                                    "decoders": decoder_,
                                    "message_sens": msg_sens
                                }

                                wandb.init(
                                    project=config["wandb_project_name"],
                                    name=run_name,
                                    config=run_config,
                                    settings=wandb.Settings(start_method="thread")
                                )
                                # print(encoder)
                                # print(decoder)
                                # exit(-1)
                                autoencoder = MRGAE(encoder, decoder, projections=config["projections"]).to(device)
                                if task == "GraphMAE_Recons_X":
                                    autoencoder.init_x_mask_token(data.num_features, device=device)
                                    if config.get("graphmae_structure_masking") == "learnable":
                                        autoencoder.init_structural_mask_scorer(
                                            data.num_features,
                                            hidden_channels=config.get("graphmae_learnable_scorer_hidden"),
                                            device=device,
                                        )
                                optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
                                local_data = copy.deepcopy(data)
                                train_x_fn = train_GraphMAE_X_reconstruction if task == "GraphMAE_Recons_X" else train_X_reconstruction
                                train_x_loss = [config.get("graphmae_loss_fn", "SCE")] if task == "GraphMAE_Recons_X" else ["MSE"]
                                performances = train_x_fn(autoencoder, local_data, optimizer, config["num_epochs"],
                                            gdp, file_name,device, config,loss_fct=train_x_loss, save_dir = save_dir,
                                            wandb=wandb, seed = config["seed"])
                                results.append(performances)
                                wandb.finish()

                        else:  # Si num_bases n'est pas utilisé
                            if encoder_ == "GCN":
                                encoder = GCNEncoder(data, out_channels, config["num_layers"], message_sens=msg_sens).to(device)

                            elif encoder_ == "TransGCN_conv":
                                encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                          kg_score_fn='TransE', variant='conv',
                                                          use_edges_info=config["use_edges_info"], activation='relu',
                                                          bias=False).to(device)

                            elif encoder_ == "TransGCN_attn":
                                encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                          kg_score_fn='TransE', variant='attn',
                                                          use_edges_info=config["use_edges_info"], activation='relu',
                                                          bias=False).to(device)

                            elif encoder_ == "RotatEGCN_conv":
                                encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                          kg_score_fn='RotatE', variant='conv',
                                                          use_edges_info=config["use_edges_info"], activation='relu',
                                                          bias=False).to(device)

                            elif encoder_ == "RotatEGCN_attn":
                                encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                          kg_score_fn='RotatE', variant='attn',
                                                          use_edges_info=config["use_edges_info"], activation='relu',
                                                          bias=False).to(device)

                            elif encoder_ == "GAT":
                                encoder = GATEncoder(data, out_channels, config["num_layers"])


                            if decoder_ == "GCN":
                                decoder = GCNDecoder(encoder, data, config["alpha"], message_sens=msg_sens).to(device)
                            elif decoder_ == "MLP":
                                decoder = MLPDecoder(encoder, data, config["alpha"]).to(device)

                            elif decoder_ == "TransGCN_conv":
                                decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                                          kg_score_fn='TransE',
                                                          variant='conv',
                                                          use_edges_info=config["use_edges_info"]).to(device)

                            elif decoder_ == "TransGCN_attn":
                                decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                                          kg_score_fn='TransE', variant='attn',
                                                          use_edges_info=config["use_edges_info"]).to(device)
                            elif decoder_ == "RotatEGCN_conv":
                                decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                                          kg_score_fn='RotatE', variant='conv',
                                                          use_edges_info=config["use_edges_info"]).to(device)

                            elif decoder_ == "RotatEGCN_attn":
                                decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                                          kg_score_fn='RotatE', variant='attn',
                                                          use_edges_info=config["use_edges_info"]).to(device)

                            elif decoder_ == "GAT":
                                decoder = GATDecoder(encoder, data, heads=4, alpha=0.01, dropout=0.3)



                            else:
                                print("Error: RGCN decoder requires num_bases but is not defined!")
                                raise ValueError("Invalid encoder type!")


                            run_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_dec-{decoder_}"
                            file_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_dec-{decoder_}"
                            run_config = {
                                "device": config["device"],
                                "num_layers": 2,
                                "alpha": config["alpha"],
                                "max_masking_percentage": config["max_masking_percentage"],
                                "total_drop_rate": config["total_drop_rate"],
                                "learning_rate": config["learning_rate"],
                                "batch_size": config["batch_size"],
                                "num_neighbors": [500, 500],
                                "num_epochs": 100,
                                "out_channels": out_channels,
                                "training_task": config["training_task"],
                                "graphmae_mask_rate": config.get("graphmae_mask_rate"),
                                "graphmae_replace_rate": config.get("graphmae_replace_rate"),
                                "graphmae_loss_fn": config.get("graphmae_loss_fn"),
                                "graphmae_sce_alpha": config.get("graphmae_sce_alpha"),
                                "graphmae_decoder_remask": config.get("graphmae_decoder_remask"),
                                "graphmae_structure_masking": config.get("graphmae_structure_masking"),
                                "graphmae_structure_alpha": config.get("graphmae_structure_alpha"),
                                "graphmae_structure_schedule": config.get("graphmae_structure_schedule"),
                                "encoders": encoder_,
                                "decoders": decoder_,
                                "message_sens": msg_sens
                            }

                            wandb.init(
                                project=config["wandb_project_name"],
                                name=run_name,
                                config=run_config,
                                settings=wandb.Settings(start_method="thread")
                            )
                            local_data = copy.deepcopy(data)

                            autoencoder = MRGAE(encoder, decoder, projections=config["projections"]).to(device)
                            if task == "GraphMAE_Recons_X":
                                autoencoder.init_x_mask_token(data.num_features, device=device)
                                if config.get("graphmae_structure_masking") == "learnable":
                                    autoencoder.init_structural_mask_scorer(
                                        data.num_features,
                                        hidden_channels=config.get("graphmae_learnable_scorer_hidden"),
                                        device=device,
                                    )
                            optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
                            train_x_fn = train_GraphMAE_X_reconstruction if task == "GraphMAE_Recons_X" else train_X_reconstruction
                            train_x_loss = [config.get("graphmae_loss_fn", "SCE")] if task == "GraphMAE_Recons_X" else ["MSE"]
                            performances = train_x_fn(autoencoder, local_data, optimizer, config["num_epochs"],
                                            gdp, file_name, device, config,save_dir=save_dir, loss_fct=train_x_loss,
                                        wandb=wandb, seed = config["seed"])
                            results.append(performances)

                            wandb.finish()

        elif task == "Recons_A":

            for out_channels in config["hyperparams_grid"]["out_channels"]:
                for encoder_ in config["encoders"]:
                    if encoder_ == "RGCN":
                        for num_bases in config["hyperparams_grid"]["num_bases"]:
                            encoder = RGCNEncoder(data, out_channels, config["num_layers"], num_bases,
                                              message_sens=msg_sens).to(device)
                            run_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_GAE"
                            file_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_GAE"
                            run_config = {
                                "device": config["device"],
                                "num_layers": 2,
                                "alpha": config["alpha"],
                                "max_masking_percentage": config["max_masking_percentage"],
                                "total_drop_rate": config["total_drop_rate"],
                                "learning_rate": config["learning_rate"],
                                "batch_size": config["batch_size"],
                                "num_neighbors": [500, 500],
                                "num_epochs": 100,
                                "bases": num_bases,
                                "out_channels": out_channels,
                                "training_task": task,
                                "encoders": encoder_,
                                "decoders": "Dot Product",
                                "message_sens": msg_sens
                            }

                            wandb.init(
                                project=config["wandb_project_name"],
                                name=run_name,
                                config=run_config,
                                settings=wandb.Settings(start_method="thread")
                            )
                            local_data = copy.deepcopy(data)

                            autoencoder = GAE(encoder).to(device)
                            optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
                            performances = train_GAE(autoencoder, local_data, optimizer, config["num_epochs"], gdp,save_file = file_name,
                                         save_dir=config["root_save_dir"],device = device, wandb=wandb, seed=config["seed"])

                            results.append(performances)
                            wandb.finish()
                    else:
                        if encoder_ == "GCN":
                            encoder = GCNEncoder(data, out_channels, config["num_layers"],
                                                 message_sens=msg_sens).to(device)

                        elif encoder_ == "TransGCN_conv":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='TransE', variant='conv',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "TransGCN_attn":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='TransE', variant='attn',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "RotatEGCN_conv":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='RotatE', variant='conv',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "RotatEGCN_attn":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='RotatE', variant='attn',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)



                        elif encoder_ == "GAT":
                            encoder = GATEncoder(data, out_channels, config["num_layers"])
                        else:
                            print("invalid encoder type!")
                            raise ValueError("Invalid encoder type!")



                        run_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_GAE"
                        file_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_GAE"
                        run_config = {
                            "device": config["device"],
                            "num_layers": 2,
                            "alpha": config["alpha"],
                            "max_masking_percentage": config["max_masking_percentage"],
                            "total_drop_rate": config["total_drop_rate"],
                            "learning_rate": config["learning_rate"],
                            "batch_size": config["batch_size"],
                            "num_neighbors": [500, 500],
                            "num_epochs": 100,
                            "out_channels": out_channels,
                            "training_task": task,
                            "encoders": encoder_,
                            "decoders": "Dot Product",
                            "message_sens": msg_sens
                        }

                        wandb.init(
                            project=config["wandb_project_name"],
                            name=run_name,
                            config=run_config,
                            settings=wandb.Settings(start_method="thread")
                        )
                        local_data = copy.deepcopy(data)

                        autoencoder = GAE(encoder).to(device)
                        optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
                        performances = train_GAE(autoencoder, local_data, optimizer, config["num_epochs"], gdp,
                                                 save_file=file_name,
                                                 save_dir=config["root_save_dir"], device=device, wandb=wandb, seed = config["seed"])

                        results.append(performances)
                        wandb.finish()

        elif task in ("Recons_R", "Recons_R_with_onto"):
            use_onto = task == "Recons_R_with_onto"
            recons_r_mode = config.get("recons_r_training_mode")
            if removed_edge_indices is None:
                if recons_r_mode == "random_static_masked_only":
                    print("\n--- Preparing random static relation mask ONCE ---\n")
                    _, removed_edge_indices, removed_edge_types = random_edge_dropping(
                        data, config["total_drop_rate"], random_seed=active_seed
                    )
                elif recons_r_mode in ("removed_only", "random_masked_only", "balanced_static_masked_only"):
                    print("\n--- Preparing type-balanced static relation mask ONCE ---\n")
                    _, removed_edge_indices, removed_edge_types = relation_based_edge_dropping_balanced(
                        data, config["total_drop_rate"], max_drop_fraction_per_node=0.3, random_seed=active_seed
                    )
                if removed_edge_indices is not None:
                    removed_edge_indices = removed_edge_indices.to(device=device, dtype=torch.long)
                    removed_edge_types = removed_edge_types.to(device)

            for out_channels in config["hyperparams_grid"]["out_channels"]:
                for encoder_ in config["encoders"]:
                    if encoder_ == "RGCN":
                        for num_bases in config["hyperparams_grid"]["num_bases"]:
                            encoder = RGCNEncoder(data, out_channels, config["num_layers"], num_bases,
                                              message_sens=msg_sens).to(device)
                            run_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_Dismult"
                            file_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_Dismult"
                            run_config = {
                                "device": config["device"],
                                "num_layers": 2,
                                "alpha": config["alpha"],
                                "max_masking_percentage": config["max_masking_percentage"],
                                "total_drop_rate": config["total_drop_rate"],
                                "learning_rate": config["learning_rate"],
                                "batch_size": config["batch_size"],
                                "num_neighbors": config["num_neighbors"],
                                "num_epochs": config["num_epochs"],
                                "bases": num_bases,
                                "out_channels": out_channels,
                                "training_task": task,
                                "recons_r_training_mode": config.get("recons_r_training_mode"),
                                "recons_r_target_relation_field": config.get("recons_r_target_relation_field"),
                                "lambda_onto": config.get("lambda_onto"),
                                "lambda_align": config.get("lambda_align"),
                                "relation_alignment_loss": config.get("relation_alignment_loss"),
                                "lambda_core_contrastive": config.get("lambda_core_contrastive"),
                                "core_contrastive_temperature": config.get("core_contrastive_temperature"),
                                "lambda_core_align": config.get("lambda_core_align"),
                                "core_alignment_loss": config.get("core_alignment_loss"),
                                "lambda_domain_range": config.get("lambda_domain_range"),
                                "domain_range_temperature": config.get("domain_range_temperature"),
                                "lambda_domain_range_embedding": config.get("lambda_domain_range_embedding"),
                                "domain_range_embedding_temperature": config.get("domain_range_embedding_temperature"),
                                "lambda_onto_hierarchy": config.get("lambda_onto_hierarchy"),
                                "negative_sampling_mode": config.get("negative_sampling_mode"),
                                "negative_corruption_mode": config.get("negative_corruption_mode"),
                                "negative_entity_sampling_scope": config.get("negative_entity_sampling_scope"),
                                "kg_negative_sampling_seed": config.get("kg_negative_sampling_seed"),
                                "soft_type_negative_ratio": config.get("soft_type_negative_ratio"),
                                "soft_type_top_k": config.get("soft_type_top_k"),
                                "soft_type_temperature": config.get("soft_type_temperature"),
                                "encoders": encoder_,
                                "decoders": "DisMult",
                                "message_sens": msg_sens
                            }

                            wandb.init(
                                project=config["wandb_project_name"],
                                name=run_name,
                                config=run_config,
                                settings=wandb.Settings(start_method="thread")
                            )
                            r_decoder = DistMultDecoder(data.num_recons_edge_types, out_channels[-1])
                            autoencoder = MRGAE(encoder,x_decoder = None, r_decoder= r_decoder).to(device)
                            onto_r_decoder = None
                            relation_projector = None
                            core_projector = None
                            optimizer_params = list(autoencoder.parameters())
                            if use_onto:
                                onto_r_decoder = DistMultDecoder(int(onto_data.edge_type.max().item()) + 1, out_channels[-1]).to(device)
                                optimizer_params += list(onto_r_decoder.parameters())
                                if kg_relation_align_ids is not None and config.get("lambda_align", 0) != 0:
                                    relation_projector = torch.nn.Linear(out_channels[-1], out_channels[-1]).to(device)
                                    optimizer_params += list(relation_projector.parameters())
                                if kg_core_ids is not None and config.get("lambda_core_align", 0) != 0:
                                    core_projector = torch.nn.Linear(out_channels[-1], out_channels[-1]).to(device)
                                    optimizer_params += list(core_projector.parameters())
                            optimizer = optim.Adam(optimizer_params, lr=config["learning_rate"])

                            local_data = copy.deepcopy(data)

                            if use_onto:
                                performances = train_DisMult_with_onto(
                                    autoencoder, local_data, onto_data, onto_r_decoder, optimizer,
                                    config["num_epochs"], gdp, file_name, device,
                                    save_dir=config["root_save_dir"]+"/"+task, wandb=wandb, seed=config["seed"],
                                    removed_edge_indices=removed_edge_indices, removed_edge_types=removed_edge_types,
                                    lambda_onto=config["lambda_onto"],
                                    relation_projector=relation_projector,
                                    kg_relation_align_ids=kg_relation_align_ids,
                                    onto_relation_align_ids=onto_relation_align_ids,
                                    lambda_align=config["lambda_align"],
                                    relation_alignment_loss=config["relation_alignment_loss"],
                                    onto_gdp=onto_gdp,
                                    shared_relations=shared_relations,
                                    visualizations_dir=os.path.join(config["root_save_dir"], "visualizations", file_name),
                                    kg_core_ids=kg_core_ids,
                                    onto_core_ids=onto_core_ids,
                                    lambda_core_contrastive=config["lambda_core_contrastive"],
                                    core_contrastive_temperature=config["core_contrastive_temperature"],
                                    lambda_core_align=config["lambda_core_align"],
                                    core_alignment_loss=config["core_alignment_loss"],
                                    core_projector=core_projector,
                                    domain_range_type_ids=domain_range_type_ids,
                                    domain_mask_by_relation=domain_mask_by_relation,
                                    range_mask_by_relation=range_mask_by_relation,
                                    lambda_domain_range=config["lambda_domain_range"],
                                    domain_range_temperature=config["domain_range_temperature"],
                                    domain_range_embedding_type_ids=domain_range_embedding_type_ids,
                                    domain_explicit_mask_by_relation=domain_explicit_mask_by_relation,
                                    range_explicit_mask_by_relation=range_explicit_mask_by_relation,
                                    domain_allowed_mask_by_relation=domain_allowed_mask_by_relation,
                                    range_allowed_mask_by_relation=range_allowed_mask_by_relation,
                                    lambda_domain_range_embedding=config["lambda_domain_range_embedding"],
                                    domain_range_embedding_temperature=config["domain_range_embedding_temperature"],
                                    onto_hierarchy_child_ids=onto_hierarchy_child_ids,
                                    onto_hierarchy_parent_ids=onto_hierarchy_parent_ids,
                                    lambda_onto_hierarchy=config["lambda_onto_hierarchy"],
                                    negative_sampling_mode=config["negative_sampling_mode"],
                                    negative_corruption_mode=config["negative_corruption_mode"],
                                    soft_type_candidates=soft_type_negative_candidates,
                                    soft_type_negative_ratio=config["soft_type_negative_ratio"],
                                )
                            else:
                                performances = train_DisMult(
                                    autoencoder, local_data, optimizer, config["num_epochs"], gdp, file_name, device,
                                    save_dir=config["root_save_dir"]+"/"+task, wandb=wandb, seed=config["seed"],
                                    removed_edge_indices=removed_edge_indices, removed_edge_types=removed_edge_types,
                                    domain_range_type_ids=domain_range_type_ids,
                                    domain_mask_by_relation=domain_mask_by_relation,
                                    range_mask_by_relation=range_mask_by_relation,
                                    lambda_domain_range=config["lambda_domain_range"],
                                    domain_range_temperature=config["domain_range_temperature"],
                                    domain_range_embedding_type_ids=domain_range_embedding_type_ids,
                                    domain_explicit_mask_by_relation=domain_explicit_mask_by_relation,
                                    range_explicit_mask_by_relation=range_explicit_mask_by_relation,
                                    domain_allowed_mask_by_relation=domain_allowed_mask_by_relation,
                                    range_allowed_mask_by_relation=range_allowed_mask_by_relation,
                                    lambda_domain_range_embedding=config["lambda_domain_range_embedding"],
                                    domain_range_embedding_temperature=config["domain_range_embedding_temperature"],
                                    onto_hierarchy_child_ids=onto_hierarchy_child_ids,
                                    onto_hierarchy_parent_ids=onto_hierarchy_parent_ids,
                                    lambda_onto_hierarchy=config["lambda_onto_hierarchy"],
                                    negative_sampling_mode=config["negative_sampling_mode"],
                                    negative_corruption_mode=config["negative_corruption_mode"],
                                    soft_type_candidates=soft_type_negative_candidates,
                                    soft_type_negative_ratio=config["soft_type_negative_ratio"],
                                )
                            # performances = train_GAE(autoencoder, data, optimizer, config["num_epochs"], gdp,save_file = file_name,
                            #              save_dir=config["root_save_dir"],device = device, wandb=wandb)
                            #
                            #
                            results.append(performances)

                            wandb.finish()
                    else:
                        if encoder_ == "GCN":
                            encoder = GCNEncoder(data, out_channels, config["num_layers"],
                                             message_sens=msg_sens).to(device)


                        elif encoder_ == "TransGCN_conv":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='TransE', variant='conv',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "TransGCN_attn":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='TransE', variant='attn',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "RotatEGCN_conv":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='RotatE', variant='conv',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "RotatEGCN_attn":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='RotatE', variant='attn',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)





                        elif encoder_ == "GAT":
                            encoder = GATEncoder(data, out_channels, config["num_layers"])

                        else:
                            print("invalid encoder type!")
                            raise ValueError("Invalid encoder type!")


                        run_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_Dismult"
                        file_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_Dismult"
                        run_config = {
                            "device": config["device"],
                            "num_layers": 2,
                            "alpha": config["alpha"],
                            "max_masking_percentage": config["max_masking_percentage"],
                            "total_drop_rate": config["total_drop_rate"],
                            "learning_rate": config["learning_rate"],
                            "batch_size": config["batch_size"],
                            "num_neighbors": config["num_neighbors"],
                            "num_epochs": config["num_epochs"],
                            "out_channels": out_channels,
                            "training_task": task,
                            "recons_r_training_mode": config.get("recons_r_training_mode"),
                            "recons_r_target_relation_field": config.get("recons_r_target_relation_field"),
                            "lambda_onto": config.get("lambda_onto"),
                            "lambda_align": config.get("lambda_align"),
                            "relation_alignment_loss": config.get("relation_alignment_loss"),
                            "lambda_core_contrastive": config.get("lambda_core_contrastive"),
                            "core_contrastive_temperature": config.get("core_contrastive_temperature"),
                            "lambda_core_align": config.get("lambda_core_align"),
                            "core_alignment_loss": config.get("core_alignment_loss"),
                            "lambda_domain_range": config.get("lambda_domain_range"),
                            "domain_range_temperature": config.get("domain_range_temperature"),
                            "lambda_domain_range_embedding": config.get("lambda_domain_range_embedding"),
                            "domain_range_embedding_temperature": config.get("domain_range_embedding_temperature"),
                            "lambda_onto_hierarchy": config.get("lambda_onto_hierarchy"),
                            "negative_sampling_mode": config.get("negative_sampling_mode"),
                            "negative_corruption_mode": config.get("negative_corruption_mode"),
                            "negative_entity_sampling_scope": config.get("negative_entity_sampling_scope"),
                            "kg_negative_sampling_seed": config.get("kg_negative_sampling_seed"),
                            "soft_type_negative_ratio": config.get("soft_type_negative_ratio"),
                            "soft_type_top_k": config.get("soft_type_top_k"),
                            "soft_type_temperature": config.get("soft_type_temperature"),
                            "encoders": encoder_,
                            "decoders": "DisMult",
                            "message_sens": msg_sens
                        }

                        wandb.init(
                            project=config["wandb_project_name"],
                            name=run_name,
                            config=run_config,
                            settings=wandb.Settings(start_method="thread")
                        )
                        r_decoder = DistMultDecoder(data.num_recons_edge_types, out_channels[-1])
                        autoencoder = MRGAE(encoder, x_decoder=None, r_decoder=r_decoder).to(device)
                        onto_r_decoder = None
                        relation_projector = None
                        core_projector = None
                        optimizer_params = list(autoencoder.parameters())
                        if use_onto:
                            onto_r_decoder = DistMultDecoder(int(onto_data.edge_type.max().item()) + 1, out_channels[-1]).to(device)
                            optimizer_params += list(onto_r_decoder.parameters())
                            if kg_relation_align_ids is not None and config.get("lambda_align", 0) != 0:
                                relation_projector = torch.nn.Linear(out_channels[-1], out_channels[-1]).to(device)
                                optimizer_params += list(relation_projector.parameters())
                            if kg_core_ids is not None and config.get("lambda_core_align", 0) != 0:
                                core_projector = torch.nn.Linear(out_channels[-1], out_channels[-1]).to(device)
                                optimizer_params += list(core_projector.parameters())
                        optimizer = optim.Adam(optimizer_params, lr=config["learning_rate"])

                        if use_onto:
                            performances = train_DisMult_with_onto(
                                autoencoder, data, onto_data, onto_r_decoder, optimizer, config["num_epochs"], gdp, file_name,
                                device, save_dir=config["root_save_dir"]+"/"+task, wandb=wandb, seed=config["seed"],
                                removed_edge_indices=removed_edge_indices, removed_edge_types=removed_edge_types,
                                lambda_onto=config["lambda_onto"],
                                relation_projector=relation_projector,
                                kg_relation_align_ids=kg_relation_align_ids,
                                onto_relation_align_ids=onto_relation_align_ids,
                                lambda_align=config["lambda_align"],
                                relation_alignment_loss=config["relation_alignment_loss"],
                                onto_gdp=onto_gdp,
                                shared_relations=shared_relations,
                                visualizations_dir=os.path.join(config["root_save_dir"], "visualizations", file_name),
                                kg_core_ids=kg_core_ids,
                                onto_core_ids=onto_core_ids,
                                lambda_core_contrastive=config["lambda_core_contrastive"],
                                core_contrastive_temperature=config["core_contrastive_temperature"],
                                lambda_core_align=config["lambda_core_align"],
                                core_alignment_loss=config["core_alignment_loss"],
                                core_projector=core_projector,
                                domain_range_type_ids=domain_range_type_ids,
                                domain_mask_by_relation=domain_mask_by_relation,
                                range_mask_by_relation=range_mask_by_relation,
                                lambda_domain_range=config["lambda_domain_range"],
                                domain_range_temperature=config["domain_range_temperature"],
                                domain_range_embedding_type_ids=domain_range_embedding_type_ids,
                                domain_explicit_mask_by_relation=domain_explicit_mask_by_relation,
                                range_explicit_mask_by_relation=range_explicit_mask_by_relation,
                                domain_allowed_mask_by_relation=domain_allowed_mask_by_relation,
                                range_allowed_mask_by_relation=range_allowed_mask_by_relation,
                                lambda_domain_range_embedding=config["lambda_domain_range_embedding"],
                                domain_range_embedding_temperature=config["domain_range_embedding_temperature"],
                                onto_hierarchy_child_ids=onto_hierarchy_child_ids,
                                onto_hierarchy_parent_ids=onto_hierarchy_parent_ids,
                                lambda_onto_hierarchy=config["lambda_onto_hierarchy"],
                                negative_sampling_mode=config["negative_sampling_mode"],
                                negative_corruption_mode=config["negative_corruption_mode"],
                                soft_type_candidates=soft_type_negative_candidates,
                                soft_type_negative_ratio=config["soft_type_negative_ratio"],
                            )
                        else:
                            performances = train_DisMult(
                                autoencoder, data, optimizer, config["num_epochs"], gdp, file_name,
                                device, save_dir=config["root_save_dir"]+"/"+task, wandb=wandb, seed=config["seed"],
                                removed_edge_indices=removed_edge_indices, removed_edge_types=removed_edge_types,
                                domain_range_type_ids=domain_range_type_ids,
                                domain_mask_by_relation=domain_mask_by_relation,
                                range_mask_by_relation=range_mask_by_relation,
                                lambda_domain_range=config["lambda_domain_range"],
                                domain_range_temperature=config["domain_range_temperature"],
                                domain_range_embedding_type_ids=domain_range_embedding_type_ids,
                                domain_explicit_mask_by_relation=domain_explicit_mask_by_relation,
                                range_explicit_mask_by_relation=range_explicit_mask_by_relation,
                                domain_allowed_mask_by_relation=domain_allowed_mask_by_relation,
                                range_allowed_mask_by_relation=range_allowed_mask_by_relation,
                                lambda_domain_range_embedding=config["lambda_domain_range_embedding"],
                                domain_range_embedding_temperature=config["domain_range_embedding_temperature"],
                                onto_hierarchy_child_ids=onto_hierarchy_child_ids,
                                onto_hierarchy_parent_ids=onto_hierarchy_parent_ids,
                                lambda_onto_hierarchy=config["lambda_onto_hierarchy"],
                                negative_sampling_mode=config["negative_sampling_mode"],
                                negative_corruption_mode=config["negative_corruption_mode"],
                                soft_type_candidates=soft_type_negative_candidates,
                                soft_type_negative_ratio=config["soft_type_negative_ratio"],
                            )
                        results.append(performances)

                        wandb.finish()

        elif task == "Double_reconstruction":
            if masked_features_data is None:
                print("\n--- Preparing feature mask ONCE for double reconstruction ---\n")
                masked_features_data = view_partial_features_masking(
                    data, max_masking_percentage=config["max_masking_percentage"], random_seed=active_seed
                )
            if removed_edge_indices is None:
                print("\n--- Preparing relation mask ONCE for double reconstruction ---\n")
                _, removed_edge_indices, removed_edge_types = relation_based_edge_dropping_balanced(
                    data, config["total_drop_rate"], max_drop_fraction_per_node=0.3, random_seed=active_seed
                )
                removed_edge_indices = removed_edge_indices.to(device)
                removed_edge_types = removed_edge_types.to(device)

            for cmb in config["param_combinations"]:
                if cmb["encoder"] == "GCN":
                    encoder = GCNEncoder(data, cmb["out_channels"], config["num_layers"],
                                             message_sens=msg_sens).to(device)
                elif cmb["encoder"] == "RGCN":
                    encoder = RGCNEncoder(data, cmb["out_channels"], config["num_layers"], 5,
                                          message_sens=msg_sens).to(device)
                elif cmb["encoder"] == "TransGCN":
                    encoder = TransGCNEncoder(data, cmb["out_channels"], config["num_layers"], dropout=0.2,
                                              kg_score_fn=config["kg_score_fn"], variant=config["variant"],
                                              use_edges_info=config["use_edges_info"], activation='relu',
                                              bias=False).to(device)


                else:
                    print("invalid encoder type!")
                    raise ValueError("Invalid encoder type!")


                if cmb["decoder"] == "GCN":
                    decoder = GCNDecoder(encoder, data, config["alpha"], message_sens=msg_sens).to(device)
                elif cmb["decoder"] == "RGCN":
                    decoder = RGCNDecoder(encoder, data, 5, config["alpha"],
                                          message_sens=msg_sens).to(device)
                elif cmb["decoder"] == "MLP":
                    decoder = MLPDecoder(encoder, data, config["alpha"]).to(device)
                elif cmb["decoder"] == "TransGCN":
                    decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                              kg_score_fn=config["kg_score_fn"], variant=config["variant"],
                                              use_edges_info=config["use_edges_info"]).to(device)


                else:
                    print("invalid decoder type!")
                    raise ValueError("Invalid encoder type!")


                run_name = f"{task}_channels_{'-'.join(map(str, cmb['out_channels']))}_enc-{cmb['encoder']}_dec-{cmb['decoder']}_R_Dismult"
                file_name = f"{task}_channels_{'-'.join(map(str, cmb['out_channels']))}_enc-{cmb['encoder']}_dec-{cmb['decoder']}_R_Dismult"
                run_config = {
                    "device": config["device"],
                    "num_layers": 2,
                    "alpha": config["alpha"],
                    "max_masking_percentage": config["max_masking_percentage"],
                    "total_drop_rate": config["total_drop_rate"],
                    "learning_rate": config["learning_rate"],
                    "batch_size": config["batch_size"],
                    "num_neighbors": config["num_neighbors"],
                    "num_epochs": config["num_epochs"],
                    "out_channels": cmb["out_channels"],
                    "training_task": task,
                    "encoder": cmb["encoder"],
                    "decoder": cmb["decoder"],
                    "r_decoder": "DisMult",
                    "message_sens": msg_sens
                }

                wandb.init(
                    project=config["wandb_project_name"],
                    name=run_name,
                    config=run_config,
                    settings=wandb.Settings(start_method="thread")
                )



                r_decoder = DistMultDecoder(data.num_edge_types, cmb["out_channels"][-1])
                local_data = copy.deepcopy(data)

                autoencoder = MRGAE(encoder, x_decoder=decoder, r_decoder=r_decoder).to(device)
                optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])

                performances = train_Double_Reconstruction(
                    autoencoder, local_data, optimizer, config["num_epochs"], gdp, file_name,
                    device, save_dir=config["root_save_dir"] + "/" + task, wandb=wandb, seed=config["seed"],
                    masked_features_data=masked_features_data,
                    removed_edge_indices=removed_edge_indices,
                    removed_edge_types=removed_edge_types,
                )
                results.append(performances)

                wandb.finish()

        elif task == "Contrastive":
            if masked_features_data is None:
                print("\n--- Preparing feature mask ONCE for contrastive learning ---\n")
                masked_features_data = view_partial_features_masking(
                    data, max_masking_percentage=config["max_masking_percentage"], random_seed=active_seed
                )
            if removed_edge_indices is None:
                print("\n--- Preparing relation mask ONCE for contrastive learning ---\n")
                _, removed_edge_indices, _ = relation_based_edge_dropping_balanced(
                    data, config["total_drop_rate"], max_drop_fraction_per_node=0.3, random_seed=active_seed
                )
                removed_edge_indices = removed_edge_indices.to(device)
                        
            for out_channels in config["hyperparams_grid"]["out_channels"]:
                for encoder_ in config["encoders"]:
                    if encoder_ == "RGCN":
                        for num_bases in config["hyperparams_grid"]["num_bases"]:
                            encoder = RGCNEncoder(data, out_channels, config["num_layers"], num_bases,
                                                  message_sens=msg_sens).to(device)

                            run_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}"
                            file_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}"
                            run_config = {
                                "device": config["device"],
                                "num_layers": config["num_layers"],
                                "learning_rate": config["learning_rate"],
                                "batch_size": config["batch_size"],
                                "num_neighbors": config["num_neighbors"],
                                "num_epochs": config["num_epochs"],
                                "bases": num_bases,
                                "out_channels": out_channels,
                                "training_task": task,
                                "encoders": encoder_,
                                "projections": config["projections"]
                            }

                            wandb.init(
                                project=config["wandb_project_name"],
                                name=run_name,
                                config=run_config,
                                settings=wandb.Settings(start_method="thread")
                            )

                            autoencoder = MRGAE(encoder, x_decoder=None, r_decoder=None, projections=[out_channels[-1], out_channels[-1]]).to(
                                device)
                            optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
                            local_data = copy.deepcopy(data)

                            performances = train_Contrastive(
                                autoencoder, local_data, optimizer, config["num_epochs"], gdp, file_name,
                                masked_features_data, removed_edge_indices,
                                device=device, save_dir=config["root_save_dir"] + "/" + task,
                                wandb=wandb, seed=config["seed"]
                            )
                            results.append(performances)
                            wandb.finish()

                    else:  # GCN, TransGCN, etc.
                        if encoder_ == "GCN":
                            encoder = GCNEncoder(data, out_channels, config["num_layers"], message_sens=msg_sens).to(device)

                        elif encoder_ == "TransGCN_conv":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='TransE', variant='conv',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "TransGCN_attn":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='TransE', variant='attn',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "RotatEGCN_conv":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='RotatE', variant='conv',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "RotatEGCN_attn":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='RotatE', variant='attn',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "GAT":
                            encoder = GATEncoder(data, out_channels, config["num_layers"]).to(device)

                        else:
                            raise ValueError("Invalid encoder for Contrastive task!")

                        run_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}"
                        file_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}"
                        run_config = {
                            "device": config["device"],
                            "num_layers": config["num_layers"],
                            "learning_rate": config["learning_rate"],
                            "batch_size": config["batch_size"],
                            "num_neighbors": config["num_neighbors"],
                            "num_epochs": config["num_epochs"],
                            "out_channels": out_channels,
                            "training_task": task,
                            "encoders": encoder_,
                            "projections": config["projections"]
                        }

                        wandb.init(
                            project=config["wandb_project_name"],
                            name=run_name,
                            config=run_config,
                            settings=wandb.Settings(start_method="thread")
                        )
                        autoencoder = MRGAE(encoder, x_decoder=None, r_decoder=None, projections= [out_channels[-1], out_channels[-1]]).to(device)
                        optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
                        local_data = copy.deepcopy(data)


                        performances = train_Contrastive(
                            autoencoder, local_data, optimizer, config["num_epochs"], gdp, file_name,
                            masked_features_data, removed_edge_indices,
                            device=device, save_dir=config["root_save_dir"] + "/" + task,
                            wandb=wandb, seed = config["seed"]
                        )                       
                        results.append(performances)
                        wandb.finish()

    os.makedirs(config["root_save_dir"], exist_ok=True)
    results_path = os.path.join(config["root_save_dir"], f"results_seed_{config['seed']}.xlsx")
    if results and not os.path.exists(results_path):
        pd.DataFrame([row for row in results if row]).to_excel(results_path, index=False)
    print(f"Results saved incrementally in: {results_path}")
#
# def main():
#     print(44)

if __name__ == "__main__":
    main()
