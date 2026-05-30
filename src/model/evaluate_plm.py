import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def _add_project_paths():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    paths = [
        repo_root,
        script_dir,
        script_dir.parent / "layers",
        script_dir.parent / "bert_embedding",
        repo_root / "data",
        repo_root / "utilities",
    ]
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_add_project_paths()


def _read_gs(gs_path):
    suffix = Path(gs_path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(gs_path)
    return pd.read_excel(gs_path, sheet_name=0)


def _parse_core_concepts(raw_core_concepts, gs_path):
    if raw_core_concepts:
        return [concept.strip() for concept in raw_core_concepts.split(",") if concept.strip()]

    gs_df = _read_gs(gs_path)
    if "label" not in gs_df.columns:
        raise ValueError("Le fichier GS doit contenir une colonne 'label'.")

    labels = gs_df["label"].dropna().astype(str).unique().tolist()
    return [label for label in labels if label.lower() not in {"o", "other"}]


def _to_numpy_embedding(value):
    if hasattr(value, "detach"):
        value = value.squeeze().detach().cpu().numpy()
    return np.asarray(value).squeeze()


def _load_embedding_cache(entities_cache_path):
    if not entities_cache_path:
        return {}
    if not os.path.exists(entities_cache_path):
        return {}
    with open(entities_cache_path, "rb") as file:
        return pickle.load(file) or {}


def _read_models_file(models_file):
    models = []
    with open(models_file, "r", encoding="utf-8") as file:
        for line in file:
            model = line.strip()
            if model and not model.startswith("#"):
                models.append(model)
    return models


def _safe_model_name(model_name):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    return safe_name or "model"


def _export_path_for_model(export_preds_path, model_name, multiple_models):
    if not export_preds_path or not multiple_models:
        return export_preds_path

    export_path = Path(export_preds_path)
    suffix = export_path.suffix or ".xlsx"
    stem = export_path.stem if export_path.suffix else export_path.name
    return str(export_path.with_name(f"{stem}_{_safe_model_name(model_name)}{suffix}"))


def _save_summary(summary_df, summary_path):
    if not summary_path:
        return

    suffix = Path(summary_path).suffix.lower()
    if suffix == ".xlsx":
        summary_df.to_excel(summary_path, index=False)
    elif suffix == ".json":
        summary_df.to_json(summary_path, orient="records", indent=2)
    else:
        summary_df.to_csv(summary_path, index=False)


def _embed_terms(terms, embedder, cache):
    embeddings = {}
    for term in tqdm(terms, desc="Embedding"):
        if term in cache:
            embeddings[term] = _to_numpy_embedding(cache[term])
        else:
            embeddings[term] = _to_numpy_embedding(embedder.embed_entity(term))
    return embeddings


def _classify_by_cosine(gs_embeddings, core_concepts_embeddings):
    classifications = {}
    for term, term_embedding in gs_embeddings.items():
        best_class = None
        best_similarity = -1
        for concept, concept_embedding in core_concepts_embeddings.items():
            similarity = cosine_similarity([term_embedding], [concept_embedding])[0, 0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_class = concept
        classifications[term] = best_class
    return classifications


def _compute_classification_metrics(true_labels, predicted_labels):
    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
    }
    for average in ["macro", "micro", "weighted"]:
        metrics[f"f1_{average}"] = f1_score(true_labels, predicted_labels, average=average, zero_division=0)
        metrics[f"precision_{average}"] = precision_score(
            true_labels, predicted_labels, average=average, zero_division=0
        )
        metrics[f"recall_{average}"] = recall_score(true_labels, predicted_labels, average=average, zero_division=0)

    metrics["f1_score"] = metrics["f1_macro"]
    metrics["precision"] = metrics["precision_macro"]
    metrics["recall"] = metrics["recall_macro"]
    return metrics


def evaluate_plm(gs_path, plm_model, core_concepts, export_preds_path=None, entities_cache_path=None):
    from BertEmbedder import BertEmbedder

    gs_df = _read_gs(gs_path)
    if "term" not in gs_df.columns or "label" not in gs_df.columns:
        raise ValueError("Le fichier GS doit contenir les colonnes 'term' et 'label'.")

    terms = gs_df["term"].astype(str).tolist()
    true_labels = gs_df["label"].astype(str).tolist()

    cache = _load_embedding_cache(entities_cache_path)
    embedder = BertEmbedder(plm_model)
    gs_embeddings = _embed_terms(terms, embedder, cache)
    core_concepts_embeddings = _embed_terms(core_concepts, embedder, cache)
    classifications = _classify_by_cosine(gs_embeddings, core_concepts_embeddings)
    predicted_labels = [str(classifications[term]) for term in terms]

    if export_preds_path:
        pd.DataFrame(
            {
                "term": terms,
                "Predictions": predicted_labels,
                "Labels": true_labels,
            }
        ).to_excel(export_preds_path, index=False)

    print("\nRapport de Classification:\n")
    print(classification_report(true_labels, predicted_labels, zero_division=0))

    return _compute_classification_metrics(true_labels, predicted_labels)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluer un Gold Standard avec un modele PLM via similarite cosinus."
    )
    parser.add_argument("gs_path_pos", nargs="?", help="Chemin vers le fichier Gold Standard .xlsx.")
    parser.add_argument("plm_model_pos", nargs="?", help="Nom ou chemin HuggingFace du modele PLM.")
    parser.add_argument("--gs-path", default=None, help="Chemin vers le fichier Gold Standard .xlsx.")
    parser.add_argument("--plm-model", default=None, help="Nom ou chemin HuggingFace du modele PLM.")
    parser.add_argument(
        "--plm-models-file",
        default=None,
        help="Fichier .txt contenant un modele PLM par ligne. Alternative a --plm-model.",
    )
    parser.add_argument(
        "--core-concepts",
        default=None,
        help="Liste de concepts separes par des virgules. Par defaut: labels uniques du GS hors 'o'/'other'.",
    )
    parser.add_argument(
        "--entities-cache-path",
        default=None,
        help="Chemin optionnel vers un pickle d'embeddings d'entites deja calcules.",
    )
    parser.add_argument(
        "--export-preds-path",
        default=None,
        help="Chemin optionnel .xlsx pour exporter les predictions. En mode multi-modeles, un suffixe est ajoute.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Chemin optionnel pour exporter le resume des performances (.csv, .xlsx ou .json).",
    )
    args = parser.parse_args()

    gs_path_arg = args.gs_path or args.gs_path_pos
    plm_model_arg = args.plm_model or args.plm_model_pos
    if not gs_path_arg:
        parser.error("fournis le chemin GS en positionnel ou via --gs-path.")
    if args.plm_models_file and plm_model_arg:
        parser.error("utilise soit un modele PLM unique, soit --plm-models-file, pas les deux.")
    if args.plm_models_file:
        plm_models = _read_models_file(args.plm_models_file)
    elif plm_model_arg:
        plm_models = [plm_model_arg]
    else:
        parser.error("fournis un modele PLM ou un fichier --plm-models-file.")
    if not plm_models:
        parser.error("aucun modele PLM trouve.")

    gs_path = os.path.abspath(gs_path_arg)
    core_concepts = _parse_core_concepts(args.core_concepts, gs_path)
    multiple_models = len(plm_models) > 1
    summary_rows = []

    for plm_model in plm_models:
        print(f"\n===== Evaluation PLM: {plm_model} =====\n")
        row = {"model": plm_model}
        try:
            metrics = evaluate_plm(
                gs_path=gs_path,
                plm_model=plm_model,
                core_concepts=core_concepts,
                export_preds_path=_export_path_for_model(args.export_preds_path, plm_model, multiple_models),
                entities_cache_path=args.entities_cache_path,
            )
            row.update(metrics)
        except Exception as exc:
            row["error"] = str(exc)
            print(f"Erreur avec le modele {plm_model}: {exc}")
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print("\nSummary:\n")
    print(summary_df.to_string(index=False))
    print("\nSummary JSON:\n")
    print(json.dumps(summary_rows, indent=2))
    _save_summary(summary_df, args.summary_path)


if __name__ == "__main__":
    main()
