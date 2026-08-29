import os
import pickle
from html import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

try:
    import plotly.graph_objects as go
    from plotly.offline import plot as plotly_plot
except ImportError:
    go = None
    plotly_plot = None


def _split_encoder_output(encoder_output):
    if isinstance(encoder_output, tuple):
        return encoder_output[0]
    return encoder_output


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def _safe_tsne(values):
    if values.shape[0] < 2:
        return None
    if values.shape[0] == 2:
        return np.array([[0.0, 0.0], [1.0, 0.0]])
    perplexity = min(30, values.shape[0] - 1)
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(values)


def _write_fallback_html(coords, labels, output_path, title, groups=None):
    if coords.size == 0:
        return
    x = coords[:, 0]
    y = coords[:, 1]
    x_range = max(x.max() - x.min(), 1e-9)
    y_range = max(y.max() - y.min(), 1e-9)
    width = 1000
    height = 800
    pad = 70
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    group_values = list(dict.fromkeys(groups or ["points"]))
    group_to_color = {group: colors[i % len(colors)] for i, group in enumerate(group_values)}

    points = []
    for i, label in enumerate(labels):
        group = groups[i] if groups is not None else "points"
        cx = pad + ((x[i] - x.min()) / x_range) * (width - 2 * pad)
        cy = height - pad - ((y[i] - y.min()) / y_range) * (height - 2 * pad)
        points.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="7" fill="{group_to_color[group]}">'
            f'<title>{escape(group)}: {escape(label)}</title></circle>'
            f'<text x="{cx + 9:.2f}" y="{cy + 4:.2f}" font-size="10">{escape(label)}</text>'
        )

    legend = "".join(
        f'<span style="margin-right:16px"><span style="display:inline-block;width:10px;height:10px;'
        f'background:{group_to_color[group]};border-radius:50%"></span> {escape(group)}</span>'
        for group in group_values
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
    svg {{ border: 1px solid #e5e7eb; width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h2>{escape(title)}</h2>
  <div>{legend}</div>
  <p>Hover on points to inspect labels. Plotly is not installed, so this fallback SVG was generated.</p>
  <svg viewBox="0 0 {width} {height}" role="img">
    {''.join(points)}
  </svg>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html)


def _write_plotly_html(coords, labels, output_path, title, groups=None):
    if go is None or plotly_plot is None:
        _write_fallback_html(coords, labels, output_path, title, groups=groups)
        return

    fig = go.Figure()
    if groups is None:
        fig.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1], mode="markers+text",
            text=labels, textposition="top center", hovertext=labels,
            hoverinfo="text", marker={"size": 10},
        ))
    else:
        for group in list(dict.fromkeys(groups)):
            idx = [i for i, value in enumerate(groups) if value == group]
            fig.add_trace(go.Scatter(
                x=coords[idx, 0], y=coords[idx, 1], mode="markers+text",
                text=[labels[i] for i in idx], textposition="top center",
                hovertext=[labels[i] for i in idx], hoverinfo="text",
                marker={"size": 10}, name=group,
            ))
    fig.update_layout(
        title=title,
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        template="plotly_white",
        hovermode="closest",
    )
    plotly_plot(fig, filename=output_path, auto_open=False, include_plotlyjs=True)


def _plot_points(points, labels, png_output_path, title, groups=None, html_output_path=None):
    coords = _safe_tsne(points)
    if coords is None:
        return

    os.makedirs(os.path.dirname(png_output_path), exist_ok=True)

    plt.figure(figsize=(10, 8))
    if groups is None:
        plt.scatter(coords[:, 0], coords[:, 1], s=70)
    else:
        unique_groups = list(dict.fromkeys(groups))
        for group in unique_groups:
            idx = [i for i, value in enumerate(groups) if value == group]
            plt.scatter(coords[idx, 0], coords[idx, 1], s=70, label=group)
        plt.legend()

    for i, label in enumerate(labels):
        plt.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=8, alpha=0.85)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(png_output_path, dpi=300)
    plt.close()
    if html_output_path is not None:
        os.makedirs(os.path.dirname(html_output_path), exist_ok=True)
        _write_plotly_html(coords, labels, html_output_path, title, groups=groups)


def _node_embeddings_by_name(model, data, gdp):
    model.eval()
    with torch.no_grad():
        embeddings = _split_encoder_output(model.encode(data))
    index_to_name = gdp.decode_indexes()
    return {index_to_name[idx]: embeddings[idx] for idx in index_to_name}


def _select_core_embeddings(named_embeddings, core_concepts):
    by_lower = {name.lower(): (name, emb) for name, emb in named_embeddings.items()}
    selected = {}
    for concept in core_concepts:
        item = by_lower.get(concept.lower())
        if item is not None:
            selected[item[0]] = item[1]
    return selected


def _pack_named_embeddings(named_embeddings):
    labels = list(named_embeddings.keys())
    if not labels:
        return {"labels": [], "embeddings": np.empty((0, 0))}
    return {
        "labels": labels,
        "embeddings": _to_numpy(torch.stack([named_embeddings[label] for label in labels])),
    }


def run_recons_r_with_onto_visualizations(
    model,
    data,
    gdp,
    onto_data,
    onto_gdp,
    onto_r_decoder,
    relation_projector,
    kg_relation_align_ids,
    onto_relation_align_ids,
    shared_relations,
    core_concepts,
    output_dir,
    prefix,
):
    os.makedirs(output_dir, exist_ok=True)

    artifacts = {}

    if relation_projector is not None and kg_relation_align_ids is not None and onto_relation_align_ids is not None:
        with torch.no_grad():
            kg_rel = model.r_decoder.relation_embedding[kg_relation_align_ids.to(model.r_decoder.relation_embedding.device)]
            onto_rel = onto_r_decoder.relation_embedding[onto_relation_align_ids.to(onto_r_decoder.relation_embedding.device)]
            projected_kg_rel = F.normalize(relation_projector(kg_rel), p=2, dim=1)
            onto_rel = F.normalize(onto_rel, p=2, dim=1)

        relation_points = torch.cat((projected_kg_rel, onto_rel), dim=0)
        relation_labels = [f"KG:{rel}" for rel in shared_relations] + [f"ONTO:{rel}" for rel in shared_relations]
        relation_groups = ["KG projected"] * len(shared_relations) + ["Ontology"] * len(shared_relations)
        artifacts["relation_projected_embeddings"] = {
            "labels": relation_labels,
            "groups": relation_groups,
            "embeddings": _to_numpy(relation_points),
        }
        _plot_points(
            _to_numpy(relation_points),
            relation_labels,
            os.path.join(output_dir, "relations_projected_tsne.png"),
            "Shared relation embeddings in projected ontology space",
            groups=relation_groups,
            html_output_path=os.path.join(output_dir, "relations_projected_tsne.html"),
        )

    kg_named_embeddings = _node_embeddings_by_name(model, data, gdp)
    artifacts["kg_all_node_embeddings"] = _pack_named_embeddings(kg_named_embeddings)
    kg_core_embeddings = _select_core_embeddings(kg_named_embeddings, core_concepts)
    artifacts["kg_core_concept_embeddings"] = {
        "labels": list(kg_core_embeddings.keys()),
        "embeddings": _to_numpy(torch.stack(list(kg_core_embeddings.values()))) if kg_core_embeddings else np.empty((0, 0)),
    }
    if kg_core_embeddings:
        _plot_points(
            artifacts["kg_core_concept_embeddings"]["embeddings"],
            artifacts["kg_core_concept_embeddings"]["labels"],
            os.path.join(output_dir, "kg_core_concepts_tsne.png"),
            "Core concepts in KG GNN space",
            html_output_path=os.path.join(output_dir, "kg_core_concepts_tsne.html"),
        )

    onto_named_embeddings = _node_embeddings_by_name(model, onto_data, onto_gdp)
    artifacts["onto_all_node_embeddings"] = _pack_named_embeddings(onto_named_embeddings)
    onto_core_embeddings = _select_core_embeddings(onto_named_embeddings, core_concepts)
    artifacts["onto_core_concept_embeddings"] = {
        "labels": list(onto_core_embeddings.keys()),
        "embeddings": _to_numpy(torch.stack(list(onto_core_embeddings.values()))) if onto_core_embeddings else np.empty((0, 0)),
    }
    if onto_core_embeddings:
        _plot_points(
            artifacts["onto_core_concept_embeddings"]["embeddings"],
            artifacts["onto_core_concept_embeddings"]["labels"],
            os.path.join(output_dir, "onto_core_concepts_tsne.png"),
            "Core concepts in ontology GNN space",
            html_output_path=os.path.join(output_dir, "onto_core_concepts_tsne.html"),
        )

    with open(os.path.join(output_dir, "embeddings.pkl"), "wb") as file:
        pickle.dump(artifacts, file)

    return artifacts
