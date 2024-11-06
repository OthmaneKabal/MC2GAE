import torch
config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "out_channels": [128, 64],
    "num_layers": 2,
    "num_bases": 30,
    "alpha": 0.01,
    "max_masking_percentage": 0.3,
    "total_drop_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 10,
    "Entities_path": "../outputs/EntitiesBertEmbeddingAugmented.pickle",
    "Edges_path": "../outputs/PredicatesBertEmbeddingAugmented.pickle",
    "KG_path": "../../data/graph_filtred_gpt_20.json"
}
