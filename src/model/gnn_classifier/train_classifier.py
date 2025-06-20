import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score
from torch import nn
import torch.optim as optim
from utilities.utilities import set_seed
import os
import json
import torch
import pandas as pd


## Full Batch Training
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    mask = data.train_mask & (data.y >= 0)  # only labeled nodes
    loss = criterion(out[mask], data.y[mask])
    # loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

## Full batch evaluation
@torch.no_grad()
def evaluate(model, data, mask_name):
    model.eval()
    out = model(data)
    mask = getattr(data, f"{mask_name}_mask") & (data.y >= 0)  # ignore unlabeled (-1)
    y_true = data.y[mask].cpu()
    y_pred = out[mask].argmax(dim=1).cpu()
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def training_loop(model, data, config, epochs=100, seed = 42):
    set_seed(seed)

    device = torch.device(config["device"])
    model = model.to(device)
    data = data.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 0.001), weight_decay=config.get("weight_decay", 5e-4))
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    best_test_f1 = 0.0

    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer, criterion)

        train_metrics = evaluate(model, data, "train")
        test_metrics = evaluate(model, data, "test")

        if test_metrics["f1"] > best_test_f1:
            best_test_f1 = test_metrics["f1"]

        print(
            f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}"
        )

    print(f"\nBest Test F1-score: {best_test_f1:.4f}")
##### Mini Batching ######


## MINI batch Training
def batch_train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    mask_input_id = torch.isin(data.n_id, data.input_id)
    mask = mask_input_id & data.train_mask & (data.y >= 0)
    loss = criterion(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

### Mini Batch testing
@torch.no_grad()
def evaluate_batch(model, data_loader, mask_name, config):
    model.eval()
    set_seed(42)
    y_true = []
    preds = []
    for batch in data_loader:
        batch.to(config["device"])
        out = model(batch)
        mask_input_id = torch.isin(batch.n_id, batch.input_id)
        mask = mask_input_id & getattr(batch, mask_name) & (batch.y >= 0)
        batch_y_true = batch.y[mask].cpu()
        batch_y_pred = out[mask].argmax(dim=1).cpu()
        y_true.append(batch_y_true)
        preds.append(batch_y_pred)

    # Concatenate all batches
    y_true = torch.cat(y_true, dim=0).numpy()
    preds = torch.cat(preds, dim=0).numpy()

    # Calculate metrics with sklearn
    accuracy = accuracy_score(y_true, preds)
    recall = recall_score(y_true, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_true, preds, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'recall': recall,
        'f1': f1
    }


def training_loop_minibatch(model, train_loader, test_loader, config, epochs=100, seed=42):
    set_seed(seed)
    device = torch.device(config["device"])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 0.001),
                           weight_decay=config.get("weight_decay", 5e-4))
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    best_test_acc = 0.0
    print("Starting training...")
    print("=" * 60)
    best_model_state = None
    for epoch in range(1, epochs + 1):
        loss = 0
        for batch in train_loader:
            batch.to("cuda")
            batch_loss = batch_train(model, batch, optimizer, criterion)
            loss += batch_loss

        train_metrics = evaluate_batch(model, train_loader, "train_mask", config)
        test_metrics = evaluate_batch(model, test_loader, "test_mask", config)
        avg_loss = loss / len(train_loader)

        # Update best accuracy
        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']
            best_test_metrics = test_metrics
            best_model_state = model.state_dict()

        # Clean display
        print(f"Epoch {epoch:3d}/{epochs}")
        print(f"  Loss:     {avg_loss:.4f}")
        print(
            f"  Train -   Acc: {train_metrics['accuracy']:.4f} | Recall: {train_metrics['recall']:.4f} | F1: {train_metrics['f1']:.4f}")
        print(
            f"  Test  -   Acc: {test_metrics['accuracy']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")
        print(f"  Best Test Acc: {best_test_acc:.4f}")
        print("-" * 60)
        print(f"\nTraining completed! Best Test Accuracy: {best_test_acc:.4f}")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)  # Optionnel : recharger dans le modèle courant
        save_best_classifier_and_config(model, config, best_test_metrics)


def save_best_classifier_and_config(model, config, test_metrics, directory="checkpoints", excel_file="classifier_results.xlsx"):
    """
    Save the best model and its config, and append results to Excel.

    Parameters:
    - model: trained PyTorch model
    - config: configuration dictionary
    - test_metrics: dict containing 'accuracy', 'recall', 'f1'
    - directory: where to save the model/config
    - excel_file: where to log results
    """
    os.makedirs(directory, exist_ok=True)

    encoder_name = config.get("classifier_encoder", "unknownEncoder")
    dataset_name = config.get("dataset", "noDataset")

    # Clés spécifiques selon l'encodeur
    keys_to_include = []
    if encoder_name == 'RGCN':
        keys_to_include.append("num_bases")

    config_str_parts = []
    for k in keys_to_include:
        v = config.get(k)
        if isinstance(v, list):
            v = "-".join(map(str, v))
        config_str_parts.append(f"{k}{v}")

    # Nom MLP
    mlp = "MLP_8"
    if config.get('MLP_layers'):
        mlp = "MLP_" + '_'.join(map(str, config.get('MLP_layers'))) + "_8"

    encoder_out_channels = '_'.join(map(str, config.get("encoder_out_channels", [])))
    base_name = "_".join([dataset_name, encoder_name, encoder_out_channels] + config_str_parts + [mlp])

    # Sauvegarde modèle et config
    model_path = os.path.join(directory, f"{base_name}.pt")
    config_path = os.path.join(directory, f"{base_name}.json")

    torch.save(model.state_dict(), model_path)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Model saved to: {model_path}")
    print(f"Config saved to: {config_path}")

    # Préparation ligne Excel
    row_data = {
        "Graph": dataset_name,
        "classifier_encoder": encoder_name,
        "Nb_GNN_layers": config.get("num_layers", "_"),
        "encoder_out_channels": encoder_out_channels,
        "MLP_layers": "-".join(map(str, config.get("MLP_layers", []))) if config.get("MLP_layers") else "8",
        "num_neighbors": "-".join(map(str, config.get("num_neighbors", []))) if config.get("num_neighbors") else "_",
        "train_batch_size": config.get("train_batch_size", ""),
        "test_batch_size": config.get("test_batch_size", ""),
        "num_bases": "-".join(map(str, config.get("num_bases", []))) if encoder_name == "RGCN" else "_",
        "Test_Accuracy": round(test_metrics.get("accuracy", 0), 4),
        "Test_Recall": round(test_metrics.get("recall", 0), 4),
        "Test_F1": round(test_metrics.get("f1", 0), 4)
    }

    # Ajout au fichier Excel
    if os.path.exists(excel_file):
        df_existing = pd.read_excel(excel_file)
        df_updated = pd.concat([df_existing, pd.DataFrame([row_data])], ignore_index=True)
    else:
        df_updated = pd.DataFrame([row_data])

    df_updated.to_excel(excel_file, index=False)
    print(f"Results appended to: {excel_file}")

