import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.nn import GAE

from src.layers.GATEncoder import GATEncoder
from src.layers.GCNEncoder import GCNEncoder
from src.layers.RGCNEncoder import RGCNEncoder
from src.layers.TransGCNEncoder import TransGCNEncoder
from src.model.MRGAE import MRGAE
from src.model.gnn_classifier.classifier_utils import instantiate_decoder


def load_gold_standard_labels(gs_path):
    """
    Charge les labels du gold standard à partir d'un fichier Excel.

    :param gs_path: Chemin vers le fichier Excel contenant le GS
    :return: DataFrame avec les colonnes 'term' et 'label'
    """
    # Charger la feuille contenant les termes et labels
    gold_standard_df = pd.read_excel(gs_path, sheet_name='Sheet1')

    # Vérifier que les colonnes nécessaires sont présentes
    if 'term' not in gold_standard_df.columns or 'label' not in gold_standard_df.columns:
        raise ValueError("Le fichier GS doit contenir les colonnes 'term' et 'label'.")

    return gold_standard_df[['term', 'label']]


def save_model(model, optimizer, epoch, save_dir="checkpoints", name = False):
    """
    Saves the model and optimizer state dictionaries.

    Parameters:
    - model (nn.Module): The model to save.
    - optimizer (torch.optim.Optimizer): The optimizer to save.
    - epoch (int): The current epoch number.
    - save_dir (str): Directory where the checkpoint will be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define the checkpoint path
    checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")

    if name:
        checkpoint_path = os.path.join(save_dir, f"best_model.pth")


    # Save the model and optimizer states

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    print(f"Model saved at '{checkpoint_path}'")


def load_model_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer states from a checkpoint file.

    Parameters:
    - model (nn.Module): The model to load the state dictionary into.
    - optimizer (torch.optim.Optimizer): The optimizer to load the state dictionary into.
    - checkpoint_path (str): Path to the checkpoint file.

    Returns:
    - model: The model with loaded weights.
    - optimizer: The optimizer with loaded state.
    - start_epoch (int): The epoch to resume training from.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    # Load state dictionaries
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Retrieve the last completed epoch
    start_epoch = checkpoint['epoch']

    print(f"Checkpoint loaded from '{checkpoint_path}', resuming from epoch {start_epoch}")

    return model, optimizer, start_epoch


def generate_relation_embeddings_tensor(relations, embedding_size, device, seed=42):
    """
    Génère un tenseur d'embedding pour des relations uniques.

    Args:
        relations (list): Liste des relations uniques.
        embedding_size (int): Taille des vecteurs d'embedding.
        seed (int): Seed pour garantir la reproductibilité.

    Returns:
        torch.Tensor: Tenseur contenant les embeddings de taille [nb_relations, embedding_size].
    """
    # Fixer le seed pour la reproductibilité
    torch.manual_seed(seed)

    # Nombre de relations uniques
    num_relations = len(relations)

    # Définir une couche d'embedding
    embedding_layer = torch.nn.Embedding(num_relations, embedding_size)

    # Générer les embeddings pour tous les indices
    indices = torch.arange(num_relations)
    embeddings = embedding_layer(indices)  # Shape: [num_relations, embedding_size]

    return embeddings.to(device)





# def removed_edges_train_test_split(indices: torch.Tensor, labels: torch.Tensor, test_size=0.2, random_state=42,output_device="cuda"):
#     """
#     Split GPU tensors into training and testing sets, preserving class balance, with a specified output device.
#
#     Args:
#         indices (torch.Tensor): Tensor of indices (on GPU).
#         labels (torch.Tensor): Tensor of labels (on GPU).
#         test_size (float): Proportion of the dataset to include in the test split.
#         random_state (int): Seed for reproducibility.
#         output_device (str): Device for the output tensors (e.g., "cuda" or "cpu").
#
#     Returns:
#         train_indices (torch.Tensor): Tensor of training indices (on output_device).
#         test_indices (torch.Tensor): Tensor of testing indices (on output_device).
#         train_labels (torch.Tensor): Tensor of training labels (on output_device).
#         test_labels (torch.Tensor): Tensor of testing labels (on output_device).
#     """
#
#     # Move tensors to CPU for compatibility with StratifiedShuffleSplit
#     indices_cpu = indices.cpu().numpy()
#     labels_cpu = labels.cpu().numpy()
#
#     # Initialize StratifiedShuffleSplit
#     splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
#
#     # Perform the split
#     for train_idx, test_idx in splitter.split(indices_cpu, labels_cpu):
#         train_indices = torch.tensor(indices_cpu[train_idx], device=output_device)
#         test_indices = torch.tensor(indices_cpu[test_idx], device=output_device)
#         train_labels = torch.tensor(labels_cpu[train_idx], device=output_device)
#         test_labels = torch.tensor(labels_cpu[test_idx], device=output_device)
#
#     return train_indices, test_indices, train_labels, test_labels
#
#




def removed_edges_train_test_split(indices: torch.Tensor, labels: torch.Tensor, test_size=0.25, random_state=42, output_device="cuda"):
    """
    Split GPU tensors into training and testing sets, preserving class balance, with a specified output device.

    Args:
        indices (torch.Tensor): Tensor of indices (on GPU).
        labels (torch.Tensor): Tensor of labels (on GPU).
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility.
        output_device (str): Device for the output tensors (e.g., "cuda" or "cpu").

    Returns:
        train_indices (torch.Tensor): Tensor of training indices (on output_device).
        test_indices (torch.Tensor): Tensor of testing indices (on output_device).
        train_labels (torch.Tensor): Tensor of training labels (on output_device).
        test_labels (torch.Tensor): Tensor of testing labels (on output_device).
    """

    # Move tensors to CPU for compatibility with train_test_split
    indices_cpu = indices.cpu().numpy()
    labels_cpu = labels.cpu().numpy()

    # Perform stratified train-test split
    train_idx, test_idx, train_labels_cpu, test_labels_cpu = train_test_split(
        indices_cpu,
        labels_cpu,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_cpu
    )

    # Convert the results back to PyTorch tensors on the specified device
    train_indices = torch.tensor(train_idx, device=output_device)
    test_indices = torch.tensor(test_idx, device=output_device)
    train_labels = torch.tensor(train_labels_cpu, device=output_device)
    test_labels = torch.tensor(test_labels_cpu, device=output_device)

    return train_indices, test_indices, train_labels, test_labels


def save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir="ckpt",
                                is_best_acc=False):
    os.makedirs(save_dir, exist_ok=True)
    base_filename = f"best_model_bases{num_bases}_channels{'-'.join(map(str, out_channels))}"
    file_name = ""
    if is_best_acc:
        file_name = f"{base_filename}_best_acc.pth"
    else:
        file_name = f"{base_filename}_best_loss.pth"

    checkpoint_path = os.path.join(save_dir, file_name)


    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_bases': num_bases,
        'out_channels': out_channels
    }, checkpoint_path)
    print(f"Model saved at '{checkpoint_path}'")

def save_model(model, optimizer, epoch, save_dir="ckpt",file_name = "_",
                                is_best_acc=False):
    os.makedirs(save_dir, exist_ok=True)
    if is_best_acc:
        file_name = f"{file_name}_best_acc.pth"
    else:
        file_name = f"{file_name}_best_loss.pth"

    checkpoint_path = os.path.join(save_dir, file_name)


    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Model saved at '{checkpoint_path}'")


def calculate_metrics(predictions, true_labels):
    """
    Calculate accuracy, precision, recall, and F1 score.
    """
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    return accuracy, precision, recall, f1

def instantiate_encoder(config_, data):
    encoder_type = config_["classifier_encoder"]
    out_channels = config_["encoder_out_channels"]
    device = torch.device(config_["device"])
    num_layers = config_["num_layers"]
    use_edges_info = config_.get("use_edges_info", False)
    num_bases = config_.get("num_bases", None)
    msg_sens = config_.get("message_sens", "source_to_target")

    if encoder_type == "GCN":
        encoder = GCNEncoder(data, out_channels, num_layers,
                             message_sens=msg_sens).to(device)

    elif encoder_type == "RGCN":
        encoder = RGCNEncoder(data, out_channels, num_layers, num_bases,
                              message_sens=msg_sens).to(device)

    elif encoder_type in ["TransGCN_conv", "TransGCN_attn"]:
        variant = "conv" if "conv" in encoder_type else "attn"
        encoder = TransGCNEncoder(
            data, out_channels, num_layers, dropout=0.2,
            kg_score_fn='TransE', variant=variant,
            use_edges_info=use_edges_info, activation='relu',
            bias=False
        ).to(device)

    elif encoder_type in ["RotatEGCN_conv", "RotatEGCN_attn"]:
        variant = "conv" if "conv" in encoder_type else "attn"
        encoder = TransGCNEncoder(
            data, out_channels, num_layers, dropout=0.2,
            kg_score_fn='RotatE', variant=variant,
            use_edges_info=use_edges_info, activation='relu',
            bias=False
        ).to(device)

    elif encoder_type == "GAT":
        encoder = GATEncoder(data, out_channels, num_layers).to(device)

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return encoder



import os
import re
import torch


def parse_model_name(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]

    encoder_patterns = [
        'TransGCN_conv', 'TransGCN_attn',
        'RotatEGCN_conv', 'RotatEGCN_attn',
        'RGCN', 'GAT', 'GCN'
    ]

    decoder_patterns = [
        'TransGCN_conv', 'TransGCN_attn',
        'RotatEGCN_conv', 'RotatEGCN_attn',
        'RGCN', 'GAT', 'GCN', 'MLP', 'Dismult', 'GAE'
    ]

    info = {
        'task_type': None,
        'encoder_type': None,
        'decoder_type': None,
        'out_channels': None,
        'num_bases': None,
        'epoch': None,
        'is_best': False,
    }

    for task in ['Recons_X', 'Recons_A', 'Recons_R', 'Double_reconstruction']:
        if task in basename:
            info['task_type'] = task
            break

    for enc in encoder_patterns:
        if f'enc-{enc}' in basename:
            info['encoder_type'] = enc
            break

    for dec in decoder_patterns:
        if f'dec-{dec}' in basename or f'_{dec}' in basename:
            info['decoder_type'] = dec
            break

    match = re.search(r'channels_([\d\-]+)', basename)

    if match:
        channels_str = match.group(1)
        info['out_channels'] = [int(x) for x in re.split(r'[-_]', channels_str)]

    match = re.search(r'bases-(\d+)', basename)
    if match:
        info['num_bases'] = int(match.group(1))

    match = re.search(r'epoch[_-](\d+)', basename, re.IGNORECASE)
    if match:
        info['epoch'] = int(match.group(1))

    if 'best' in basename.lower():
        info['is_best'] = True

    return info


def build_config_from_filename(filename, default_device='cuda'):
    info = parse_model_name(filename)
    config = {
        "device": default_device if torch.cuda.is_available() else "cpu",
        "classifier_encoder": info['encoder_type'],
        "classifier_decoder": info['decoder_type'],
        "encoder_out_channels": info['out_channels'],
        "num_bases": info['num_bases'],
        "num_layers": 2,
        "alpha": 0.01,
        "message_sens": "source_to_target",
        "use_edges_info": True,
        "task_type": info['task_type']
    }
    return config, info


def load_model_from_checkpoint(filename, data, strict=True, load_full_model=True):


    config, model_info = build_config_from_filename(filename)

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    encoder = instantiate_encoder(config, data)
    model = encoder
    device = torch.device(config["device"])
    checkpoint = torch.load(filename, map_location=device)

    if load_full_model and config["classifier_decoder"]:
        try:
            decoder = instantiate_decoder(config, data, encoder)
            if model_info['task_type'] == 'Recons_A':
                model = GAE(encoder).to(device)
            elif model_info['task_type'] == 'Double_reconstruction':
                r_decoder = instantiate_decoder({**config, "classifier_decoder": "Dismult"}, data, encoder)
                model = MRGAE(encoder, x_decoder=decoder, r_decoder=r_decoder).to(device)
            elif model_info['task_type'] == 'Recons_R':
                model = MRGAE(encoder, x_decoder=None, r_decoder=decoder).to(device)
            else:
                model = MRGAE(encoder, x_decoder=decoder).to(device)
        except Exception as e:
            print(f"[Warning] Could not load decoder. Using encoder only. Reason: {e}")
            model = encoder
        state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
        model.load_state_dict(checkpoint.get(state_key, checkpoint), strict=strict)
        model.eval()
        return model, model_info, checkpoint



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False