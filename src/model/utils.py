import os
import torch


def save_model(model, optimizer, epoch, save_dir="checkpoints"):
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