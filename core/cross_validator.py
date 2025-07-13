import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose, Resize

from dataset import PatchDatasetFromJson
import global_vars as gv
from model_m import get_levit_model, train_epoch, evaluate
from helpers import read_args

def cross_validate(dataset: PatchDatasetFromJson) -> tuple[int, float]:
    """Performs cross-validation on the given dataset.

    Args:
        dataset (PatchDatasetFromJson): The dataset to use for cross-validation.

    Returns:
        tuple[int, float]: The best batch size and learning rate.
    """
    # Get the labels from the dataset max 2 subsets deep
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        if isinstance(base_dataset, Subset):
            base_base_dataset = base_dataset.dataset
            indices = dataset.indices
            y_labels = [base_base_dataset.samples[i][1] for i in indices]
        else:
            indices = dataset.indices
            y_labels = [base_dataset.samples[i][1] for i in indices]
    else:
        y_labels = [label for _, label in dataset.samples]
        
    # Get a StratifiedKFold instance
    skf = StratifiedKFold(n_splits= gv.K_SPLITS, shuffle=True, random_state=gv.SPLIT_SEED)
    
    # Initialize variables to keep track of the best hyperparameters
    best_acc = 0.0
    best_params = {}

    # Iterate over hyperparameter combinations
    tqdm.write("Starting cross-validation...")
    for batch_size in tqdm(gv.BATCH_SIZE_LIST):
        for lr in tqdm(gv.LR_LIST):
            fold_scores = []

            # Perform Cross validation
            tqdm.write(
                f"Running cross validation over {len(list(skf.split(np.zeros(len(y_labels)), y_labels)))} folds with batch size {batch_size} and learning rate {lr}")
            
            for train_idx, val_idx in tqdm(skf.split(
                                            np.zeros(len(y_labels)), y_labels), 
                                           desc=f"Batch size: {batch_size}, LR: {lr}"):
                # Create train and validation datasets
                tqdm.write(
                    f"Training on fold with {len(train_idx)} samples, validating on {len(val_idx)} samples")
                train_ds = Subset(dataset, train_idx)
                val_ds = Subset(dataset, val_idx)

                # Create data loaders
                train_loader = DataLoader(
                    train_ds, batch_size=batch_size, shuffle=True, num_workers=gv.NUM_WORKERS)
                val_loader = DataLoader(
                    val_ds, batch_size=batch_size, shuffle=False, num_workers=gv.NUM_WORKERS)

                # Initialize model, optimizer and loss function
                model = get_levit_model(num_classes=gv.NUM_CLASSES).to(gv.DEVICE)
                optimizer = optim.AdamW(
                    model.parameters(), lr=lr, weight_decay=1e-2)
                criterion = nn.CrossEntropyLoss()

                # Train the model for a NUM_CROSS_VAL_EPOCHS epochs
                for _ in tqdm(range(gv.NUM_CROSS_VAL_EPOCHS), desc="Training Epochs"):
                    train_epoch(model, train_loader, criterion, optimizer)

                # Evaluate the model on the validation set
                acc = evaluate(model, val_loader)
                tqdm.write(f"Fold Accuracy: {acc:.4f}")
                fold_scores.append(acc)

            # Calculate average score for the current hyperparameter combination
            avg_score = np.mean(fold_scores)
            tqdm.write(
                f"[lr={lr}, batch_size={batch_size}] CV Acc: {avg_score:.4f}")

            # Save the best model based on average score
            if avg_score > best_acc:
                best_acc = avg_score
                best_tuple = (batch_size, lr)
                best_params = {'batch_size': batch_size, 'lr': lr}
                torch.save(model.state_dict(), "best_model.pth")

    # Print the best hyperparameters and accuracy
    tqdm.write("Best parameters:")
    tqdm.write(str(best_params))
    tqdm.write(f"Best CV Accuracy: {best_acc:.4f}")
    return best_tuple

def full_cross_validate() -> None:
    """Perform full cross-validation on the dataset and print the best hyperparameters.
    """    
    # transform to resize images to 224x224
    transform = Compose([
        Resize((224, 224))  # expects torch.Tensor C×H×W
    ])

    # Read command line arguments
    arguments = vars(read_args())
    tqdm.write(f"Arguments: {arguments}")
    if arguments["db"]:
        default_path = arguments["db"]
        tqdm.write(f"Using database path: {default_path}")

    if os.path.isdir(default_path):
        
        # Build Dataset from channel list with transformer to resize
        dataset = PatchDatasetFromJson(
            default_path, transform=transform, channel_indices=gv.CHANNEL_LIST)

        # Split dataset into training and test sets
        train_size = int(0.75 * len(dataset))
        test_size = len(dataset) - train_size
        random_generator = torch.Generator().manual_seed(gv.SPLIT_SEED)
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], random_generator)

        cross_validate(train_dataset)
        
if __name__ == "__main__":
    # Run the model training and evaluation
    full_cross_validate()