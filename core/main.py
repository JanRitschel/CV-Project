import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import timm
import numpy as np
from dataset import PatchDatasetFromJson
from tqdm import tqdm
from torchvision.transforms import Compose, Resize
import copy

from helpers import set_module_by_name

# Global Constants
NUM_CLASSES = 8  # Number of classes in the dataset
NUM_EPOCHS = 20  # Number of epochs for final training
NUM_CROSS_VAL_EPOCHS = 3  # Number of epochs for each cross-validation fold
NUM_WORKERS = 4  # Number of workers for DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# List of batch sizes to test during cross-validation
BATCH_SIZE_LIST = [16, 32, 64]
# List of learning rates to test during cross-validation
LR_LIST = [1e-2, 1e-3, 1e-4]
K_SPLITS = 3  # Number of splits for cross-validation
DROP_RATE = 0.2  # Dropout rate for the model
SPLIT_SEED = 42  # Seed for random splits. Used for comparability between channels
CHANNEL_LIST = [1]  # Use first channel, can be modified to use more channels


def get_levit_model(model_name: str = "levit_128s", num_classes: int = 8,
                    input_channels: int = 2) -> nn.Module:
    """Get a LeViT model with specified parameters.

    Args:
        model_name (str, optional): Name of the model architecture. Defaults to "levit_128s".
        num_classes (int, optional): Number of output classes. Defaults to 8.
        input_channels (int, optional): Number of input channels. Defaults to 2.

    Returns:
        nn.Module: The constructed LeViT model.
    """
    # Create the model with the specified architecture
    model = timm.create_model(
        model_name, pretrained=False, num_classes=num_classes, drop_rate=DROP_RATE)

    # Find the first Conv2d layer with 3 input channels
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
            # exchange the input layer to match the number of input channels
            print(f"Replacing input layer: {name}")
            new_conv = torch.nn.Conv2d(
                input_channels, module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None
            )
            with torch.no_grad():
                # copy input_channels of 3 input channels
                new_conv.weight[:,
                                :input_channels] = module.weight[:, :input_channels]
            # set the new conv in the model
            set_module_by_name(model, name, new_conv)
            break

    return model


def train_epoch(model: nn.Module, loader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer) -> float:
    """Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The data loader for the training data.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer for the model.

    Returns:
        float: The average loss for the epoch.
    """    
    model.train()
    running_loss = 0
    for x, y in tqdm(loader):
        x = x.to(dtype=torch.float32, device=DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Evaluates the model on the given data loader.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The data loader for the evaluation data.

    Returns:
        float: The accuracy of the model on the evaluation data.
    """
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x = x.to(dtype=torch.float32, device=DEVICE)
            out = model(x)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pred)
            labels.extend(y.numpy())
    return accuracy_score(labels, preds)


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
    skf = StratifiedKFold(n_splits=K_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    
    # Initialize variables to keep track of the best hyperparameters
    best_acc = 0.0
    best_params = {}

    # Iterate over hyperparameter combinations
    tqdm.write("Starting cross-validation...")
    for batch_size in tqdm(BATCH_SIZE_LIST):
        for lr in tqdm(LR_LIST):
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
                    train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
                val_loader = DataLoader(
                    val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

                # Initialize model, optimizer and loss function
                model = get_levit_model(num_classes=NUM_CLASSES).to(DEVICE)
                optimizer = optim.AdamW(
                    model.parameters(), lr=lr, weight_decay=1e-2)
                criterion = nn.CrossEntropyLoss()

                # Train the model for a NUM_CROSS_VAL_EPOCHS epochs
                for _ in tqdm(range(NUM_CROSS_VAL_EPOCHS), desc="Training Epochs"):
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


def train_final_model(dataset: PatchDatasetFromJson, batch_size: int, lr: float, num_epochs: int = NUM_EPOCHS, patience: int = 3,
                      val_split: float = 0.2, loss_file: str|None = None, num_chanels: int = 2,
                      model_path: str = "final_model.pth") -> nn.Module:
    """Train the final model on a given dataset.

    Args:
        dataset (PatchDatasetFromJson): The dataset to train on.
        batch_size (int): The batch size to use during training.
        lr (float): The learning rate for the optimizer.
        num_epochs (int, optional): The number of epochs to train for. Defaults to NUM_EPOCHS.
        patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 3.
        val_split (float, optional): The proportion of the dataset to use for validation. Defaults to 0.2.
        loss_file (str|None, optional): The file to save loss information to. Defaults to None.
        num_chanels (int, optional): The number of input channels for the model. Defaults to 2.
        model_path (str, optional): The path to save the trained model to. Defaults to "final_model.pth".
        
    Returns:
        nn.Module: The trained model.
    """
    # Split dataset in train/val
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], torch.Generator().manual_seed(SPLIT_SEED))

    # load data
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=NUM_WORKERS)

    # Initialize model, optimizer and loss function
    model = get_levit_model(
        model_name="levit_384", num_classes=NUM_CLASSES, input_channels=num_chanels).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    tqdm.write(
        f"Training final model with batch_size={batch_size}, lr={lr}, epochs={num_epochs}")
    for epoch in tqdm(range(num_epochs), desc="Final Training Epochs"):
        # run training epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        # calculate validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validation"):
                x = x.to(dtype=torch.float32, device=DEVICE)
                y = y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save losses in txt file
        with open(loss_file, "a") as f:
            f.write(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")

        # Check for early stopping and save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            tqdm.write(f"Early stopping triggered after {epoch+1} epochs.")
            break

    tqdm.write(f"Final model saved as {model_path}")
    return best_model


def read_args() -> argparse.Namespace:
    """Reads command line arguments for the script.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a LeViT model with cross-validation on a dataset")
    parser.add_argument("-db", metavar="database",
                        required=True, help="Path to the database directory")
    parser.add_argument("-lf", metavar="loss_file",
                        required=False, help="Path to the loss file",
                        default='loss.txt')
    parser.add_argument("-mf", metavar="model_file", required=False,
                        help="Path to the model file", default='final_model.pth')
    parser.add_argument("-lr", metavar="learning_rate", required=False,
                        help="Learning rate for training", default=1e-4)
    parser.add_argument("-bs", metavar="batch_size", required=False,
                        help="Batch size for training", default=16)
    
    # Print help if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(default_path=None):
    """Main function to train and evaluate the model.

    Args:
        default_path (str, optional): The default path to the database specified by -db. Defaults to None.
        
    Saves the trained model to the path specified by the `-mf` argument or defaults to 'final_model.pth'.
    Saves the training and validation losses to the file specified by the `-lf` argument.
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
            default_path, transform=transform, channel_indices=CHANNEL_LIST)

        # Split dataset into training and test sets
        train_size = int(0.75 * len(dataset))
        test_size = len(dataset) - train_size
        random_generator = torch.Generator().manual_seed(SPLIT_SEED)
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], random_generator)

        """
        # Split dataset into training and validation sets
        train_size = int(0.5 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        cross_train_dataset, cross_val_dataset = random_split(train_dataset, [train_size, val_size], random_generator)

        # Run cross validation
        best_batch, best_lr = cross_validate(cross_train_dataset)
         """

        # Use the best hyperparameters found during cross-validation
        best_batch = arguments["bs"]
        best_lr = arguments["lr"]

        # get test loader and train model
        test_loader = DataLoader(test_dataset, batch_size=best_batch,
                                 shuffle=False, num_workers=NUM_WORKERS)
        final_model = train_final_model(train_dataset, batch_size=best_batch,
                                        lr=best_lr, num_epochs=NUM_EPOCHS,
                                        loss_file=arguments["lf"], num_chanels=1,
                                        model_path=arguments["mf"])

        # evaluate on test set
        test_acc = evaluate(final_model, test_loader)
        tqdm.write(f"Test Accuracy: {test_acc:.4f}")
        with open(arguments["lf"], "a") as f:
            print(f"Best Batch Size: {best_batch}", file=f)
            print(f"Best Learning Rate: {best_lr}", file=f)
            print(f"Test Accuracy: {test_acc:.4f}", file=f)

    else:
        tqdm.write(f"Provided path '{default_path}' is not a valid directory.")
        return


if __name__ == "__main__":
    # Run the model training and evaluation
    main()
