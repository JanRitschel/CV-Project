import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import timm
from tqdm import tqdm
import copy

from dataset import PatchDatasetFromJson
from helpers import set_module_by_name
import global_vars as gv

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
        model_name, pretrained=False, num_classes=num_classes, drop_rate=gv.DROP_RATE)

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
        x = x.to(dtype=torch.float32, device=gv.DEVICE)
        y = y.to(gv.DEVICE)
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
            x = x.to(dtype=torch.float32, device=gv.DEVICE)
            out = model(x)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pred)
            labels.extend(y.numpy())
    return accuracy_score(labels, preds)

def train_final_model(dataset: PatchDatasetFromJson, batch_size: int, lr: float, num_epochs: int = gv.NUM_EPOCHS, patience: int = 3,
                      val_split: float = 0.2, loss_file: str|None = None, num_chanels: int = 2,
                      model_path: str = "final_model.pth") -> nn.Module:
    """Train the final model on a given dataset.

    Args:
        dataset (PatchDatasetFromJson): The dataset to train on.
        batch_size (int): The batch size to use during training.
        lr (float): The learning rate for the optimizer.
        num_epochs (int, optional): The number of epochs to train for. Defaults to gv.NUM_EPOCHS.
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
        dataset, [train_size, val_size], torch.Generator().manual_seed(gv.SPLIT_SEED))

    # load data
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=gv.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=gv.NUM_WORKERS)

    # Initialize model, optimizer and loss function
    model = get_levit_model(
        model_name="levit_384", num_classes=gv.NUM_CLASSES, input_channels=num_chanels).to(gv.DEVICE)
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
                x = x.to(dtype=torch.float32, device=gv.DEVICE)
                y = y.to(gv.DEVICE)
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