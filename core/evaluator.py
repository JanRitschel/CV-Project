import os
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision.transforms import Compose, Resize
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import PatchDatasetFromJson
from model_m import get_levit_model
from helpers import read_args
import global_vars as gv

def evaluate_model(default_path:str, model_path:str, model_architecture:str = "levit_384") -> None:
    """Evaluate a pre-trained model on a dataset. 
    This function loads a model from the specified path, evaluates it on the test set of the dataset,
    and prints the accuracy. It also generates a UMAP projection of the model's outputs and
    a confusion matrix for the predictions.

    Args:
        default_path (str): Path to the dataset directory.
        model_path (str): Path to the model file.
        model_architecture (str): Architecture of the model to be evaluated. Defaults to "levit_384".
        
    The accuracy of the model on the test set is printed.
    """
    # Load the model
    model = get_levit_model(model_name=model_architecture, num_classes=gv.NUM_CLASSES, input_channels=len(gv.CHANNEL_LIST))
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()
    model = model.to(gv.DEVICE)

    # Get a transform to resize images
    transform = Compose([
        Resize((224, 224))  # expects torch.Tensor C×H×W
    ])
    
    # Load the dataset
    dataset = PatchDatasetFromJson(default_path, transform=transform, channel_indices=gv.CHANNEL_LIST)
    
    # Split dataset into training and validation sets
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    random_generator = torch.Generator().manual_seed(gv.SPLIT_SEED)
    _, test_dataset = random_split(dataset, [train_size, val_size], random_generator)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=gv.NUM_WORKERS)

    # Evaluate the model on the test set
    outputs, preds, labels = get_predictions_and_labels(model, test_loader)

    # Accuracy
    accuracy = (np.array(preds) == np.array(labels)).mean()
    tqdm.write(f"Test Accuracy: {accuracy:.2%}")

    # UMAP
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(outputs.numpy())
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(embedding[:,0], embedding[:,1], c=labels, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter)
    plt.title("UMAP projection")
    plt.savefig("umap_projection.png")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

def get_predictions_and_labels(model, dataloader):
    """Get predictions and labels from the model for the given dataloader.
    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the dataset.
    Returns:
        tuple: A tuple containing:
            - all_outputs (torch.Tensor): Model outputs for all samples.
            - all_preds (list): List of predicted labels.
            - all_labels (list): List of true labels.
    """
    model.eval()
    all_outputs = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Predicting"):
            inputs = inputs.to(gv.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_outputs.append(outputs.cpu())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    all_outputs = torch.cat(all_outputs)
    return all_outputs, all_preds, all_labels

def load_and_evaluate(default_path=None) -> None:
    """Load a model and dataset and evaluate it on a test set.

    Args:
        default_path (str, optional): Path to the dataset directory. Defaults to None.
    """
    # Get command line arguments
    arguments = vars(read_args())
    tqdm.write(f"Arguments: {arguments}")
    
    # Set default path if not provided
    if arguments["db"]:
        default_path = arguments["db"]
        tqdm.write(f"Using database path: {default_path}")
    if os.path.isdir(default_path):
        evaluate_model(default_path, arguments["mf"])
    else:
        tqdm.write(f"{default_path} is not a valid directory")

if __name__ == "__main__":
    # load and evaluate the model
    load_and_evaluate()