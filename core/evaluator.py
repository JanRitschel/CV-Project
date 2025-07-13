import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision.transforms import Compose, Resize
from torch.utils.data import DataLoader

from dataset import PatchDatasetFromJson
from model_m import get_levit_model
from helpers import read_args
import global_vars as gv

def evaluate_model(default_path:str, model_path:str, model_architecture:str = "levit_384") -> None:
    """Evaluate a pre-trained model on a dataset.

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
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on test set"):
            outputs = model(inputs.to(gv.DEVICE))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(gv.DEVICE)).sum().item()

    test_accuracy = correct / total
    tqdm.write(f"Test Accuracy: {test_accuracy:.2%}")

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