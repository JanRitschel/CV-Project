# load model from state_dict and evaluate model on test data  
import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from dataset import PatchDatasetFromJson
from tqdm import tqdm
from torchvision.transforms import Compose, Resize
from torch.utils.data import DataLoader
from dataset import PatchDatasetFromJson
from main import DEVICE, NUM_WORKERS, get_levit_model

def evaluate_model(default_path):
    model = get_levit_model(model_name="levit_384", num_classes=8, input_channels=1)
    model.load_state_dict(torch.load("/home/group.kurse/cviwo020/CV-Project/core/final_model.pth"))
    model.eval()
    model = model.to(DEVICE)
    transform = Compose([
        Resize((224, 224))  # expects torch.Tensor C×H×W
    ])
    dataset = PatchDatasetFromJson(default_path, transform=transform, channel_indices=[0])  # Use second channel
    # Split dataset into training and validation sets
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    random_generator = torch.Generator().manual_seed(42)
    _, test_dataset = random_split(dataset, [train_size, val_size], random_generator)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on test set"):
            outputs = model(inputs.to(DEVICE))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(DEVICE)).sum().item()

    train_accuracy = correct / total
    tqdm.write(f"Train Accuracy: {train_accuracy:.2%}")

def main(default_path=None):

    parser = argparse.ArgumentParser(
        description="Trains a LeViT model with cross-validation on a dataset")
    #    parser.add_argument("--version", action="version", version='%(prog)s ' + __version__)
    parser.add_argument("-db", metavar="database", required=True,
                        help="Please provide the path to the database directory: ")
    parser.add_argument("-lf", metavar="loss_file", required=True,
                        help="Please provide the path to the loss file: ")

    arguments = vars(parser.parse_args())
    tqdm.write(f"Arguments: {arguments}")
    if arguments["db"]:
        default_path = arguments["db"]
        tqdm.write(f"Using database path: {default_path}")
    if os.path.isdir(default_path):
        evaluate_model(default_path)


if __name__ == "__main__":
    # Load dataset
    root_path = '/Users/Noah/Uni Koeln Git/Computer Vision/Project/Michigan-Dataset'

    # Run cross-validation
    main(root_path)