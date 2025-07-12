import os
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

# Constants
NUM_CLASSES = 8
NUM_EPOCHS = 20
NUM_CROSS_VAL_EPOCHS = 3  # Number of epochs for each cross-validation fold
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_LIST = [16,32,64]
LR_LIST = [1e-2, 1e-3, 1e-4]
K_SPLITS = 3
DROP_RATE = 0.2  # Dropout rate for the model


def get_levit_model(model_name="levit_128s", num_classes=8, input_channels=2):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, drop_rate = DROP_RATE)

    # Find the first Conv2d layer with 3 input channels
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
            print(f"Replacing input layer: {name}")
            new_conv = torch.nn.Conv2d(
                input_channels, module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None
            )
            with torch.no_grad():
                new_conv.weight[:, :2] = module.weight[:, :2]  # copy 2 of 3 input channels
            # set the new conv in the model
            set_module_by_name(model, name, new_conv)
            break

    return model

# Helper function to replace a nested module by name
def set_module_by_name(model, name, module):
    parts = name.split('.')
    for p in parts[:-1]:
        model = getattr(model, p)
    setattr(model, parts[-1], module)



def train_epoch(model, loader, criterion, optimizer):
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


def evaluate(model, loader):
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


def cross_validate(dataset: PatchDatasetFromJson):
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
    skf = StratifiedKFold(n_splits=K_SPLITS, shuffle=True, random_state=42)

    best_acc = 0.0
    best_params = {}

    # Iterate over hyperparameter combinations
    tqdm.write("Starting cross-validation...")
    for batch_size in tqdm(BATCH_SIZE_LIST):
        for lr in tqdm(LR_LIST):
            fold_scores = []

            #Perform Cross validation
            tqdm.write(f"Running cross validation over {len(list(skf.split(np.zeros(len(y_labels)), y_labels)))} folds with batch size {batch_size} and learning rate {lr}")
            for train_idx, val_idx in tqdm(skf.split(np.zeros(len(y_labels)), y_labels), desc=f"Batch size: {batch_size}, LR: {lr}"):
                tqdm.write(f"Training on fold with {len(train_idx)} samples, validating on {len(val_idx)} samples")
                train_ds = Subset(dataset, train_idx)
                val_ds = Subset(dataset, val_idx)

                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

                model = get_levit_model(num_classes=NUM_CLASSES).to(DEVICE)
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
                criterion = nn.CrossEntropyLoss()

                for _ in tqdm(range(NUM_CROSS_VAL_EPOCHS), desc="Training Epochs"):
                    train_epoch(model, train_loader, criterion, optimizer)

                acc = evaluate(model, val_loader)
                tqdm.write(f"Fold Accuracy: {acc:.4f}")
                fold_scores.append(acc)

            # Calculate average score for the current hyperparameter combination
            avg_score = np.mean(fold_scores)
            tqdm.write(f"[lr={lr}, batch_size={batch_size}] CV Acc: {avg_score:.4f}")

            # Save the best model based on average score
            if avg_score > best_acc:
                best_acc = avg_score
                best_tuple = (batch_size, lr)
                best_params = {'batch_size': batch_size, 'lr': lr}
                torch.save(model.state_dict(), "best_model.pth")

    tqdm.write("Best parameters:")
    tqdm.write(str(best_params))
    tqdm.write(f"Best CV Accuracy: {best_acc:.4f}")
    return best_tuple

def train_final_model(dataset, batch_size, lr, num_epochs=NUM_EPOCHS):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    model = get_levit_model(model_name="levit_384",num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    tqdm.write(f"Training final model with batch_size={batch_size}, lr={lr}, epochs={num_epochs}")
    for epoch in tqdm(range(num_epochs), desc="Final Training Epochs"):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), "final_model.pth")
    tqdm.write("Final model saved as 'final_model.pth'")
    return model

def main(default_path=None):

    transform = Compose([
        Resize((224, 224))  # expects torch.Tensor C×H×W
    ])

    parser = argparse.ArgumentParser(
        description="Trains a LeViT model with cross-validation on a dataset")
    #    parser.add_argument("--version", action="version", version='%(prog)s ' + __version__)
    parser.add_argument("-db", metavar="database", required=True,
                        help="Please provide the path to the database directory: ")
    
    arguments = vars(parser.parse_args())
    tqdm.write(f"Arguments: {arguments}")
    if arguments["db"]:
        default_path = arguments["db"]
        tqdm.write(f"Using database path: {default_path}")
    if os.path.isdir(default_path):
        
        dataset = PatchDatasetFromJson(default_path, transform=transform)
        # Split dataset into training and validation sets
        train_size = int(0.75 * len(dataset))
        val_size = len(dataset) - train_size
        random_generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], random_generator)
        
        """
        # Split dataset into training and validation sets
        train_size = int(0.5 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        cross_train_dataset, cross_val_dataset = random_split(train_dataset, [train_size, val_size], random_generator)

        
        best_batch, best_lr = cross_validate(cross_train_dataset)
         """
        # Use the best hyperparameters found during cross-validation
        best_batch = 16
        best_lr = 1e-4
        #NUM_EPOCHS = 1
        
        #get test loader and train model
        val_loader = DataLoader(train_dataset, batch_size=best_batch, shuffle=False, num_workers=NUM_WORKERS)
        final_model = train_final_model(train_dataset, batch_size=best_batch, lr=best_lr, num_epochs=NUM_EPOCHS)
        
        # Evaluate on training set
        """ train_loader = DataLoader(train_dataset, batch_size=best_batch, shuffle=False, num_workers=NUM_WORKERS)
        model = get_levit_model(model_name="levit_128s", num_classes=8, input_channels=2)
        model.load_state_dict(torch.load("/home/group.kurse/cviwo021/CV-Project/core/final_model.pth"))
        model.eval()
        model = model.to(DEVICE)

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(train_loader, desc="Evaluating on training set"):
                outputs = model(inputs.to(DEVICE))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(DEVICE)).sum().item()

        train_accuracy = correct / total
        print(f"Train Accuracy: {train_accuracy:.2%}") """
        
        #evaluate on test set
        val_acc = evaluate(final_model, val_loader)
        tqdm.write(f"Validation Accuracy: {val_acc:.4f}")
        with open("/home/group.kurse/cviwo021/RESULTS/results_big_model.txt", "w") as f:
            print(f"Best Batch Size: {best_batch}", file=f)
            print(f"Best Learning Rate: {best_lr}", file=f)
            print(f"Validation Accuracy: {val_acc:.4f}", file=f)
        
    else:
        tqdm.write(f"Provided path '{default_path}' is not a valid directory.")
        return


if __name__ == "__main__":
    # Load dataset
    root_path = '/Users/Noah/Uni Koeln Git/Computer Vision/Project/Michigan-Dataset'

    # Run cross-validation
    main(root_path)
