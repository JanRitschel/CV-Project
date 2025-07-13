import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision.transforms import Compose, Resize

from dataset import PatchDatasetFromJson
import global_vars as gv
from model_m import train_final_model, evaluate
from helpers import read_args

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
            default_path, transform=transform, channel_indices=gv.CHANNEL_LIST)

        # Split dataset into training and test sets
        train_size = int(0.75 * len(dataset))
        test_size = len(dataset) - train_size
        random_generator = torch.Generator().manual_seed(gv.SPLIT_SEED)
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], random_generator)

        # Use the best hyperparameters found during cross-validation
        best_batch = arguments["bs"]
        best_lr = arguments["lr"]

        # get test loader and train model
        test_loader = DataLoader(test_dataset, batch_size=best_batch,
                                 shuffle=False, num_workers=gv.NUM_WORKERS)
        final_model = train_final_model(train_dataset, batch_size=best_batch,
                                        lr=best_lr, num_epochs=gv.NUM_EPOCHS,
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