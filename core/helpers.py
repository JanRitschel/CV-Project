import sys
import torch.nn as nn
import argparse

def set_module_by_name(model: nn.Module, name: str, module: nn.Module) -> None:
    """Sets a module in the model by its name.

    Args:
        model (nn.Module): The model to modify.
        name (str): The name of the module to replace.
        module (nn.Module): The new module to set.
    """    
    parts = name.split('.')
    for p in parts[:-1]:
        model = getattr(model, p)
    setattr(model, parts[-1], module)
    

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