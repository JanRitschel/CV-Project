import torch.nn as nn

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