import torch
from torch import nn
from utils import losses


def clone_state_dict(thing: nn.Module | dict[str, torch.Tensor]):
    if isinstance(object, nn.Module):
        state_dict = thing.state_dict()
    elif isinstance(object, dict):
        state_dict = thing
    else:
        raise TypeError(f"Expected `nn.Module` or `dict[str, torch.Tensor]` for `thing` but got `{repr(thing)}` instead.")
    
    return {key: val.clone().detach().cpu() for key, val in state_dict.items()}
