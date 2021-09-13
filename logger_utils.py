
from typing import Any, Callable, Union, Dict

import numpy as np

import torchvision.utils as vutils
import torch
import torch.utils.tensorboard as tb


def make_recursive_func(func: Callable) -> Callable:
    """Convert a function into recursive style to handle nested dict/list/tuple variables

    Args:
        func: input function

    Returns:
        recursive style function
    """

    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper

@make_recursive_func
def tensor2numpy(vars: Any) -> np.ndarray:
    """Convert tensor to numpy array"""
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))

def save_images(logger: tb.SummaryWriter, mode: str, images: Any, global_step: int) -> None:
    """Log images stored in the image dictionary

    Args:
        logger: tensorboard summary writer
        mode: mode name used in writing summaries
        images_dict: python dictionary stores the key and image pairs to be recorded
        global_step: step index where the logger should write
    """
    images = tensor2numpy(images)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    name = "{}/img".format(mode)
    logger.add_image(name, preprocess(name, images), global_step, dataformats='CHW')
