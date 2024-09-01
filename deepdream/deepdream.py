import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict, Any, Union, Optional

from IPython.display import clear_output, display
from tqdm.notebook import tqdm

from collections import OrderedDict

from .helper_function.utils import ImagePlotter, create_logger, close_all_hooks, display_img, create_wandb_logger

logger = create_logger(__name__)


def total_variation_loss(img, weight=0.1):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)


class ObjectiveHook:
    def __init__(
        self,
        model: nn.Module,
        layer_names: List,
        device: torch.device,
        layer_weights: Optional[List] = None,
    ):

        self.layer_names = layer_names
        self.device = device
        self.model = model.to(self.device)

        self.outputs = {name: None for name in self.layer_names}

        self.forward_hooks = []
        self.backward_hooks = []

        self.forward_output = None

        self.losses = 0

        # consider weights for each layer
        if layer_weights is None:
            logger.debug("No layer weights provided. Setting all layer weights to 1.")
            layer_weights = [1] * len(self.layer_names)

            self.layer_weights = layer_weights
        else:
            if len(layer_weights) != len(self.layer_names):

                m = "Number of layer weights should be equal to the number of layers."
                logger.error( m)

                raise ValueError(
                    m
                )

            self.layer_weights = layer_weights

        for layer_name, weight in zip(self.layer_names, self.layer_weights):
            logger.debug(f"Adding forward hook for layer {layer_name}")
            self.model.get_submodule(layer_name).register_forward_hook(
                lambda module, input, output, name=layer_name, weight=weight: self.store_weighted_loss(
                    name, output, weight
                )
            )

    def __add__(self, other: "ObjectiveHook"):

        logger.debug("Adding two ObjectiveHook objects.")

        new_hook = ObjectiveHook(
            self.model,
            self.layer_names + other.layer_names,
            self.device,
        )
        new_hook.losses = self.losses + other.losses
        return new_hook

    def __mul__(self, other: Union[int, float]):

        logger.debug("Multiplying a ObjectiveHook object with a scalar.")

        if isinstance(other, (int, float)):
            new_losses = self.losses * other
            new_hook = ObjectiveHook(
                self.model, self.layer_names, self.device, self.layer_weights
            )
            new_hook.losses = new_losses
            return new_hook

        else:
            raise TypeError("Multiplication is only supported with int or float.")

    def __enter__(self):

        return self

    def __exit__(self, type, value, traceback):

        close_all_hooks(self.model)

    def store_weighted_loss(self, name, output, weight):

        # use total variation loss
        if name == self.layer_names[-1]:
            self.losses += weight * total_variation_loss(output)
        else:
            self.losses += weight * torch.mean(output, dim=None)


class DeepDream:

    """
    DeepDream class to generate deep dream images using the given model and layer name.

    Args:
    model: nn.Module
        Pytorch model to generate deep dream images.

    img: torch.Tensor
        Input image to generate deep dream images.

    layer_name: str
        Layer name to generate deep dream images.

    device: torch.device
        Device to run the model.

    """

    def __init__(self,  img: torch.Tensor, hook_obj: ObjectiveHook, device: torch.device):
        self.img = img.to(device)
        self.layer_hook = hook_obj
        self.device = device

    @classmethod
    def from_layer_name(
        cls,
        model: nn.Module,
        img: torch.Tensor,
        layer_name: str,
        device: torch.device,
        layer_weight: Optional[float] = None,
    ) -> "DeepDream":

        layer_hook = ObjectiveHook(
            model, [layer_name], device, layer_weights=[layer_weight]
        )

        return cls(img, layer_hook, device)

    def deep_dream(self, iterations=20, lr=0.01):

        image_displayer = ImagePlotter(figsize=(8, 8))

        # self.optimizer = optim.Adam([self.img], lr=lr)
        self.optimizer = optim.SGD([self.img], lr=lr)

        self.img.requires_grad = True

        for i in tqdm(range(iterations)):

            model = self.layer_hook.model

            model.zero_grad()

            # inference model in eval mode
            model.eval()
            _ = model(self.img)

            loss = -self.layer_hook.losses  # weighted sum of losses of all the layers

            # calculate the gradient
            loss.backward()

            # normalize the gradient
            self.img.grad.data /= self.img.grad.data.std() + 1e-8

            # update the image
            self.optimizer.step()

            # clamp the pixel values between -1 and 1
            self.img.data = torch.clamp(self.img.data, -1, 1)
            self.img.grad.data.zero_()

            self.layer_hook.losses = 0

            if i % 5 == 0:
                image_displayer.update_image(self.img, title=f"Iteration: {i}")

        return self.img

    def close(self):
        self.layer_hook.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
