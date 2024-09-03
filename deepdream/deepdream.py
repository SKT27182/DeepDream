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

from .helper_function.utils import (
    HTMLImageDisplayer,
    ImagePlotter,
    create_animation,
    create_logger,
    close_all_hooks,
    display_img,
    create_wandb_logger,
)

logger = create_logger(__name__, "info")


def total_variation_loss(img, weight=0.1):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)


class ObjectiveHook:

    """
    ObjectiveHook class to store the weighted loss of the given layers.

    Args:
    model: nn.Module
        Pytorch model to generate deep dream images.

    layer_names: List
        List of layer names to store the loss.

    device: torch.device
        Device to run the model.

    layer_weights: Optional[List]
        List of weights for each layer. If not provided, all the layer weights are set to 1.

    """

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

        self.losses = 0

        # consider weights for each layer
        if layer_weights is None:
            logger.debug("No layer weights provided. Setting all layer weights to 1.")
            layer_weights = [1] * len(self.layer_names)

            self.layer_weights = layer_weights
        else:
            if len(layer_weights) != len(self.layer_names):

                m = "Number of layer weights should be equal to the number of layers."
                logger.error(m)

                raise ValueError(m)

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

    def __exit__(self, type: Any, value: Any, traceback: Any):

        close_all_hooks(self.model)

    def store_weighted_loss(self, name: str, output: torch.Tensor, weight: float):

        """
        
        Store the weighted loss of the given layer.
        
        Args:
        name: str
            Layer name to store the loss.
            
        output: torch.Tensor
            Output of the layer.
            
        weight: float
            Weight to multiply with the output.
            
        """

        output = output

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

    def __init__(
        self, img: torch.Tensor, objective: ObjectiveHook, device: torch.device
    ):
        self.img = img.to(device)
        self.objective = objective
        self.device = device

    @classmethod
    def from_layer_name(
        cls,
        model: nn.Module,
        img: torch.Tensor,
        layer_names: List,
        device: torch.device,
        layer_weight: Optional[float] = None,
    ) -> "DeepDream":

        """
        Create DeepDream object using the given model, image, layer name, and device.

        Args:
        model: nn.Module
            Pytorch model to generate deep dream images.

        img: torch.Tensor
            Input image to generate deep dream images.

        layer_names: List
            List of layer names to store the loss.

        device: torch.device
            Device to run the model.

        layer_weight: Optional[float]
            Weight for the layer. If not provided, the weight is set to 1.

        Returns:
        DeepDream object

        """

        if isinstance(layer_names, str):
            layer_names = [layer_names]

        objective = ObjectiveHook(
            model, layer_names, device, layer_weights=[layer_weight]
        )

        return cls(img, objective, device)

    def deep_dream(self, iterations=20, lr=0.01, display_interval=5) -> torch.Tensor:

        """
        Generate deep dream images using the given model and layer name.

        Args:
        iterations: int
            Number of iterations to generate deep dream images.

        lr: float
            Learning rate for the optimizer.

        display_interval: int
            Display the image after every display_interval iterations.  

        Returns:
        torch.Tensor: Deep dream image.

        """

        image_displayer = HTMLImageDisplayer()

        self.optimizer = optim.Adam([self.img], lr=lr)

        self.img.requires_grad = True

        to_display_imgs = [self.img.detach().cpu().clone()]

        for i in tqdm(range(iterations+1)):

            model = self.objective.model

            model.zero_grad()

            # inference model in eval mode
            model.eval()
            _ = model(self.img)

            loss = -self.objective.losses  # weighted sum of losses of all the layers

            # calculate the gradient
            loss.backward()

            # normalize the gradient
            self.img.grad.data /= self.img.grad.data.std() + 1e-8

            # update the image
            self.optimizer.step()

            # clamp the pixel values between -1 and 1
            self.img.data = torch.clamp(self.img.data, -1, 1)
            # self.img = self.img.detach()  # Detach to free the graph
            # self.img.requires_grad = True  # Re-enable gradients
            self.img.grad.data.zero_()

            logger.debug(f"Iteration: {i}, Loss: {loss.item()}")

            self.objective.losses = 0

            to_display_imgs.append(self.img.detach().cpu().clone())

            if (i+1) % display_interval == 0:
                image_displayer.clear()

                # check if all the images are same or not
                if len(set([str(img) for img in to_display_imgs])) == 1:
                    logger.debug("All the images are same.")
                    raise ValueError("All the images are same. Stopping the training.")

                image_displayer.display_grid(
                    to_display_imgs,
                    base_title=f"Iteration: {i}",
                    save_path="dreemed_images",
                )

                create_animation(
                    to_display_imgs,
                    f"deep_dream_{i}.gif",
                    fps=2,
                    title=f"Iteration: {i}",
                )

                to_display_imgs = []

        return self.img

    def close(self):
        self.objective.close()

    def __enter__(self):
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.close()
