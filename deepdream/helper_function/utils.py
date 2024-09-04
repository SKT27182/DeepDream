import base64
from io import BytesIO
import os
import torch
import torch.nn as nn
from collections import OrderedDict
import einops


import logging
from logging import Logger

from typing import List, Optional, Dict, Callable, Literal, Union, Tuple, Any
from copy import deepcopy

from IPython.display import clear_output, display, HTML
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import wandb


# Color mappings
COLOUR_MAPPING = {
    "BLACK": "\033[30m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
    "END": "\033[0m",
    "HEADER": "\033[95m",
    "OKBLUE": "\033[94m",
    "OKGREEN": "\033[92m",
    "WARNING": "\033[93m",
    "FAIL": "\033[91m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
}

LOG_LEVEL_MAPPING = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

############################################# Logger #######################################

def set_log_level_to_mine_logger(level: int| str) -> None:
    """Set the log level of the mine logger.

    Args:
        level: Logging level.
    """
    all_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]

    level = LOG_LEVEL_MAPPING[level] if isinstance(level, str) else level

    for logger in all_loggers:
        if hasattr(logger, 'mine'):
            logger.setLevel(level)


def add_logging_level(
    level_name: str, level_num: int, method_name: Optional[str] = None
):
    if not method_name:
        method_name = level_name.lower()

    def log_for_level(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)

    def log_to_root(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name.upper())
    setattr(logging, level_name.upper(), level_num)
    setattr(logging.getLoggerClass(), method_name, log_for_level)
    setattr(logging, method_name, log_to_root)


class CustomFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": COLOUR_MAPPING["CYAN"],
        "INFO": COLOUR_MAPPING["GREEN"],
        "VERBOSE": COLOUR_MAPPING["WHITE"],
        "WARNING": COLOUR_MAPPING["YELLOW"],
        "ERROR": COLOUR_MAPPING["RED"] + COLOUR_MAPPING["BOLD"],
        "CRITICAL": COLOUR_MAPPING["RED"]
        + COLOUR_MAPPING["BOLD"]
        + COLOUR_MAPPING["UNDERLINE"],
    }

    def format(self, record):
        formatted_record = deepcopy(record)

        level_name = formatted_record.levelname
        color = self.COLORS.get(level_name, "")

        # Adjust name based on whether it's in the main module or not
        formatted_record.name = (
            formatted_record.name
            if formatted_record.funcName == "<module>"
            else f"{formatted_record.name}.{formatted_record.funcName}"
        )

        # Create the log message without color first
        custom_format = (
            "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s"
        )
        formatter = logging.Formatter(custom_format, datefmt="%Y-%m-%d %H:%M:%S")
        log_message = formatter.format(formatted_record)

        # Then apply color to the entire message
        colored_message = f"{color}{log_message}{COLOUR_MAPPING['END']}"

        return colored_message


def create_logger(
    name: str,
    level: Literal[
        "notset",
        "debug",
        "info",
        "warning",
        "error",
        "critical",
    ] = "info",
    consolidate_file_loggers: bool = True,
    use_json: bool = True,
    suppress_loggers: Optional[List[str]] = None,
) -> Logger:
    """
    Create a logger with the specified name and level, including color formatting for console
    and JSON formatting for file output, with optional custom filters.

    Args:
        name: Name of the logger.
        level: Logging level. Defaults to "info".
        log_file: Name of the log file. Defaults to None.
        consolidate_file_loggers: Whether to consolidate file loggers. Defaults to True.
        use_json: Whether to use JSON formatting for file logging. Defaults to True.
        filters: List of custom filter functions to apply to the logger. Defaults to None.
        custom_levels: Dictionary of custom log level names and their corresponding integer values.


    allowed_colours = ["black", "red", "green", "yellow", "blue", "cyan", "white",
                    "bold_black", "bold_red", "bold_green", "bold_yellow", "bold_blue",
                     "bold_cyan", "bold_white",
                    ]



    Returns:
        Configured logger object.
    """

    level_to_int_map: Dict[str, int] = {
        "notset": logging.NOTSET,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    logger: Logger = logging.getLogger(name)
    level_int: int = (
        level_to_int_map[level.lower()] if isinstance(level, str) else level
    )
    logger.setLevel(level_int)

    custom_formatter = CustomFormatter(
        "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler with color formatting
    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setLevel(level_int)
    console_handler.setFormatter(custom_formatter)
    logger.addHandler(console_handler)

    logger.mine = True

    set_log_level_to_mine_logger(level_int)

    return logger


logger = create_logger(__name__, level="info")


def close_all_hooks(model: nn.Module):
    for module in model.children():
        if len(list(module.children())) > 0:
            close_all_hooks(module)
        if hasattr(module, "forward_hooks"):
            for hook in module.forward_hooks:
                hook.remove()
            module.forward_hooks = OrderedDict()
        if hasattr(module, "backward_hooks"):
            for hook in module.backward_hooks:
                hook.remove()
            module.backward_hooks = OrderedDict()


def img_to_tensor(file_path: str) -> torch.Tensor:
    """
    Convert an image to a tensor.

    Args:
        file_path: Path to the image file.

    Returns:
        Tensor representation of the image of shape (1, C, H, W).
    """
    # print(logger.level)

    logger.debug(f"Converting image to tensor: {file_path}")
    img = plt.imread(file_path)  # (H, W, C)
    # add batch =1 axis and move channel axis to the first
    img = einops.rearrange(img, "h w c -> 1 c h w")
    img = torch.tensor(img, dtype=torch.float32)  # (C, H, W)
    # img = img.unsqueeze(0)  # (B, C, H, W)
    img = normalize_img(img)   # -1 to 1
    return img


def tensor_to_img(tensor: torch.Tensor, file_path: str) -> None:
    """
    Denormalize and save an image tensor to a file.

    Args:
        tensor: Tensor representation of the image of shape (1, C, H, W).
        file_path: Path to save the image file.
    """

    # img = tensor.squeeze().permute(1, 2, 0).byte().numpy()
    img = einops.rearrange(tensor, "1 c h w -> h w c").byte().numpy()
    img = denormalize_img(img)  # 0 to 255
    img = Image.fromarray(img)
    img.save(file_path)


def normalize_img(img: torch.Tensor) -> torch.Tensor:
    """
    Normalize an image tensor between -1 and 1.

    Args:
        img: Tensor representation of the image of shape (1, C, H, W).

    Returns:
        Normalized image tensor.
    """
    return 2 * (img.float() / 255.0) - 1.0


def denormalize_img(img: torch.Tensor) -> torch.Tensor:
    """
    Normalize an image tensor between 0 and 255.

    Args:
        img: Tensor representation of the image of shape (1, C, H, W).

    Returns:
        Normalized image tensor.
    """

    img = 255 * (img + 1.0) / 2.0
    return img.to(torch.uint8)


def display_img(img: torch.Tensor) -> None:
    """
    Display an image tensor.

    Args:
        img: Tensor representation of the image of shape (1, C, H, W).
    """

    img = denormalize_img(img)

    # img = img.squeeze().permute(1, 2, 0).byte().numpy()
    img = einops.rearrange(img, "1 c h w -> h w c").byte().numpy()
    clear_output(wait=True)
    display(Image.fromarray(img))


def is_jupyter_notebook() -> bool:
    """Checks if the code is being run in a Jupyter notebook.

    Returns
    -------
    bool
        True if the code is being run in a Jupyter notebook, False otherwise.
    """
    is_jupyter = False
    try:
        # noinspection PyUnresolvedReferences
        from IPython import get_ipython

        # noinspection PyUnresolvedReferences
        if get_ipython() is None or "IPKernelApp" not in get_ipython().config:
            pass
        else:
            is_jupyter = True
    except (ImportError, NameError):
        pass
    if is_jupyter:
        logger.debug("Running in Jupyter notebook.")
    else:
        logger.debug("Not running in a Jupyter notebook.")
    return is_jupyter


def create_animation(
    images: torch.Tensor,
    file_path: str,
    fps: int = 10,
    dpi: int = 100,
    title: str = "",
    denormalize: bool = True,
) -> None:
    """
    Create an animation from a list of images.

    Args:
        images: tensor representation of the images of shape (B, C, H, W).
        file_path: Path to save the animation.
        fps: Frames per second. Defaults to 10.
        dpi: Dots per inch. Defaults to 100.
        title: Title of the animation. Defaults to "".
        denormalize: Whether to denormalize the images. Defaults to True. (-1 to 1 -> 0 to 255)
    """

    fig = plt.figure()
    plt.axis("off")
    plt.title(title)

    # images is a tensor of shape (B, C, H, W)

    # images = denormalize_img(images) if denormalize else images

    # images = einops.rearrange(images, "b c h w -> b h w c").numpy()

    ims = []
    for img in images:
        im = HTMLImageDisplayer._convert_img(img, denormalize)

        im = plt.imshow(img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=1000 // fps, blit=True)
    ani.save(file_path, writer="imagemagick", dpi=dpi)


class HTMLImageDisplayer:
    """A class to display images in a Jupyter notebook using HTML."""

    def __init__(self):
        self.html = ""

    def _save_img(self, img: Image.Image, file_path: str) -> None:
        """Save an image to a file."""

        img.save(f"{file_path}.png")

    @staticmethod
    def _convert_img(
         img: Union[np.ndarray, Image.Image, torch.Tensor], denormalize: bool = True
    ) -> Image.Image:
        """Convert an image tensor to a PIL image.
        
        Args:
            img: Image tensor of shape (H, W, C) 
            denormalize: Whether to denormalize the image. Defaults to True.

        """
        logger.debug(f"Converting image to PIL image.")

        if isinstance(img, (torch.Tensor, np.ndarray)):
            logger.debug(f"Image is a tensor or numpy array.")
            img = denormalize_img(img) if denormalize else img
            img = img.detach().cpu().numpy()
            logger.debug(f"Img shape after denormalize: {img.shape}")

            # if isinstance(img, torch.Tensor):
            #     img = denormalize_img(img)
            #     logger.debug(f"Img shape in tensor after denormalize: {img.shape}")

            if len(img.shape) == 4:
                logger.error(f"Image tensor has more than 3 dimensions.")
                raise ValueError("Image tensor has more than 3 dimensions.")
                # img = img.squeeze().detach().cpu().numpy()
                # img = np.transpose(img, (1, 2, 0))

        if isinstance(img, Image.Image):
            img = np.array(img)

        return Image.fromarray(img)

    def _single_img_to_html(
        self,
        html: str,
        img: Union[np.ndarray, Image.Image, torch.Tensor],
        title: str,
        save_path: Optional[str] = None,
        height=200,
        width=200,
        denormalize: bool = True,
    ) -> str:

        """
        Convert a single image to HTML.

        Args:
            html: HTML string.
            img: Image tensor of shape (H, W, C).
            title: Title of the image.
            save_path: Path to save the image.
            height: Height of the image.
            width: Width of the image.
            denormalize: Whether to denormalize the image. Defaults to True.
        
        """

        img = self._convert_img(img, denormalize)

        if save_path:
            self._save_img(img, os.path.join(save_path, title))

        img_io = BytesIO()
        img.save(img_io, "PNG")
        img_io.seek(0)
        img_ = base64.b64encode(img_io.getvalue()).decode()

        html += f"""
            <div style="text-align:center; width:{width}px; margin-bottom: 20px;">
                <h3 style="margin-bottom: 5px;">{title}</h3>
                <a href="data:image/png;base64,{img_}" target="_blank">
                    <img src="data:image/png;base64,{img_}" height={height} width={width}" loading="lazy">
                </a>
            </div>
        """

        return html

    def update_image(
        self,
        images: Union[List[torch.Tensor], torch.Tensor],
        base_title: str = "",
        height=200,
        width=200,
        save_path: Optional[str] = None,
        denormalize: bool = True,
    ) -> None:
        """Display multiple image tensors.
        
        Args:
            images: List of image tensors of shape (B, C, H, W) or List[(H, W, C)].
            base_title: Base title for the images.
            height: Height of the image.
            width: Width of the image.
            save_path: Path to save the images
            denormalize: Whether to denormalize the images. Defaults to True.
        """
        # if isinstance(images, torch.Tensor):
        #     images = [images]

        # check if images is a tensor and have 4 dimensions
        if isinstance(images, torch.Tensor) and len(images.shape) == 4:
            images = einops.rearrange(images, "b c h w -> b h w c")

        if isinstance(images, List) and len(images[0].shape) == 4:
            m = "Images is a list of tensors with 4 dimensions. Please provide a list of 3-d tensors."
            logger.error(m)
            raise ValueError(m)

        for i, img in enumerate(images):  # if 4-d tensor, loop through the batch
            logger.debug(f"Displaying image {i} of shape {img.shape}")
            title = f"{base_title}: {i}"
            self.html += self._single_img_to_html(
                self.html, img, title, save_path, height, width, denormalize=denormalize
            )
            self.html += f"<br><br>"

        if is_jupyter_notebook():
            display(HTML(self.html))

    def clear(self) -> None:
        """Clear the displayed images."""
        self.html = ""
        clear_output(wait=True)

    def save(self, file_path: str) -> None:
        """Save the displayed images to a file."""
        with open(file_path, "w") as f:
            f.write(self.html)

    def display_grid(
        self,
        images: Union[List[torch.Tensor], torch.Tensor],
        captions: Optional[List[str]] = None,
        base_title: str = "",
        cols: int = 8,
        height=200,
        width=200,
        save_path: Optional[str] = None,
        denormalize: bool = False,
    ) -> None:
        """Display images in a grid layout with relative padding.
        
        Args:
            images: List of image tensors of shape (B, C, H, W) or List[(H, W, C)].
            
            captions: List of captions for the images. Defaults to None.
            
            base_title: Base title for the images.
            
            cols: Number of columns in the grid. Defaults to 8.
            
            height: Height of the image. Defaults to 200.
            
            width: Width of the image. Defaults to 200.
            
            save_path: Path to save the images. Defaults to None.
            
            denormalize: Whether to denormalize the images. Defaults to False.
            
        """
        if isinstance(images, torch.Tensor):
            images = [images]

        if captions is None:
            captions = [f"{base_title}: {i}" for i in range(len(images))]

        rows = (len(images) + cols - 1) // cols  # Calculate number of rows needed
        html = '<div style="display: flex; flex-wrap: wrap; justify-content: center;">'

        for i, img in enumerate(images):
            title = captions[i] if captions else f"{base_title}: {i}"
            img_html = self._single_img_to_html("", img, title, save_path, height, width, denormalize=denormalize)
            html += f'''
                <div style="flex: 1 0 {100 // cols - 2}%;" class="responsive-img">
                    {img_html}
                </div>
            '''

        html += "</div>"
        self.html += html

        # Add CSS for responsive images with relative padding
        # self.html += f"""
        # <style>
        #     .responsive-img {{
        #         padding: 1px;
        #     }}
        #     @media screen and (max-width: 768px) {{
        #         .responsive-img {{
        #             flex: 1 0 100%;
        #             max-width: 100%;
        #         }}
        #     }}
        # </style>
        # """

        if is_jupyter_notebook():
            display(HTML(self.html))

class ImagePlotter:
    """A class to display images. It can be used to display images in a loop."""

    def __init__(
        self,
        cmap: str = "viridis",
        **kwargs: dict[str, any],
    ):
        """Initializes the ImagePlotter object.

        Parameters
        ----------
        title : str, optional
            The title of the figure. Default is "".
        cmap : str, optional
            The colormap to be used. Default is "viridis".
        kwargs
            Additional keyword arguments to be passed to the `plt.subplots` method.
        """

        self.fig, self.ax = plt.subplots(**kwargs)
        self.im = None
        self.cmap = cmap

    def update_image(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        title: str = "",
        path_to_save: Union[str, None] = None,
    ) -> None:
        # convert pil image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        # convert tensor to numpy array
        if isinstance(image, torch.Tensor):
            image = denormalize_img(image)
            image = einops.rearrange(image, "1 c h w -> h w c").detach().cpu().numpy()
            # image = image.squeeze().detach().cpu().numpy()
            # # permute the channels to the last dimension
            # image = np.transpose(image, (1, 2, 0))

        channels = image.shape[-1]
        if channels == 1 and self.cmap not in ["gray", "Greys"]:
            cmap = "gray"
        else:
            cmap = self.cmap

        if self.im is None:
            self.im = self.ax.imshow(image, cmap=cmap)
        else:
            self.im.set_data(image)
        self.ax.set_title(title)
        self.ax.title.set_fontsize(15)
        self.ax.axis("off")
        plt.draw()
        plt.pause(0.01)
        # display the figure if running in a Jupyter notebook
        if is_jupyter_notebook():
            # clear_output(wait=True)
            display(self.fig, clear=True)
        if path_to_save is not None:
            self.fig.savefig(path_to_save)


def create_wandb_logger(
    project: str,
    entity: str,
    tags: List[str],
    name: str,
    config: Dict[str, Any],
    notes: str,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
) -> None:
    """Create a Weights & Biases logger.

    Parameters
    ----------
    project : str
        The name of the project.
    entity : str
        The entity to which the project belongs.
    tags : List[str]
        A list of tags for the run.
    name : str
        The name of the run.
    config : Dict[str, Any]
        A dictionary containing the configuration parameters.
    notes : str
        Notes for the run.
    group : str, optional
        The name of the group. Default is None.
    job_type : str, optional
        The type of job. Default is None.
    """

    wandb.init(
        project=project,
        entity=entity,
        tags=tags,
        name=name,
        config=config,
        notes=notes,
        group=group,
        job_type=job_type,
    )

    return wandb
