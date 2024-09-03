import base64
from io import BytesIO
import torch
import torch.nn as nn
from collections import OrderedDict

import logging
from logging import Logger

from typing import List, Optional, Dict, Callable, Literal, Union, Tuple, Any
from copy import deepcopy

from IPython.display import clear_output, display, HTML
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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


############################################# Logger #######################################


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
    img = plt.imread(file_path)  # (H, W, C)
    # Normalize the image between -1 and 1
    img = np.moveaxis(img, -1, 0)  # (C, H, W)
    img = torch.tensor(img, dtype=torch.float32)  # (C, H, W)
    img = img.unsqueeze(0)  # (B, C, H, W)
    img = normalize_img(img)
    return img


def tensor_to_img(tensor: torch.Tensor, file_path: str) -> None:
    """
    Convert a tensor to an image and save it.

    Args:
        tensor: Tensor representation of the image of shape (1, C, H, W).
        file_path: Path to save the image file.
    """
    from PIL import Image

    img = tensor.squeeze().permute(1, 2, 0).byte().numpy()
    img = Image.fromarray(img)
    img.save(file_path)


def normalize_img(img: torch.Tensor) -> torch.Tensor:
    """
    Normalize an image tensor.

    Args:
        img: Tensor representation of the image of shape (1, C, H, W).

    Returns:
        Normalized image tensor.
    """
    return 2 * (img.float() / 255.0) - 1.0


def denormalize_img(img: torch.Tensor) -> torch.Tensor:
    """
    Normalize an image tensor.

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

    img = img.squeeze().permute(1, 2, 0).byte().numpy()
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


class HTMLImageDisplayer:
    """A class to display images in a Jupyter notebook using HTML."""

    def __init__(self):
        self.html = ""

    def _convert_img(
        self, img: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> Image.Image:
        """Convert an image tensor to a PIL image.

        Parameters
        ----------
        img : torch.Tensor
            Tensor representation of the image of shape (1, C, H, W).

        Returns
        -------
        Image.Image
            PIL image.
        """

        logger.debug(f"Converting image to PIL image.")

        if isinstance(img, (torch.Tensor, np.ndarray)):

            logger.debug(f"Img shape in tensor: {img.shape}")

            if isinstance(img, torch.Tensor):
                img = denormalize_img(img)
                logger.debug(f"Img shape in tensor after denormalize: {img.shape}")

            if len(img.shape) == 4:

                img = img.squeeze().detach().cpu().numpy()

                img = np.transpose(img, (1, 2, 0))

        if isinstance(img, Image.Image):
            img = np.array(img)

        return Image.fromarray(img)

    def _single_img_to_html(
        self,
        html: str,
        img: Union[np.ndarray, Image.Image, torch.Tensor],
        title: str,
        height=400,
        width=400,
    ) -> str:

        img = self._convert_img(img)

        # BytesIO object to display image in Jupyter notebook
        img_io = BytesIO()
        img.save(img_io, "PNG")
        img_io.seek(0)
        img_ = base64.b64encode(img_io.getvalue()).decode()

        html += f"""
            <div style="text-align:center; width:{width}px; margin-bottom: 20px;">
                <h3 style="margin-bottom: 5px;">{title}</h3>
                <img src="data:image/png;base64,{img_}" height={height} width={width}>
            </div>
        """

        return html

    def update_image(
        self,
        images: Union[List[torch.Tensor], torch.Tensor],
        base_title: str = "",
        height=400,
        width=400,
    ) -> None:
        """Display multiple image tensors.

        Parameters
        ----------
        images : Union[List[np.ndarray, Image.Image, torch.Tensor], str]
            List of tensor representations of the images of shape (1, C, H, W).
        """

        if isinstance(images, torch.Tensor):
            images = [images]

        for i, img in enumerate(images):

            logger.debug(f"Img shape: {img.shape}")
            title = f"{base_title}: {i}"
            self.html += self._single_img_to_html(self.html, img, title, height, width)
            self.html += f"<br><br>"

        if is_jupyter_notebook():
            display(HTML(self.html))

    def clear(self) -> None:
        """Clear the displayed images."""
        self.html = ""
        clear_output(wait=True)

    def save(self, file_path: str) -> None:
        """Save the displayed images to a file.

        Parameters
        ----------
        file_path : str
            Path to save the file.
        """
        with open(file_path, "w") as f:
            f.write(self.html)


class HTMLImageDisplayer:
    """A class to display images in a Jupyter notebook using HTML."""

    def __init__(self):
        self.html = ""

    def _convert_img(
        self, img: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> Image.Image:
        """Convert an image tensor to a PIL image."""
        logger.debug(f"Converting image to PIL image.")

        if isinstance(img, (torch.Tensor, np.ndarray)):
            logger.debug(f"Img shape in tensor: {img.shape}")

            if isinstance(img, torch.Tensor):
                img = denormalize_img(img)
                logger.debug(f"Img shape in tensor after denormalize: {img.shape}")

            if len(img.shape) == 4:
                img = img.squeeze().detach().cpu().numpy()
                img = np.transpose(img, (1, 2, 0))

        if isinstance(img, Image.Image):
            img = np.array(img)

        return Image.fromarray(img)

    def _single_img_to_html(
        self,
        html: str,
        img: Union[np.ndarray, Image.Image, torch.Tensor],
        title: str,
        height=400,
        width=400,
    ) -> str:
        img = self._convert_img(img)

        img_io = BytesIO()
        img.save(img_io, "PNG")
        img_io.seek(0)
        img_ = base64.b64encode(img_io.getvalue()).decode()

        html += f"""
            <div style="text-align:center; width:{width}px; margin-bottom: 20px;">
                <h3 style="margin-bottom: 5px;">{title}</h3>
                <img src="data:image/png;base64,{img_}" height={height} width={width}>
            </div>
        """

        return html

    def update_image(
        self,
        images: Union[List[torch.Tensor], torch.Tensor],
        base_title: str = "",
        height=400,
        width=400,
    ) -> None:
        """Display multiple image tensors."""
        if isinstance(images, torch.Tensor):
            images = [images]

        for i, img in enumerate(images):
            logger.debug(f"Img shape: {img.shape}")
            title = f"{base_title}: {i}"
            self.html += self._single_img_to_html(self.html, img, title, height, width)
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
        base_title: str = "",
        cols: int = 3,
        height=400,
        width=400,
    ) -> None:
        """Display images in a grid layout."""
        if isinstance(images, torch.Tensor):
            images = [images]

        rows = (len(images) + cols - 1) // cols  # Calculate number of rows needed
        html = '<div style="display: flex; flex-wrap: wrap;">'

        for i, img in enumerate(images):
            title = f"{base_title}: {i}"
            img_html = self._single_img_to_html("", img, title, height, width)
            html += f'<div style="flex: 1 0 {100 // cols}%;">{img_html}</div>'

        html += "</div>"
        self.html += html

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
            image = image.squeeze().detach().cpu().numpy()
            # permute the channels to the last dimension
            image = np.transpose(image, (1, 2, 0))

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
