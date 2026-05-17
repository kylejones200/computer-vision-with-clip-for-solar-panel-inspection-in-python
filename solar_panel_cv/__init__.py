"""solar_panel_cv — split from `legacy`."""

from .download_dataset import download_dataset
from .download_dataset_2 import download_dataset_2
from .download_dataset_3 import download_dataset_3
from .fine_tuning import fine_tuning
from .matplotlib_inline_jupyter_only import matplotlib_inline_jupyter_only
from .matplotlib_inline_jupyter_only_2 import matplotlib_inline_jupyter_only_2
from .set_image_dimensions import set_image_dimensions
from .set_image_dimensions_2 import set_image_dimensions_2
from .set_image_dimensions_3 import set_image_dimensions_3
from .set_image_dimensions_4 import set_image_dimensions_4
from .set_image_dimensions_5 import set_image_dimensions_5
from .set_image_dimensions_6 import set_image_dimensions_6
from .set_up_binary_classification_directories import set_up_binary_classification_directories
from . import steps

from .steps import main

__all__ = ['download_dataset', 'download_dataset_2', 'download_dataset_3', 'fine_tuning', 'main', 'matplotlib_inline_jupyter_only', 'matplotlib_inline_jupyter_only_2', 'set_image_dimensions', 'set_image_dimensions_2', 'set_image_dimensions_3', 'set_image_dimensions_4', 'set_image_dimensions_5', 'set_image_dimensions_6', 'set_up_binary_classification_directories']
