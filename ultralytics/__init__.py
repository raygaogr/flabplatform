# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.3.109"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics.models import NAS, SAM, FastSAM
from ultralytics.utils.downloads import download


# settings = SETTINGS
__all__ = (
    "__version__",
    "NAS",
    "SAM",
    "FastSAM",
    "download",
)
