# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.3.109"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics.models import NAS, RTDETR, SAM, FastSAM, YOLO
from ultralytics.utils.downloads import download

from flabplatform.flabdet.utils.yolos import ASSETS, SETTINGS
from flabplatform.flabdet.utils.yolos.checks import check_yolo as checks
# from flabplatform.utils.engine import YOLORunnerWarper, YOLOWorld, YOLOE

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    # "YOLOWorld",
    # "YOLOE",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
)
