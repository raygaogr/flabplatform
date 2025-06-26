
from pathlib import Path
from typing import Any, Dict, List, Union
import torch
import numpy as np
from PIL import Image
import inspect

from flabplatform.flabdet.registry import MODELS, TRAINERS, PREDICTORS, VALIDATORS
from flabplatform.flabdet.utils.yolos import (
    ASSETS,  DEFAULT_CFG_DICT, LOGGER, RANK,
    emojis, yaml_load, yaml_save,
    YOLO_ROOT, checks, SETTINGS
)
from flabplatform.core.config import Config
from flabplatform.flabdet.models import attempt_load_one_weight, guess_model_task, yaml_model_load
from flabplatform.flabdet.configs import get_cfg, get_save_dir
from .baserunner import BaseRunner
import os

class YOLORunner(BaseRunner):
    """
    A base class for implementing YOLO models, unifying APIs across different model types.

    This class provides a common interface for various operations related to YOLO models, such as training,
    validation, prediction, exporting, and benchmarking. It handles different types of models, including those
    loaded from local files, Ultralytics HUB, or Triton Server.

    Attributes:
        callbacks (dict): A dictionary of callback functions for various events during model operations.
        predictor (BasePredictor): The predictor object used for making predictions.
        model (torch.nn.Module): The underlying PyTorch model.
        trainer (BaseTrainer): The trainer object used for training the model.
        ckpt (dict): The checkpoint data if the model is loaded from a *.pt file.
        cfg (str): The configuration of the model if loaded from a *.yaml file.
        ckpt_path (str): The path to the checkpoint file.
        overrides (dict): A dictionary of overrides for model configuration.
        metrics (dict): The latest training/validation metrics.
        task (str): The type of task the model is intended for.
        model_name (str): The name of the model.

    Methods:
        __call__: Alias for the predict method, enabling the model instance to be callable.
        _new: Initializes a new model based on a configuration file.
        _load: Loads a model from a checkpoint file.
        _check_is_pytorch_model: Ensures that the model is a PyTorch model.
        reset_weights: Resets the model's weights to their initial state.
        load: Loads model weights from a specified file.
        save: Saves the current state of the model to a file.
        info: Logs or returns information about the model.
        fuse: Fuses Conv2d and BatchNorm2d layers for optimized inference.
        predict: Performs object detection predictions.
        track: Performs object tracking.
        val: Validates the model on a dataset.
        benchmark: Benchmarks the model on various export formats.
        export: Exports the model to different formats.
        train: Trains the model on a dataset.
        tune: Performs hyperparameter tuning.
        _apply: Applies a function to the model's tensors.
        add_callback: Adds a callback function for an event.
        clear_callback: Clears all callbacks for an event.
        reset_callbacks: Resets all callbacks to their default functions.

    Examples:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> results = model.predict("image.jpg")
        >>> model.train(data="coco8.yaml", epochs=3)
        >>> metrics = model.val()
        >>> model.export(format="onnx")
    """

    def __init__(
        self,
        overrides: dict,
        verbose: bool = False,
    ) -> None:
        """
        Initialize a new instance of the YOLO model class.

        This constructor sets up the model based on the provided model path or name. It handles various types of
        model sources, including local files, Ultralytics HUB models, and Triton Server models. The method
        initializes several important attributes of the model and prepares it for operations like training,
        prediction, or export.

        Args:
            model (str | Path): Path or name of the model to load or create. Can be a local file path, a
                model name from Ultralytics HUB, or a Triton Server model.
            task (str | None): The task type associated with the YOLO model, specifying its application domain.
            verbose (bool): If True, enables verbose output during the model's initialization and subsequent
                operations.

        Raises:
            FileNotFoundError: If the specified model file does not exist or is inaccessible.
            ValueError: If the model file or configuration is invalid or unsupported.
            ImportError: If required dependencies for specific model types (like HUB SDK) are not installed.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model = Model("path/to/model.yaml", task="detect")
            >>> model = Model("hub_model", verbose=True)
        """
        from ultralytics.utils import callbacks
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = {}  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.metrics = None  # validation/training metrics

        self.overrides = overrides
        self.task = overrides.get("task", None)  # task type
        self.pretrainDir= overrides.get("pretrainDir", None)

        model = str(self.overrides["model"]).strip()
        if not Path(model).suffix:
            model = model + ".yaml"
        __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # to avoid deterministic warnings
        if Path(model).suffix in {".yaml", ".yml"} and not self.pretrainDir:
            self._new(model, task=self.task, verbose=verbose)
        else:
            if not model.endswith("pt"):
                model = os.path.join(self.pretrainDir, "best.pt")
                if not os.path.exists(model):
                    raise FileNotFoundError(f"Model file '{model}' does not exist. Please check the path or model name.")
            self._load(model, task=self.task)

        # Delete super().training for accessing self.model.training
        del self.training

    def __call__(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """
        Alias for the predict method, enabling the model instance to be callable for predictions.

        This method simplifies the process of making predictions by allowing the model instance to be called
        directly with the required arguments.

        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple): The source of
                the image(s) to make predictions on. Can be a file path, URL, PIL image, numpy array, PyTorch
                tensor, or a list/tuple of these.
            stream (bool): If True, treat the input source as a continuous stream for predictions.
            **kwargs (Any): Additional keyword arguments to configure the prediction process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, each encapsulated in a
                Results object.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model("https://ultralytics.com/images/bus.jpg")
            >>> for r in results:
            ...     print(f"Detected {len(r)} objects in image")
        """
        return self.predict(source, stream, **kwargs)

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """
        Initialize a new model and infer the task type from model definitions.

        Creates a new model instance based on the provided configuration file. Loads the model configuration, infers
        the task type if not specified, and initializes the model using the appropriate class from the task map.

        Args:
            cfg (str): Path to the model configuration file in YAML format.
            task (str | None): The specific task for the model. If None, it will be inferred from the config.
            model (torch.nn.Module | None): A custom model instance. If provided, it will be used instead of creating
                a new one.
            verbose (bool): If True, displays model information during loading.

        Raises:
            ValueError: If the configuration file is invalid or the task cannot be inferred.
            ImportError: If the required dependencies for the specified task are not installed.

        Examples:
            >>> model = Model()
            >>> model._new("yolo11n.yaml", task="detect", verbose=True)
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        model_cfg = {}
        model_cfg["type"] = self._smart_load("model")
        model_cfg["cfg"] = cfg_dict
        self.model = MODELS.build(model_cfg)

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task

    def _load(self, weights: str, task=None) -> None:
        """
        Load a model from a checkpoint file or initialize it from a weights file.

        This method handles loading models from either .pt checkpoint files or other weight file formats. It sets
        up the model, task, and related attributes based on the loaded weights.

        Args:
            weights (str): Path to the model weights file to be loaded.
            task (str | None): The task associated with the model. If None, it will be inferred from the model.

        Raises:
            FileNotFoundError: If the specified weights file does not exist or is inaccessible.
            ValueError: If the weights file format is unsupported or invalid.

        Examples:
            >>> model = Model()w
            >>> model._load("yolo11n.pt")
            >>> model._load("path/to/weights.pth", task="detect")
        """
        weights = checks.check_model_file_from_stem(weights)  # add suffix, i.e. yolo11n -> yolo11n.pt

        if Path(weights).suffix == ".pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args["task"]
            weights = self.model.args.get("model", "yolo11n").strip().split(".")[0]
            self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else: #TODO 
            weights = checks.check_file(weights)  # runs in all cases, not redundant with above call
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task

    def _check_is_pytorch_model(self) -> None:
        """
        Check if the model is a PyTorch model and raise TypeError if it's not.

        This method verifies that the model is either a PyTorch module or a .pt file. It's used to ensure that
        certain operations that require a PyTorch model are only performed on compatible model types.

        Raises:
            TypeError: If the model is not a PyTorch module or a .pt file. The error message provides detailed
                information about supported model formats and operations.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model._check_is_pytorch_model()  # No error raised
            >>> model = Model("yolo11n.onnx")
            >>> model._check_is_pytorch_model()  # Raises TypeError
        """
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt"
        pt_module = isinstance(self.model, torch.nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolo11n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def reset_weights(self) -> "YOLORunner":
        """
        Reset the model's weights to their initial state.

        This method iterates through all modules in the model and resets their parameters if they have a
        'reset_parameters' method. It also ensures that all parameters have 'requires_grad' set to True,
        enabling them to be updated during training.

        Returns:
            (Model): The instance of the class with reset weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.reset_weights()
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: Union[str, Path] = "yolo11n.pt") -> "YOLORunner":
        """
        Load parameters from the specified weights file into the model.

        This method supports loading weights from a file or directly from a weights object. It matches parameters by
        name and shape and transfers them to the model.

        Args:
            weights (Union[str, Path]): Path to the weights file or a weights object.

        Returns:
            (Model): The instance of the class with loaded weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model()
            >>> model.load("yolo11n.pt")
            >>> model.load(Path("path/to/weights.pt"))
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            self.overrides["pretrained"] = weights  # remember the weights for DDP training
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def save(self, filename: Union[str, Path] = "saved_model.pt") -> None:
        """
        Save the current model state to a file.

        This method exports the model's checkpoint (ckpt) to the specified filename. It includes metadata such as
        the date, Ultralytics version, license information, and a link to the documentation.

        Args:
            filename (str | Path): The name of the file to save the model to.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.save("my_model.pt")
        """
        self._check_is_pytorch_model()
        from copy import deepcopy
        from datetime import datetime

        from ultralytics import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, torch.nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "",
            "docs": "",
        }
        torch.save({**self.ckpt, **updates}, filename)

    def info(self, detailed: bool = False, verbose: bool = True):
        """
        Display model information.

        This method provides an overview or detailed information about the model, depending on the arguments
        passed. It can control the verbosity of the output and return the information as a list.

        Args:
            detailed (bool): If True, shows detailed information about the model layers and parameters.
            verbose (bool): If True, prints the information. If False, returns the information as a list.

        Returns:
            (List[str]): A list of strings containing various types of information about the model, including
                model summary, layer details, and parameter counts. Empty if verbose is True.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.info()  # Prints model summary
            >>> info_list = model.info(detailed=True, verbose=False)  # Returns detailed info as a list
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self) -> None:
        """
        Fuse Conv2d and BatchNorm2d layers in the model for optimized inference.

        This method iterates through the model's modules and fuses consecutive Conv2d and BatchNorm2d layers
        into a single layer. This fusion can significantly improve inference speed by reducing the number of
        operations and memory accesses required during forward passes.

        The fusion process typically involves folding the BatchNorm2d parameters (mean, variance, weight, and
        bias) into the preceding Conv2d layer's weights and biases. This results in a single Conv2d layer that
        performs both convolution and normalization in one step.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.fuse()
            >>> # Model is now fused and ready for optimized inference
        """
        self._check_is_pytorch_model()
        self.model.fuse()

    def embed(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """
        Generate image embeddings based on the provided source.

        This method is a wrapper around the 'predict()' method, focusing on generating embeddings from an image
        source. It allows customization of the embedding process through various keyword arguments.

        Args:
            source (str | Path | int | List | Tuple | np.ndarray | torch.Tensor): The source of the image for
                generating embeddings. Can be a file path, URL, PIL image, numpy array, etc.
            stream (bool): If True, predictions are streamed.
            **kwargs (Any): Additional keyword arguments for configuring the embedding process.

        Returns:
            (List[torch.Tensor]): A list containing the image embeddings.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> image = "https://ultralytics.com/images/bus.jpg"
            >>> embeddings = model.embed(image)
            >>> print(embeddings[0].shape)
        """
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]  # embed second-to-last layer if no indices passed
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs: Any,
    ):
        """
        Performs predictions on the given image source using the YOLO model.

        This method facilitates the prediction process, allowing various configurations through keyword arguments.
        It supports predictions with custom predictors or the default predictor method. The method handles different
        types of image sources and can operate in a streaming mode.

        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple): The source
                of the image(s) to make predictions on. Accepts various types including file paths, URLs, PIL
                images, numpy arrays, and torch tensors.
            stream (bool): If True, treats the input source as a continuous stream for predictions.
            predictor (BasePredictor | None): An instance of a custom predictor class for making predictions.
                If None, the method uses a default predictor.
            **kwargs (Any): Additional keyword arguments for configuring the prediction process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, each encapsulated in a
                Results object.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.predict(source="path/to/image.jpg", conf=0.25)
            >>> for r in results:
            ...     print(r.boxes.data)  # print detection bounding boxes

        Notes:
            - If 'source' is not provided, it defaults to the ASSETS constant with a warning.
            - The method sets up a new predictor if not already present and updates its arguments with each call.
            - For SAM-type models, 'prompts' can be passed as a keyword argument.
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        custom = {"batch": 1, "save": False, "mode": "predict", "rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right
        if args.get("conf") is None:
            args["conf"] = 0.25
        prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor:
            predictor_cfg = {}
            predictor_cfg["type"] = self._smart_load("predictor")
            predictor_cfg["overrides"] = args
            predictor_cfg["_callbacks"] = self.callbacks
            self.predictor = PREDICTORS.build(predictor_cfg)
            self.predictor.setup_model(model=self.model, verbose=False)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        return self.predictor(source=source, stream=stream)

    def track(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs: Any,
    ) :
        """
        Conducts object tracking on the specified input source using the registered trackers.

        This method performs object tracking using the model's predictors and optionally registered trackers. It handles
        various input sources such as file paths or video streams, and supports customization through keyword arguments.
        The method registers trackers if not already present and can persist them between calls.

        Args:
            source (Union[str, Path, int, List, Tuple, np.ndarray, torch.Tensor], optional): Input source for object
                tracking. Can be a file path, URL, or video stream.
            stream (bool): If True, treats the input source as a continuous video stream.
            persist (bool): If True, persists trackers between different calls to this method.
            **kwargs (Any): Additional keyword arguments for configuring the tracking process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of tracking results, each a Results object.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.track(source="path/to/video.mp4", show=True)
            >>> for r in results:
            ...     print(r.boxes.id)  # print tracking IDs

        Notes:
            - This method sets a default confidence threshold of 0.1 for ByteTrack-based tracking.
            - The tracking mode is explicitly set in the keyword arguments.
            - Batch size is set to 1 for tracking in videos.
        """
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs["batch"] = kwargs.get("batch") or 1  # batch-size 1 for tracking in videos
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def val(
        self,
        **kwargs: Any,
    ):
        """
        Validate the model using a specified dataset and validation configuration.

        This method facilitates the model validation process, allowing for customization through various settings. It
        supports validation with a custom validator or the default validation approach. The method combines default
        configurations, method-specific defaults, and user-provided arguments to configure the validation process.

        Args:
            validator (ultralytics.engine.validator.BaseValidator | None): An instance of a custom validator class for
                validating the model.
            **kwargs (Any): Arbitrary keyword arguments for customizing the validation process.

        Returns:
            (ultralytics.utils.metrics.DetMetrics): Validation metrics obtained from the validation process.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.val(data="coco8.yaml", imgsz=640)
            >>> print(results.box.map)  # Print mAP50-95
        """
        custom = {"rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator_cfg = {}
        validator_cfg["type"] = self._smart_load("validator")
        validator_cfg["args"] = args
        validator_cfg["_callbacks"] = self.callbacks
        validator = VALIDATORS.build(validator_cfg)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(
        self,
        **kwargs: Any,
    ):
        """
        Benchmark the model across various export formats to evaluate performance.

        This method assesses the model's performance in different export formats, such as ONNX, TorchScript, etc.
        It uses the 'benchmark' function from the ultralytics.utils.benchmarks module. The benchmarking is
        configured using a combination of default configuration values, model-specific arguments, method-specific
        defaults, and any additional user-provided keyword arguments.

        Args:
            **kwargs (Any): Arbitrary keyword arguments to customize the benchmarking process. Common options include:
                - data (str): Path to the dataset for benchmarking.
                - imgsz (int | List[int]): Image size for benchmarking.
                - half (bool): Whether to use half-precision (FP16) mode.
                - int8 (bool): Whether to use int8 precision mode.
                - device (str): Device to run the benchmark on (e.g., 'cpu', 'cuda').
                - verbose (bool): Whether to print detailed benchmark information.
                - format (str): Export format name for specific benchmarking.

        Returns:
            (dict): A dictionary containing the results of the benchmarking process, including metrics for
                different export formats.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.benchmark(data="coco8.yaml", imgsz=640, half=True)
            >>> print(results)
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        custom = {"verbose": False}  # method defaults
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        return benchmark(
            model=self,
            data=kwargs.get("data"),  # if no 'data' argument passed set data=None for default datasets
            imgsz=args["imgsz"],
            half=args["half"],
            int8=args["int8"],
            device=args["device"],
            verbose=kwargs.get("verbose", False),
            format=kwargs.get("format", ""),
        )

    def export(
        self,
        **kwargs: Any,
    ) -> str:
        """
        Export the model to a different format suitable for deployment.

        This method facilitates the export of the model to various formats (e.g., ONNX, TorchScript) for deployment
        purposes. It uses the 'Exporter' class for the export process, combining model-specific overrides, method
        defaults, and any additional arguments provided.

        Args:
            **kwargs (Any): Arbitrary keyword arguments to customize the export process. These are combined with
                the model's overrides and method defaults. Common arguments include:
                format (str): Export format (e.g., 'onnx', 'engine', 'coreml').
                half (bool): Export model in half-precision.
                int8 (bool): Export model in int8 precision.
                device (str): Device to run the export on.
                workspace (int): Maximum memory workspace size for TensorRT engines.
                nms (bool): Add Non-Maximum Suppression (NMS) module to model.
                simplify (bool): Simplify ONNX model.

        Returns:
            (str): The path to the exported model file.

        Raises:
            AssertionError: If the model is not a PyTorch model.
            ValueError: If an unsupported export format is specified.
            RuntimeError: If the export process fails due to errors.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.export(format="onnx", dynamic=True, simplify=True)
            'path/to/exported/model.onnx'
        """
        self._check_is_pytorch_model()
        from flabplatform.flabdet.export import Exporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,  # reset to avoid multi-GPU errors
            "verbose": False,
        }  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def build_trainer(self):
        save_dir = self.overrides.get("save_dir", None)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        params_yaml = os.path.join(save_dir, "params.yaml")
        params_content = {
            "model": {
                    "name": "yolov8n",
                    "version": "v1.0",
                    "type": "cv",
                    "task": "detect",
                    "framework": "pytorch"
            },
            "input": [
                {
                    "type": "image",
                    "shape": [1, 3, 384, 384],
                    "dtype": "float32",
                    "normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                        "color_space": "RGB"
                    }
                },
                {
                    "type": "image",
                    "shape": [1, 3, 384, 384],
                    "dtype": "float32",
                    "normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                        "color_space": "RGB"
                    }
                }
            ],
            "output": [
                {
                    "type": "class_probabilities",
                    "shape": [1, 1000],
                    "dtype": "float32",
                    "class_labels": "labels.txt"
                },
                {
                    "type": "bounding_boxes",
                    "shape": [1, 4]
                }
            ],
            "deployment": {
                "min_hardware": "CPU",
                "recommended_hardware": "GPU with 4GB VRAM",
                "quantization": "int8",
                "supported_formats": ["onnx", "torchscript"]
            },
            "metrics": {
                "accuracy": 94.2,
                "precision": 93.8,
                "recall": 94.1,
                "f1_score": 93.9,
                "latency": "15ms (T4 GPU)"
            },
            "training_dataset": "ImageNet-1K",
            "created_date": "2025-06-05 18:00:00",
            "author": "Detection Team"
        }
        yaml_save(params_yaml, params_content)

        self._check_is_pytorch_model()

        overrides = self.overrides
        args = {**overrides, "mode": "train"}  # highest priority args on the right
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        trainer_cfg = {}
        trainer_cfg["type"] = self._smart_load("trainer")
        trainer_cfg["overrides"] = args
        trainer_cfg["_callbacks"] = self.callbacks
        trainer = TRAINERS.build(trainer_cfg)
        if not args.get("resume"):  # manually set model only if not resuming
            trainer.model = trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = trainer.model
        return trainer

    def train(
        self,
    ):
        """
        Trains the model using the specified dataset and training configuration.

        This method facilitates model training with a range of customizable settings. It supports training with a
        custom trainer or the default training approach. The method handles scenarios such as resuming training
        from a checkpoint, integrating with Ultralytics HUB, and updating model and configuration after training.

        When using Ultralytics HUB, if the session has a loaded model, the method prioritizes HUB training
        arguments and warns if local arguments are provided. It checks for pip updates and combines default
        configurations, method-specific defaults, and user-provided arguments to configure the training process.

        Args:
            trainer (BaseTrainer | None): Custom trainer instance for model training. If None, uses default.
            **kwargs (Any): Arbitrary keyword arguments for training configuration. Common options include:
                data (str): Path to dataset configuration file.
                epochs (int): Number of training epochs.
                batch_size (int): Batch size for training.
                imgsz (int): Input image size.
                device (str): Device to run training on (e.g., 'cuda', 'cpu').
                workers (int): Number of worker threads for data loading.
                optimizer (str): Optimizer to use for training.
                lr0 (float): Initial learning rate.
                patience (int): Epochs to wait for no observable improvement for early stopping of training.

        Returns:
            (Dict | None): Training metrics if available and training is successful; otherwise, None.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.train(data="coco8.yaml", epochs=3)
        """
        self.trainer = self.build_trainer()
        self.trainer.train()
        # Update model and cfg after training
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, self.ckpt = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP
        return self.metrics

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Conducts hyperparameter tuning for the model, with an option to use Ray Tune.

        This method supports two modes of hyperparameter tuning: using Ray Tune or a custom tuning method.
        When Ray Tune is enabled, it leverages the 'run_ray_tune' function from the ultralytics.utils.tuner module.
        Otherwise, it uses the internal 'Tuner' class for tuning. The method combines default, overridden, and
        custom arguments to configure the tuning process.

        Args:
            use_ray (bool): Whether to use Ray Tune for hyperparameter tuning. If False, uses internal tuning method.
            iterations (int): Number of tuning iterations to perform.
            *args (Any): Additional positional arguments to pass to the tuner.
            **kwargs (Any): Additional keyword arguments for tuning configuration. These are combined with model
                overrides and defaults to configure the tuning process.

        Returns:
            (dict): Results of the hyperparameter search, including best parameters and performance metrics.

        Raises:
            TypeError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.tune(data="coco8.yaml", iterations=5)
            >>> print(results)

            # Use Ray Tune for more advanced hyperparameter search
            >>> results = model.tune(use_ray=True, iterations=20, data="coco8.yaml")
        """
        self._check_is_pytorch_model()
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune

            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from ultralytics.engine.tuner import Tuner

            custom = {}  # method defaults
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    def _apply(self, fn) -> "YOLORunner":
        """
        Apply a function to model tensors that are not parameters or registered buffers.

        This method extends the functionality of the parent class's _apply method by additionally resetting the
        predictor and updating the device in the model's overrides. It's typically used for operations like
        moving the model to a different device or changing its precision.

        Args:
            fn (Callable): A function to be applied to the model's tensors. This is typically a method like
                to(), cpu(), cuda(), half(), or float().

        Returns:
            (Model): The model instance with the function applied and updated attributes.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model = model._apply(lambda t: t.cuda())  # Move model to GPU
        """
        self._check_is_pytorch_model()
        self = super()._apply(fn)  # noqa
        self.predictor = None  # reset predictor as device may have changed
        self.overrides["device"] = self.device  # was str(self.device) i.e. device(type='cuda', index=0) -> 'cuda:0'
        return self

    @property
    def names(self) -> Dict[int, str]:
        """
        Retrieves the class names associated with the loaded model.

        This property returns the class names if they are defined in the model. It checks the class names for validity
        using the 'check_class_names' function from the ultralytics.nn.autobackend module. If the predictor is not
        initialized, it sets it up before retrieving the names.

        Returns:
            (Dict[int, str]): A dictionary of class names associated with the model, where keys are class indices and
                values are the corresponding class names.

        Raises:
            AttributeError: If the model or predictor does not have a 'names' attribute.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.names)
            {0: 'person', 1: 'bicycle', 2: 'car', ...}
        """
        from ultralytics.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        if not self.predictor:  # export formats will not have predictor defined until predict() is called
            self.predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=False)
        return self.predictor.model.names

    @property
    def device(self) -> torch.device:
        """
        Get the device on which the model's parameters are allocated.

        This property determines the device (CPU or GPU) where the model's parameters are currently stored. It is
        applicable only to models that are instances of torch.nn.Module.

        Returns:
            (torch.device): The device (CPU/GPU) of the model.

        Raises:
            AttributeError: If the model is not a torch.nn.Module instance.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.device)
            device(type='cuda', index=0)  # if CUDA is available
            >>> model = model.to("cpu")
            >>> print(model.device)
            device(type='cpu')
        """
        return next(self.model.parameters()).device if isinstance(self.model, torch.nn.Module) else None

    @property
    def transforms(self):
        """
        Retrieves the transformations applied to the input data of the loaded model.

        This property returns the transformations if they are defined in the model. The transforms
        typically include preprocessing steps like resizing, normalization, and data augmentation
        that are applied to input data before it is fed into the model.

        Returns:
            (object | None): The transform object of the model if available, otherwise None.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> transforms = model.transforms
            >>> if transforms:
            ...     print(f"Model transforms: {transforms}")
            ... else:
            ...     print("No transforms defined for this model.")
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        """
        Add a callback function for a specified event.

        This method allows registering custom callback functions that are triggered on specific events during
        model operations such as training or inference. Callbacks provide a way to extend and customize the
        behavior of the model at various stages of its lifecycle.

        Args:
            event (str): The name of the event to attach the callback to. Must be a valid event name recognized
                by the Ultralytics framework.
            func (Callable): The callback function to be registered. This function will be called when the
                specified event occurs.

        Raises:
            ValueError: If the event name is not recognized or is invalid.

        Examples:
            >>> def on_train_start(trainer):
            ...     print("Training is starting!")
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", on_train_start)
            >>> model.train(data="coco8.yaml", epochs=1)
        """
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """
        Clears all callback functions registered for a specified event.

        This method removes all custom and default callback functions associated with the given event.
        It resets the callback list for the specified event to an empty list, effectively removing all
        registered callbacks for that event.

        Args:
            event (str): The name of the event for which to clear the callbacks. This should be a valid event name
                recognized by the Ultralytics callback system.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", lambda: print("Training started"))
            >>> model.clear_callback("on_train_start")
            >>> # All callbacks for 'on_train_start' are now removed

        Notes:
            - This method affects both custom callbacks added by the user and default callbacks
              provided by the Ultralytics framework.
            - After calling this method, no callbacks will be executed for the specified event
              until new ones are added.
            - Use with caution as it removes all callbacks, including essential ones that might
              be required for proper functioning of certain operations.
        """
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        """
        Reset all callbacks to their default functions.

        This method reinstates the default callback functions for all events, removing any custom callbacks that were
        previously added. It iterates through all default callback events and replaces the current callbacks with the
        default ones.

        The default callbacks are defined in the 'callbacks.default_callbacks' dictionary, which contains predefined
        functions for various events in the model's lifecycle, such as on_train_start, on_epoch_end, etc.

        This method is useful when you want to revert to the original set of callbacks after making custom
        modifications, ensuring consistent behavior across different runs or experiments.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", custom_function)
            >>> model.reset_callbacks()
            # All callbacks are now reset to their default functions
        """
        from ultralytics.utils import callbacks
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        """
        Reset specific arguments when loading a PyTorch model checkpoint.

        This method filters the input arguments dictionary to retain only a specific set of keys that are
        considered important for model loading. It's used to ensure that only relevant arguments are preserved
        when loading a model from a checkpoint, discarding any unnecessary or potentially conflicting settings.

        Args:
            args (dict): A dictionary containing various model arguments and settings.

        Returns:
            (dict): A new dictionary containing only the specified include keys from the input arguments.

        Examples:
            >>> original_args = {"imgsz": 640, "data": "coco.yaml", "task": "detect", "batch": 16, "epochs": 100}
            >>> reset_args = Model._reset_ckpt_args(original_args)
            >>> print(reset_args)
            {'imgsz': 640, 'data': 'coco.yaml', 'task': 'detect'}
        """
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    def _smart_load(self, key: str):
        """
        Intelligently loads the appropriate module based on the model task.

        This method dynamically selects and returns the correct module (model, trainer, validator, or predictor)
        based on the current task of the model and the provided key. It uses the task_map dictionary to determine
        the appropriate module to load for the specific task.

        Args:
            key (str): The type of module to load. Must be one of 'model', 'trainer', 'validator', or 'predictor'.

        Returns:
            (object): The loaded module class corresponding to the specified key and current task.

        Raises:
            NotImplementedError: If the specified key is not supported for the current task.

        Examples:
            >>> model = Model(task="detect")
            >>> predictor_class = model._smart_load("predictor")
            >>> trainer_class = model._smart_load("trainer")
        """
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")
            ) from e

    @property
    def task_map(self) -> dict:
        """
        Provides a mapping from model tasks to corresponding classes for different modes.

        This property method returns a dictionary that maps each supported task (e.g., detect, segment, classify)
        to a nested dictionary. The nested dictionary contains mappings for different operational modes
        (model, trainer, validator, predictor) to their respective class implementations.

        The mapping allows for dynamic loading of appropriate classes based on the model's task and the
        desired operational mode. This facilitates a flexible and extensible architecture for handling
        various tasks and modes within the Ultralytics framework.

        Returns:
            (Dict[str, Dict[str, Any]]): A dictionary mapping task names to nested dictionaries. Each nested dictionary
            contains mappings for 'model', 'trainer', 'validator', and 'predictor' keys to their respective class
            implementations for that task.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> task_map = model.task_map
            >>> detect_predictor = task_map["detect"]["predictor"]
            >>> segment_trainer = task_map["segment"]["trainer"]
        """
        raise NotImplementedError("Please provide task map for your model!")

    def eval(self):
        """
        Sets the model to evaluation mode.

        This method changes the model's mode to evaluation, which affects layers like dropout and batch normalization
        that behave differently during training and evaluation. In evaluation mode, these layers use running statistics
        rather than computing batch statistics, and dropout layers are disabled.

        Returns:
            (Model): The model instance with evaluation mode set.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.eval()
            >>> # Model is now in evaluation mode for inference
        """
        self.model.eval()
        return self

    def __getattr__(self, name):
        """
        Enable accessing model attributes directly through the Model class.

        This method provides a way to access attributes of the underlying model directly through the Model class
        instance. It first checks if the requested attribute is 'model', in which case it returns the model from
        the module dictionary. Otherwise, it delegates the attribute lookup to the underlying model.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            (Any): The requested attribute value.

        Raises:
            AttributeError: If the requested attribute does not exist in the model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.stride)  # Access model.stride attribute
            >>> print(model.names)  # Access model.names attribute
        """
        return self._modules["model"] if name == "model" else getattr(self.model, name)


class YOLORunnerWarpper(YOLORunner):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, cfg: dict, verbose=True):
        """
        Initialize a YOLO model.

        This constructor initializes a YOLO model, automatically switching to specialized model types
        (YOLOWorld or YOLOE) based on the model filename.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'yolo11n.pt', 'yolov8n.yaml'.
            task (str | None): YOLO task specification, i.e. 'detect', 'segment', 'classify', 'pose', 'obb'.
                Defaults to auto-detection based on model.
            verbose (bool): Display model info on load.

        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n detection model
            >>> model = YOLO("yolov8n-seg.pt")  # load a pretrained YOLOv8n segmentation model
            >>> model = YOLO("yolo11n.pt")  # load a pretrained YOLOv11n detection model
        """
        path = Path(cfg["model"])
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(cfg, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "yoloe" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOE PyTorch model
            new_instance = YOLOE(cfg, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(cfg, verbose=verbose)

    @classmethod
    def from_cfg(cls, cfg: Config):
        cfg_dict = cfg.to_dict()
        special_keys = ["model", "optimizer"]
        ignore_keys = ["mqTopic", "rootDir", "type", "pipelineId", "runId", "nodeId"] #TODO

        def parse_dict(input_dict: dict, output_dict: dict) -> dict:
            for k, v in input_dict.items():
                if k in ignore_keys:
                    continue
                elif k == "outputDir":
                    output_dict["save_dir"] = v
                    SETTINGS.update(dict(runs_dir=v))
                elif k == "operation":
                    if "training" in v:
                        output_dict["mode"] = "train"
                    elif "eval" in v or "val" in v or "test" in v:
                        output_dict["mode"] = "val"
                    else:
                        output_dict["mode"] = v
                elif k == "metafile":
                    output_dict["data"] = v
                elif k in special_keys:
                    output_dict[k] = v["type"]
                    parse_dict(v, output_dict)
                else:
                    if isinstance(v, dict):
                        parse_dict(v, output_dict)
                    else:
                        output_dict[k] = v
            return output_dict
        
        def parse_data(input_dict: dict) -> dict:
            """Parse data dictionary to ensure it has the correct structure."""
            data_cfg = Config.fromfile(input_dict["data"])
            input_dict["data"] = dict()
            input_dict["data"]["path"] = cfg_dict["commonParams"]["datasets"]["rootDir"]
            SETTINGS.update(dict(datasets_dir=input_dict["data"]["path"]))
            input_dict["data"]["names"] = data_cfg.get("labels", [])
            input_dict["data"]["train"] = list()
            input_dict["data"]["val"] = list()
            input_dict["data"]["test"] = list()
            valid_purpose = ["train", "eval", "test"]
            exist_purpose = []
            for dataset in data_cfg["datasets"]:
                dataset_purpose = dataset["purpose"]
                if dataset_purpose in valid_purpose:
                    if dataset["purpose"] == "eval" or dataset["purpose"] == "test":
                        dataset_purpose = "val"
                    exist_purpose.append(dataset_purpose)
                    if len(dataset["samples"]) == 0:
                        input_dict["data"][dataset_purpose].append(dataset["sourceRoot"])
                    else:
                        for sample in dataset["samples"]:
                            input_dict["data"][dataset_purpose].append(os.path.join(dataset["sourceRoot"], sample["image"]))
            if input_dict["mode"] not in exist_purpose and input_dict["mode"] in valid_purpose:
                raise ValueError(
                    f"Invalid operation '{input_dict['mode']}' specified. "
                )
            if len(input_dict["data"]["val"]) == 0 and input_dict["mode"] == "train":
                input_dict["data"]["val"] = input_dict["data"]["train"]
            return input_dict

        overrides = parse_dict(cfg_dict, {})
        overrides = parse_data(overrides)
        return cls(cfg=overrides)
        
    @property
    def task_map(self):
        from ultralytics.models import yolo
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": "ClassificationModel",
                "trainer": "ClassificationTrainer",
                "validator": "ClassificationValidator",
                "predictor": "ClassificationPredictor",
            },
            "detect": {
                "model": "DetectionModel",
                "trainer": "DetectionTrainer",
                "validator": "DetectionValidator",
                "predictor": "DetectionPredictor",
            },
            "segment": {
                "model": "SegmentationModel",
                "trainer": "SegmentationTrainer",
                "validator": "SegmentationValidator",
                "predictor": "SegmentationPredictor",
            },
            "pose": {
                "model": "PoseModel",
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": "OBBModel",
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(YOLORunner):
    """YOLO-World object detection model."""

    def __init__(self, cfg, verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(cfg, verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(YOLO_ROOT / "configs/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        from ultralytics.models import yolo
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": "WorldModel",
                "validator": "DetectionValidator",
                "predictor": "DetectionPredictor",
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        """
        Set the model's class names for detection.

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes


class YOLOE(YOLORunner):
    """YOLOE object detection and segmentation model."""

    def __init__(self, cfg, verbose=False) -> None:
        """
        Initialize YOLOE model with a pre-trained model file.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            task (str, optional): Task type for the model. Auto-detected if None.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(cfg, verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(YOLO_ROOT / "configs/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        from ultralytics.models import yolo
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": "YOLOEModel",
                "validator": yolo.yoloe.YOLOEDetectValidator,
                "predictor": "DetectionPredictor",
                "trainer": yolo.yoloe.YOLOETrainer,
            },
            "segment": {
                "model": "YOLOESegModel",
                "validator": yolo.yoloe.YOLOESegValidator,
                "predictor": yolo.segment.SegmentationPredictor,
                "trainer": yolo.yoloe.YOLOESegTrainer,
            },
        }

    def get_text_pe(self, texts):
        """Get text positional embeddings for the given texts."""
        # assert isinstance(self.model, YOLOEModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        """
        Get visual positional embeddings for the given image and visual features.

        This method extracts positional embeddings from visual features based on the input image. It requires
        that the model is an instance of YOLOEModel.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features extracted from the image.

        Returns:
            (torch.Tensor): Visual positional embeddings.

        Examples:
            >>> model = YOLOE("yoloe-v8s.pt")
            >>> img = torch.rand(1, 3, 640, 640)
            >>> visual_features = model.model.backbone(img)
            >>> pe = model.get_visual_pe(img, visual_features)
        """
        # assert isinstance(self.model, YOLOEModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab, names):
        """
        Set vocabulary and class names for the YOLOE model.

        This method configures the vocabulary and class names used by the model for text processing and
        classification tasks. The model must be an instance of YOLOEModel.

        Args:
            vocab (list): Vocabulary list containing tokens or words used by the model for text processing.
            names (list): List of class names that the model can detect or classify.

        Raises:
            AssertionError: If the model is not an instance of YOLOEModel.

        Examples:
            >>> model = YOLOE("yoloe-v8s.pt")
            >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])
        """
        # assert isinstance(self.model, YOLOEModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        """Get vocabulary for the given class names."""
        # assert isinstance(self.model, YOLOEModel)
        return self.model.get_vocab(names)

    def set_classes(self, classes, embeddings):
        """
        Set the model's class names and embeddings for detection.

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
            embeddings (torch.Tensor): Embeddings corresponding to the classes.
        """
        # assert isinstance(self.model, YOLOEModel)
        self.model.set_classes(classes, embeddings)
        # Verify no background class is present
        assert " " not in classes
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes

    def val(
        self,
        validator=None,
        load_vp=False,
        refer_data=None,
        **kwargs,
    ):
        """
        Validate the model using text or visual prompts.

        Args:
            validator (callable, optional): A callable validator function. If None, a default validator is loaded.
            load_vp (bool): Whether to load visual prompts. If False, text prompts are used.
            refer_data (str, optional): Path to the reference data for visual prompts.
            **kwargs (Any): Additional keyword arguments to override default settings.

        Returns:
            (dict): Validation statistics containing metrics computed during validation.
        """
        custom = {"rect": not load_vp}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
        self.metrics = validator.metrics
        return validator.metrics

    def predict(
        self,
        source=None,
        stream: bool = False,
        visual_prompts: dict = {},
        refer_image=None,
        predictor=None,
        **kwargs,
    ):
        """
        Run prediction on images, videos, directories, streams, etc.

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): Source for prediction. Accepts image paths,
                directory paths, URL/YouTube streams, PIL images, numpy arrays, or webcam indices.
            stream (bool): Whether to stream the prediction results. If True, results are yielded as a
                generator as they are computed.
            visual_prompts (dict): Dictionary containing visual prompts for the model. Must include 'bboxes' and
                'cls' keys when non-empty.
            refer_image (str | PIL.Image | np.ndarray, optional): Reference image for visual prompts.
            predictor (callable, optional): Custom predictor function. If None, a predictor is automatically
                loaded based on the task.
            **kwargs (Any): Additional keyword arguments passed to the predictor.

        Returns:
            (List | generator): List of Results objects or generator of Results objects if stream=True.

        Examples:
            >>> model = YOLOE("yoloe-v8s-seg.pt")
            >>> results = model.predict("path/to/image.jpg")
            >>> # With visual prompts
            >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
            >>> results = model.predict("path/to/image.jpg", visual_prompts=prompts)
        """
        if len(visual_prompts):
            assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
                f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
            )
            assert len(visual_prompts["bboxes"]) == len(visual_prompts["cls"]), (
                f"Expected equal number of bounding boxes and classes, but got {len(visual_prompts['bboxes'])} and "
                f"{len(visual_prompts['cls'])} respectively"
            )
        self.predictor = (predictor or self._smart_load("predictor"))(
            overrides={
                "task": self.model.task,
                "mode": "predict",
                "save": False,
                "verbose": refer_image is None,
                "batch": 1,
            },
            _callbacks=self.callbacks,
        )

        if len(visual_prompts):
            num_cls = (
                max(len(set(c)) for c in visual_prompts["cls"])
                if isinstance(source, list)  # means multiple images
                else len(set(visual_prompts["cls"]))
            )
            self.model.model[-1].nc = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())

        self.predictor.setup_model(model=self.model)

        if refer_image is None and source is not None:
            from ultralytics.data.build import load_inference_source
            dataset = load_inference_source(source)
            if dataset.mode in {"video", "stream"}:
                # NOTE: set the first frame as refer image for videos/streams inference
                refer_image = next(iter(dataset))[1][0]
        if refer_image is not None and len(visual_prompts):
            from ultralytics.models import yolo
            vpe = self.predictor.get_vpe(refer_image)
            self.model.set_classes(self.model.names, vpe)
            # self.task = "segment" if isinstance(self.predictor, SegmentationPredictor) else "detect"
            self.predictor = None  # reset predictor

        return super().predict(source, stream, **kwargs)
