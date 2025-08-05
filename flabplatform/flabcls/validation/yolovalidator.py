from pathlib import Path
import torch
import json
import copy
import numpy as np
from ultralytics.data import build_dataloader
from ultralytics.utils.torch_utils import get_flops
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images
from flabplatform.flabdet.datasets.yolos import ClassificationDataset
from flabplatform.flabdet.utils.yolos import LOGGER
from flabplatform.flabdet.validation import BaseValidator
from flabplatform.flabdet.registry import VALIDATORS

@VALIDATORS.register_module()
class ClassificationValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a classification model.

    This validator handles the validation process for classification models, including metrics calculation,
    confusion matrix generation, and visualization of results.

    Attributes:
        targets (List[torch.Tensor]): Ground truth class labels.
        pred (List[torch.Tensor]): Model predictions.
        metrics (ClassifyMetrics): Object to calculate and store classification metrics.
        names (dict): Mapping of class indices to class names.
        nc (int): Number of classes.
        confusion_matrix (ConfusionMatrix): Matrix to evaluate model performance across classes.

    Methods:
        get_desc: Return a formatted string summarizing classification metrics.
        init_metrics: Initialize confusion matrix, class names, and tracking containers.
        preprocess: Preprocess input batch by moving data to device.
        update_metrics: Update running metrics with model predictions and batch targets.
        finalize_metrics: Finalize metrics including confusion matrix and processing speed.
        postprocess: Extract the primary prediction from model output.
        get_stats: Calculate and return a dictionary of metrics.
        build_dataset: Create a ClassificationDataset instance for validation.
        get_dataloader: Build and return a data loader for classification validation.
        print_results: Print evaluation metrics for the classification model.
        plot_val_samples: Plot validation image samples with their ground truth labels.
        plot_predictions: Plot images with their predicted class labels.

    Examples:
        >>> from ultralytics.models.yolo.classify import ClassificationValidator
        >>> args = dict(model="yolo11n-cls.pt", data="imagenet10")
        >>> validator = ClassificationValidator(args=args)
        >>> validator()

    Notes:
        Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize ClassificationValidator with dataloader, save directory, and other parameters.

        This validator handles the validation process for classification models, including metrics calculation,
        confusion matrix generation, and visualization of results.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (str | Path, optional): Directory to save results.
            pbar (bool, optional): Display a progress bar.
            args (dict, optional): Arguments containing model and validation configuration.
            _callbacks (list, optional): List of callback functions to be called during validation.

        Examples:
            >>> from ultralytics.models.yolo.classify import ClassificationValidator
            >>> args = dict(model="yolo11n-cls.pt", data="imagenet10")
            >>> validator = ClassificationValidator(args=args)
            >>> validator()
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task = "classify"
        self.metrics = ClassifyMetrics()
        self.fitness = -1
    def get_desc(self):
        """Return a formatted string summarizing classification metrics."""
        return ("%22s" + "%11s" * 3) % ("classes", "top1_acc", "top5_acc","f1_score")

    def init_metrics(self, model):
        """Initialize confusion matrix, class names, and tracking containers for predictions and targets."""
        self.names = model.names
        self.nc = len(model.names)
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf, task="classify")
        self.pred = []
        self.targets = []
        if not self.training:
            self.im_files = []



    def preprocess(self, batch):
        """Preprocess input batch by moving data to device and converting to appropriate dtype."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        """
        Update running metrics with model predictions and batch targets.

        Args:
            preds (torch.Tensor): Model predictions, typically logits or probabilities for each class.
            batch (dict): Batch data containing images and class labels.

        This method appends the top-N predictions (sorted by confidence in descending order) to the
        prediction list for later evaluation. N is limited to the minimum of 5 and the number of classes.
        """
        n5 = min(len(self.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu())
        self.targets.append(batch["cls"].type(torch.int32).cpu())
        if not self.training:
            self.im_files.extend([Path(i).name for i in batch["im_file"]])


    def finalize_metrics(self, *args, **kwargs):
        """
        Finalize metrics including confusion matrix and processing speed.

        This method processes the accumulated predictions and targets to generate the confusion matrix,
        optionally plots it, and updates the metrics object with speed information.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Examples:
            >>> validator = ClassificationValidator()
            >>> validator.pred = [torch.tensor([[0, 1, 2]])]  # Top-3 predictions for one sample
            >>> validator.targets = [torch.tensor([0])]  # Ground truth class
            >>> validator.finalize_metrics()
            >>> print(validator.metrics.confusion_matrix)  # Access the confusion matrix
        """
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir
        if not self.training:
            self.im_files = np.array(self.im_files)
            preds, targets = torch.cat(self.pred)[:, 0].numpy(), torch.cat(self.targets).numpy()
            confusion_matrix_json = {}
            confusion_matrix_json["labels"] = list(self.names.values())
            for i in range(len(targets)):
                gt_cls = self.names[targets[i]]
                pred_cls = self.names[preds[i]]
                img_name = self.im_files[i]
                if gt_cls not in confusion_matrix_json:
                    confusion_matrix_json[gt_cls] = {}
                if pred_cls not in confusion_matrix_json[gt_cls]:
                    confusion_matrix_json[gt_cls][pred_cls] = {"num": 0, "imagePath": []}
                confusion_matrix_json[gt_cls][pred_cls]["num"] += 1
                confusion_matrix_json[gt_cls][pred_cls]["imagePath"].append(img_name)   
 

            with open(Path(self.save_dir / 'confusion_matrix.json'),'w',encoding="utf-8") as f:
                json.dump(confusion_matrix_json,f,indent=4)


    def postprocess(self, preds):
        """Extract the primary prediction from model output if it's in a list or tuple format."""
        return preds[0] if isinstance(preds, (list, tuple)) else preds

    def get_stats(self):
        """Calculate and return a dictionary of metrics by processing targets and predictions."""
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def build_dataset(self, img_path):
        """Create a ClassificationDataset instance for validation."""
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)

    def get_dataloader(self, dataset_path, batch_size):
        """
        Build and return a data loader for classification validation.

        Args:
            dataset_path (str | Path): Path to the dataset directory.
            batch_size (int): Number of samples per batch.

        Returns:
            (torch.utils.data.DataLoader): DataLoader object for the classification validation dataset.
        """
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self):
        """Print evaluation metrics for the classification model."""
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5, self.metrics.f1_score))

    def plot_val_samples(self, batch, ni):
        """
        Plot validation image samples with their ground truth labels.

        Args:
            batch (dict): Dictionary containing batch data with 'img' (images) and 'cls' (class labels).
            ni (int): Batch index used for naming the output file.

        Examples:
            >>> validator = ClassificationValidator()
            >>> batch = {"img": torch.rand(16, 3, 224, 224), "cls": torch.randint(0, 10, (16,))}
            >>> validator.plot_val_samples(batch, 0)
        """
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=batch["cls"].view(-1),  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """
        Plot images with their predicted class labels and save the visualization.

        Args:
            batch (dict): Batch data containing images and other information.
            preds (torch.Tensor): Model predictions with shape (batch_size, num_classes).
            ni (int): Batch index used for naming the output file.

        Examples:
            >>> validator = ClassificationValidator()
            >>> batch = {"img": torch.rand(16, 3, 224, 224)}
            >>> preds = torch.rand(16, 10)  # 16 images, 10 classes
            >>> validator.plot_predictions(batch, preds, 0)
        """
        plot_images(
            batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=torch.argmax(preds, dim=1),
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_to_json(self, stats, model):
        """
            Save validation metrics to a JSON file.
            Args:
                stats (dict): Dictionary containing validation statistics.
        """
        # get GPU name if available
        gpu_name = "cpu"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            
        # only compute FLOPs if not already done
        if not hasattr(self, 'flops') or self.flops is None:
            self.flops = get_flops(copy.deepcopy(model).float().to(self.device), imgsz=640) # calculate FLOPs if not already done

        fps = 1000 / (self.dataloader.batch_size * (self.speed['preprocess']  + self.speed['inference'] +self.speed['postprocess']))
        val_metrics = {
            "operation":self.args.task,
            "performance": {
                "device": gpu_name,
                "fps": round(fps),
                "flops": f"{self.flops:.2f} GFLOPs",
            },
            f"{self.args.task}":{}
            }
        
        if self.training:
            cur_fitness = stats.get("fitness", 0.0)
            if cur_fitness >= self.fitness:
                self.fitness = cur_fitness
                val_metrics[self.args.task]['top1'] = round(stats.get('metrics/accuracy_top1',0.0), 2)
                val_metrics[self.args.task]['f1_score'] = round(stats.get('metrics/f1_score',0.0),2)
        else:
            val_metrics[f"{self.args.task}"]["top1"]= round(stats.get('metrics/accuracy_top1', 0.0),2)
            val_metrics[f"{self.args.task}"]["f1_score"] = round(stats.get('metrics/f1_score', 0.0), 2)


        with open(Path(self.save_dir / "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=4)


    def preds_to_labelme(self, preds, batch):
        """
        Convert predictions to LabelMe format.

        Args:
            preds (List[torch.Tensor]): List of predictions from the model.
            batch (dict): Batch data containing images and annotations.

        Returns:
            
        """
        batch_size = len(preds)
        save_path = Path(self.save_dir / "label")
        save_path.mkdir(parents=True, exist_ok=True)
        im_file = batch["im_file"] # a batch of image files with absolute path
        ori_shape = batch["ori_shape"] # original shape of the images
        for b in range(batch_size):
            self.preds_to_labelme_single(im_file[b], ori_shape[b],
                                         preds[b], save_path)
    
    def preds_to_labelme_single(self, 
                              im_file: str, 
                              im_ori_shape:list,
                              pred:torch.Tensor,
                              save_path:Path):
        """
        Convert a single prediction to LabelMe format and save it.
        args:
            im_file (str): Path to the image file.
            im_ori_shape [h,w] (list): Original shape of the image.
            pred [n,6] (torch.Tensor): Predictions for the image.
            save_path (Path): Path to save the LabelMe JSON file.
            save_conf (float): Confidence threshold for saving predictions.
        """
        
        standard_json = {
                "flags": {},
                "version": "5.5.0",
                "imageData": None,
                "imagePath": Path(im_file).name,
                "imageHeight": im_ori_shape[0],
                "imageWidth": im_ori_shape[1],
            }
        
        class_idx = torch.argmax(pred).item()
        shapes =[{
            "label":self.names[class_idx],
            "points": [],
            "group_id": None,
            "description": None,
            "shape_type": "classification",
            "score": round(pred[class_idx].item(),4)
        }]
        standard_json["shapes"] = shapes
        with open(save_path / f"{Path(im_file).stem}.json", 'w', encoding='utf-8') as f:
            json.dump(standard_json, f, indent=4)
