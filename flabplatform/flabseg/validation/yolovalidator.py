from multiprocessing.pool import ThreadPool
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
import copy
from flabplatform.flabdet.validation import DetectionValidator
from flabplatform.flabdet.utils.yolos import LOGGER, NUM_THREADS
from flabplatform.flabdet.utils.yolos.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
from ultralytics.utils.plotting import output_to_target, plot_images
from flabplatform.flabdet.registry import VALIDATORS
from ultralytics.utils.torch_utils import get_flops
from flabplatform.flabdet.utils import LOGGER
from ultralytics.utils import ops

@VALIDATORS.register_module()
class SegmentationValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a segmentation model.

    This validator handles the evaluation of segmentation models, processing both bounding box and mask predictions
    to compute metrics such as mAP for both detection and segmentation tasks.

    Attributes:
        plot_masks (list): List to store masks for plotting.
        process (callable): Function to process masks based on save_json and save_txt flags.
        args (namespace): Arguments for the validator.
        metrics (SegmentMetrics): Metrics calculator for segmentation tasks.
        stats (dict): Dictionary to store statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationValidator
        >>> args = dict(model="yolo11n-seg.pt", data="coco8-seg.yaml")
        >>> validator = SegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (Any, optional): Progress bar for displaying progress.
            args (namespace, optional): Arguments for the validator.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics(save_dir=self.save_dir)
        self.fitness = -1.0 

    def preprocess(self, batch):
        """Preprocess batch by converting masks to float and sending to device."""
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        return batch

    def init_metrics(self, model):
        """
        Initialize metrics and select mask processing function based on save_json flag.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        self.plot_masks = []
        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")
        # more accurate vs faster
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask
        self.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[], m_iou=[])

    def get_desc(self):
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 11) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95",
            "mIoU)"
        )

    def postprocess(self, preds):
        """
        Post-process YOLO predictions and return output detections with proto.

        Args:
            preds (list): Raw predictions from the model.

        Returns:
            p (torch.Tensor): Processed detection predictions.
            proto (torch.Tensor): Prototype masks for segmentation.
        """
        p = super().postprocess(preds[0])
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        return p, proto

    def _prepare_batch(self, si, batch):
        """
        Prepare a batch for training or inference by processing images and targets.

        Args:
            si (int): Batch index.
            batch (dict): Batch data containing images and targets.

        Returns:
            (dict): Prepared batch with processed images and targets.
        """
        prepared_batch = super()._prepare_batch(si, batch)
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        prepared_batch["masks"] = batch["masks"][midx]
        return prepared_batch

    def _prepare_pred(self, pred, pbatch, proto):
        """
        Prepare predictions for evaluation by processing bounding boxes and masks.

        Args:
            pred (torch.Tensor): Raw predictions from the model.
            pbatch (dict): Prepared batch data.
            proto (torch.Tensor): Prototype masks for segmentation.

        Returns:
            predn (torch.Tensor): Processed bounding box predictions.
            pred_masks (torch.Tensor): Processed mask predictions.
        """
        predn = super()._prepare_pred(pred, pbatch)
        pred_masks = self.process(proto, pred[:, 6:], pred[:, :4], shape=pbatch["imgsz"])
        return predn, pred_masks

    def update_metrics(self, preds, batch):
        """
        Update metrics with the current batch predictions and targets.

        Args:
            preds (list): Predictions from the model.
            batch (dict): Batch data containing images and targets.
        """
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_m=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Masks
            gt_masks = pbatch.pop("masks")
            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn, pred_masks = self._prepare_pred(pred, pbatch, proto)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                stat["tp_m"],stat['m_iou'] = self._process_batch(
                    predn, bbox, cls, pred_masks, gt_masks, self.args.overlap_mask, masks=True
                ) # compute mask IoU for segmentation task 
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if self.args.plots and self.batch_i < 3:
                self.plot_masks.append(pred_masks[:50].cpu())  # Limit plotted items for speed
                if pred_masks.shape[0] > 50:
                    LOGGER.warning("WARNING ⚠️ Limiting validation plots to first 50 items per image for speed...")

            # Save
            if self.args.save_json:
                self.pred_to_json(
                    predn,
                    batch["im_file"][si],
                    ops.scale_image(
                        pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                        pbatch["ori_shape"],
                        ratio_pad=batch["ratio_pad"][si],
                    ),
                )
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    pred_masks,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )


    def finalize_metrics(self, *args, **kwargs):
        """
        Finalize evaluation metrics by setting the speed attribute in the metrics object.

        This method is called at the end of validation to set the processing speed for the metrics calculations.
        It transfers the validator's speed measurement to the metrics object for reporting.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.
        """
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_masks=None, gt_masks=None, overlap=False, masks=False):
        """
        Compute correct prediction matrix for a batch based on bounding boxes and optional masks.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detected bounding boxes and
                associated confidence scores and class indices. Each row is of the format [x1, y1, x2, y2, conf, class].
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground truth bounding box coordinates.
                Each row is of the format [x1, y1, x2, y2].
            gt_cls (torch.Tensor): Tensor of shape (M,) representing ground truth class indices.
            pred_masks (torch.Tensor, optional): Tensor representing predicted masks, if available. The shape should
                match the ground truth masks.
            gt_masks (torch.Tensor, optional): Tensor of shape (M, H, W) representing ground truth masks, if available.
            overlap (bool): Flag indicating if overlapping masks should be considered.
            masks (bool): Flag indicating if the batch contains mask data.

        Returns:
            (torch.Tensor): A correct prediction matrix of shape (N, 10), where 10 represents different IoU levels.

        Note:
            - If `masks` is True, the function computes IoU between predicted and ground truth masks.
            - If `overlap` is True and `masks` is True, overlapping masks are taken into account when computing IoU.

        Examples:
            >>> detections = torch.tensor([[25, 30, 200, 300, 0.8, 1], [50, 60, 180, 290, 0.75, 0]])
            >>> gt_bboxes = torch.tensor([[24, 29, 199, 299], [55, 65, 185, 295]])
            >>> gt_cls = torch.tensor([1, 0])
            >>> correct_preds = validator._process_batch(detections, gt_bboxes, gt_cls)
        """
        if masks:
            if overlap:
                nl = len(gt_cls)
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
                gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
            tp_m = self.match_predictions(detections[:, 5], gt_cls, iou)
            m_iou = self.compute_mask_iou(detections[:, 5], gt_cls, iou)
        else:  # boxes
            iou = box_iou(gt_bboxes, detections[:, :4])
            tp = self.match_predictions(detections[:, 5], gt_cls, iou)
        return (tp_m,m_iou) if masks else tp # tp for box, iou for mask 

    def plot_val_samples(self, batch, ni):
        """
        Plot validation samples with bounding box labels and masks.

        Args:
            batch (dict): Batch data containing images and targets.
            ni (int): Batch index.
        """
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """
        Plot batch predictions with masks and bounding boxes.

        Args:
            batch (dict): Batch data containing images.
            preds (list): Predictions from the model.
            ni (int): Batch index.
        """
        plot_images(
            batch["img"],
            *output_to_target(preds[0], max_det=50),  # not set to self.args.max_det due to slow plotting speed
            torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred
        self.plot_masks.clear()

    def save_one_txt(self, predn, pred_masks, save_conf, shape, file):
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (torch.Tensor): Predictions in the format [x1, y1, x2, y2, conf, cls].
            pred_masks (torch.Tensor): Predicted masks.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple): Original image shape.
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
            masks=pred_masks,
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename, pred_masks):
        """
        Save one JSON result for COCO evaluation.

        Args:
            predn (torch.Tensor): Predictions in the format [x1, y1, x2, y2, conf, cls].
            filename (str): Image filename.
            pred_masks (numpy.ndarray): Predicted masks.

        Examples:
             >>> result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        """
        from pycocotools.mask import encode  # noqa

        def single_encode(x):
            """Encode predicted masks as RLE and append results to jdict."""
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        pred_masks = np.transpose(pred_masks, (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                    "segmentation": rles[i],
                }
            )
    
    def preds_to_labelme(self, preds, batch):
        """
        Convert predictions to LabelMe format.
        Args:
            preds (List[torch.Tensor]): List of predictions from the model.
            batch (dict): Batch data containing images and annotations.
        Returns:
            None
        """
    
        save_path = Path(self.save_dir / "label")
        save_path.mkdir(parents=True, exist_ok=True)
        im_file = batch["im_file"] # a batch of image files with absolute path
        ori_shape = batch["ori_shape"] # original shape of the images
        save_conf = self.args.conf + 0.25 # whether to save confidence scores
        img_shape = batch["img"].shape[2:] # shape of the images
        preds, protos = preds[0],preds[1]

        for i in range(len(im_file)):
            self.pred_to_labelme_single(preds[i],im_file[i],protos[i],
                                        save_conf,ori_shape[i],img_shape,save_path)
        

    def pred_to_labelme_single(self, pred,
                               img_path, 
                               proto, 
                               save_conf,
                               ori_shape,
                               img_shape,
                               save_path):
        """
        Convert a single prediction to LabelMe format and save it.

        Args:
            pred (torch.Tensor): Single prediction tensor.
            ori_image (torch.Tensor): Original image tensor.
            img_path (str): Path to the image file.
            proto (torch.Tensor): Prototype masks for segmentation.
            save_conf (bool): threshold to save confidence scores.
            img_shape (tuple): Shape of the image after preprocess.
        """
        standard_json = {
                "flags": {},
                "version": "5.0.1",
                "imageData": None,
                "imagePath": Path(img_path).name,
                "imageHeight": ori_shape[0],
                "imageWidth": ori_shape[1],
            }
        shapes = []
        if pred.shape[0]:
            indices = torch.where(pred[:, 4] > save_conf)[0]
            pred = pred[indices]
            if pred.shape[0]:
                masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img_shape, upsample=True)  # HWC
                # pred[:, :4] = ops.scale_boxes(img_shape, pred[:, :4], ori_shape[::-1])
                keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
                masks = masks[keep]
                from ultralytics.engine.results import Masks
                xy = Masks(masks, ori_shape)
                pixel_coords =xy.xy
                for i in range(masks.shape[0]):
                    cls_idx = int(pred[i][5].item())
                    temp_unit = {'flags': [], 'group_id': None, 'shape_type': 'polygon'}
                    temp_unit['points'] = pixel_coords[i].tolist()
                    temp_unit["label"] = self.data['names'][cls_idx]
                    shapes.append(temp_unit)
        standard_json["shapes"] = shapes
        with open(save_path / f"{Path(img_path).stem}.json", 'w', encoding='utf-8') as f:
            json.dump(standard_json, f, indent=4)


    def eval_json(self, stats):
        """Return COCO-style object detection evaluation metrics."""
        if self.args.save_json and (self.is_lvis or self.is_coco) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # predictions

            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # annotations

            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # init annotations api
                    pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                    vals = [COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm")]
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))
                    pred = anno._load_json(str(pred_json))
                    vals = [LVISEval(anno, pred, "bbox"), LVISEval(anno, pred, "segm")]

                for i, eval in enumerate(vals):
                    eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # im to eval
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    if self.is_lvis:
                        eval.print_results()
                    idx = i * 4 + 2
                    # update mAP50-95 and mAP50
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = (
                        eval.stats[:2] if self.is_coco else [eval.results["AP"], eval.results["AP50"]]
                    )
                    if self.is_lvis:
                        tag = "B" if i == 0 else "M"
                        stats[f"metrics/APr({tag})"] = eval.results["APr"]
                        stats[f"metrics/APc({tag})"] = eval.results["APc"]
                        stats[f"metrics/APf({tag})"] = eval.results["APf"]

                if self.is_lvis:
                    stats["fitness"] = stats["metrics/mAP50-95(B)"]

            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats

    def compute_mask_iou(self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor
        ) -> torch.Tensor:
        """
        compute the maximum IoU for each predicted class against the true classes.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape (N,).
            true_classes (torch.Tensor): Target class indices of shape (M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground truth.
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape (N, 10) for 10 IoU thresholds.
        """

        class_mask = true_classes.unsqueeze(1) == pred_classes.unsqueeze(0) 
        masked_iou = iou.clone()
        masked_iou[~class_mask] = -torch.inf
        max_ious, _ = torch.max(masked_iou, dim=1)
        max_ious = torch.nan_to_num(max_ious, nan=0.0, neginf=0.0)
        return max_ious
    

    def save_val_json(self, stats, model):
        """
        Save validation metrics to a JSON file.
        Args:
            stats (dict): Dictionary containing validation statistics.
            model: current model 
        """
        # get GPU name if available
        gpu_name = "cpu"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)

        # only compute FLOPs if not already done
        if not hasattr(self, 'flops') or self.flops is None:
            self.flops = get_flops(copy.deepcopy(model).float().to(self.device), imgsz=640) 

        fps = 1000 / (self.speed['preprocess']  + self.speed['inference'] +self.speed['postprocess'])

        val_metrics = {
            "operation":self.args.task,
            "performance": {
                "device": gpu_name,
                "fps": round(fps),
                "flops": f"{self.flops:.2f} GFLOPs",
            },
            f"{self.args.task}":{},
        }

         # update val_metrics during training with best fitness (best model)
        if self.training:
            cur_fitness = stats.get("fitness", 0.0)
            if cur_fitness > self.fitness:
                self.fitness = cur_fitness
                val_metrics[self.args.task]['ap'] = round(stats.get('metrics/mAP50(M)', 0.0),2)
                val_metrics[self.args.task]['mIoU'] = round(stats.get('metrics/mIoU(M)', 0.0), 2)
        else:
            val_metrics[self.args.task]['ap'] = round(stats.get('metrics/mAP50(M)', 0.0),2)
            val_metrics[self.args.task]['mIoU'] = round(stats.get('metrics/mIoU(M)', 0.0), 2)

        with open(Path(self.save_dir / "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=4)
            # LOGGER.info(f"Validation metrics saved to {self.save_dir / 'metrics.json'}")

