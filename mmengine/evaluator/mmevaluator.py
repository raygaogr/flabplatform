import torch 
import os,json,copy
import numpy as np
from flabplatform.flabdet.configs import get_cfg, get_save_dir
from typing import Callable, Dict, List, Optional, Sequence, Union
from mmengine.runner.amp import autocast
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from flabplatform.flabdet.utils.mmdet.structures import DetDataSample 
from pathlib import Path
from flabplatform.core.registry import EVALUATOR
from mmengine.device import get_device
@EVALUATOR.register_module()
class MMEvaluator:

    def __init__(self,):
        
        self.metrics = DetMetrics()
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.class_map = None
        self.device = get_device()
        self.save_json = True
        self.fitness = -1.0


    def init_metrics(self,datasets,runner=None):

        self.runner = runner
        self.save_dir = runner.work_dir
        classes = datasets.metainfo['classes']
        self.names = {val:classes[val] for key,val in datasets.cat2label.items()}
        self.nc = len(self.names)
        self.metrics.names = self.names
        self.metrics.plot = False
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=0.25)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.conf_matrix_json = {}
        self.fps = 0.0
        self.flops = 0.0
        
    def update_metrics(self, preds: DetDataSample,
                             batch: dict):
        """
        Update metrics with new predictions and ground truth.

        Args:
            preds (List[torch.Tensor]): List of predictions from the model.
            batch (dict): Batch data containing ground truth.

        """
        # change mm preds and batch to yolo format
        preds,batch = self._prepare_mm_to_yolo(preds, batch)

        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = batch[si]
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.runner.epoch == self.runner.max_epochs:
                        cur_confu_matrix = self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                        self.update_conf_matrix_json_single(cur_confu_matrix,pbatch["im_file"])
                continue

            # Predictions
            # if self.args.single_cls:
            #     pred[:, 5] = 0
            # predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = pred[:, 4]
            stat["pred_cls"] = pred[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(pred, bbox, cls)
            if self.runner.epoch == self.runner.max_epochs:
                cur_confu_matrix = self.confusion_matrix.process_batch(pred, bbox, cls)
                self.update_conf_matrix_json_single(cur_confu_matrix,pbatch["im_file"])
            for k in self.stats.keys():
                self.stats[k].append(stat[k])


    
    def _prepare_mm_to_yolo(self,preds:DetDataSample, 
                                 batch):
        
        
        # Convert predictions to YOLO format
        tmp_preds =[]
        tmp_batch = []
        for pred,batch_sample,img in zip(preds,batch['data_samples'],batch['inputs']):
    
            meta_info = {}
            meta_info['im_file'] = batch_sample.img_path
            meta_info['imgsz'] = batch_sample.img_shape
            meta_info['ori_shape'] = batch_sample.ori_shape
            meta_info['img'] = img
            meta_info['bbox'] = batch_sample.gt_instances.bboxes.tensor
            meta_info['cls'] = batch_sample.gt_instances.labels
            tmp_batch.append(meta_info)
            tmp_preds.append(torch.cat((pred.pred_instances.bboxes,
                                        pred.pred_instances.scores.unsqueeze(1),
                                        pred.pred_instances.labels.unsqueeze(1)),dim=1).cpu())
        
        return tmp_preds, tmp_batch
    


    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)
    

    def match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
    ) -> torch.Tensor:
        """
        Match predictions to ground truth objects using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape (N,).
            true_classes (torch.Tensor): Target class indices of shape (M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground truth.
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape (N, 10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
    

    def get_stats(self):
        """
        Calculate and return metrics statistics.

        Returns:
            (dict): Dictionary containing metrics results.
        """
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats):
            self.metrics.process(**stats, on_plot=False)
        return self.metrics.results_dict

    def print_results(self,logger):
        """Print training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        pf_head = self.get_desc()  # print header
        logger.info(pf_head)
        logger.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            logger.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")
    
    def get_desc(self):
        """Return a formatted string summarizing class metrics of current model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")
    


    def save_to_json(self, stats):
        """
        Save validation metrics to a JSON file.
        Args:
            stats (dict): Dictionary containing validation statistics.
        """
        # get GPU name if available
        gpu_name = "cpu"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            
        val_metrics = {
            "operation":'detect',
            "performance": {
                "device": gpu_name,
                "fps": round(self.fps),
                "flops": f'{self.flops}FLOPs',
            },
            f"{'detect'}":{}
            }
    
        # update val_metrics during training with best fitness (best model)
        if self.runner.training:
            cur_fitness = stats.get("fitness", 0.0)
            if cur_fitness >= self.fitness:
                self.fitness = cur_fitness
                val_metrics['detect']['mAP'] = round(stats.get('metrics/mAP50(B)',0.0), 2)
                val_metrics['detect']['precision'] = round(stats.get('metrics/precision(B)',0.0),2)
                val_metrics['detect']['recall'] = round(stats.get('metrics/recall(B)',0.0),2)
        else:
            val_metrics[f"{'detect'}"]["mAP"]= round(stats.get('metrics/mAP50-95(B)', 0.0),2)
            val_metrics[f"{'detect'}"]["precision"] = round(stats.get('metrics/precision(B)', 0.0), 2)
            val_metrics[f"{'detect'}"]["recall"] = round(stats.get('metrics/recall(B)', 0.0), 2)


        with open(Path(Path(self.save_dir) / "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=4)
    

    def update_conf_matrix_json_single(self,cur_con:np.array,im_file:str):
        '''
        Update confusion matrix json with current image 
        args:
            cur_con: numpy.array [num_classes+1,num_classes+1], current image's confusion matrix 
            im_file: str, image path
        output: 
            None  
        '''
        im_file = Path(im_file).name
        gt_tmp_names,p_tmp_names = copy.deepcopy(self.names),copy.deepcopy(self.names)
        gt_tmp_names[len(self.names)] = "unlabeled"
        p_tmp_names[len(self.names)] = "undetected"
        pre_idx, gt_idx = np.where(cur_con > 0)
        for i in range(len(gt_idx)):
            g = int(gt_idx[i])
            p = int(pre_idx[i])
            if gt_tmp_names[g] not in self.conf_matrix_json:
                self.conf_matrix_json[gt_tmp_names[g]] = {}
            if p_tmp_names[p] not in self.conf_matrix_json[gt_tmp_names[g]]:
                self.conf_matrix_json[gt_tmp_names[g]][p_tmp_names[p]] = {"num": 0, "imagePath": []}
            self.conf_matrix_json[gt_tmp_names[g]][p_tmp_names[p]]["num"] += int(cur_con[p, g])
            self.conf_matrix_json[gt_tmp_names[g]][p_tmp_names[p]]["imagePath"].append(im_file)


    def preds_to_labelme(self, preds, batch):
        """
        Convert predictions to LabelMe format.

        Args:
            preds (List[torch.Tensor]): List of predictions from the model.
            batch (dict): Batch data containing images and annotations.

        Returns:
            
        """
        save_dir = os.path.join(self.save_dir, 'label')
        os.makedirs(save_dir, exist_ok=True)
        for pred in preds:
            self.preds_to_labelme_single(pred,save_dir)
    
    def preds_to_labelme_single(self,pred,save_dir):

        im_file = Path(pred.img_path).name
        standard_json = {
                "flags": {},
                "version": "5.0.1",
                "imageData": None,
                "imagePath": im_file,
                "imageHeight": pred.ori_shape[0],
                "imageWidth": pred.ori_shape[1],
            }
        shapes =[]
        iou_threshold = 0.25
        pred_instances = pred.pred_instances[pred.pred_instances.scores > iou_threshold]
        class_names = self.runner.val_dataloader.dataset.metainfo['classes']
        for box,label,score in zip (pred_instances.bboxes,
                                    pred_instances.labels,
                                    pred_instances.scores):
            temp_unit = {'flags': [], 'group_id': None, 'shape_type': 'rectangle'}
            temp_unit['points'] = box.cpu().numpy().reshape((2, 2)).tolist()
            temp_unit["label"] = class_names[label.item()]
            temp_unit["score"] = round(score.item(),4)
            shapes.append(temp_unit)
        standard_json["shapes"] = shapes
        with open(os.path.join(save_dir,f"{Path(im_file).stem}.json"), 'w', encoding='utf-8') as f:
            json.dump(standard_json, f, indent=4)

    def finalize_metrics(self, *args, **kwargs):
        """
        Set final values for metrics speed and confusion matrix.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.
        """
       
        if self.runner.epoch == self.runner.max_epochs:
            self.conf_matrix_json["labels"]= list(self.names.values())
            with open(os.path.join(self.save_dir,"confusion_matrix.json"),'w', encoding="utf-8") as f:
                json.dump(self.conf_matrix_json,f,indent=4)
