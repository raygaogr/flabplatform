import os
import os.path as osp
import time
import yaml
import logging
from fdcv import ClassificationData, Classifier, ClassificationEvaluator, ONNXRuntimeInference, LabelmeClassificationData
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

logger = logging.getLogger()


class ClassificationRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.task_path = cfg['commonParams']['outputDir']
        os.makedirs(self.task_path, exist_ok=True)
        self.model_path = cfg['commonParams']['modelDir']
        os.makedirs(self.model_path, exist_ok=True)

        # Dataset
        # self.data = ClassificationData(
        #     data_dir=cfg['commonParams']['datasets']['rootDir'],
        #     batch_size=cfg['training']['algoParams']['batch'],
        #     num_workers=cfg['training']['algoParams']['workers'],
        #     split_ratio=cfg['training']['algoParams']['split_ratio'],
        #     split_flag=cfg['training']['algoParams']['split_flag'],
        #     input_size=tuple(cfg['training']['algoParams'].get('input_size', (224, 224))),
        #     augmentation=cfg['training']['algoParams'].get('augmentation', True),
        #     mixup=cfg['training']['algoParams'].get('mixup', False),
        #     mixup_alpha=cfg['training']['algoParams'].get('mixup_alpha', 1.0),
        #     label_smoothing=cfg['training']['algoParams'].get('label_smoothing', 0.0),
        # )

        self.data = LabelmeClassificationData(
            metadata_path=cfg['commonParams']['datasets']['metafile'],
            root_dir=cfg['commonParams']['datasets']['rootDir'],
            batch_size=cfg['training']['algoParams']['batch'],
            num_workers=cfg['training']['algoParams']['workers'],
            input_size=tuple(cfg['training']['algoParams'].get(
                'input_size', (224, 224))),
            augmentation=cfg['training']['algoParams'].get(
                'augmentation', True),
            mixup=cfg['training']['algoParams'].get('mixup', False),
            mixup_alpha=cfg['training']['algoParams'].get('mixup_alpha', 1.0),
            label_smoothing=cfg['training']['algoParams'].get(
                'label_smoothing', 0.0),
        )
        self.data.setup()

        self.model = Classifier(
            num_classes=self.data.num_classes,
            backbone=cfg['training']['algoParams']['model']['backbone'],
            img_c=cfg['training']['algoParams']['model']['img_channels'],
            lr=cfg['training']['algoParams']['model']['lr'],
            pretrained=cfg['training']['algoParams']['model']['pretrained'],
            multi_label=cfg['training']['algoParams']['model']['multi_label'],
            fcn_head=cfg['training']['algoParams']['model']['fcn_head'],
            dropout=cfg['training']['algoParams']['model']['dropout'],
        )

        if cfg['training']['algoParams']['model'].get('freeze_backbone', False):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            self.model.backbone.eval()
        else:
            self.model.backbone.train()

    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg)

    def train(self):
        cfg = self.cfg
        monitor_metric = cfg['training']['algoParams']['monitor']
        monitor_mode = cfg['training']['algoParams']['monitor_mode']
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor_metric,
            dirpath=self.task_path,
            filename=f"model-{{epoch:02d}}-{{{monitor_metric}:.2f}}",
            mode=monitor_mode,
            save_last=True,
            save_top_k=1,
            verbose=False,
        )
        callbacks = [checkpoint_callback]
        if cfg['training']['algoParams']['early_stop']:
            callbacks.append(EarlyStopping(
                monitor=monitor_metric,
                patience=cfg['training']['algoParams']['patience'],
                mode=monitor_mode,
            ))
        min_epochs = cfg['training']['algoParams'].get('epochs', 24)
        max_epochs = cfg['training']['algoParams'].get('epochs', 100)
        trainer = L.Trainer(
            default_root_dir=self.task_path,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            devices=[cfg['training']['algoParams']['device']],
            accelerator=cfg['training']['algoParams']['accelerator'],
            check_val_every_n_epoch=cfg['training']['algoParams']['check_val_every_n_epoch'],
            callbacks=callbacks,
        )
        trainer.fit(self.model, self.data)
        trainer.save_checkpoint(osp.join(self.task_path, 'base_model.pth'))
        trainer.save_checkpoint(osp.join(self.model_path, 'base_model.pth'))

    def val(self, *args, **kwargs):
        model = Classifier.load_from_checkpoint(
            osp.join(self.model_path, 'base_model.pth'))
        model.eval()
        model.freeze()
        model.cuda()

        evaluator = ClassificationEvaluator(
            num_classes=self.data.num_classes,
            multi_label=self.cfg['training']['algoParams']['model']['multi_label'],
            labels=self.data.val_dataset.classes,
            save_path=self.task_path,
        )

        start_time = time.time()
        metrics = evaluator.evaluate(model, self.data.test_dataloader())
        end_time = time.time()
        print(
            f"[ClassificationRunner] Inference time: {end_time - start_time:.2f}s")
        return metrics

    def export(self, format="onnx"):
        input_size = self.cfg.get("training", {}).get(
            "algoParams", {}).get("input_size", [224, 224])
        input_size = list(input_size) if isinstance(
            input_size, (tuple, list)) else [224, 224]
        if len(input_size) == 2:
            input_size = [1, 3] + input_size
        if len(input_size) != 4:
            raise ValueError(f"Invalid input size: {input_size}, must be 4D")
        self.model.export_onnx(
            onnx_path=osp.join(self.model_path, 'base_model.onnx'),
            input_shape=tuple(input_size)
        )
