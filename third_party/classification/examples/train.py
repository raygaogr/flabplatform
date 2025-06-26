import argparse
import os
import os.path as osp
import time
from fdcv import ClassificationData, Classifier, ClassificationEvaluator, ONNXRuntimeInference
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import yaml
import logging
logging.basicConfig(level=logging.INFO, filename='./classification_train.log',
                    format='%(asctime)s[%(levelname)s]: %(message)s')
logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate a classification model')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--eval', action='store_true', help='Run evaluation after training')
    parser.add_argument('--export_onnx', action='store_true', help='Export model to ONNX format')
    parser.add_argument('--onnx_eval', action='store_true', help='Run ONNX inference')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Create output directory
    os.makedirs(cfg['TASK_PATH'], exist_ok=True)
    
    # ---------------------- Dataset ----------------------
    data = ClassificationData(
        data_dir=cfg['DATA']['DATA_DIR'],
        batch_size=cfg['DATA']['BATCH_SIZE'],
        num_workers=cfg['DATA']['NUM_WORKERS'],
        split_ratio=cfg['DATA']['SPLIT_RATIO'],
        split_flag=cfg['DATA']['SPLIT_FLAG'],
        input_size=tuple(cfg['DATA'].get('INPUT_SIZE', (224, 224))),
        augmentation=cfg['DATA'].get('AUGMENTATION', True),
        mixup=cfg['DATA'].get('MIXUP', False),
        mixup_alpha=cfg['DATA'].get('MIXUP_ALPHA', 1.0),
        label_smoothing=cfg['DATA'].get('LABEL_SMOOTHING', 0.0),
    )
    data.setup()
    logger.info(f"Number of classes: {data.num_classes}")
    logger.info(f"Train sample shape: {data.train_dataloader().dataset[0][0].shape}, "
          f"Train batches: {len(data.train_dataloader())}, "
          f"Test batches: {len(data.test_dataloader())}")
    
    # ---------------------- Model ----------------------
    model = Classifier(
        num_classes=data.num_classes,
        backbone=cfg['MODEL']['BACKBONE'],
        img_c=cfg['MODEL']['IMG_CHANNELS'],
        lr=cfg['MODEL']['LR'],
        pretrained=cfg['MODEL']['PRETRAINED'],
        multi_label=cfg['MODEL']['MULTI_LABEL'],
        fcn_head=cfg['MODEL']['FCN_HEAD'],
        dropout=cfg['MODEL']['DROPOUT'],
    )

    if cfg['MODEL'].get('FREEZE_BACKBONE', False):
        for param in model.backbone.parameters():
            param.requires_grad = False
        model.backbone.eval()
        logger.info("Freezing backbone layers.")
    else:
        model.backbone.train()
        logger.info("Training backbone layers.")

    # ---------------------- Trainer ----------------------
    monitor_metric = cfg['TRAIN']['MONITOR']
    monitor_mode = cfg['TRAIN']['MODE']
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=cfg['TASK_PATH'],
        filename=f"model-{{epoch:02d}}-{{{monitor_metric}:.2f}}",
        mode=monitor_mode,
        save_last=True,
        save_top_k=1,
        verbose=False,
    )
    
    if cfg['TRAIN']['EARLY_STOP']:
        early_stop = EarlyStopping(monitor=monitor_metric, patience=cfg['TRAIN']['PATIENCE'], mode=monitor_mode)
        callbacks = [checkpoint_callback, early_stop]
    else:
        callbacks = [checkpoint_callback]
    
    trainer = L.Trainer(
        default_root_dir=cfg['TASK_PATH'],
        min_epochs=cfg['TRAIN']['MIN_EPOCHS'],
        max_epochs=cfg['TRAIN']['MAX_EPOCHS'],
        devices=cfg['TRAIN']['DEVICES'],
        accelerator=cfg['TRAIN']['ACCELERATOR'],
        check_val_every_n_epoch=cfg['TRAIN']['CHECK_VAL_EVERY_N_EPOCH'],
        callbacks=callbacks,
    )
    
    trainer.fit(model, data)
    
    best_model_path = checkpoint_callback.best_model_path
    logger.info(f"Best model saved at: {best_model_path}")
    trainer.save_checkpoint(osp.join(cfg['TASK_PATH'], 'base_model.pth'))
    
    # ---------------------- Evaluation ----------------------
    eval_flag = args.eval if args.eval else cfg.get('EVAL', False)
    if eval_flag:
        model = Classifier.load_from_checkpoint(osp.join(cfg['TASK_PATH'], 'base_model.pth'))
        model.eval()
        model.freeze()
        model.cuda()
        
        evaluator = ClassificationEvaluator(
            num_classes=data.num_classes,
            multi_label=cfg['MODEL']['MULTI_LABEL'],
            labels=data.val_dataset.classes,
            save_path=cfg['TASK_PATH'],
        )
        
        start_time = time.time()
        metrics = evaluator.evaluate(model, data.test_dataloader())
        end_time = time.time()
        logger.info(f"Inference time: {end_time - start_time}")
    
    # ---------------------- Export ONNX ----------------------
    export_onnx_flag = args.export_onnx if args.export_onnx else cfg.get('EXPORT_ONNX', False)
    if export_onnx_flag:
        input_size = tuple(cfg['ONNX'].get('ONNX_INPUT_SIZE', (1, 3, 224, 224)))
        model.export_onnx(onnx_path=osp.join(cfg['TASK_PATH'], 'base_model.onnx'), input_shape=input_size)
    
    # ---------------------- ONNX Eval ----------------------
    onnx_eval_flag = args.onnx_eval if args.onnx_eval else cfg.get('ONNX_EVAL', False)
    if onnx_eval_flag:
        input_size = tuple(cfg['ONNX'].get('ONNX_INPUT_SIZE', [224, 224])[-2:])
        eval_path =  cfg['ONNX'].get('ONNX_EVAL_PATH', cfg['DATA']['DATA_DIR'] + '/val/')
        eval_batch_size = cfg['ONNX'].get('EVAL_BATCH_SIZE', 32)

        inference = ONNXRuntimeInference(
            osp.join(cfg['TASK_PATH'], 'base_model.onnx'),
            task='classification',
            input_size=input_size,
            num_classes=data.num_classes,
            providers=cfg['ONNX'].get('PROVIDERS', ['CUDAExecutionProvider']),
        )
        metrics = inference.evaluate(eval_path, eval_batch_size)

if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=5 python train.py classification_config.yaml
