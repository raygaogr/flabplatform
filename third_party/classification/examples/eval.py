from fdcv import *
import argparse
import os
import os.path as osp
import time
import yaml
import logging
logging.basicConfig(level=logging.INFO, filename='./classification_eval.log',
                    format='%(asctime)s[%(levelname)s]: %(message)s')
logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate and export a classification model')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--eval', action='store_true', help='Run evaluation on the test dataset')
    parser.add_argument('--export_onnx', action='store_true', help='Export model to ONNX format')
    parser.add_argument('--onnx_eval', action='store_true', help='Run ONNX inference on the dataset')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_config(args.config)

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

    # ---------------------- Model ----------------------
    model = Classifier(
        num_classes=data.num_classes,
        backbone=cfg['MODEL']['BACKBONE'],
        img_c=cfg['MODEL']['IMG_CHANNELS'],
        pretrained=cfg['MODEL']['PRETRAINED'],
        multi_label=cfg['MODEL']['MULTI_LABEL'],
        fcn_head=cfg['MODEL']['FCN_HEAD'],
        dropout=cfg['MODEL']['DROPOUT'],
    )

    # ---------------------- Evaluation ----------------------
    if args.eval:
        # Load the pretrained model (if available)
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
        logger.info(f"Evaluation metrics: {metrics}")

    # ---------------------- Export ONNX ----------------------
    if args.export_onnx:
        input_size = tuple(cfg['ONNX'].get('ONNX_INPUT_SIZE', (1, 3, 224, 224)))
        model.export_onnx(onnx_path=osp.join(cfg['TASK_PATH'], 'base_model.onnx'), input_shape=input_size)

    # ---------------------- ONNX Eval ----------------------
    if args.onnx_eval:
        input_size = tuple(cfg['ONNX'].get('ONNX_INPUT_SIZE', [224, 224])[-2:])
        eval_path = cfg['ONNX'].get('ONNX_EVAL_PATH', cfg['DATA']['DATA_DIR'] + '/val/')
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

# CUDA_VISIBLE_DEVICES=5 python eval.py classification_config.yaml --eval --export_onnx --onnx_eval
