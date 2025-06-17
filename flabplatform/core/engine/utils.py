from flabplatform.core.config import Config
import os.path as osp
import os
from flabplatform.core.logging import MMLogger
from confluent_kafka import Producer, Consumer, TopicPartition
import time
import json
import signal
from PIL import Image

class AiAnnotation:
    def __init__(self, cfg : dict):
        self.cfg = cfg
        self.commonParams = self.cfg["commonParams"]
        self.root_dir = self.commonParams["datasets"]["rootDir"]
        self.save_dir = self.commonParams["outputDir"]
        self.meta_json = self.load_config(self.commonParams["datasets"]["metafile"])
        self.topic = self.commonParams["mqTopic"]
        self.pipelineData = self.commonParams["pipelineData"]
        self.task = self.cfg['task']
        self.logger = MMLogger(name="flabplatform", logger_name="flabplatform", log_level="INFO")

        self.CONFIG = {
            "producer_conf": {
                "bootstrap.servers": "worker01-cdp05.byd.com:9092,worker02-cdp05.byd.com:9092,worker03-cdp05.byd.com:9092",
                "security.protocol": "SASL_PLAINTEXT",
                "sasl.mechanism": "GSSAPI",
                "sasl.kerberos.service.name": "kafka",
                "sasl.kerberos.keytab": "config/fdt.dfs.keytab",
                "sasl.kerberos.principal": "fdt.dfs@BYD.COM"
            },
            "producer_topic": self.topic,
            "maximum_allow_timediff": 2,
            "time_check": 10
        }

        self.producer = None
        self.running = True

        try:
            self.producer = Producer(self.CONFIG["producer_conf"])
        except Exception as e:
            self.logger.error(f"Error initializing PrepareData: {e}")
            self.shutdown()
        
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_signal(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def send_message(self, message):
        try:
            self.producer.produce(self.topic, 
                                  json.dumps(message).encode('utf-8'),)
            self.producer.flush()
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
    
    def load_config(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            message = json.load(f)
        return message

    def process_message(self, runner):
        datasets = self.meta_json["datasets"]
        datapath = []
        for dataset in datasets:
            if dataset["purpose"] == "ai-annotation":
                if len(dataset["samples"]) > 0:
                    for sample in dataset["samples"]:
                        datapath.append(osp.join(self.root_dir, dataset["sourceRoot"], sample["image"]))
                else:
                    datapath.append(osp.join(self.root_dir, dataset["sourceRoot"], "image"))
        assert len(datapath) > 0, "No data found in the dataset for AI annotation."

        target_data_list = []
        for path in datapath:
            if osp.isdir(path):
                path_list = os.listdir(path)  # Ensure the directory exists
                for p in path_list:
                    target_data_list.append(osp.join(path, p))
            else:
                target_data_list.append(path)
        total_target_img = len(target_data_list)

        for idx, name in enumerate(target_data_list):
            target_img = Image.open(name)

            standard_json = {
                "flags": "Alice",
                "version": "5.0.1",
                "imageData": None,
                "imagePath": osp.basename(name),
                "imageHeight": target_img.size[1],
                "imageWidth": target_img.size[0],
            }

            results = runner.predict(name)[0]
            shapes = []
            label_names = results.names
            if self.task == "detect":
                for i, box in enumerate(results.boxes):
                    temp_unit = {'flags': [], 'group_id': None, 'shape_type': 'rectangle'}
                    temp_unit['points'] = box.xyxy[0].reshape((2, 2)).tolist()
                    temp_unit["label"] = label_names[int(box.cls[0].item())]
                    shapes.append(temp_unit)
                standard_json["shapes"] = shapes
            elif self.task == "segment":
                for i, mask in enumerate(results.masks):
                    temp_unit = {'flags': [], 'group_id': None, 'shape_type': 'polygon'}
                    temp_unit['points'] = mask.xy[0].tolist()
                    cls_id = results.boxes.cls[i].item()
                    temp_unit["label"] = label_names[cls_id]
                    shapes.append(temp_unit)
                standard_json["shapes"] = shapes
            elif self.task == "classify":
                pass
            else:
                self.logger.error(f"Unsupported task type: {self.task}")
                continue

            with open(osp.join(self.save_dir, osp.basename(name)[:-4] + ".json"), 'w', encoding='utf-8') as f:
                json.dump(standard_json, f, indent=4)
            
            json_msg = {
                "header": {
                    "version": "1.0",
                    "type": "event",
                    "bizType": "prompt",
                    "action": "progressUpdate|jobCompleted",
                    "eventId": "UUID",
                    "correlationId": "",
                    "timestamp": "2025-05-14T12:00:00Z",
                    "sender": ""
                },
                "payload": {
                    "metadata": {
                        "userId": "user-789",
                        "appId": "app-012"
                    },
                    "progress": "0.00%",
                    "pipelineData": self.pipelineData,
                }
            }

            if idx == total_target_img - 1:
                json_msg["header"]["action"] = "jobCompleted"
            else:
                json_msg["header"]["action"] = "progressUpdate"
            taskID = round((idx + 1) / total_target_img * 100, 2)
            json_msg["payload"]["progress"] = taskID

            self.logger.info(f"Construct message: {json_msg}")
            self.send_message(json_msg)


    def run(self):
        # Implement data preparation logic here
        try:
            start_1 = time.time()
            
            self.process_message()

            overhead = time.time() - start_1
            self.logger.info(f'Data preparation overhead: {overhead:.4f} seconds')
            
        except Exception as e:
            self.logger.error(f'Error during data preparation: {e}')
        finally:
            self.shutdown()

    def shutdown(self):
        # Implement any cleanup logic here
        if not self.running:
            return

        self.logger.info('Shutting down data preparation...')
        self.running = False

        try:
            if self.producer:
                self.producer.flush(10)
                self.logger.info('Data preparation completed.')
        except Exception as e:
            self.logger.error(f'Error during data preparation shutdown: {e}')
        self.logger.info('Data preparation shutdown complete.')

def merge_args(cfg, args):
    cfg.work_dir = osp.join('./res', osp.splitext(osp.basename(args.config))[0])

    # # enable automatic-mixed-precision training
    # if args.amp is True:
    cfg.optim_wrapper.type = 'AmpOptimWrapper'
    cfg.optim_wrapper.loss_scale = 'dynamic'

    # # enable automatically scaling LR
    # if args.auto_scale_lr:
    #     if 'auto_scale_lr' in cfg and \
    #             'enable' in cfg.auto_scale_lr and \
    #             'base_batch_size' in cfg.auto_scale_lr:
    #         cfg.auto_scale_lr.enable = True
    #     else:
    #         raise RuntimeError('Can not find "auto_scale_lr" or '
    #                            '"auto_scale_lr.enable" or '
    #                            '"auto_scale_lr.base_batch_size" in your'
    #                            ' configuration file.')
    # resume is determined in this priority: resume from > auto_resume
    # if args.resume == 'auto':
    # cfg.resume = True
    # cfg.load_from = None
    # elif args.resume is not None:
    #     cfg.resume = True
    #     cfg.load_from = args.resume
    return cfg


def create_runner(args):
    """Create a runner instance."""
    cfg = Config.fromfile(args.config)
    operation = cfg.operation
    if "training" in operation:
        operation = "training"
    modelname = cfg[operation]['algoParams']['model']['type']

    if 'yolo' in modelname:
        from .yolorunner import YOLORunnerWarpper
        return YOLORunnerWarpper.from_cfg(cfg)
    else:
        from .mmrunner import MMRunner
        cfg = merge_args(cfg, args)
        return MMRunner.from_cfg(cfg)