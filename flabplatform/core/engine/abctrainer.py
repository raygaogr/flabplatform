from abc import ABCMeta, abstractmethod
from torch import nn, optim
from flabplatform.flabdet.utils import LOGGER

class ABCTrainer(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.model = None
        self.validator = None
        self.optimizer = None
        self.scheduler = None

    def train(self):
        """Train the model."""
        raise NotImplementedError("train function not implemented in trainer")

    def build_dataloader(self, dataset_path, batch_size=16, rank=0):
        """Get dataloader for the given dataset."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")
    
    def build_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")
    
    def build_optimizer(self, name="auto", lr=0.001, momentum=0.9, decay=1e-5):
        raise NotImplementedError(
            "build_optimizer function not implemented in trainer. "
            "Please implement this method to build the optimizer for your model."
        )
    
    def build_scheduler(self):
        """Returns a NotImplementedError when the build_scheduler function is called."""
        raise NotImplementedError("build_scheduler function not implemented in trainer")

    def resume_training(self, ckpt):
        """Resume training from a checkpoint."""
        return NotImplementedError(
            "resume_training function not implemented in trainer. "
            "Please implement this method to resume training from a checkpoint."
        )
    
    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths."""
        return NotImplementedError(
            "preprocess_batch function not implemented in trainer. "
            "Please implement this method to preprocess the batch for your model."
        )
    
    def save_metrics(self, metrics):
        """Save training metrics to a json file."""
        raise NotImplementedError("save_metrics function not implemented in trainer")

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        raise NotImplementedError("save_model function not implemented in trainer")