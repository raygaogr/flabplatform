from abc import ABCMeta, abstractmethod

class ABCValidator(metaclass=ABCMeta):
    def __init__(self):
        """
        Initialize a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm, optional): Progress bar for displaying progress.
            args (SimpleNamespace, optional): Configuration for the validator.
            _callbacks (dict, optional): Dictionary to store various callback functions.
        """
        super().__init__()
        self.dataloader = None
        self.training = True
        self.stats = None

    @abstractmethod
    def __call__(self):
        raise NotImplementedError("The __call__ method must be implemented in the validator subclass.")

    @abstractmethod
    def build_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    @abstractmethod
    def build_dataset(self, img_path):
        """Build dataset from image path."""
        raise NotImplementedError("build_dataset function not implemented in validator")

    @abstractmethod
    def preprocess(self, batch):
        """Preprocess an input batch."""
        return batch

    @abstractmethod
    def postprocess(self, preds):
        """Postprocess the predictions."""
        return preds

    @abstractmethod
    def save_to_json(self, model):
        raise NotImplementedError("save_to_json function not implemented in validator")
