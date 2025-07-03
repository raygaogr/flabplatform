

from abc import ABCMeta, abstractmethod

class ABCPredictor(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.save_dir = None
        self.model = None
        self.dataset = None
        self.results = None

    @abstractmethod
    def preprocess(self, image):
        raise NotImplementedError("Preprocess method not implemented in predictor")


    @abstractmethod
    def postprocess(self, preds, img, orig_imgs):
        """Post-process predictions for an image and return them."""
        return preds


    @abstractmethod
    def __call__(self, source=None, model=None):
        raise NotImplementedError(
            "The __call__ method is not implemented in the predictor. "
            "Please implement this method to run the predictor with the given source and model."
        )


    @abstractmethod
    def save_results(self):
        raise NotImplementedError(
            "The write_results method is not implemented in the predictor. "
            "Please implement this method to write results for the given image."
        )

