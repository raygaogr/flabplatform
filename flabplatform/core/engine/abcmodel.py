
import torch
from abc import abstractmethod


class ABCModel(torch.nn.Module):
    @abstractmethod
    def forward(self, x, *args, **kwargs):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        raise NotImplementedError(
            f"forward() function not implemented in {self.__class__.__name__}. "
            "Please implement this method to perform a forward pass through your model."
        )

    @abstractmethod
    def predict(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        raise NotImplementedError(
            f"predict() function not implemented in {self.__class__.__name__}. "
            "Please implement this method to perform inference with your model."
        )

    @abstractmethod
    def load(self, weights):
        """
        Load weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress.
        """
        raise NotImplementedError(
            f"load() function not implemented in {self.__class__.__name__}. "
            "Please implement this method to load pre-trained weights into your model."
        )

    @abstractmethod
    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        raise NotImplementedError(
            f"loss() function not implemented in {self.__class__.__name__}. "
            "Please implement this method to compute the loss for your model."
        )

    @abstractmethod
    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")
