import logging
from typing import Optional, Tuple, Union

import numpy as np

from src.config import FLOAT_NUMPY
from src.defences.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


def broadcastable_mean_std(
    x: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    if mean.ndim == 1 and mean.shape[0] > 1 and mean.shape[0] != x.shape[-1]:
        # allow input shapes NC* (batch) and C* (non-batch)
        channel_idx = 1 if x.shape[1] == mean.shape[0] else 0
        broadcastable_shape = [1] * x.ndim
        broadcastable_shape[channel_idx] = mean.shape[0]

        # expand mean and std to new shape
        mean = mean.reshape(broadcastable_shape)
        std = std.reshape(broadcastable_shape)

    return mean, std


class StandardisationMeanStd(Preprocessor):
    """
    Implement the standardisation with mean and standard deviation.
    """

    params = ["mean", "std", "apply_fit", "apply_predict"]

    def __init__(
        self,
        mean: Union[float, np.ndarray] = 0.0,
        std: Union[float, np.ndarray] = 1.0,
        apply_fit: bool = True,
        apply_predict: bool = True,
    ):
        """
        Create an instance of StandardisationMeanStd.

        :param mean: Mean.
        :param std: Standard Deviation.
        """
        super().__init__(
            is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict
        )
        self.mean = np.asarray(mean, dtype=FLOAT_NUMPY)
        self.std = np.asarray(std, dtype=FLOAT_NUMPY)

        # init broadcastable mean and std for lazy loading
        self._broadcastable_mean: Optional[np.ndarray] = None
        self._broadcastable_std: Optional[np.ndarray] = None

    def __call__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply StandardisationMeanStd inputs `x`.

        :param x: Input samples to standardise.
        :param y: Label data, will not be affected by this preprocessing.
        :return: Standardise input samples and unmodified labels.
        """
        if x.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:  # pragma: no cover
            raise TypeError(
                f"The data type of input data `x` is {x.dtype} and cannot represent negative values. Consider "
                f"changing the data type of the input data `x` to a type that supports negative values e.g. "
                f"np.float32."
            )

        if self._broadcastable_mean is None:
            self._broadcastable_mean, self._broadcastable_std = broadcastable_mean_std(
                x, self.mean, self.std
            )

        x_norm = x - self._broadcastable_mean
        x_norm = x_norm / self._broadcastable_std
        x_norm = x_norm.astype(FLOAT_NUMPY)

        return x_norm, y

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Provide an estimate of the gradients of preprocessor for the backward pass. If the preprocessor is not
        differentiable, this is an estimate of the gradient, most often replacing the computation performed by the
        preprocessor with the identity function (the default).

        :param x: Input data for which the gradient is estimated. First dimension is the batch size.
        :param grad: Gradient value so far.
        :return: The gradient (estimate) of the defence.
        """
        _, std = broadcastable_mean_std(x, self.mean, self.std)
        gradient_back = grad / std

        return gradient_back
