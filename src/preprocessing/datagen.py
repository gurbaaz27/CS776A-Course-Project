from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import inspect
import logging
from typing import Any, Generator, Optional, Union

import keras

logger = logging.getLogger(__name__)


class DataGenerator(abc.ABC):

    def __init__(self, size: Optional[int], batch_size: int) -> None:
        self._size = size
        self._batch_size = batch_size
        self._iterator: Optional[Any] = None

    @abc.abstractmethod
    def get_batch(self) -> tuple:
        raise NotImplementedError

    @property
    def iterator(self):
        return self._iterator

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def size(self) -> Optional[int]:
        return self._size


class KerasDataGenerator(DataGenerator):

    def __init__(
        self,
        iterator: Union[
            "keras.utils.Sequence",
            "keras.preprocessing.image.ImageDataGenerator",
            Generator,
        ],
        size: Optional[int],
        batch_size: int,
    ) -> None:
        super().__init__(size=size, batch_size=batch_size)
        self._iterator = iterator

    def get_batch(self) -> tuple:
        if inspect.isgeneratorfunction(self.iterator):
            return next(self.iterator)

        iter_ = iter(self.iterator)
        return next(iter_)
