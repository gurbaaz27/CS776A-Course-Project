import abc
import logging
from typing import Any, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class Attack(abc.ABC):
    _estimator_requirements: Optional[Union[Tuple[Any, ...], Tuple[()]]] = None

    def __init__(
        self,
        estimator,
    ):
        super().__init__()

        if self.estimator_requirements is None:
            raise ValueError(
                "Estimator requirements have not been defined in `_estimator_requirements`."
            )

        self._estimator = estimator

    @property
    def estimator(self):
        return self._estimator

    @property
    def estimator_requirements(self):
        return self._estimator_requirements

    def set_params(self, **kwargs) -> None:
        pass


class EvasionAttack(Attack):

    def __init__(self, **kwargs) -> None:
        self._targeted = False
        super().__init__(**kwargs)

    @abc.abstractmethod
    def generate(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        raise NotImplementedError

    @property
    def targeted(self) -> bool:
        return self._targeted

    @targeted.setter
    def targeted(self, targeted) -> None:
        self._targeted = targeted


class PoisoningAttackBlackBox(Attack):

    def __init__(self):
        super().__init__(None)  # type: ignore

    @abc.abstractmethod
    def poison(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
