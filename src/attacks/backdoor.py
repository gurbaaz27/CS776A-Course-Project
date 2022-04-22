import logging
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from src.attacks.attack import PoisoningAttack


logger = logging.getLogger(__name__)


class PoisoningAttackBackdoor(PoisoningAttack):
    _estimator_requirements = ()

    def __init__(self, perturbation: Union[Callable, List[Callable]]) -> None:

        super().__init__()
        self.perturbation = perturbation

    def poison(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, broadcast=False, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:

        if broadcast:
            y_attack = np.broadcast_to(y, (x.shape[0], y.shape[0]))
        else:
            y_attack = np.copy(y)

        num_poison = len(x)

        poisoned = np.copy(x)

        if callable(self.perturbation):
            return self.perturbation(poisoned), y_attack

        for perturb in self.perturbation:
            poisoned = perturb(poisoned)

        return poisoned, y_attack
