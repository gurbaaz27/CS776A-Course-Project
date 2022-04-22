import logging
from typing import Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

from src.attacks.attack import PoisoningAttackBlackBox
from src.attacks.projected_gradient_descent import ProjectedGradientDescent
from src.attacks.backdoor import PoisoningAttackBackdoor

if TYPE_CHECKING:
    from src.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE


logger = logging.getLogger(__name__)


class PoisoningAttackCleanLabelBackdoor(PoisoningAttackBlackBox):
    _estimator_requirements = ()

    def __init__(
        self,
        backdoor: PoisoningAttackBackdoor,
        trained_classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        target: np.ndarray,
        pp_poison: float = 0.33,
        norm: Union[int, float, str] = np.inf,
        eps: float = 0.3,
        eps_step: float = 0.1,
        max_iter: int = 100,
        num_random_init: int = 0,
    ) -> None:
        super().__init__()
        self.backdoor = backdoor
        self.trained_classifier = trained_classifier
        self.target = target
        self.pp_poison = pp_poison
        self.attack = ProjectedGradientDescent(
            trained_classifier,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=False,
            num_random_init=num_random_init,
        )

    def poison(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        broadcast: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = np.copy(x)
        estimated_labels = (
            self.trained_classifier.predict(data) if y is None else np.copy(y)
        )

        # Selected target indices to poison
        all_indices = np.arange(len(data))
        target_indices = all_indices[np.all(estimated_labels == self.target, axis=1)]
        num_poison = int(self.pp_poison * len(target_indices))
        selected_indices = np.random.choice(target_indices, num_poison)

        # Run untargeted PGD on selected points, making it hard to classify correctly
        perturbed_input = self.attack.generate(data[selected_indices])

        # Add backdoor and poison with the same label
        poisoned_input, _ = self.backdoor.poison(
            perturbed_input, self.target, broadcast=broadcast
        )
        data[selected_indices] = poisoned_input

        return data, estimated_labels
