import logging
from typing import Optional, Union, TYPE_CHECKING

import numpy as np

from src.defences.adversarial_trainer import AdversarialTrainer
from src.attacks.projected_gradient_descent import ProjectedGradientDescent

if TYPE_CHECKING:
    from src.preprocessing.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE


logger = logging.getLogger(__name__)


class AdversarialTrainerMadryPGD:
    def __init__(
        self,
        classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        nb_epochs: int = 391,
        batch_size: int = 128,
        eps: Union[int, float] = 8,
        eps_step: Union[int, float] = 2,
        max_iter: int = 7,
        num_random_init: int = 1,
    ) -> None:
        self._classifier = classifier
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs

        self.attack = ProjectedGradientDescent(
            classifier,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            num_random_init=num_random_init,
        )

        self.trainer = AdversarialTrainer(classifier, self.attack, ratio=1.0) 

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        self.trainer.fit(
            x,
            y,
            validation_data=validation_data,
            nb_epochs=self.nb_epochs,
            batch_size=self.batch_size,
            **kwargs
        )

    def get_classifier(self) -> "CLASSIFIER_LOSS_GRADIENTS_TYPE":
        return self.trainer.get_classifier()
