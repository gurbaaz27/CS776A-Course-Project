from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange, tqdm


if TYPE_CHECKING:
    from src.preprocessing.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE
    from src.attacks.attack import AdversarialAttack
    from src.preprocessing.datagen import DataGenerator

logger = logging.getLogger(__name__)


class AdversarialTrainer:

    def __init__(
        self,
        classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        attacks: Union["AdversarialAttack", List["AdversarialAttack"]],
        ratio: float = 0.5,
    ) -> None:
        from src.attacks.attack import AdversarialAttack

        self._classifier = classifier
        if isinstance(attacks, AdversarialAttack):
            self.attacks = [attacks]
        elif isinstance(attacks, list):
            self.attacks = attacks
        else:
            raise ValueError(
                "Only AdversarialAttack instances or list of attacks supported."
            )

        if ratio <= 0 or ratio > 1:
            raise ValueError(
                "The `ratio` of adversarial samples in each batch has to be between 0 and 1."
            )
        self.ratio = ratio

        self._precomputed_adv_samples: List[Optional[np.ndarray]] = []
        self.x_augmented: Optional[np.ndarray] = None
        self.y_augmented: Optional[np.ndarray] = None

    def fit_generator(
        self, generator: "DataGenerator", nb_epochs: int = 20, **kwargs
    ) -> None:
        logger.info(
            "Performing adversarial training using %i attacks.", len(self.attacks)
        )
        size = generator.size
        if size is None:
            raise ValueError("Generator size is required and cannot be None.")
        batch_size = generator.batch_size
        nb_batches = int(np.ceil(size / batch_size))  # type: ignore
        ind = np.arange(generator.size)
        attack_id = 0

        # Precompute adversarial samples for transferred attacks
        logged = False
        self._precomputed_adv_samples = []
        for attack in tqdm(self.attacks, desc="Precompute adversarial examples."):

            if attack.estimator != self._classifier:
                if not logged:
                    logger.info("Precomputing transferred adversarial samples.")
                    logged = True

                for batch_id in range(nb_batches):
                    # Create batch data
                    x_batch, y_batch = generator.get_batch()
                    x_adv_batch = attack.generate(x_batch, y=y_batch)
                    if batch_id == 0:
                        next_precomputed_adv_samples = x_adv_batch
                    else:
                        next_precomputed_adv_samples = np.append(
                            next_precomputed_adv_samples, x_adv_batch, axis=0
                        )
                self._precomputed_adv_samples.append(next_precomputed_adv_samples)
            else:
                self._precomputed_adv_samples.append(None)

        for _ in trange(nb_epochs, desc="Adversarial training epochs"):
            # Shuffle the indices of precomputed examples
            np.random.shuffle(ind)

            for batch_id in range(nb_batches):
                # Create batch data
                x_batch, y_batch = generator.get_batch()
                x_batch = x_batch.copy()

                # Choose indices to replace with adversarial samples
                attack = self.attacks[attack_id]

                # If source and target models are the same, craft fresh adversarial samples
                if attack.estimator == self._classifier:
                    nb_adv = int(np.ceil(self.ratio * x_batch.shape[0]))

                    if self.ratio < 1:
                        adv_ids = np.random.choice(
                            x_batch.shape[0], size=nb_adv, replace=False
                        )
                    else:
                        adv_ids = np.array(list(range(x_batch.shape[0])))
                        np.random.shuffle(adv_ids)

                    x_batch[adv_ids] = attack.generate(
                        x_batch[adv_ids], y=y_batch[adv_ids]
                    )

                # Otherwise, use precomputed adversarial samples
                else:
                    batch_size_current = min(batch_size, size - batch_id * batch_size)
                    nb_adv = int(np.ceil(self.ratio * batch_size_current))
                    if self.ratio < 1:
                        adv_ids = np.random.choice(
                            batch_size_current, size=nb_adv, replace=False
                        )
                    else:
                        adv_ids = np.array(list(range(batch_size_current)))
                        np.random.shuffle(adv_ids)

                    x_adv = self._precomputed_adv_samples[attack_id]
                    if x_adv is not None:
                        x_adv = x_adv[
                            ind[
                                batch_id
                                * batch_size : min((batch_id + 1) * batch_size, size)
                            ]
                        ][adv_ids]
                    x_batch[adv_ids] = x_adv

                # Fit batch
                self._classifier.fit(
                    x_batch,
                    y_batch,
                    nb_epochs=1,
                    batch_size=x_batch.shape[0],
                    verbose=0,
                    **kwargs
                )
                attack_id = (attack_id + 1) % len(self.attacks)

    def fit(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
        nb_epochs: int = 20,
        **kwargs
    ) -> None:
        logger.info(
            "Performing adversarial training using %i attacks.", len(self.attacks)
        )
        nb_batches = int(np.ceil(len(x) / batch_size))
        ind = np.arange(len(x))
        attack_id = 0

        # Precompute adversarial samples for transferred attacks
        logged = False
        self._precomputed_adv_samples = []
        for attack in tqdm(self.attacks, desc="Precompute adv samples"):

            if attack.estimator != self._classifier:
                if not logged:
                    logger.info("Precomputing transferred adversarial samples.")
                    logged = True
                self._precomputed_adv_samples.append(attack.generate(x, y=y))
            else:
                self._precomputed_adv_samples.append(None)

        for _ in trange(nb_epochs, desc="Adversarial training epochs"):
            # Shuffle the examples
            np.random.shuffle(ind)

            for batch_id in range(nb_batches):
                # Create batch data
                x_batch = x[
                    ind[
                        batch_id
                        * batch_size : min((batch_id + 1) * batch_size, x.shape[0])
                    ]
                ].copy()
                y_batch = y[
                    ind[
                        batch_id
                        * batch_size : min((batch_id + 1) * batch_size, x.shape[0])
                    ]
                ]

                # Choose indices to replace with adversarial samples
                nb_adv = int(np.ceil(self.ratio * x_batch.shape[0]))
                attack = self.attacks[attack_id]
                if self.ratio < 1:
                    adv_ids = np.random.choice(
                        x_batch.shape[0], size=nb_adv, replace=False
                    )
                else:
                    adv_ids = np.array(list(range(x_batch.shape[0])))
                    np.random.shuffle(adv_ids)

                # If source and target models are the same, craft fresh adversarial samples
                if attack.estimator == self._classifier:
                    x_batch[adv_ids] = attack.generate(
                        x_batch[adv_ids], y=y_batch[adv_ids]
                    )

                # Otherwise, use precomputed adversarial samples
                else:
                    x_adv = self._precomputed_adv_samples[attack_id]
                    if x_adv is not None:
                        x_adv = x_adv[
                            ind[
                                batch_id
                                * batch_size : min(
                                    (batch_id + 1) * batch_size, x.shape[0]
                                )
                            ]
                        ][adv_ids]
                    x_batch[adv_ids] = x_adv

                # Fit batch
                self._classifier.fit(
                    x_batch,
                    y_batch,
                    nb_epochs=1,
                    batch_size=x_batch.shape[0],
                    verbose=0,
                    **kwargs
                )
                attack_id = (attack_id + 1) % len(self.attacks)

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._classifier.predict(x, **kwargs)

    def get_classifier(self) -> "CLASSIFIER_LOSS_GRADIENTS_TYPE":
        return self._classifier
