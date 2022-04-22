import logging
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
import numpy as np
from scipy.stats import truncnorm
from tqdm.auto import trange

from src.classifiers.estimator import BaseEstimator, LossGradientsMixin
from src.attacks.attack import EvasionAttack
from src.attacks.fast_gradient import FastGradientMethod
from src.config import FLOAT_NUMPY
from src.classifiers.classifier import ClassifierMixin
from src.classifiers.estimator import BaseEstimator, LossGradientsMixin
from src.utils import (
    compute_success,
    get_labels_np_array,
    check_and_transform_label_format,
    compute_success_array,
)

if TYPE_CHECKING:
    from src.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class ProjectedGradientDescent(EvasionAttack):
    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        verbose: bool = True,
    ):
        super().__init__(estimator=estimator)  # , summary_writer=False)

        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.targeted = targeted
        self.num_random_init = num_random_init
        self.batch_size = batch_size
        self.random_eps = random_eps
        self.verbose = verbose

        self._attack = ProjectedGradientDescentNumpy(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
            verbose=verbose,
        )

    def generate(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        logger.info("Creating adversarial samples.")
        print(self._attack.__dict__)
        return self._attack.generate(x=x, y=y, **kwargs)

    def set_params(self, **kwargs) -> None:
        super().set_params(**kwargs)
        self._attack.set_params(**kwargs)


class ProjectedGradientDescentCommon(FastGradientMethod):
    """
    Common class for different variations of implementation of the Projected Gradient Descent attack. The attack is an
    iterative method in which, after each iteration, the perturbation is projected on an lp-ball of specified radius (in
    addition to clipping the values of the adversarial sample so that it lies in the permitted data range). This is the
    attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """

    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            minimal=False,
        )
        self.max_iter = max_iter
        self.random_eps = random_eps
        self.verbose = verbose

        lower: Union[int, float, np.ndarray]
        upper: Union[int, float, np.ndarray]
        var_mu: Union[int, float, np.ndarray]
        sigma: Union[int, float, np.ndarray]

        if self.random_eps:
            if isinstance(eps, (int, float)):
                lower, upper = 0, eps
                var_mu, sigma = 0, (eps / 2)
            else:
                lower, upper = np.zeros_like(eps), eps
                var_mu, sigma = np.zeros_like(eps), (eps / 2)

            self.norm_dist = truncnorm(
                (lower - var_mu) / sigma,
                (upper - var_mu) / sigma,
                loc=var_mu,
                scale=sigma,
            )

    def _random_eps(self):
        """
        Check whether random eps is enabled, then scale eps and eps_step appropriately.
        """
        if self.random_eps:
            ratio = self.eps_step / self.eps

            if isinstance(self.eps, (int, float)):
                self.eps = np.round(self.norm_dist.rvs(1)[0], 10)
            else:
                self.eps = np.round(self.norm_dist.rvs(size=self.eps.shape), 10)

            self.eps_step = ratio * self.eps

    def _set_targets(
        self, x: np.ndarray, y: Optional[np.ndarray], classifier_mixin: bool = True
    ) -> np.ndarray:
        if classifier_mixin:
            if y is not None:
                y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:  # pragma: no cover
                raise ValueError(
                    "Target labels `y` need to be provided for a targeted attack."
                )

            # Use model predictions as correct outputs
            if classifier_mixin:
                targets = get_labels_np_array(
                    self.estimator.predict(x, batch_size=self.batch_size)
                )
            else:
                targets = self.estimator.predict(x, batch_size=self.batch_size)

        else:
            targets = y

        return targets


class ProjectedGradientDescentNumpy(ProjectedGradientDescentCommon):

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        verbose: bool = True,
    ) -> None:

        super().__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
            verbose=verbose,
        )

        self._project = True

    def generate(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        mask = self._get_mask(x, **kwargs)

        # Ensure eps is broadcastable
        self._check_compatibility_input_and_eps(x=x)

        # Check whether random eps is enabled
        self._random_eps()

        if isinstance(self.estimator, ClassifierMixin):
            # Set up targets
            targets = self._set_targets(x, y)

            # Start to compute adversarial examples
            adv_x = x.astype(FLOAT_NUMPY)

            for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):

                self._batch_id = batch_id

                for rand_init_num in trange(
                    max(1, self.num_random_init),
                    desc="PGD - Random Initializations",
                    disable=not self.verbose,
                ):
                    batch_index_1, batch_index_2 = (
                        batch_id * self.batch_size,
                        (batch_id + 1) * self.batch_size,
                    )
                    batch_index_2 = min(batch_index_2, x.shape[0])
                    batch = x[batch_index_1:batch_index_2]
                    batch_labels = targets[batch_index_1:batch_index_2]
                    mask_batch = mask

                    if mask is not None:
                        if len(mask.shape) == len(x.shape):
                            mask_batch = mask[batch_index_1:batch_index_2]

                    for i_max_iter in trange(
                        self.max_iter,
                        desc="PGD - Iterations",
                        leave=False,
                        disable=not self.verbose,
                    ):
                        self._i_max_iter = i_max_iter

                        batch = self._compute(
                            batch,
                            x[batch_index_1:batch_index_2],
                            batch_labels,
                            mask_batch,
                            self.eps,
                            self.eps_step,
                            self._project,
                            self.num_random_init > 0 and i_max_iter == 0,
                            self._batch_id,
                        )

                    if rand_init_num == 0:
                        # initial (and possibly only) random restart: we only have this set of
                        # adversarial examples for now
                        adv_x[batch_index_1:batch_index_2] = np.copy(batch)
                    else:
                        # replace adversarial examples if they are successful
                        attack_success = compute_success_array(
                            self.estimator,  # type: ignore
                            x[batch_index_1:batch_index_2],
                            targets[batch_index_1:batch_index_2],
                            batch,
                            self.targeted,
                            batch_size=self.batch_size,
                        )
                        adv_x[batch_index_1:batch_index_2][attack_success] = batch[
                            attack_success
                        ]

            logger.info(
                "Success rate of attack: %.2f%%",
                100
                * compute_success(
                    self.estimator,  # type: ignore
                    x,
                    targets,
                    adv_x,
                    self.targeted,
                    batch_size=self.batch_size,  # type: ignore
                ),
            )
        else:
            if self.num_random_init > 0:  # pragma: no cover
                raise ValueError(
                    "Random initialisation is only supported for classification."
                )

            # Set up targets
            targets = self._set_targets(x, y, classifier_mixin=False)

            # Start to compute adversarial examples
            if x.dtype == object:
                adv_x = x.copy()
            else:
                adv_x = x.astype(FLOAT_NUMPY)

            for i_max_iter in trange(
                self.max_iter, desc="PGD - Iterations", disable=not self.verbose
            ):
                self._i_max_iter = i_max_iter

                adv_x = self._compute(
                    adv_x,
                    x,
                    targets,
                    mask,
                    self.eps,
                    self.eps_step,
                    self._project,
                    self.num_random_init > 0 and i_max_iter == 0,
                )

        return adv_x
