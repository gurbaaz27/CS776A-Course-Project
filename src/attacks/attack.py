from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import logging
from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.utils import CLASSIFIER_TYPE, GENERATOR_TYPE

logger = logging.getLogger(__name__)


class Attack(abc.ABC):
    """
    Abstract base class for all attack abstract base classes.
    """

    attack_params: List[str] = []
    # The _estimator_requirements define the requirements an estimator must satisfy to be used as a target for an
    # attack. They should be a tuple of requirements, where each requirement is either a class the estimator must
    # inherit from, or a tuple of classes which define a union, i.e. the estimator must inherit from at least one class
    # in the requirement tuple.
    _estimator_requirements: Optional[Union[Tuple[Any, ...], Tuple[()]]] = None

    def __init__(
        self,
        estimator,
    ):
        """
        :param estimator: An estimator.
        """
        super().__init__()

        if self.estimator_requirements is None:
            raise ValueError("Estimator requirements have not been defined in `_estimator_requirements`.")

        self._estimator = estimator

        print("Hi")
        print(self._estimator)

        Attack._check_params(self)

    @property
    def estimator(self):
        """The estimator."""
        print("Bi")
        return self._estimator

    @property
    def estimator_requirements(self):
        """The estimator requirements."""
        return self._estimator_requirements

    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: A dictionary of attack-specific parameters.
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        self._check_params()

    def _check_params(self) -> None:
        pass


    @staticmethod
    def is_estimator_valid(estimator, estimator_requirements) -> bool:
        """
        Checks if the given estimator satisfies the requirements for this attack.

        :param estimator: The estimator to check.
        :param estimator_requirements: Estimator requirements.
        :return: True if the estimator is valid for the attack.
        """

        for req in estimator_requirements:
            # A requirement is either a class which the estimator must inherit from, or a tuple of classes and the
            # estimator is required to inherit from at least one of the classes
            if isinstance(req, tuple):
                if all(p not in type(estimator).__mro__ for p in req):
                    return False
            elif req not in type(estimator).__mro__:
                return False
        return True


class EvasionAttack(Attack):
    """
    Abstract base class for evasion attack classes.
    """

    def __init__(self, **kwargs) -> None:
        self._targeted = False
        super().__init__(**kwargs)

    @abc.abstractmethod
    def generate(  # lgtm [py/inheritance/incorrect-overridden-signature]
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """
        Generate adversarial examples and return them as an array. This method should be overridden by all concrete
        evasion attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y: Correct labels or target labels for `x`, depending if the attack is targeted
                  or not. This parameter is only used by some of the attacks.
        :return: An array holding the adversarial examples.
        """
        raise NotImplementedError

    @property
    def targeted(self) -> bool:
        """
        Return Boolean if attack is targeted. Return None if not applicable.
        """
        return self._targeted

    @targeted.setter
    def targeted(self, targeted) -> None:
        self._targeted = targeted


class PoisoningAttack(Attack):
    """
    Abstract base class for poisoning attack classes
    """

    def __init__(self, classifier: Optional["CLASSIFIER_TYPE"]) -> None:
        """
        :param classifier: A trained classifier (or none if no classifier is needed)
        """
        super().__init__(classifier)

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y=Optional[np.ndarray], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate poisoning examples and return them as an array. This method should be overridden by all concrete
        poisoning attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y:  Target labels for `x`. Untargeted attacks set this value to None.
        :return: An tuple holding the (poisoning examples, poisoning labels).
        """
        raise NotImplementedError


class PoisoningAttackBlackBox(PoisoningAttack):
    """
    Abstract base class for poisoning attack classes that have no access to the model (classifier object).
    """

    def __init__(self):
        """
        Initializes black-box data poisoning attack.
        """
        super().__init__(None)  # type: ignore

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate poisoning examples and return them as an array. This method should be overridden by all concrete
        poisoning attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y:  Target labels for `x`. Untargeted attacks set this value to None.
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """
        raise NotImplementedError