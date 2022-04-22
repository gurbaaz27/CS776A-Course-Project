from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import shutil
import sys
import tarfile
import zipfile
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import six
from scipy.special import gammainc
from tqdm.auto import tqdm

from src import constants


logger = logging.getLogger(__name__)


CLIP_VALUES_TYPE = Tuple[
    Union[int, float, np.ndarray], Union[int, float, np.ndarray]
]

if TYPE_CHECKING:
    from src.defences.preprocessor import Preprocessor

    PREPROCESSING_TYPE = Optional[
        Union[
            Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]],
            Preprocessor,
            Tuple[Preprocessor, ...],
        ]
    ]

    from src.classifiers.classifier import (
        Classifier,
        ClassifierClassLossGradients,
        ClassifierLossGradients,
        ClassifierNeuralNetwork,
    )
    from src.classifiers.kerasclassifier import KerasClassifier

    CLASSIFIER_LOSS_GRADIENTS_TYPE = Union[
        ClassifierLossGradients,
        KerasClassifier,
    ]

    CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE = Union[  # pylint: disable=C0103
        ClassifierClassLossGradients,
        KerasClassifier,
    ]

    CLASSIFIER_NEURALNETWORK_TYPE = Union[  # pylint: disable=C0103
        ClassifierNeuralNetwork,
        KerasClassifier,
    ]

    CLASSIFIER_TYPE = Union[  # pylint: disable=C0103
        Classifier,
        KerasClassifier,
        CLASSIFIER_NEURALNETWORK_TYPE,
    ]


def projection_l1_1(
    values: np.ndarray, eps: Union[int, float, np.ndarray]
) -> np.ndarray:
    shp = values.shape
    a = values.copy()
    n = np.prod(a.shape[1:])
    m = a.shape[0]
    a = a.reshape((m, n))
    sgns = np.sign(a)
    a = np.abs(a)

    a_argsort = a.argsort(axis=1)
    a_sorted = np.zeros((m, n))
    for i in range(m):
        a_sorted[i, :] = a[i, a_argsort[i, :]]
    a_argsort_inv = a.argsort(axis=1).argsort(axis=1)
    mat = np.zeros((m, 2))

    #   if  a_sorted[i, n-1]  >= a_sorted[i, n-2] + eps,  then the projection is  [0,...,0,eps]
    done = False
    active = np.array([1] * m)
    after_vec = np.zeros((m, n))
    proj = a_sorted.copy()
    j = n - 2
    while j >= 0:
        mat[:, 0] = (
            mat[:, 0] + a_sorted[:, j + 1]
        )  # =  sum(a_sorted[: i] :  i = j + 1,...,n-1
        mat[:, 1] = a_sorted[:, j] * (n - j - 1) + eps
        #  Find the max in each problem  max{ sum{a_sorted[:, i] : i=j+1,..,n-1} , a_sorted[:, j] * (n-j-1) + eps }
        row_maxes = np.max(mat, axis=1)
        #  Set to  1  if  max >  a_sorted[:, j] * (n-j-1) + eps  >  sum ;  otherwise, set to  0
        ind_set = np.sign(np.sign(row_maxes - mat[:, 0]))
        #  ind_set = ind_set.reshape((m, 1))
        #   Multiplier for activation
        act_multiplier = (1 - ind_set) * active
        act_multiplier = np.transpose([np.transpose(act_multiplier)] * n)
        #  if done, the projection is supported by the current indices  j+1,..,n-1   and the amount by which each
        #  has to be reduced is  delta
        delta = (mat[:, 0] - eps) / (n - j - 1)
        #    The vector of reductions
        delta_vec = np.array([delta] * (n - j - 1))
        delta_vec = np.transpose(delta_vec)
        #   The sub-vectors:  a_sorted[:, (j+1):]
        a_sub = a_sorted[:, (j + 1) :]
        #   After reduction by delta_vec
        a_after = a_sub - delta_vec
        after_vec[:, (j + 1) :] = a_after
        proj = (act_multiplier * after_vec) + ((1 - act_multiplier) * proj)
        active = active * ind_set
        if sum(active) == 0:
            done = True
            break
        j -= 1
    if not done:
        proj = active * a_sorted + (1 - active) * proj

    for i in range(m):
        proj[i, :] = proj[i, a_argsort_inv[i, :]]

    proj = sgns * proj
    proj = proj.reshape(shp)

    return proj


def projection_l1_2(
    values: np.ndarray, eps: Union[int, float, np.ndarray]
) -> np.ndarray:
    shp = values.shape
    a = values.copy()
    n = np.prod(a.shape[1:])
    m = a.shape[0]
    a = a.reshape((m, n))
    sgns = np.sign(a)
    a = np.abs(a)
    a_argsort = a.argsort(axis=1)
    a_sorted = np.zeros((m, n))
    for i in range(m):
        a_sorted[i, :] = a[i, a_argsort[i, :]]

    a_argsort_inv = a.argsort(axis=1).argsort(axis=1)
    row_sums = np.sum(a, axis=1)
    mat = np.zeros((m, 2))
    mat0 = np.zeros((m, 2))
    a_var = a_sorted.copy()
    for j in range(n):
        mat[:, 0] = (row_sums - eps) / (n - j)
        mat[:, 1] = a_var[:, j]
        mat0[:, 1] = np.min(mat, axis=1)
        min_t = np.max(mat0, axis=1)
        if np.max(min_t) < 1e-8:
            break
        row_sums = row_sums - a_var[:, j] * (n - j)
        a_var[:, (j + 1) :] = a_var[:, (j + 1) :] - np.matmul(
            min_t.reshape((m, 1)), np.ones((1, n - j - 1))
        )
        a_var[:, j] = a_var[:, j] - min_t
    proj = np.zeros((m, n))
    for i in range(m):
        proj[i, :] = a_var[i, a_argsort_inv[i, :]]

    proj = sgns * proj
    proj = proj.reshape(shp)
    return proj


def projection(
    values: np.ndarray,
    eps: Union[int, float, np.ndarray],
    norm_p: Union[int, float, str],
) -> np.ndarray:
    tol = 10e-8
    values_tmp = values.reshape((values.shape[0], -1))

    if norm_p == 2:
        if isinstance(eps, np.ndarray):
            raise NotImplementedError(
                "The parameter `eps` of type `np.ndarray` is not supported to use with norm 2."
            )

        values_tmp = values_tmp * np.expand_dims(
            np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1) + tol)), axis=1
        )

    elif norm_p == 1:
        if isinstance(eps, np.ndarray):
            raise NotImplementedError(
                "The parameter `eps` of type `np.ndarray` is not supported to use with norm 1."
            )

        values_tmp = values_tmp * np.expand_dims(
            np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1, ord=1) + tol)),
            axis=1,
        )
    elif norm_p == 1.1:
        values_tmp = projection_l1_1(values_tmp, eps)
    elif norm_p == 1.2:
        values_tmp = projection_l1_2(values_tmp, eps)

    elif norm_p in [np.inf, "inf"]:
        if isinstance(eps, np.ndarray):
            eps = eps * np.ones_like(values)
            eps = eps.reshape([eps.shape[0], -1])  # type: ignore

        values_tmp = np.sign(values_tmp) * np.minimum(abs(values_tmp), eps)

    else:
        raise NotImplementedError(
            'Values of `norm_p` different from 1, 2, `np.inf` and "inf" are currently not '
            "supported."
        )

    values = values_tmp.reshape(values.shape)

    return values


def random_sphere(
    nb_points: int,
    nb_dims: int,
    radius: Union[int, float, np.ndarray],
    norm: Union[int, float, str],
) -> np.ndarray:
    """
    Generate randomly `m x n`-dimension points with radius `radius` and centered around 0.

    :param nb_points: Number of random data points.
    :param nb_dims: Dimensionality of the sphere.
    :param radius: Radius of the sphere.
    :param norm: Current support: 1, 2, np.inf, "inf".
    :return: The generated random sphere.
    """
    if norm == 1:
        if isinstance(radius, np.ndarray):
            raise NotImplementedError(
                "The parameter `radius` of type `np.ndarray` is not supported to use with norm 1."
            )

        a_tmp = np.zeros(shape=(nb_points, nb_dims + 1))
        a_tmp[:, -1] = np.sqrt(np.random.uniform(0, radius**2, nb_points))

        for i in range(nb_points):
            a_tmp[i, 1:-1] = np.sort(np.random.uniform(0, a_tmp[i, -1], nb_dims - 1))

        res = (a_tmp[:, 1:] - a_tmp[:, :-1]) * np.random.choice(
            [-1, 1], (nb_points, nb_dims)
        )

    elif norm == 2:
        if isinstance(radius, np.ndarray):
            raise NotImplementedError(
                "The parameter `radius` of type `np.ndarray` is not supported to use with norm 2."
            )

        a_tmp = np.random.randn(nb_points, nb_dims)
        s_2 = np.sum(a_tmp**2, axis=1)
        base = (
            gammainc(nb_dims / 2.0, s_2 / 2.0) ** (1 / nb_dims) * radius / np.sqrt(s_2)
        )
        res = a_tmp * (np.tile(base, (nb_dims, 1))).T

    elif norm in [np.inf, "inf"]:
        if isinstance(radius, np.ndarray):
            radius = radius * np.ones(shape=(nb_points, nb_dims))

        res = np.random.uniform(-radius, radius, (nb_points, nb_dims))

    else:
        raise NotImplementedError(f"Norm {norm} not supported")

    return res


def to_categorical(
    labels: Union[np.ndarray, List[float]], nb_classes: Optional[int] = None
) -> np.ndarray:
    labels = np.array(labels, dtype=int)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical


def check_and_transform_label_format(
    labels: np.ndarray, nb_classes: Optional[int] = None, return_one_hot: bool = True
) -> np.ndarray:
    labels_return = labels

    if len(labels.shape) == 2 and labels.shape[1] > 1:  # multi-class, one-hot encoded
        if not return_one_hot:
            labels_return = np.argmax(labels, axis=1)
            labels_return = np.expand_dims(labels_return, axis=1)
    elif (
        len(labels.shape) == 2
        and labels.shape[1] == 1
        and nb_classes is not None
        and nb_classes > 2
    ):  # multi-class, index labels
        if return_one_hot:
            labels_return = to_categorical(labels, nb_classes)
        else:
            labels_return = np.expand_dims(labels, axis=1)
    elif (
        len(labels.shape) == 2
        and labels.shape[1] == 1
        and nb_classes is not None
        and nb_classes == 2
    ):  # binary, index labels
        if return_one_hot:
            labels_return = to_categorical(labels, nb_classes)
    elif len(labels.shape) == 1:  # index labels
        if return_one_hot:
            labels_return = to_categorical(labels, nb_classes)
        else:
            labels_return = np.expand_dims(labels, axis=1)
    else:
        raise ValueError(
            "Shape of labels not recognised."
            "Please provide labels in shape (nb_samples,) or (nb_samples, nb_classes)"
        )

    return labels_return


def get_labels_np_array(preds: np.ndarray) -> np.ndarray:
    if len(preds.shape) >= 2:
        preds_max = np.amax(preds, axis=1, keepdims=True)
    else:
        preds_max = np.round(preds)
    y = preds == preds_max
    y = y.astype(np.uint8)
    return y


def compute_success_array(
    classifier: "CLASSIFIER_TYPE",
    x_clean: np.ndarray,
    labels: np.ndarray,
    x_adv: np.ndarray,
    targeted: bool = False,
    batch_size: int = 1,
) -> float:

    adv_preds = classifier.predict(x_adv, batch_size=batch_size)
    if len(adv_preds.shape) >= 2:
        adv_preds = np.argmax(adv_preds, axis=1)
    else:
        adv_preds = np.round(adv_preds)
    if targeted:
        attack_success = adv_preds == np.argmax(labels, axis=1)
    else:
        preds = classifier.predict(x_clean, batch_size=batch_size)
        if len(preds.shape) >= 2:
            preds = np.argmax(preds, axis=1)
        else:
            preds = np.round(preds)
        attack_success = adv_preds != preds

    return attack_success


def compute_success(
    classifier: "CLASSIFIER_TYPE",
    x_clean: np.ndarray,
    labels: np.ndarray,
    x_adv: np.ndarray,
    targeted: bool = False,
    batch_size: int = 1,
) -> float:
    attack_success = compute_success_array(
        classifier, x_clean, labels, x_adv, targeted, batch_size
    )
    return np.sum(attack_success) / x_adv.shape[0]


def compute_accuracy(
    preds: np.ndarray, labels: np.ndarray, abstain: bool = True
) -> Tuple[float, float]:
    has_pred = np.sum(preds, axis=1)
    idx_pred = np.where(has_pred)[0]
    labels = np.argmax(labels[idx_pred], axis=1)
    num_correct = np.sum(np.argmax(preds[idx_pred], axis=1) == labels)
    coverage_rate = len(idx_pred) / preds.shape[0]

    if abstain:
        acc_rate = num_correct / preds[idx_pred].shape[0]
    else:
        acc_rate = num_correct / preds.shape[0]

    return acc_rate, coverage_rate


def load_cifar10(
    raw: bool = False,
):
    def load_batch(fpath: str) -> Tuple[np.ndarray, np.ndarray]:
        with open(fpath, "rb") as file_:
            if sys.version_info < (3,):
                content = six.moves.cPickle.load(file_)
            else:
                content = six.moves.cPickle.load(file_, encoding="bytes")
                content_decoded = {}
                for key, value in content.items():
                    content_decoded[key.decode("utf8")] = value
                content = content_decoded
        data = content["data"]
        labels = content["labels"]

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels

    path = get_file(
        "cifar-10-batches-py",
        extract=True,
        path=constants.DATASET_PATH,
        url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    )

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype=np.uint8)
    y_train = np.zeros((num_train_samples,), dtype=np.uint8)

    for i in range(1, 6):
        fpath = os.path.join(path, "data_batch_" + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000 : i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000 : i * 10000] = labels

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = load_batch(fpath)
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # Set channels last
    x_train = x_train.transpose((0, 2, 3, 1))
    x_test = x_test.transpose((0, 2, 3, 1))

    min_, max_ = 0.0, 255.0
    if not raw:
        min_, max_ = 0.0, 1.0
        x_train, y_train = preprocess(x_train, y_train, clip_values=(0, 255))
        x_test, y_test = preprocess(x_test, y_test, clip_values=(0, 255))

    return (x_train, y_train), (x_test, y_test), min_, max_


def load_mnist(
    raw: bool = False,
):
    path = get_file(
        "mnist.npz",
        path=constants.DATASET_PATH,
        url="https://s3.amazonaws.com/img-datasets/mnist.npz",
    )

    dict_mnist = np.load(path)
    x_train = dict_mnist["x_train"]
    y_train = dict_mnist["y_train"]
    x_test = dict_mnist["x_test"]
    y_test = dict_mnist["y_test"]
    dict_mnist.close()

    # Add channel axis
    min_, max_ = 0.0, 255.0
    if not raw:
        min_, max_ = 0.0, 1.0
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        x_train, y_train = preprocess(x_train, y_train)
        x_test, y_test = preprocess(x_test, y_test)

    return (x_train, y_train), (x_test, y_test), min_, max_


def _extract(full_path: str, path: str) -> bool:
    archive: Union[zipfile.ZipFile, tarfile.TarFile]
    if full_path.endswith("tar"):  # pragma: no cover
        if tarfile.is_tarfile(full_path):
            archive = tarfile.open(full_path, "r:")
    elif full_path.endswith("tar.gz"):  # pragma: no cover
        if tarfile.is_tarfile(full_path):
            archive = tarfile.open(full_path, "r:gz")
    elif full_path.endswith("zip"):  # pragma: no cover
        if zipfile.is_zipfile(full_path):
            archive = zipfile.ZipFile(full_path)  # pylint: disable=R1732
        else:
            return False
    else:
        return False

    try:
        archive.extractall(path)
    except (tarfile.TarError, RuntimeError, KeyboardInterrupt):  # pragma: no cover
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
        raise
    return True


def get_file(
    filename: str,
    url: str,
    path: Optional[str] = None,
    extract: bool = False,
    verbose: bool = False,
) -> str:
    if path is None:
        path_ = os.path.expanduser(constants.DATASET_PATH)
    else:
        path_ = os.path.expanduser(path)
    if not os.access(path_, os.W_OK):
        path_ = os.path.join("/tmp", ".xyz")

    if not os.path.exists(path_):
        os.makedirs(path_)

    if extract:
        extract_path = os.path.join(path_, filename)
        full_path = extract_path + ".tar.gz"
    else:
        full_path = os.path.join(path_, filename)

    # Determine if dataset needs downloading
    download = not os.path.exists(full_path)

    if download:
        logger.info("Downloading data from %s", url)
        error_msg = "URL fetch failure on {}: {} -- {}"
        try:
            try:
                from six.moves.urllib.error import HTTPError, URLError
                from six.moves.urllib.request import urlretrieve

                # The following two lines should prevent occasionally occurring
                # [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:847)
                import ssl

                ssl._create_default_https_context = (
                    ssl._create_unverified_context
                )  # pylint: disable=W0212

                if verbose:
                    with tqdm() as t_bar:
                        last_block = [0]

                        def progress_bar(
                            blocks: int = 1,
                            block_size: int = 1,
                            total_size: Optional[int] = None,
                        ):
                            """
                            :param blocks: Number of blocks transferred so far [default: 1].
                            :param block_size: Size of each block (in tqdm units) [default: 1].
                            :param total_size: Total size (in tqdm units). If [default: None] or -1, remains unchanged.
                            """
                            if total_size not in (None, -1):
                                t_bar.total = total_size
                            displayed = t_bar.update(
                                (blocks - last_block[0]) * block_size
                            )
                            last_block[0] = blocks
                            return displayed

                        urlretrieve(url, full_path, reporthook=progress_bar)
                else:
                    urlretrieve(url, full_path)

            except HTTPError as exception:  # pragma: no cover
                raise Exception(error_msg.format(url, exception.code, exception.msg)) from HTTPError  # type: ignore
            except URLError as exception:  # pragma: no cover
                raise Exception(error_msg.format(url, exception.errno, exception.reason)) from HTTPError  # type: ignore
        except (Exception, KeyboardInterrupt):  # pragma: no cover
            if os.path.exists(full_path):
                os.remove(full_path)
            raise

    if extract:
        if not os.path.exists(extract_path):
            _extract(full_path, path_)
        return extract_path

    return full_path


def preprocess(
    x: np.ndarray,
    y: np.ndarray,
    nb_classes: int = 10,
    clip_values: Optional["CLIP_VALUES_TYPE"] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scales `x` to [0, 1] and converts `y` to class categorical confidences.

    :param x: Data instances.
    :param y: Labels.
    :param nb_classes: Number of classes in dataset.
    :param clip_values: Original data range allowed value for features, either one respective scalar or one value per
           feature.
    :return: Rescaled values of `x`, `y`.
    """
    if clip_values is None:
        min_, max_ = np.amin(x), np.amax(x)
    else:
        min_, max_ = clip_values

    normalized_x = (x - min_) / (max_ - min_)
    categorical_y = to_categorical(y, nb_classes)

    return normalized_x, categorical_y
