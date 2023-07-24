import copy
import logging
from typing import Optional, Literal

import attrs
import numba as nb
import numpy as np
import scipy.sparse as sp

from Recommenders.BaseMatrixFactorizationRecommender import (
    BaseMatrixFactorizationRecommender,
)
from Recommenders.Incremental_Training_Early_Stopping import (
    Incremental_Training_Early_Stopping,
)
from Recommenders.Recommender_utils import check_matrix
from recsys_framework_extensions.data.io import DataIO
from recsys_framework_extensions.recommenders.base import (
    SearchHyperParametersBaseRecommender,
)
from skopt.space import Integer, Categorical, Real

from impressions_evaluation.impression_recommenders.matrix_factorization.jax.model_sfc import (
    SFCModel,
    create_dict_binned_frequency,
)

logger = logging.getLogger(__name__)


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersSFCRecommender(SearchHyperParametersBaseRecommender):
    epochs: Categorical = attrs.field(
        default=Categorical(  # The paper does not specify.
            [1],
        )
    )
    frequency_num_bins: Categorical = attrs.field(
        default=Categorical(  # The paper uses 26
            [26],
        )
    )
    frequency_mode: Categorical = attrs.field(
        default=Categorical(  # The paper uses global or two content-based variants of item.
            [
                "global",
                "user",
                "item",
            ],
        )
    )
    batch_size: Integer = attrs.field(
        default=Integer(  # The paper does not specify ranges.
            low=2**7,  # 128
            high=2**14,  # 16384
            prior="uniform",
        ),
    )
    embedding_size: Integer = attrs.field(
        default=Integer(  # The paper does not specify ranges.
            low=2**0,
            high=2**10,
            prior="uniform",
        ),
    )
    learning_rate: Real = attrs.field(
        default=Real(  # The paper does not specify ranges.
            low=1e-5,
            high=1e-2,
            prior="log-uniform",
        )
    )
    l2_reg: Real = attrs.field(
        default=Real(  # The paper does not specify ranges.
            low=1e-5,
            high=1e-2,
            prior="log-uniform",
        )
    )
    scheduler_alpha: Real = attrs.field(
        default=Real(  # The paper does not specify ranges.
            low=1e-5,
            high=1e-2,
            prior="log-uniform",
        )
    )
    scheduler_beta: Real = attrs.field(
        default=Real(  # The paper does not specify ranges.
            low=1e-5,
            high=1e-2,
            prior="log-uniform",
        )
    )


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersSFCRecommenderDEBUG(SearchHyperParametersSFCRecommender):
    frequency_mode: Categorical = attrs.field(
        default=Categorical(
            ["user"],
        )
    )
    batch_size: Integer = attrs.field(
        default=Categorical(
            [9883],
        )
    )
    embedding_size: Integer = attrs.field(
        default=Categorical(
            [724],
        )
    )


FREQUENCY_MODE = Literal["global", "user", "item"]


@nb.njit
def _get_frequency_embedding_index(
    frequency_mode: FREQUENCY_MODE,
    user_id: int,
    item_id: int,
) -> int:
    if frequency_mode == "global":
        return 0
    elif frequency_mode == "user":
        return user_id
    elif frequency_mode == "item":
        return item_id

    # return value needed for numba
    return 0  # type: ignore


@nb.njit
def _get_frequency_num_embeddings(
    frequency_mode: FREQUENCY_MODE,
    num_users: int,
    num_items: int,
) -> int:
    if frequency_mode == "global":
        return 1
    elif frequency_mode == "user":
        return num_users
    elif frequency_mode == "item":
        return num_items

    # return value needed for numba
    return 1  # type: ignore


@nb.njit
def _compute_array_frequencies(
    arr_user_ids: np.ndarray,
    arr_item_ids: np.ndarray,
    arr_frequency_factors: np.ndarray,
    dict_frequency: dict[tuple[int, int], int],
    num_items: int,
    frequency_mode: FREQUENCY_MODE,
) -> np.ndarray:
    num_users = len(arr_user_ids)

    arr_frequencies = np.zeros(
        (num_users, num_items),
        dtype=np.float32,
    )

    for idx_user, user in enumerate(arr_user_ids):
        for item in arr_item_ids:
            freq = (
                dict_frequency[(user, item)]
                if (user, item) in dict_frequency
                else int(0)
            )

            idx = _get_frequency_embedding_index(
                frequency_mode=frequency_mode,
                user_id=user,
                item_id=item,
            )

            factor = arr_frequency_factors[idx, freq]

            arr_frequencies[idx_user, item] = factor

    return arr_frequencies


@nb.njit
def _compute_soft_frequency_capping_score(
    arr_item_scores: np.ndarray,
    arr_user_ids: np.ndarray,
    arr_item_ids: np.ndarray,
    arr_global_bias: np.ndarray,
    arr_user_factors: np.ndarray,
    arr_item_factors: np.ndarray,
    arr_frequency_factors: np.ndarray,
) -> np.ndarray:
    assert arr_global_bias.shape == (1,)
    assert arr_user_factors.shape[1] == arr_item_factors.shape[1]
    assert arr_frequency_factors.shape == arr_item_scores.shape

    arr_item_scores[:, arr_item_ids] = (
        arr_global_bias[0]
        + np.dot(arr_user_factors[arr_user_ids, :], arr_item_factors.T)
        + arr_frequency_factors
    )[:, arr_item_ids]

    return arr_item_scores


class SoftFrequencyCappingRecommender(
    BaseMatrixFactorizationRecommender,
    Incremental_Training_Early_Stopping,
):
    """"""

    RECOMMENDER_NAME = "SFCRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        use_gpu: bool = True,
        verbose: bool = True,
    ):
        super().__init__(
            URM_train=urm_train,
            verbose=verbose,
        )
        self.urm_train = check_matrix(urm_train.copy(), "csr", dtype=np.float32)
        self.urm_train.eliminate_zeros()

        self.uim_train = check_matrix(uim_train.copy(), "csr", dtype=np.float32)
        self.uim_train.eliminate_zeros()

        self.uim_frequency = check_matrix(uim_frequency.copy(), "csr", dtype=np.int32)
        self.uim_frequency.eliminate_zeros()

        self.dict_binned_frequency: dict[tuple[int, int], int] = {}

        self.BIAS_factors: np.ndarray = np.array([])
        self.USER_factors: np.ndarray = np.array([])
        self.ITEM_factors: np.ndarray = np.array([])
        self.FREQUENCY_factors: np.ndarray = np.array([])

        self.BIAS_factors_best: np.ndarray = np.array([])
        self.USER_factors_best: np.ndarray = np.array([])
        self.ITEM_factors_best: np.ndarray = np.array([])
        self.FREQUENCY_factors_best: np.ndarray = np.array([])

        self.model_state: dict[str, np.ndarray] = {}
        self.model_state_best: dict[str, np.ndarray] = {}

        self.model: Optional[SFCModel] = None

        self.frequency_mode: Optional[FREQUENCY_MODE] = None
        self.frequency_num_bins: Optional[int] = None
        self.frequency_num_embeddings: Optional[int] = None

        self.epochs: Optional[int] = None
        self.batch_size: Optional[int] = None
        self.embedding_size: Optional[int] = None

        self.learning_rate: Optional[float] = None
        self.l2_reg: Optional[float] = None
        self.initial_step_size: Optional[float] = None
        self.scheduler_alpha: Optional[float] = None
        self.scheduler_beta: Optional[float] = None

        self.use_gpu = use_gpu

    def _compute_item_score(
        self,
        user_id_array: np.ndarray,
        items_to_compute: Optional[np.ndarray] = None,
    ):
        """
        USER_factors is n_users x n_factors
        ITEM_factors is n_items x n_factors

        The prediction for cold users will always be -inf for ALL items

        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        assert (
            self.USER_factors.shape[1] == self.ITEM_factors.shape[1]
        ), "{}: User and Item factors have inconsistent shape".format(
            self.RECOMMENDER_NAME
        )

        assert self.USER_factors.shape[0] > np.max(
            user_id_array
        ), "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
            self.RECOMMENDER_NAME, self.USER_factors.shape[0], np.max(user_id_array)
        )

        assert self.frequency_mode is not None
        assert self.frequency_num_bins is not None

        arr_item_scores = (
            np.ones(
                (len(user_id_array), self.n_items),
                dtype=np.float32,
            )
            * np.NINF
        )

        arr_user_ids = np.asarray(user_id_array, dtype=np.int32)
        arr_item_ids = (
            np.asarray(items_to_compute, dtype=np.int32)
            if items_to_compute is not None
            else np.arange(start=0, stop=self.n_items, dtype=np.int32)
        )

        arr_frequency_factors = _compute_array_frequencies(
            arr_user_ids=arr_user_ids,
            arr_item_ids=arr_item_ids,
            arr_frequency_factors=self.FREQUENCY_factors,
            dict_frequency=self.dict_binned_frequency,
            num_items=self.n_items,
            frequency_mode=self.frequency_mode,
        )

        arr_item_scores = _compute_soft_frequency_capping_score(
            arr_item_scores=arr_item_scores,
            arr_user_ids=arr_user_ids,
            arr_item_ids=arr_item_ids,
            arr_global_bias=self.BIAS_factors,
            arr_user_factors=self.USER_factors,
            arr_item_factors=self.ITEM_factors,
            arr_frequency_factors=arr_frequency_factors,
        )

        return arr_item_scores

    def fit(
        self,
        *,
        epochs: int,
        frequency_mode: FREQUENCY_MODE,
        frequency_num_bins: int,
        batch_size: int,
        embedding_size: int,
        learning_rate: float,
        l2_reg: float,
        scheduler_alpha: float,
        scheduler_beta: float,
        **earlystopping_kwargs,
    ):
        self.epochs = int(epochs)

        self.frequency_mode = frequency_mode
        self.frequency_num_bins = int(frequency_num_bins)
        self.frequency_num_embeddings = _get_frequency_num_embeddings(
            frequency_mode=self.frequency_mode,
            num_users=self.n_users,
            num_items=self.n_items,
        )

        self.batch_size = int(batch_size)
        self.embedding_size = int(embedding_size)

        self.learning_rate = float(learning_rate)
        self.l2_reg = float(l2_reg)

        self.scheduler_alpha = float(scheduler_alpha)
        self.scheduler_beta = float(scheduler_beta)

        uim_frequency_coo = self.uim_frequency.tocoo(copy=True)
        self.dict_binned_frequency = create_dict_binned_frequency(
            uim_frequency_coo_row=uim_frequency_coo.row,
            uim_frequency_coo_col=uim_frequency_coo.col,
            uim_frequency_coo_data=uim_frequency_coo.data,
            num_bins=self.frequency_num_bins,
        )

        self.model = SFCModel(
            urm_train=self.urm_train,
            uim_train=self.uim_train,
            uim_frequency=self.uim_frequency,
            num_users=self.n_users,
            num_items=self.n_items,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            scheduler_beta=self.scheduler_beta,
            scheduler_alpha=self.scheduler_alpha,
            frequency_mode=self.frequency_mode,
            frequency_num_bins=self.frequency_num_bins,
            frequency_num_embeddings=self.frequency_num_embeddings,
            embedding_size=self.embedding_size,
            use_gpu=self.use_gpu,
        )

        ############################################################
        ### This is a standard training with early stopping part ###
        ############################################################

        # Initializing for epoch 0
        self._prepare_model_for_validation()
        self._update_best_model()
        self._train_with_early_stopping(
            epochs,
            algorithm_name=self.RECOMMENDER_NAME,
            **earlystopping_kwargs,
        )
        self._print("Training complete")

        self.BIAS_factors = self.BIAS_factors_best.copy()
        self.USER_factors = self.USER_factors_best.copy()
        self.ITEM_factors = self.ITEM_factors_best.copy()
        self.FREQUENCY_factors = self.FREQUENCY_factors_best.copy()
        self.model_state = self.model_state_best.copy()

    def _prepare_model_for_validation(self):
        assert self.model is not None
        assert self.frequency_num_embeddings is not None

        # Expected shape: (1, )
        self.BIAS_factors = np.asarray(
            self.model.embedding_bias,
            dtype=np.float32,
        )

        # Expected shape: (num_users, num_factors)
        self.USER_factors = np.asarray(
            self.model.embedding_user,
            dtype=np.float32,
        )
        # Expected shape: (num_items, num_factors)
        self.ITEM_factors = np.asarray(
            self.model.embedding_item,
            dtype=np.float32,
        )
        # expected shape: (frequency_num_embeddings, frequency_num_bins)
        self.FREQUENCY_factors = np.asarray(
            self.model.embedding_frequencies,
            dtype=np.float32,
        )

    def _update_best_model(self):
        self.BIAS_factors_best = self.BIAS_factors.copy()
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self.FREQUENCY_factors_best = self.FREQUENCY_factors.copy()
        self.model_state_best = copy.deepcopy(self.model_state)

    def _run_epoch(self, num_epoch: int):
        assert self.model is not None
        epoch_loss = self.model.run_epoch()

        self._print("Loss {:.2E}".format(epoch_loss))

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {
            "BIAS_factors": self.BIAS_factors,
            "USER_factors": self.USER_factors,
            "ITEM_factors": self.ITEM_factors,
            "FREQUENCY_factors": self.FREQUENCY_factors,
            "model_state": self.model_state,
        }

        DataIO.s_save_data(
            folder_path=folder_path,
            file_name=file_name,
            data_dict_to_save=data_dict_to_save,
        )

        self._print("Saving complete")


__all__ = [
    "SoftFrequencyCappingRecommender",
    "SearchHyperParametersSFCRecommender",
    "SearchHyperParametersSFCRecommenderDEBUG",
]
