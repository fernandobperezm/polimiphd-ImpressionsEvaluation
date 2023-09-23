import enum
from typing import Optional, Callable, Any, Literal

import attrs
import numpy as np
import scipy.sparse as sp
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Recommender_utils import check_matrix
from recsys_framework_extensions.data.io import DataIO, attach_to_extended_json_decoder
from recsys_framework_extensions.recommenders.base import (
    SearchHyperParametersBaseRecommender,
    AbstractExtendedBaseRecommender,
)
import logging
from skopt.space import Real, Categorical


logger = logging.getLogger(__name__)


@attach_to_extended_json_decoder
class EImpressionsDiscountingFunctions(enum.Enum):
    LINEAR = "LINEAR"
    INVERSE = "INVERSE"
    EXPONENTIAL = "EXPONENTIAL"
    LOGARITHMIC = "LOGARITHMIC"
    QUADRATIC = "QUADRATIC"
    SQUARE_ROOT = "SQUARE_ROOT"


_all_enum_values = [en.value for en in list(EImpressionsDiscountingFunctions)]


_original_paper_enum_values = [
    en.value
    for en in [
        EImpressionsDiscountingFunctions.LINEAR,
        EImpressionsDiscountingFunctions.INVERSE,
        EImpressionsDiscountingFunctions.EXPONENTIAL,
        EImpressionsDiscountingFunctions.QUADRATIC,
    ]
]

_only_linear_enum_values = [
    en.value
    for en in [
        EImpressionsDiscountingFunctions.LINEAR,
    ]
]

T_SIGN = Literal[-1, 1]


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersImpressionsDiscountingRecommender(
    SearchHyperParametersBaseRecommender
):
    sign_user_frequency: Categorical = attrs.field(
        default=Categorical(
            categories=[-1, 1],
        )
    )
    sign_uim_frequency: Categorical = attrs.field(
        default=Categorical(
            categories=[-1, 1],
        )
    )
    sign_uim_position: Categorical = attrs.field(
        default=Categorical(
            categories=[-1, 1],
        )
    )
    sign_uim_last_seen: Categorical = attrs.field(
        default=Categorical(
            categories=[-1, 1],
        )
    )

    reg_user_frequency: Real = attrs.field(
        default=Real(
            low=1e-5,
            high=1,
            prior="log-uniform",
            base=10,
        )
    )
    reg_uim_frequency: Real = attrs.field(
        default=Real(
            low=1e-5,
            high=1,
            prior="log-uniform",
            base=10,
        )
    )
    reg_uim_position: Real = attrs.field(
        default=Real(
            low=1e-5,
            high=1,
            prior="log-uniform",
            base=10,
        )
    )
    reg_uim_last_seen: Real = attrs.field(
        default=Real(
            low=1e-5,
            high=1,
            prior="log-uniform",
            base=10,
        )
    )

    func_user_frequency: Categorical = attrs.field(
        default=Categorical(_all_enum_values)
    )
    func_uim_frequency: Categorical = attrs.field(default=Categorical(_all_enum_values))
    func_uim_position: Categorical = attrs.field(default=Categorical(_all_enum_values))
    func_uim_last_seen: Categorical = attrs.field(default=Categorical(_all_enum_values))


DICT_SEARCH_CONFIGS = {
    "REPRODUCIBILITY_ORIGINAL_PAPER": SearchHyperParametersImpressionsDiscountingRecommender(
        sign_user_frequency=Categorical(categories=[1]),
        sign_uim_frequency=Categorical(categories=[1]),
        sign_uim_position=Categorical(categories=[1]),
        sign_uim_last_seen=Categorical(categories=[1]),
        func_user_frequency=Categorical(categories=_original_paper_enum_values),
        func_uim_frequency=Categorical(categories=_original_paper_enum_values),
        func_uim_position=Categorical(categories=_original_paper_enum_values),
        func_uim_last_seen=Categorical(categories=_original_paper_enum_values),
    ),
    "ABLATION_ONLY_IMPRESSIONS_FEATURES": SearchHyperParametersImpressionsDiscountingRecommender(
        sign_user_frequency=Categorical(categories=[1]),
        reg_user_frequency=Categorical(categories=[0.0]),
        func_user_frequency=Categorical(categories=_only_linear_enum_values),
    ),
    "ABLATION_ONLY_UIM_FREQUENCY": SearchHyperParametersImpressionsDiscountingRecommender(
        sign_user_frequency=Categorical(categories=[1]),
        sign_uim_position=Categorical(categories=[1]),
        sign_uim_last_seen=Categorical(categories=[1]),
        reg_user_frequency=Categorical(categories=[0.0]),
        reg_uim_position=Categorical(categories=[0.0]),
        reg_uim_last_seen=Categorical(categories=[0.0]),
        func_user_frequency=Categorical(categories=_only_linear_enum_values),
        func_uim_position=Categorical(categories=_only_linear_enum_values),
        func_uim_last_seen=Categorical(categories=_only_linear_enum_values),
    ),
    "SIGNAL_ANALYSIS_NEGATIVE_ABLATION_ONLY_UIM_FREQUENCY": SearchHyperParametersImpressionsDiscountingRecommender(
        sign_uim_frequency=Categorical(categories=[-1]),
        sign_user_frequency=Categorical(categories=[1]),
        sign_uim_position=Categorical(categories=[1]),
        sign_uim_last_seen=Categorical(categories=[1]),
        reg_user_frequency=Categorical(categories=[0.0]),
        reg_uim_position=Categorical(categories=[0.0]),
        reg_uim_last_seen=Categorical(categories=[0.0]),
        func_user_frequency=Categorical(categories=_only_linear_enum_values),
        func_uim_position=Categorical(categories=_only_linear_enum_values),
        func_uim_last_seen=Categorical(categories=_only_linear_enum_values),
    ),
    "SIGNAL_ANALYSIS_POSITIVE_ABLATION_ONLY_UIM_FREQUENCY": SearchHyperParametersImpressionsDiscountingRecommender(
        sign_uim_frequency=Categorical(categories=[1]),
        sign_user_frequency=Categorical(categories=[1]),
        sign_uim_position=Categorical(categories=[1]),
        sign_uim_last_seen=Categorical(categories=[1]),
        reg_user_frequency=Categorical(categories=[0.0]),
        reg_uim_position=Categorical(categories=[0.0]),
        reg_uim_last_seen=Categorical(categories=[0.0]),
        func_user_frequency=Categorical(categories=_only_linear_enum_values),
        func_uim_position=Categorical(categories=_only_linear_enum_values),
        func_uim_last_seen=Categorical(categories=_only_linear_enum_values),
    ),
}


IMPRESSIONS_DISCOUNTING_HYPER_PARAMETER_SEARCH_CONFIGURATIONS = {
    "SIGNAL_ANALYSIS_SIGN_ALL_POSITIVE": SearchHyperParametersImpressionsDiscountingRecommender(
        sign_uim_frequency=Categorical(categories=[1]),
        sign_user_frequency=Categorical(categories=[1]),
        sign_uim_position=Categorical(categories=[1]),
        sign_uim_last_seen=Categorical(categories=[1]),
    ),
    "SIGNAL_ANALYSIS_SIGN_ALL_NEGATIVE": SearchHyperParametersImpressionsDiscountingRecommender(
        sign_uim_frequency=Categorical(categories=[-1]),
        sign_user_frequency=Categorical(categories=[-1]),
        sign_uim_position=Categorical(categories=[-1]),
        sign_uim_last_seen=Categorical(categories=[-1]),
    ),
}


def _func_linear(x):
    return x


def _func_inverse(x):
    if sp.issparse(x):
        new_sp = x.copy()
        # Reciprocal cannot be applied on a sparse matrix yet, therefore, we only apply it to the data array and let
        # numpy know that the result should be placed in the memory view of the data.
        np.reciprocal(x.data, out=new_sp.data, where=x.data != 0.0)
        return new_sp

    # NOTE: it is super important to provide an `out` array, given that using the `where` argument leaves the 0
    # values intact. Without an `out` array, this means that it will allocate a new array and will leave those
    # indices intact. The problem is that the content of that allocated array can be anything. To avoid this,
    # we pass an arrays of zeros, as we want to keep 0 values as zero.
    # See: https://stackoverflow.com/a/49461710
    new_arr = x.copy()
    np.reciprocal(x, out=new_arr, where=x != 0.0)
    return new_arr


def _func_exponential(x):
    if sp.issparse(x):
        exp_data: np.ndarray = x.data.copy()

    else:
        exp_data = x.copy()

    exp_data = np.exp(exp_data.astype(dtype=np.float32))

    mask_pos_inf: np.ndarray = np.isinf(exp_data)
    if mask_pos_inf.any():
        # This catches overflow errors, as the data enters as positive floats, then the exponential function will
        # transform overflows to positive infinite.
        # The problem is that having infinite in the scores causes items to not be recommended. Therefore,
        # we correct infinite values to be the maximum noninf & nonnan + 1.
        mask_not_pos_inf_on_exp_data_plus_1 = np.logical_not(np.isinf(exp_data + 1))
        max_value_non_pos_inf = (
            np.nanmax(
                exp_data[mask_not_pos_inf_on_exp_data_plus_1],
            )
            + 1
        )

        exp_data[mask_pos_inf] = max_value_non_pos_inf

    exp_data = exp_data.astype(dtype=np.float64)
    # Exp cannot be applied on a sparse matrix yet, therefore, we only apply it to the data array and let
    # numpy know that the result should be placed in the memory view of the data.
    if sp.issparse(x):
        new_sp: sp.csr_matrix = x.copy()
        new_sp.data = exp_data

        return new_sp
    else:
        return exp_data


def _func_logarithm(x):
    if sp.issparse(x):
        new_sp: sp.csr_matrix = x.copy()
        # Reciprocal cannot be applied on a sparse matrix yet, therefore, we only apply it to the data array and let
        # numpy know that the result should be placed in the memory view of the data.
        np.log(x.data, out=new_sp.data, where=x.data != 0.0)
        return new_sp

    # NOTE: it is super important to provide an `out` array, given that using the `where` argument leaves the 0
    # values intact. Without an `out` array, this means that it will allocate a new array and will leave those
    # indices intact. The problem is that the content of that allocated array can be anything. To avoid this,
    # we pass an arrays of zeros, as we want to keep 0 values as zero.
    # See: https://stackoverflow.com/a/49461710
    new_arr = x.copy()
    np.log(x, out=new_arr, where=x != 0.0)

    return new_arr


def _func_quadratic(x):
    if sp.issparse(x):
        return x.power(2.0)

    return np.power(x, 2.0)


def _func_sqrt(x):
    if sp.issparse(x):
        new_sp: sp.csr_matrix = x.copy()
        # Reciprocal cannot be applied on a sparse matrix yet, therefore, we only apply it to the data array and let
        # numpy know that the result should be placed in the memory view of the data.
        np.sqrt(x.data, out=new_sp.data, where=x.data >= 0.0)
        return new_sp

    # NOTE: it is super important to provide an `out` array, given that using the `where` argument leaves the 0
    # values intact. Without an `out` array, this means that it will allocate a new array and will leave those
    # indices intact. The problem is that the content of that allocated array can be anything. To avoid this,
    # we pass an arrays of zeros, as we want to keep 0 values as zero.
    # See: https://stackoverflow.com/a/49461710
    new_arr = x.copy()
    np.sqrt(x, out=new_arr, where=x >= 0.0)

    return new_arr


_DICT_IMPRESSIONS_DISCOUNTING_FUNCTIONS: dict[
    EImpressionsDiscountingFunctions, Callable[[Any], Any]
] = {
    EImpressionsDiscountingFunctions.LINEAR: _func_linear,
    EImpressionsDiscountingFunctions.INVERSE: _func_inverse,
    EImpressionsDiscountingFunctions.EXPONENTIAL: _func_exponential,
    EImpressionsDiscountingFunctions.LOGARITHMIC: _func_logarithm,
    EImpressionsDiscountingFunctions.QUADRATIC: _func_quadratic,
    EImpressionsDiscountingFunctions.SQUARE_ROOT: _func_sqrt,
}


class ImpressionsDiscountingRecommender(AbstractExtendedBaseRecommender):
    RECOMMENDER_NAME = "ImpressionsDiscountingRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        uim_position: sp.csr_matrix,
        uim_last_seen: sp.csr_matrix,
        trained_recommender: BaseRecommender,
        **kwargs,
    ):
        super().__init__(
            urm_train=urm_train,
        )

        self._trained_recommender = trained_recommender

        self._user_frequency: np.ndarray = (
            np.ediff1d(self.URM_train.indptr)
            .astype(np.float64)
            .reshape((self.n_users, 1))
        )
        self._uim_frequency = check_matrix(
            X=uim_frequency, format="csr", dtype=np.float64
        )
        self._uim_position = check_matrix(
            X=uim_position, format="csr", dtype=np.float64
        )
        self._uim_last_seen = check_matrix(
            X=uim_last_seen, format="csr", dtype=np.float64
        )

        self._arr_user_frequency_scores = np.array([], dtype=np.float64)
        self._matrix_uim_frequency_scores = sp.csr_matrix(
            np.array([], dtype=np.float64)
        )
        self._matrix_uim_position_scores = sp.csr_matrix(np.array([], dtype=np.float64))
        self._matrix_uim_last_seen_scores = sp.csr_matrix(
            np.array([], dtype=np.float64)
        )

        self._sign_user_frequency: T_SIGN = 1
        self._sign_uim_frequency: T_SIGN = 1
        self._sign_uim_position: T_SIGN = 1
        self._sign_uim_last_seen: T_SIGN = 1

        self._reg_user_frequency: float = 1.0
        self._reg_uim_frequency: float = 1.0
        self._reg_uim_position: float = 1.0
        self._reg_uim_last_seen: float = 1.0

        self._func_user_frequency: EImpressionsDiscountingFunctions = (
            EImpressionsDiscountingFunctions.LINEAR
        )
        self._func_uim_frequency: EImpressionsDiscountingFunctions = (
            EImpressionsDiscountingFunctions.LINEAR
        )
        self._func_uim_position: EImpressionsDiscountingFunctions = (
            EImpressionsDiscountingFunctions.LINEAR
        )
        self._func_uim_last_seen: EImpressionsDiscountingFunctions = (
            EImpressionsDiscountingFunctions.LINEAR
        )

        self.RECOMMENDER_NAME = (
            f"ImpressionsDiscountingRecommender_{trained_recommender.RECOMMENDER_NAME}"
        )

    def _compute_item_score(
        self,
        user_id_array: list[int],
        items_to_compute: Optional[list[int]] = None,
    ) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray
            A (M, N) numpy array that contains the score for each user-item pair.

        """
        assert user_id_array is not None
        assert len(user_id_array) > 0

        num_score_users: int = len(user_id_array)
        num_score_items: int = self.URM_train.shape[1]

        # Dense array of shape (M,N) where M is len(user_id_array) and N is the total number of users in the dataset.
        arr_scores_relevance: np.ndarray = (
            self._trained_recommender._compute_item_score(
                user_id_array=user_id_array,
                items_to_compute=items_to_compute,
            ).astype(np.float64)
        )

        # Compute the discounting scores for the current users. This results in a dense matrix array with shape
        # (num_score_users, num_score_items).
        # Eq. 8 In the original paper. The paper also says that the discounting score should be divided by the maximum
        # score in the dataset. Mathematically, however, this is not needed.
        arr_impressions_discounting_scores: np.ndarray = np.asarray(
            (
                1
                + self._arr_user_frequency_scores[user_id_array, :]
                + self._matrix_uim_frequency_scores[user_id_array, :]
                + self._matrix_uim_position_scores[user_id_array, :]
                + self._matrix_uim_last_seen_scores[user_id_array, :]
            )
        ).astype(np.float64)

        assert (
            num_score_users,
            num_score_items,
        ) == arr_impressions_discounting_scores.shape
        assert (num_score_users, num_score_items) == arr_scores_relevance.shape

        new_item_scores = arr_scores_relevance * arr_impressions_discounting_scores

        # If we are computing scores to a specific set of items, then we must set items outside this set to np.NINF,
        # so they are not
        if items_to_compute is not None:
            arr_mask_items = np.zeros_like(new_item_scores, dtype=np.bool8)
            arr_mask_items[:, items_to_compute] = True

            # If the item is in `items_to_compute`, then keep the value from `new_item_scores`.
            # Else, set to -inf.
            new_item_scores = np.where(
                arr_mask_items,
                new_item_scores,
                np.NINF,
            )

        assert (num_score_users, num_score_items) == new_item_scores.shape

        return new_item_scores

    def fit(
        self,
        sign_user_frequency: T_SIGN,
        sign_uim_frequency: T_SIGN,
        sign_uim_position: T_SIGN,
        sign_uim_last_seen: T_SIGN,
        reg_user_frequency: float,
        reg_uim_frequency: float,
        reg_uim_position: float,
        reg_uim_last_seen: float,
        func_user_frequency: str,
        func_uim_frequency: str,
        func_uim_position: str,
        func_uim_last_seen: str,
        **kwargs,
    ):
        assert sign_user_frequency == -1 or sign_user_frequency == 1
        assert sign_uim_frequency == -1 or sign_uim_frequency == 1
        assert sign_uim_position == -1 or sign_uim_position == 1
        assert sign_uim_last_seen == -1 or sign_uim_last_seen == 1

        assert reg_user_frequency >= 0.0
        assert reg_uim_frequency >= 0.0
        assert reg_uim_position >= 0.0
        assert reg_uim_last_seen >= 0.0

        self._sign_user_frequency = sign_user_frequency
        self._sign_uim_frequency = sign_uim_frequency
        self._sign_uim_position = sign_uim_position
        self._sign_uim_last_seen = sign_uim_last_seen

        self._reg_user_frequency = reg_user_frequency
        self._reg_uim_frequency = reg_uim_frequency
        self._reg_uim_position = reg_uim_position
        self._reg_uim_last_seen = reg_uim_last_seen

        self._func_user_frequency = EImpressionsDiscountingFunctions(
            func_user_frequency
        )
        self._func_uim_frequency = EImpressionsDiscountingFunctions(func_uim_frequency)
        self._func_uim_position = EImpressionsDiscountingFunctions(func_uim_position)
        self._func_uim_last_seen = EImpressionsDiscountingFunctions(func_uim_last_seen)

        # Compute the different arrays and matrices used in the calculation of the discounting function.
        selected_func = _DICT_IMPRESSIONS_DISCOUNTING_FUNCTIONS[
            self._func_user_frequency
        ]
        arr_user_frequency_scores: np.ndarray = (
            self._sign_user_frequency
            * self._reg_user_frequency
            * selected_func(self._user_frequency)
        )

        selected_func = _DICT_IMPRESSIONS_DISCOUNTING_FUNCTIONS[
            self._func_uim_frequency
        ]
        matrix_uim_frequency_scores: sp.csr_matrix = (
            self._sign_uim_frequency
            * self._reg_uim_frequency
            * selected_func(self._uim_frequency)
        )

        selected_func = _DICT_IMPRESSIONS_DISCOUNTING_FUNCTIONS[self._func_uim_position]
        matrix_uim_position_scores: sp.csr_matrix = (
            self._sign_uim_position
            * self._reg_uim_position
            * selected_func(self._uim_position)
        )

        selected_func = _DICT_IMPRESSIONS_DISCOUNTING_FUNCTIONS[
            self._func_uim_last_seen
        ]
        matrix_uim_last_seen_scores: sp.csr_matrix = (
            self._sign_uim_last_seen
            * self._reg_uim_last_seen
            * selected_func(self._uim_last_seen)
        )

        # Save all arrays and matrices
        self._arr_user_frequency_scores = arr_user_frequency_scores
        self._matrix_uim_frequency_scores = matrix_uim_frequency_scores
        self._matrix_uim_position_scores = matrix_uim_position_scores
        self._matrix_uim_last_seen_scores = matrix_uim_last_seen_scores

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        DataIO.s_save_data(
            folder_path=folder_path,
            file_name=file_name,
            data_dict_to_save={
                "_sign_user_frequency": self._reg_user_frequency,
                "_sign_uim_frequency": self._reg_uim_frequency,
                "_sign_uim_position": self._reg_uim_position,
                "_sign_uim_last_seen": self._reg_uim_last_seen,
                "_reg_user_frequency": self._reg_user_frequency,
                "_reg_uim_frequency": self._reg_uim_frequency,
                "_reg_uim_position": self._reg_uim_position,
                "_reg_uim_last_seen": self._reg_uim_last_seen,
                "_func_user_frequency": self._func_user_frequency,
                "_func_uim_frequency": self._func_uim_frequency,
                "_func_uim_position": self._func_uim_position,
                "_func_uim_last_seen": self._func_uim_last_seen,
                "_arr_user_frequency_scores": self._arr_user_frequency_scores,
                "_matrix_uim_frequency_scores": self._matrix_uim_frequency_scores,
                "_matrix_uim_position_scores": self._matrix_uim_position_scores,
                "_matrix_uim_last_seen_scores": self._matrix_uim_last_seen_scores,
            },
        )

    def validate_load_trained_recommender(self, *args, **kwargs) -> None:
        assert hasattr(self, "_sign_user_frequency")
        assert hasattr(self, "_sign_uim_frequency")
        assert hasattr(self, "_sign_uim_position")
        assert hasattr(self, "_sign_uim_last_seen")

        assert hasattr(self, "_reg_user_frequency")
        assert hasattr(self, "_reg_uim_frequency")
        assert hasattr(self, "_reg_uim_position")
        assert hasattr(self, "_reg_uim_last_seen")

        assert hasattr(self, "_func_user_frequency")
        assert hasattr(self, "_func_uim_frequency")
        assert hasattr(self, "_func_uim_position")
        assert hasattr(self, "_func_uim_last_seen")

        assert hasattr(self, "_arr_user_frequency_scores")
        assert hasattr(self, "_matrix_uim_frequency_scores")
        assert hasattr(self, "_matrix_uim_position_scores")
        assert hasattr(self, "_matrix_uim_last_seen_scores")
