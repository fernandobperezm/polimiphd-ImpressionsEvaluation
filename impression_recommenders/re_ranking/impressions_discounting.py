import enum
from typing import Optional, Callable, Any

import attrs
import numpy as np
import scipy.sparse as sp
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Recommender_utils import check_matrix
from recsys_framework_extensions.data.io import DataIO, attach_to_extended_json_decoder
from recsys_framework_extensions.recommenders.base import SearchHyperParametersBaseRecommender
from skopt.space import Real, Categorical


@attach_to_extended_json_decoder
class EImpressionsDiscountingFunctions(enum.Enum):
    LINEAR = "LINEAR"
    INVERSE = "INVERSE"
    EXPONENTIAL = "EXPONENTIAL"
    QUADRATIC = "QUADRATIC"


_all_enum_values = list(map(
    lambda en: en.value,
    EImpressionsDiscountingFunctions,
))


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersImpressionsDiscountingRecommender(SearchHyperParametersBaseRecommender):
    # Check UIM frequency and have a look at the scale range.
    reg_user_frequency: Real = attrs.field(
        default=Real(
            low=1e-5, high=1, prior="log-uniform", base=10,
        )
    )
    reg_uim_frequency: Real = attrs.field(
        default=Real(
            low=1e-5, high=1, prior="log-uniform", base=10,
        )
    )
    reg_uim_position: Real = attrs.field(
        default=Real(
            low=1e-5, high=1, prior="log-uniform", base=10,
        )
    )
    reg_uim_last_seen: Real = attrs.field(
        default=Real(
            low=1e-5, high=1, prior="log-uniform", base=10,
        )
    )

    func_user_frequency: Categorical = attrs.field(
        default=Categorical(_all_enum_values)
    )
    func_uim_frequency: Categorical = attrs.field(
        default=Categorical(_all_enum_values)
    )
    func_uim_position: Categorical = attrs.field(
        default=Categorical(_all_enum_values)
    )
    func_uim_last_seen: Categorical = attrs.field(
        default=Categorical(_all_enum_values)
    )


def _func_linear(x):
    return x


def _func_inverse(x):
    if sp.issparse(x):
        new_sp = x.copy()
        # Reciprocal cannot be applied on a sparse matrix yet, therefore, we only apply it to the data array and let
        # numpy know that the result should be placed in the memory view of the data.
        np.reciprocal(x.data, out=new_sp.data, where=x.data != 0.)
        return new_sp

    # NOTE: it is super important to provide an `out` array, given that using the `where` argument leaves the 0
    # values intact. Without an `out` array, this means that it will allocate a new array and will leave those
    # indices intact. The problem is that the content of that allocated array can be anything. To avoid this,
    # we pass an arrays of zeros, as we want to keep 0 values as zero.
    # See: https://stackoverflow.com/a/49461710
    new_arr = x.copy()
    np.reciprocal(x, out=new_arr, where=x != 0.)
    return new_arr


def _func_exponential(x):
    if sp.issparse(x):
        new_sp = x.copy()
        # Exp cannot be applied on a sparse matrix yet, therefore, we only apply it to the data array and let
        # numpy know that the result should be placed in the memory view of the data.
        np.exp(x.data, out=new_sp.data)
        return new_sp

    return np.exp(x)


def _func_quadratic(x):
    if sp.issparse(x):
        return x.power(2.)

    return np.power(x, 2)


_DICT_IMPRESSIONS_DISCOUNTING_FUNCTIONS: dict[EImpressionsDiscountingFunctions, Callable[[Any], Any]] = {
    EImpressionsDiscountingFunctions.LINEAR: _func_linear,
    EImpressionsDiscountingFunctions.INVERSE: _func_inverse,
    EImpressionsDiscountingFunctions.EXPONENTIAL: _func_exponential,
    EImpressionsDiscountingFunctions.QUADRATIC: _func_quadratic,
}


class ImpressionsDiscountingRecommender(BaseRecommender):
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
            URM_train=urm_train,
            verbose=True,
        )

        self._trained_recommender = trained_recommender

        self._user_frequency: np.ndarray = np.ediff1d(
            self.URM_train.indptr
        ).astype(
            np.float64
        ).reshape((self.n_users, 1))
        self._uim_frequency = check_matrix(X=uim_frequency, format="csr", dtype=np.float64)
        self._uim_position = check_matrix(X=uim_position, format="csr", dtype=np.float64)
        self._uim_last_seen = check_matrix(X=uim_last_seen, format="csr", dtype=np.float64)

        self._arr_user_frequency_scores = np.array([], dtype=np.float64)
        self._matrix_uim_frequency_scores = sp.csr_matrix(np.array([], dtype=np.float64))
        self._matrix_uim_position_scores = sp.csr_matrix(np.array([], dtype=np.float64))
        self._matrix_uim_last_seen_scores = sp.csr_matrix(np.array([], dtype=np.float64))

        self._max_discounting_score: float = 1.

        self._reg_user_frequency: float = 1.0
        self._reg_uim_frequency: float = 1.0
        self._reg_uim_position: float = 1.0
        self._reg_uim_last_seen: float = 1.0

        self._func_user_frequency:  EImpressionsDiscountingFunctions = EImpressionsDiscountingFunctions.LINEAR
        self._func_uim_frequency: EImpressionsDiscountingFunctions = EImpressionsDiscountingFunctions.LINEAR
        self._func_uim_position:  EImpressionsDiscountingFunctions = EImpressionsDiscountingFunctions.LINEAR
        self._func_uim_last_seen:  EImpressionsDiscountingFunctions = EImpressionsDiscountingFunctions.LINEAR

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
        arr_scores_relevance: np.ndarray = self._trained_recommender._compute_item_score(
            user_id_array=user_id_array,
            items_to_compute=items_to_compute,
        )

        # Compute the discounting scores for the current users. This results in a dense matrix array with shape
        # (num_score_users, num_score_items)
        arr_impressions_discounting_scores: np.ndarray = np.asarray(
            (
                self._arr_user_frequency_scores[user_id_array, :]
                + self._matrix_uim_frequency_scores[user_id_array, :]
                + self._matrix_uim_position_scores[user_id_array, :]
                + self._matrix_uim_last_seen_scores[user_id_array, :]
            )
            / self._max_discounting_score
        )

        assert (num_score_users, num_score_items) == arr_impressions_discounting_scores.shape
        assert (num_score_users, num_score_items) == arr_scores_relevance.shape

        new_item_scores = (
            arr_scores_relevance
            * arr_impressions_discounting_scores
        )

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
        assert reg_user_frequency > 0.
        assert reg_uim_frequency > 0.
        assert reg_uim_position > 0.
        assert reg_uim_last_seen > 0.

        self._reg_user_frequency = reg_user_frequency
        self._reg_uim_frequency = reg_uim_frequency
        self._reg_uim_position = reg_uim_position
        self._reg_uim_last_seen = reg_uim_last_seen

        self._func_user_frequency = EImpressionsDiscountingFunctions(func_user_frequency)
        self._func_uim_frequency = EImpressionsDiscountingFunctions(func_uim_frequency)
        self._func_uim_position = EImpressionsDiscountingFunctions(func_uim_position)
        self._func_uim_last_seen = EImpressionsDiscountingFunctions(func_uim_last_seen)

        # Compute the different arrays and matrices used in the calculation of the discounting function.
        selected_func = _DICT_IMPRESSIONS_DISCOUNTING_FUNCTIONS[self._func_user_frequency]
        self._arr_user_frequency_scores: np.ndarray = self._reg_user_frequency * selected_func(
            self._user_frequency
        )

        selected_func = _DICT_IMPRESSIONS_DISCOUNTING_FUNCTIONS[self._func_uim_frequency]
        self._matrix_uim_frequency_scores: np.ndarray = self._reg_uim_frequency * selected_func(
            self._uim_frequency
        )

        selected_func = _DICT_IMPRESSIONS_DISCOUNTING_FUNCTIONS[self._func_uim_position]
        self._matrix_uim_position_scores: np.ndarray = self._reg_uim_position * selected_func(
            self._uim_position
        )

        selected_func = _DICT_IMPRESSIONS_DISCOUNTING_FUNCTIONS[self._func_uim_last_seen]
        self._matrix_uim_last_seen_scores: np.ndarray = self._reg_uim_last_seen * selected_func(
            self._uim_last_seen
        )

        # `_arr_user_frequency_scores` is a column array that is added to the other sparse matrices.
        # However, doing this is space inefficient, as it breaks the sparsity of these matrices and converts them into
        # a dense array of possibly millions of users/items. We can compute the discounting factor on the fly when
        # computing the scores. To do that, we need the maximum discounting score.
        # We compute the maximum discounting score by assuming two things.
        # 1. All sparse matrices and arrays are non-negative.
        # 2. Computing the maximum over non-zero cells yields the same result as computing it on the dense matrix.

        # Step 1. Compute the row-wise maximum over all sparse matrices.
        sparse_matrices_scores: sp.csr_matrix = (
            self._matrix_uim_frequency_scores
            + self._matrix_uim_position_scores
            + self._matrix_uim_last_seen_scores
        )
        arr_max_matrices_score_by_row: np.ndarray = sparse_matrices_scores.max(axis=1).toarray()

        assert self._arr_user_frequency_scores.shape == arr_max_matrices_score_by_row.shape

        # Step 2. Sum both column arrays, `arr_max_matrices_score_by_row` is the maximum score for each user, while
        # `self._arr_user_frequency_scores` is the remaining score array to sum.
        arr_scores: np.ndarray = self._arr_user_frequency_scores + arr_max_matrices_score_by_row

        # Step 3. Compute the maximum of this new array.
        self._max_discounting_score: float = arr_scores.max(initial=0.) + 1e-6  # type: ignore

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        DataIO.s_save_data(
            folder_path=folder_path,
            file_name=file_name,
            data_dict_to_save={
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

                "_max_discounting_score": self._max_discounting_score,
            }
        )

    def load_model(
        self,
        folder_path: str,
        file_name: Optional[str] = None,
    ) -> None:
        super().load_model(
            folder_path=folder_path,
            file_name=file_name,
        )

        assert hasattr(self, "_arr_user_frequency_scores")
        assert hasattr(self, "_matrix_uim_frequency_scores")
        assert hasattr(self, "_matrix_uim_position_scores")
        assert hasattr(self, "_matrix_uim_last_seen_scores")

        assert hasattr(self, "_max_discounting_score")
