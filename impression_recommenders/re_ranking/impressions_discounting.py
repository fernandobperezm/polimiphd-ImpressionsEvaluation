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


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersCyclingRecommender(SearchHyperParametersBaseRecommender):
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
        default=Categorical(
            list(EImpressionsDiscountingFunctions),
        )
    )
    func_uim_frequency: Categorical = attrs.field(
        default=Categorical(
            list(EImpressionsDiscountingFunctions),
        )
    )
    func_uim_position: Categorical = attrs.field(
        default=Categorical(
            list(EImpressionsDiscountingFunctions),
        )
    )
    func_uim_last_seen: Categorical = attrs.field(
        default=Categorical(
            list(EImpressionsDiscountingFunctions),
        )
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
            np.float32
        ).reshape((self.n_users, 1))
        self._uim_frequency = check_matrix(X=uim_frequency, format="csr", dtype=np.float32)
        self._uim_position = check_matrix(X=uim_position, format="csr", dtype=np.float32)
        self._uim_last_seen = check_matrix(X=uim_last_seen, format="csr", dtype=np.float32)

        self._arr_user_frequency_scores = np.array([], dtype=np.float32)
        self._matrix_uim_frequency_scores = sp.csr_matrix(np.array([], dtype=np.float32))
        self._matrix_uim_position_scores = sp.csr_matrix(np.array([], dtype=np.float32))
        self._matrix_uim_last_seen_scores = sp.csr_matrix(np.array([], dtype=np.float32))
        self._arr_impressions_discounting_scores = np.array([], dtype=np.float32)

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
        This function computes the item scores using the definition of cycling.

        Cycling holds two arrays `arr_scores_presentation` and `arr_scores_relevance`. The first tells how many times
        each item (columns) has been impressed to the users (rows). The second is the relevance score given by the
        trained recommender to each user-item pair (users in rows, items in columns).

        The new relevance score is computed by assigning the rank (higher is more relevant) to each user-item pair. To
        assign this rank for each user, items are sorted first by their presentation score `arr_scores_presentation`
        in ascending order and then by their relevance score `arr_scores_relevance` in ascending order as well.

        Cycling implies that items with fewer impressions will get low rank scores (therefore, highly unlikely to be
        recommended)

        This method assigns ranks (and does return sorted item indices) to comply with the `recommend` function in
        BaseRecommender, i.e., the `recommend` function expects that each user-item pair holds a relevance score,
        where items with the highest scores are recommended.

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
        arr_scores_impressions_discounting: np.ndarray = self._arr_impressions_discounting_scores[user_id_array, :]

        assert (num_score_users, num_score_items) == arr_scores_impressions_discounting.shape
        assert (num_score_users, num_score_items) == arr_scores_relevance.shape

        # Note: `rank_data_by_row` requires that the most important array are place right-most in the tuple. In  this
        # case, we want to sort first by `arr_scores_presentation` and then by `arr_scores_relevance`.
        new_item_scores = (
            arr_scores_relevance
            * arr_scores_impressions_discounting
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

        func_user_frequency: EImpressionsDiscountingFunctions,
        func_uim_frequency: EImpressionsDiscountingFunctions,
        func_uim_position: EImpressionsDiscountingFunctions,
        func_uim_last_seen: EImpressionsDiscountingFunctions,

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

        self._func_user_frequency = func_user_frequency
        self._func_uim_frequency = func_uim_frequency
        self._func_uim_position = func_uim_position
        self._func_uim_last_seen = func_uim_last_seen

        selected_func = _DICT_IMPRESSIONS_DISCOUNTING_FUNCTIONS[self._func_user_frequency]
        self._arr_user_frequency_scores = self._reg_user_frequency * selected_func(
            self._user_frequency
        )

        selected_func = _DICT_IMPRESSIONS_DISCOUNTING_FUNCTIONS[self._func_uim_frequency]
        self._matrix_uim_frequency_scores = self._reg_uim_frequency * selected_func(
            self._uim_frequency
        )

        selected_func = _DICT_IMPRESSIONS_DISCOUNTING_FUNCTIONS[self._func_uim_position]
        self._matrix_uim_position_scores = self._reg_uim_position * selected_func(
            self._uim_position
        )

        selected_func = _DICT_IMPRESSIONS_DISCOUNTING_FUNCTIONS[self._func_uim_last_seen]
        self._matrix_uim_last_seen_scores = self._reg_uim_last_seen * selected_func(
            self._uim_last_seen
        )

        # Given that `self._arr_user_frequency_scores` sums on all rows, it converts the sparse matrix into a dense one.
        # We then keep the numpy array from here on.
        arr_impressions_discounting_scores: np.ndarray = (
            self._arr_user_frequency_scores
            + self._matrix_uim_frequency_scores
            + self._matrix_uim_position_scores
            + self._matrix_uim_last_seen_scores
        ).A
        # We use a min here as the matrix may be full of zeroes. We avoid a division by zero if this is the case.
        max_score = arr_impressions_discounting_scores.max(initial=0.) + 1e-6  # to avoid division by zero.

        self._arr_impressions_discounting_scores = arr_impressions_discounting_scores / max_score

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        DataIO.s_save_data(
            folder_path=folder_path,
            file_name=file_name,
            data_dict_to_save={
                "_arr_impressions_discounting_scores": self._arr_impressions_discounting_scores,

                "_arr_user_frequency_scores": self._arr_user_frequency_scores,
                "_matrix_uim_frequency_scores": self._matrix_uim_frequency_scores,
                "_matrix_uim_position_scores": self._matrix_uim_position_scores,
                "_matrix_uim_last_seen_scores": self._matrix_uim_last_seen_scores,

                "_reg_user_frequency": self._reg_user_frequency,
                "_reg_uim_frequency": self._reg_uim_frequency,
                "_reg_uim_position": self._reg_uim_position,
                "_reg_uim_last_seen": self._reg_uim_last_seen,

                "_func_user_frequency": self._func_user_frequency,
                "_func_uim_frequency": self._func_uim_frequency,
                "_func_uim_position": self._func_uim_position,
                "_func_uim_last_seen": self._func_uim_last_seen,
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

        assert hasattr(self, "_arr_impressions_discounting_scores")
