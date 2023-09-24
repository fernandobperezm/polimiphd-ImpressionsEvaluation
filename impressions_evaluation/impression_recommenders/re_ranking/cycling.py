from typing import Optional, Literal

import attrs
import numpy as np
import scipy.sparse as sp
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Recommender_utils import check_matrix
from recsys_framework_extensions.data.io import DataIO
from recsys_framework_extensions.recommenders.base import (
    SearchHyperParametersBaseRecommender,
    AbstractExtendedBaseRecommender,
)
from recsys_framework_extensions.recommenders.rank import rank_data_by_row
from skopt.space import Integer, Categorical


T_SIGN = Literal[-1, 1]


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersCyclingRecommender(SearchHyperParametersBaseRecommender):
    weight: Integer = attrs.field(
        default=Integer(
            low=1,
            high=50,
            prior="uniform",
            base=10,
        )
    )
    sign: Categorical = attrs.field(
        default=Categorical(
            categories=[-1, 1],
        )
    )


CYCLING_HYPER_PARAMETER_SEARCH_CONFIGURATIONS = {
    "ORIGINAL": SearchHyperParametersCyclingRecommender(),
    "REPRODUCIBILITY_ORIGINAL_PAPER": SearchHyperParametersCyclingRecommender(
        sign=Categorical(categories=[-1]),
    ),
    "SIGNAL_ANALYSIS_SIGN_POSITIVE": SearchHyperParametersCyclingRecommender(
        sign=Categorical(categories=[1]),
    ),
    "SIGNAL_ANALYSIS_SIGN_NEGATIVE": SearchHyperParametersCyclingRecommender(
        sign=Categorical(categories=[-1]),
    ),
}


def compute_presentation_score(
    uim_frequency: sp.csr_matrix,
    weight: int,
    sign: int,
) -> sp.csr_matrix:
    sp_presentation_scores = uim_frequency.copy()

    # Keep equal the values of items that have been seen at least `weight` times,
    # if they have been seen more than `weight` times, then discount them by dividing the frequency by `weight`
    # and round down (floor op).
    sp_presentation_scores.data = np.where(
        uim_frequency.data < weight,
        sign * uim_frequency.data,
        sign * np.floor(uim_frequency.data / weight),
    )

    return sp_presentation_scores


def compute_cycling_recommender_score(
    list_user_ids: list[int],
    list_item_ids: Optional[list[int]],
    arr_recommender_scores: np.ndarray,
    matrix_presentation_scores: sp.csr_matrix,
    num_score_users: int,
    num_score_items: int,
) -> np.ndarray:
    assert (num_score_users, num_score_items) == arr_recommender_scores.shape

    arr_scores_presentation: np.ndarray = matrix_presentation_scores[
        list_user_ids, :
    ].toarray()

    assert (num_score_users, num_score_items) == arr_scores_presentation.shape

    # arr_scores_presentation only holds integer values, while arr_recommender_scores hold real values. We will scale
    # them to the range [0,...,0.99] in this way, the integer value denotes the presentation score and the decimal
    # position denotes the other recommender's score. In case of ties in presentation score (integers), the
    # difference will be given other recommender's score (decimals).
    # Formula for normalization can be seen here: https://stackoverflow.com/a/50305307
    start = 0.0
    end = 0.99
    width = end - start
    arr_norm_scores_relevance = (
        arr_recommender_scores - arr_recommender_scores.min()
    ) / arr_recommender_scores.ptp() * width + start

    # We do a similar normalization to the `arr_scores_presentation` array.
    # In this case, we want to keep the range positive, while keeping the order that may exist with negative values.
    # Hence, we sum all the array by the abs(minimum value) + 1 to ensure the range [1,..., inf].
    arr_norm_scores_presentation = (
        arr_scores_presentation + np.abs(arr_scores_presentation.min()) + 1
    )

    arr_new_item_scores = arr_norm_scores_presentation + arr_norm_scores_relevance

    # Note: `rank_data_by_row` requires that the most important array are place right-most in the tuple. In  this
    # case, we want to sort first by `arr_scores_presentation` and then by `arr_recommender_scores`.
    # In the case of cycling, the presentation score is sorted in ascending order (least seen items are given a
    # higher score). relevance scores are sorted in descending order.
    # The hyper-parameter optimizer assigns a `sign` to indicate if for these datasets impressions are negative or
    # positive. If we want to reproduce the results of the original paper, then we must set sign as negative.
    # arr_new_item_scores = rank_data_by_row(
    #     keys=(arr_recommender_scores, arr_scores_presentation)
    # )

    # If we are computing scores to a specific set of items, then we must set items outside this set to np.NINF,
    # so they are not
    if list_item_ids is not None:
        arr_mask_items = np.zeros_like(arr_new_item_scores, dtype=np.bool8)
        arr_mask_items[:, list_item_ids] = True

        # If the item is in `list_item_ids`, then keep the value from `new_item_scores`.
        # Else, set to -inf.
        arr_new_item_scores = np.where(
            arr_mask_items,
            arr_new_item_scores,
            np.NINF,
        )

    assert (num_score_users, num_score_items) == arr_new_item_scores.shape

    return arr_new_item_scores


_sp_presentation_score = compute_presentation_score(
    uim_frequency=sp.csr_matrix([[1, 2], [4, 5], [7, 8]], dtype=np.float32),
    weight=3,
    sign=1,
)
assert np.array_equal(
    sp.csr_matrix([[1, 2], [1, 1], [2, 2]], dtype=np.float32).data,
    _sp_presentation_score.data,
)

_sp_presentation_score = compute_presentation_score(
    uim_frequency=sp.csr_matrix([[1, 2], [4, 5], [7, 8]], dtype=np.float32),
    weight=5,
    sign=1,
)
assert np.array_equal(
    sp.csr_matrix([[1, 2], [4, 1], [1, 1]], dtype=np.float32).data,
    _sp_presentation_score.data,
)

_sp_presentation_score = compute_presentation_score(
    uim_frequency=sp.csr_matrix([[1, 2], [4, 5], [7, 8]], dtype=np.float32),
    weight=3,
    sign=-1,
)
assert np.array_equal(
    sp.csr_matrix([[-1, -2], [-1, -1], [-2, -2]], dtype=np.float32).data,
    _sp_presentation_score.data,
)

_sp_presentation_score = compute_presentation_score(
    uim_frequency=sp.csr_matrix([[1, 2], [4, 5], [7, 8]], dtype=np.float32),
    weight=5,
    sign=-1,
)
assert np.array_equal(
    sp.csr_matrix([[-1, -2], [-4, -1], [-1, -1]], dtype=np.float32).data,
    _sp_presentation_score.data,
)


class CyclingRecommender(AbstractExtendedBaseRecommender):
    RECOMMENDER_NAME = "CyclingRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        trained_recommender: BaseRecommender,
        **kwargs,
    ):
        super().__init__(
            urm_train=urm_train,
        )

        self._trained_recommender = trained_recommender
        self._uim_frequency = uim_frequency
        self._matrix_presentation_scores = sp.csr_matrix(np.array([], dtype=np.float32))
        self._cycling_weight: int = 3
        self._cycling_sign: T_SIGN = -1

        self.RECOMMENDER_NAME = (
            f"CyclingRecommender_{trained_recommender.RECOMMENDER_NAME}"
        )

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
        arr_scores_relevance: np.ndarray = (
            self._trained_recommender._compute_item_score(
                user_id_array=user_id_array,
                items_to_compute=items_to_compute,
            )
        )

        arr_new_item_scores = compute_cycling_recommender_score(
            list_user_ids=user_id_array,
            list_item_ids=items_to_compute,
            num_score_users=num_score_users,
            num_score_items=num_score_items,
            arr_recommender_scores=arr_scores_relevance,
            matrix_presentation_scores=self._matrix_presentation_scores,
        )

        return arr_new_item_scores

    def fit(
        self,
        weight: int,
        sign: T_SIGN,
        **kwargs,
    ):
        assert weight > 0
        assert sign == 1 or sign == -1

        self._cycling_weight = weight
        self._cycling_sign = sign

        matrix_presentation_scores = compute_presentation_score(
            uim_frequency=self._uim_frequency,
            weight=self._cycling_weight,
            sign=self._cycling_sign,
        )
        self._matrix_presentation_scores = check_matrix(
            X=matrix_presentation_scores, format="csr", dtype=np.float32
        )

    def save_model(self, folder_path: str, file_name: Optional[str] = None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        DataIO.s_save_data(
            folder_path=folder_path,
            file_name=file_name,
            data_dict_to_save={
                "_matrix_presentation_scores": self._matrix_presentation_scores,
                "_cycling_weight": self._cycling_weight,
                "_cycling_sign": self._cycling_sign,
            },
        )

    def validate_load_trained_recommender(self, *args, **kwargs) -> None:
        assert hasattr(self, "_cycling_weight")
        assert hasattr(self, "_cycling_sign")
        assert (
            hasattr(self, "_matrix_presentation_scores")
            and self._matrix_presentation_scores.nnz > 0
        )

        self._matrix_presentation_scores = check_matrix(
            X=self._matrix_presentation_scores, format="csr", dtype=np.float32
        )
