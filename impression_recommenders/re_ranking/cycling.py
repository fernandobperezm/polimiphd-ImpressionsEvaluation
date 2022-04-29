from typing import Optional, Literal

import attrs
import numpy as np
import scipy.sparse as sp
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Recommender_utils import check_matrix
from recsys_framework_extensions.data.io import DataIO
from recsys_framework_extensions.recommenders.base import SearchHyperParametersBaseRecommender
from recsys_framework_extensions.recommenders.mixins import MixinLoadModel
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


DICT_SEARCH_CONFIGS = {
    "REPRODUCIBILITY_ORIGINAL_PAPER": SearchHyperParametersCyclingRecommender(
        sign=Categorical(categories=[-1]),
    )
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


sp_presentation_score = compute_presentation_score(
    uim_frequency=sp.csr_matrix([[1, 2], [4, 5], [7, 8]], dtype=np.float32), weight=3, sign=1
)
assert np.array_equal(
    sp.csr_matrix([[1, 2], [1, 1], [2, 2]], dtype=np.float32).data,
    sp_presentation_score.data
)

sp_presentation_score = compute_presentation_score(
    uim_frequency=sp.csr_matrix([[1, 2], [4, 5], [7, 8]], dtype=np.float32), weight=5, sign=1
)
assert np.array_equal(
    sp.csr_matrix([[1, 2], [4, 1], [1, 1]], dtype=np.float32).data,
    sp_presentation_score.data
)

sp_presentation_score = compute_presentation_score(
    uim_frequency=sp.csr_matrix([[1, 2], [4, 5], [7, 8]], dtype=np.float32), weight=3, sign=-1
)
assert np.array_equal(
    sp.csr_matrix([[-1, -2], [-1, -1], [-2, -2]], dtype=np.float32).data,
    sp_presentation_score.data
)

sp_presentation_score = compute_presentation_score(
    uim_frequency=sp.csr_matrix([[1, 2], [4, 5], [7, 8]], dtype=np.float32), weight=5, sign=-1
)
assert np.array_equal(
    sp.csr_matrix([[-1, -2], [-4, -1], [-1, -1]], dtype=np.float32).data,
    sp_presentation_score.data
)


class CyclingRecommender(MixinLoadModel, BaseRecommender):
    RECOMMENDER_NAME = "CyclingRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        trained_recommender: BaseRecommender,
        **kwargs,
    ):
        super().__init__(
            URM_train=urm_train,
            verbose=True,
        )

        self._trained_recommender = trained_recommender
        self._uim_frequency = uim_frequency
        self._matrix_presentation_scores = sp.csr_matrix(np.array([], dtype=np.float32))
        self._cycling_weight: int = 3
        self._cycling_sign: T_SIGN = -1

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
        arr_scores_presentation: np.ndarray = self._matrix_presentation_scores[user_id_array, :].toarray()

        assert (num_score_users, num_score_items) == arr_scores_presentation.shape
        assert (num_score_users, num_score_items) == arr_scores_relevance.shape

        # Note: `rank_data_by_row` requires that the most important array are place right-most in the tuple. In  this
        # case, we want to sort first by `arr_scores_presentation` and then by `arr_scores_relevance`.
        # In the case of cycling, the presentation score is sorted in ascending order (least seen items are given a
        # higher score). relevance scores are sorted in descending order.
        # The hyper-parameter optimizer assigns a `sign` to indicate if for these datasets impressions are negative or
        # positive. If we want to reproduce the results of the original paper, then we must set sign as negative.
        new_item_scores = rank_data_by_row(
            keys=(arr_scores_relevance, arr_scores_presentation)
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
        self._matrix_presentation_scores = check_matrix(X=matrix_presentation_scores, format="csr", dtype=np.float32)

    def save_model(self, folder_path: str, file_name: str =None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        DataIO.s_save_data(
            folder_path=folder_path,
            file_name=file_name,
            data_dict_to_save={
                "_matrix_presentation_scores": self._matrix_presentation_scores,
                "_cycling_weight": self._cycling_weight,
                "_cycling_sign": self._cycling_sign,
            }
        )

    def load_model(
        self,
        folder_path: str,
        file_name: str = None,
    ) -> None:
        super().load_model(
            folder_path=folder_path,
            file_name=file_name,
        )

        assert hasattr(self, "_cycling_weight")
        assert hasattr(self, "_cycling_sign")
        assert hasattr(self, "_matrix_presentation_scores") and self._matrix_presentation_scores.nnz > 0

        self._matrix_presentation_scores = check_matrix(X=self._matrix_presentation_scores, format="csr", dtype=np.float32)
