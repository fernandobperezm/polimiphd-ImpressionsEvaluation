from typing import cast, Optional

import attrs
import numpy as np
import scipy.sparse as sp
import scipy.stats as st
from Recommenders.BaseRecommender import BaseRecommender
from skopt.space import Real

from impression_recommenders.constants import ERankMethod


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersCyclingRecommender:
    weight: Real = attrs.field(
        default=Real(
            low=1e-5,
            high=1e2,
            prior="uniform",
            base=10,
        )
    )


class CyclingRecommender(BaseRecommender):
    RECOMMENDER_NAME = "CyclingRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        trained_recommender: BaseRecommender,
        seed: int,
        **kwargs,
    ):
        super().__init__(
            URM_train=urm_train,
            verbose=True,
        )


        self._trained_recommender = trained_recommender
        self._uim_frequency = uim_frequency
        self._matrix_presentation_scores = sp.csr_matrix(np.array([], dtype=np.float32))
        self._cycling_weight: float = 3.0
        self._rank_method: ERankMethod = ERankMethod.MIN

        self._rng = np.random.default_rng(seed=seed)

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

        arr_scores_presentation_relevance = np.array(
            [
                [
                    (arr_scores_presentation[row, col], arr_scores_relevance[row, col])
                    for col in range(num_score_items)
                ]
                for row in range(num_score_users)
            ],
            dtype=[
                ("score_presentation", arr_scores_presentation.dtype),
                ("score_relevance", arr_scores_relevance.dtype),
            ]
        )

        assert (num_score_users, num_score_items) == arr_scores_presentation_relevance.shape

        # st.rankdata assigns rank in ascending order [(0,0) -> 1 while (4, 6) -> 10], where the highest rank is the
        # most relevant item, therefore, we need to invert the scores with the minus sign (-) so it assigns the
        # highest rank to the most recent and most frequent item.
        new_item_scores = st.rankdata(
            a=arr_scores_presentation_relevance,
            method=self._rank_method.value,
            axis=1,
        ).astype(
            np.float32,
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
        weight: float,
        **kwargs,
    ):
        self._cycling_weight = weight

        self._matrix_presentation_scores = cast(
            sp.csr_matrix,
            self._uim_frequency / self._cycling_weight
        )

    def save_model(self, folder_path, file_name=None):
        pass
