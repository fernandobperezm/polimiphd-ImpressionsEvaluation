from typing import Literal, cast

import numpy as np
import scipy.sparse as sp
import scipy.stats as st

from Recommenders.BaseRecommender import BaseRecommender


T_RANK_METHOD = Literal["average", "min", "max", "dense", "ordinal"]


class CyclingRecommender(BaseRecommender):
    def __init__(
        self,
        urm_train: sp.csr_matrix,
        trained_recommender: BaseRecommender,
        urm_presentation: sp.csr_matrix,
        seed: int,
    ):
        super().__init__(
            URM_train=urm_train,
            verbose=True,
        )

        self._trained_recommender = trained_recommender
        self._matrix_presentation = urm_presentation
        self._matrix_presentation_scores = sp.csr_matrix(np.array([], dtype=np.float32))
        self._cycling_weight: float = 3.0
        self._rank_method: T_RANK_METHOD = "average"

        self._rng = np.random.default_rng(seed=seed)

    def _compute_item_score(
        self,
        user_id_array,
        items_to_compute=None
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
        # Dense array of shape (M,N) where M is len(user_id_array) and N is the total number of users in the dataset.
        arr_scores_relevance: np.ndarray = self._trained_recommender._compute_item_score(
            user_id_array=user_id_array,
            items_to_compute=items_to_compute,
        )
        arr_scores_presentation = self._matrix_presentation_scores[user_id_array]

        arr_scores_relevance = -arr_scores_relevance
        arr_scores_presentation = -arr_scores_presentation

        arr_presentation_relevance_scores = np.array(
            list(zip(arr_scores_presentation, arr_scores_relevance)),
            dtype=[
                ("score_presentation", arr_scores_presentation.dtype),
                ("score_relevance", arr_scores_relevance.dtype)
            ]
        )

        new_item_scores = st.rankdata(
            a=arr_presentation_relevance_scores,
            method=self._rank_method,
            axis=1,
        )

        assert arr_scores_relevance.shape == arr_scores_presentation.shape
        assert new_item_scores.shape[1] == arr_scores_presentation.shape[1]

        return new_item_scores

    def fit(
        self,
        weight: float,
        rank_method: T_RANK_METHOD,
        **kwargs,
    ):
        self._cycling_weight = weight

        self._matrix_presentation_scores = cast(
            sp.csr_matrix,
            self._matrix_presentation / self._cycling_weight
        )

    def save_model(self, folder_path, file_name=None):
        pass
