from typing import Literal, Optional

import numpy as np
import scipy.sparse as sp

from Recommenders.BaseRecommender import BaseRecommender


T_RANK_METHOD = Literal["average", "min", "max", "dense", "ordinal"]


class LastImpressionsRecommender(BaseRecommender):
    """
    TODO: fernando-debugger| Finish this
    LastImpressionsRecommender


    References
    ----------
    .. [1]

    """

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_position: sp.csr_matrix,
        uim_timestamp: sp.csr_matrix,
    ):
        """
        Parameters
        ----------
        urm_train: csr_matrix
            A sparse matrix of shape (M, N), where M = #Users, and N = #Items. The content of this matrix is the latest
            recorded implicit interaction for the user-item pair (u,i), i.e., urm_train[u,i] = 1

        uim_position: csr_matrix
            A sparse matrix of shape (M, N), where M = #Users, and N = #Items. The content of this matrix is the
            latest recorded position in the recommendation list for the user-item pair (u,i), i.e.,
            uim_position[u,i] = position.

        uim_timestamp: csr_matrix
            A sparse matrix of shape (M, N), where M = #Users, and N = #Items. The content of this matrix is the
            latest recorded timestamp for the user-item pair (u,i), i.e., uim_timestamp[u,i] = timestamp.
        """
        super().__init__(
            URM_train=urm_train,
            verbose=True,
        )

        self._uim_position: sp.csr_matrix = uim_position.copy()
        self._uim_timestamp: sp.csr_matrix = uim_timestamp.copy()
        self._rank_method: T_RANK_METHOD = "average"

        assert self.URM_train.shape == self._uim_position.shape
        assert self.URM_train.shape == self._uim_timestamp.shape

    def _compute_item_score(
        self,
        user_id_array: list[int],
        items_to_compute: Optional[list[int]] = None,
    ) -> np.ndarray:
        """
        Computes the preference scores of users toward items.

        Notes
        -----
            :class:`BaseRecommender` ranks items in a further step by their score. Items with the highest scores are
            positioned first in the list.

        Return
        ------
        np.ndarray
            A matrix with shape (M, N), where M = |user_id_array|, and N = #Items .
        """
        assert user_id_array is not None
        assert len(user_id_array) > 0

        num_score_users: int = len(user_id_array)
        num_score_items: int = self.URM_train.shape[1]

        item_scores: np.ndarray = np.NINF * np.ones(
            shape=(num_score_users, num_score_items)
        )
        item_scores2: np.ndarray = np.NINF * np.ones(
            shape=(num_score_users, num_score_items)
        )

        for idx_item_score, user_id in enumerate(user_id_array):
            user_timestamps: np.ndarray = self._uim_timestamp[user_id, :].toarray().ravel()
            user_positions: np.ndarray = -self._uim_position[user_id, :].toarray().ravel()

            user_max_timestamp = user_timestamps.max(initial=np.NINF)
            if user_max_timestamp == 0.0:
                # A value of zero means that the user did not receive any impression, therefore, we cannot recommend
                # anything with this recommender.
                user_max_timestamp = np.NINF

            # If we want to compute for only a certain group of items (`items_to_compute` not being None),
            # then we must have a mask that sets -INF to items outside this list.
            if items_to_compute is None:
                arr_mask_items: np.ndarray = np.ones_like(user_timestamps, dtype=np.bool8)
            else:
                arr_mask_items = np.zeros_like(user_timestamps, dtype=np.bool8)
                arr_mask_items[items_to_compute] = True

            # Option 1. Set all values in columns with np.where, in this case, if the condition is true, then it uses
            # the value in `user_positions`, else it uses `np.NINF`.
            item_scores[idx_item_score, :] = np.where(
                (user_timestamps == user_max_timestamp) & arr_mask_items,
                user_positions,
                np.NINF,
            )

            # Option 2. Set only the values in the columns in which the condition is true by using conditional
            # boolean indexing, in this case, items_to_select is a boolean array, in places where it is True,
            # then it will update the value. The number of truth values must be the same in the right. That is why we
            # also do user_positions[items_to_select], in this way we tell which values we want in those truth places.
            items_to_select = (user_timestamps == user_max_timestamp) & arr_mask_items
            item_scores2[idx_item_score, items_to_select] = user_positions[items_to_select]

        assert item_scores.shape == (num_score_users, num_score_items)
        assert np.array_equal(item_scores, item_scores2)

        return item_scores

    def save_model(self, folder_path, file_name=None):
        pass
