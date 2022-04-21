from typing import Optional

import attrs
import numpy as np
import scipy.sparse as sp
import scipy.stats as st
from Recommenders.BaseRecommender import BaseRecommender
from recsys_framework_extensions.recommenders.base import SearchHyperParametersBaseRecommender
from recsys_framework_extensions.recommenders.mixins import MixinEmptySaveModel

from impression_recommenders.constants import ERankMethod


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersFrequencyRecencyRecommender(SearchHyperParametersBaseRecommender):
    pass


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersRecencyRecommender(SearchHyperParametersBaseRecommender):
    pass


class FrequencyRecencyRecommender(MixinEmptySaveModel, BaseRecommender):
    RECOMMENDER_NAME = "FrequencyRecencyRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        uim_timestamp: sp.csr_matrix,
        **kwargs,
    ):
        """
        Parameters
        ----------
        urm_train: csr_matrix
            A sparse matrix of shape (M, N), where M = #Users, and N = #Items. The content of this matrix is the latest
            recorded implicit interaction for the user-item pair (u,i), i.e., urm_train[u,i] = 1

        uim_frequency: csr_matrix
            A sparse matrix of shape (M, N), where M = #Users, and N = #Items. The content of this matrix is the
            frequency of impressions (number of times a given item has been impressed to a given user) for the
            user-item pair (u,i), i.e., uim_frequency[u,i] = frequency.

        uim_timestamp: csr_matrix
            A sparse matrix of shape (M, N), where M = #Users, and N = #Items. The content of this matrix is the
            latest recorded timestamp for the user-item pair (u,i), i.e., uim_timestamp[u,i] = timestamp.
        """
        super().__init__(
            URM_train=urm_train,
            verbose=True,
        )

        self._uim_frequency: sp.csr_matrix = uim_frequency.copy()
        self._uim_timestamp: sp.csr_matrix = uim_timestamp.copy()
        self._rank_method: ERankMethod = ERankMethod.MIN

    def _compute_item_score(
        self,
        user_id_array: list[int],
        items_to_compute: Optional[list[int]] = None
    ) -> np.ndarray:
        """
        TODO: fernando-debbuger|Complete this.

        Return
        ------
        np.ndarray
            A matrix with shape (M, N), where M = |user_id_array|, and N = #Items .
        """
        assert user_id_array is not None
        assert len(user_id_array) > 0

        num_score_users: int = len(user_id_array)
        num_score_items: int = self.URM_train.shape[1]

        arr_scores_timestamp = self._uim_timestamp[user_id_array, :].toarray()
        arr_scores_frequency = self._uim_frequency[user_id_array, :].toarray()

        assert arr_scores_frequency.shape == arr_scores_timestamp.shape
        assert num_score_items == arr_scores_timestamp.shape[1]

        arr_frequency_timestamp_scores = np.array(
            [
                [
                    (arr_scores_frequency[row, col], arr_scores_timestamp[row, col])
                    for col in range(arr_scores_frequency.shape[1])
                ]
                for row in range(arr_scores_frequency.shape[0])
            ],
            dtype=[
                ("score_frequency", arr_scores_frequency.dtype),
                ("score_timestamp", arr_scores_timestamp.dtype)
            ]
        )

        # st.rankdata assigns rank in ascending order [(0,0) -> 1 while (4, 6) -> 10], where the highest rank is the
        # most relevant item.
        item_scores = st.rankdata(
            a=arr_frequency_timestamp_scores,
            method=self._rank_method.value,
            axis=1,
        )

        # If we want to compute for only a certain group of items (`items_to_compute` not being None),
        # then we must have a mask that sets -INF to items outside this list.
        if items_to_compute is None:
            arr_mask_items: np.ndarray = np.ones_like(arr_scores_frequency, dtype=np.bool8)
        else:
            arr_mask_items = np.zeros_like(arr_scores_frequency, dtype=np.bool8)
            arr_mask_items[:, items_to_compute] = True

        item_scores = np.where(
            (arr_scores_frequency != 0) & (arr_scores_timestamp != 0) & arr_mask_items,
            item_scores,
            np.NINF,
        )

        assert item_scores.shape == (num_score_users, num_score_items)
        assert item_scores.shape == arr_scores_timestamp.shape
        assert item_scores.shape == arr_scores_frequency.shape

        return item_scores


class RecencyRecommender(MixinEmptySaveModel, BaseRecommender):
    RECOMMENDER_NAME = "RecencyRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_timestamp: sp.csr_matrix,
        **kwargs,
    ):
        """
        Parameters
        ----------
        urm_train: csr_matrix
            A sparse matrix of shape (M, N), where M = #Users, and N = #Items. The content of this matrix is the latest
            recorded implicit interaction for the user-item pair (u,i), i.e., urm_train[u,i] = 1

        uim_timestamp: csr_matrix
            A sparse matrix of shape (M, N), where M = #Users, and N = #Items. The content of this matrix is the
            latest recorded timestamp for the user-item pair (u,i), i.e., uim_timestamp[u,i] = timestamp.
        """
        super().__init__(
            URM_train=urm_train,
            verbose=True,
        )

        self._uim_timestamp: sp.csr_matrix = uim_timestamp.copy()

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

        arr_timestamp_users: np.ndarray = self._uim_timestamp[user_id_array, :].toarray()
        if items_to_compute is None:
            arr_mask_items: np.ndarray = np.ones_like(arr_timestamp_users, dtype=np.bool8)
        else:
            arr_mask_items = np.zeros_like(arr_timestamp_users, dtype=np.bool8)
            arr_mask_items[:, items_to_compute] = True

        item_scores = np.where(
            (arr_timestamp_users != 0) & arr_mask_items,
            arr_timestamp_users,
            np.NINF,
        )

        assert item_scores.shape == (num_score_users, num_score_items)

        return item_scores
