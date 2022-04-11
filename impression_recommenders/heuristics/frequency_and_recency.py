from typing import Literal, cast

import numpy as np
import scipy.sparse as sp
import scipy.stats as st

from Recommenders.BaseRecommender import BaseRecommender


T_RANK_METHOD = Literal["average", "min", "max", "dense", "ordinal"]


class FrequencyRecencyRecommender(BaseRecommender):
    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        uim_timestamp: sp.csr_matrix,
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
        self._rank_method: T_RANK_METHOD = "average"

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
        _, num_train_items = self.URM_train.shape
        num_score_users = len(user_id_array)

        arr_timestamp_users = self._uim_timestamp[user_id_array]
        arr_frequency_users = self._uim_frequency[user_id_array]

        # st.rankdata assigns rank in ascending order, where the highest rank is the most relevant item, therefore,
        # we need to invert the scores with the minus sign (-) so it assigns the highest rank to the most recent and
        # most frequent item.
        arr_scores_timestamp = -arr_timestamp_users
        arr_scores_frequency = -arr_frequency_users

        arr_frequency_timestamp_scores = np.array(
            list(zip(arr_scores_frequency, arr_scores_timestamp)),
            dtype=[
                ("score_frequency", arr_scores_frequency.dtype),
                ("score_timestamp", arr_scores_timestamp.dtype)
            ]
        )

        item_scores = st.rankdata(
            a=arr_frequency_timestamp_scores,
            method=self._rank_method,
            axis=1,
        )

        assert item_scores.shape == (num_score_users, num_train_items)
        assert item_scores.shape == arr_scores_timestamp.shape
        assert item_scores.shape == arr_scores_frequency.shape

        return item_scores

    def save_model(self, folder_path, file_name=None):
        pass


class RecencyRecommender(BaseRecommender):
    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_timestamp: sp.csr_matrix,
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

        self._uim_timestamp: sp.csr_matrix = uim_timestamp.copy()
        self._rank_method: T_RANK_METHOD = "average"

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
        _, num_train_items = self.URM_train.shape
        num_score_users = len(user_id_array)

        arr_timestamp_users = self._uim_timestamp[user_id_array]

        # st.rankdata assigns rank in ascending order, where the highest rank is the most relevant item, therefore,
        # we need to invert the scores with the minus sign (-) so it assigns the highest rank to the most recent and
        # most frequent item.
        arr_scores_timestamp = -arr_timestamp_users

        item_scores = st.rankdata(
            a=arr_scores_timestamp,
            method=self._rank_method,
            axis=1,
        )

        assert item_scores.shape == (num_score_users, num_train_items)
        assert item_scores.shape == arr_scores_timestamp.shape

        return item_scores

    def save_model(self, folder_path, file_name=None):
        pass
