from typing import Optional, Literal

import attrs
import numpy as np
import scipy.sparse as sp
from Recommenders.Recommender_utils import check_matrix
from recsys_framework_extensions.data.io import DataIO
from recsys_framework_extensions.recommenders.base import (
    SearchHyperParametersBaseRecommender,
    AbstractExtendedBaseRecommender,
)
from recsys_framework_extensions.recommenders.rank import rank_data_by_row
from skopt.space import Categorical


T_SIGN = Literal[-1, 1]


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersFrequencyRecencyRecommender(SearchHyperParametersBaseRecommender):
    sign_frequency: Categorical = attrs.field(
        default=Categorical(
            categories=[-1, 1],
        )
    )
    sign_recency: Categorical = attrs.field(
        default=Categorical(
            categories=[-1, 1],
        )
    )


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersRecencyRecommender(SearchHyperParametersBaseRecommender):
    sign_recency: Categorical = attrs.field(
        default=Categorical(
            categories=[-1, 1],
        )
    )


class FrequencyRecencyRecommender(AbstractExtendedBaseRecommender):
    RECOMMENDER_NAME = "FrequencyRecencyRecommender"

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
            urm_train=urm_train,
        )

        self._uim_frequency: sp.csr_matrix = uim_frequency.copy()
        self._uim_timestamp: sp.csr_matrix = uim_timestamp.copy()

        self._sign_frequency: T_SIGN = 1
        self._sign_recency: T_SIGN = 1
        self._sp_matrix_frequency_scores: sp.csr_matrix = sp.csr_matrix([], dtype=np.float32)
        self._sp_matrix_timestamp_scores: sp.csr_matrix = sp.csr_matrix([], dtype=np.float32)

    def fit(
        self,
        sign_frequency: T_SIGN,
        sign_recency: T_SIGN,
        **kwargs,
    ) -> None:

        sp_matrix_frequency_scores = sign_frequency * self._uim_frequency
        sp_matrix_timestamp_scores = sign_recency * self._uim_timestamp

        self._sign_frequency = sign_frequency
        self._sign_recency = sign_recency
        self._sp_matrix_frequency_scores = check_matrix(X=sp_matrix_frequency_scores, dtype=np.float32, format="csr")
        self._sp_matrix_timestamp_scores = check_matrix(X=sp_matrix_timestamp_scores, dtype=np.float32, format="csr")

    def save_model(
        self,
        folder_path: str,
        file_name: Optional[str] = None,
    ) -> None:
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        DataIO.s_save_data(
            folder_path=folder_path,
            file_name=file_name,
            data_dict_to_save={
                "_sign_frequency": self._sign_frequency,
                "_sign_recency": self._sign_recency,
                "_sp_matrix_frequency_scores": self._sp_matrix_frequency_scores,
                "_sp_matrix_timestamp_scores": self._sp_matrix_timestamp_scores,
            }
        )

    def validate_load_trained_recommender(
        self, *args, **kwargs,
    ) -> None:
        assert hasattr(self, "_sign_frequency")
        assert hasattr(self, "_sign_recency")
        assert hasattr(self, "_sp_matrix_frequency_scores")
        assert hasattr(self, "_sp_matrix_timestamp_scores")

        self._sp_matrix_frequency_scores = check_matrix(
            X=self._sp_matrix_frequency_scores,
            format="csr",
            dtype=np.float32
        )
        self._sp_matrix_timestamp_scores = check_matrix(
            X=self._sp_matrix_timestamp_scores,
            format="csr",
            dtype=np.float32
        )

    def _compute_item_score(
        self,
        user_id_array: list[int],
        items_to_compute: Optional[list[int]] = None
    ) -> np.ndarray:
        """
        Return
        ------
        np.ndarray
            A matrix with shape (M, N), where M = |user_id_array|, and N = #Items .
        """
        assert user_id_array is not None
        assert len(user_id_array) > 0

        num_score_users: int = len(user_id_array)
        num_score_items: int = self.URM_train.shape[1]

        arr_scores_timestamp = self._sp_matrix_timestamp_scores[user_id_array, :].toarray()
        arr_scores_frequency = self._sp_matrix_frequency_scores[user_id_array, :].toarray()

        assert arr_scores_frequency.shape == arr_scores_timestamp.shape
        assert num_score_items == arr_scores_timestamp.shape[1]

        item_scores = rank_data_by_row(
            keys=(arr_scores_timestamp, arr_scores_frequency)
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


class RecencyRecommender(AbstractExtendedBaseRecommender):
    RECOMMENDER_NAME = "RecencyRecommender"

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

        uim_timestamp: csr_matrix
            A sparse matrix of shape (M, N), where M = #Users, and N = #Items. The content of this matrix is the
            latest recorded timestamp for the user-item pair (u,i), i.e., uim_timestamp[u,i] = timestamp.
        """
        super().__init__(
            urm_train=urm_train,
        )

        self._uim_timestamp: sp.csr_matrix = uim_timestamp.copy()

        self._sign_recency: T_SIGN = 1
        self._sp_matrix_timestamp_scores: sp.csr_matrix = sp.csr_matrix([], dtype=np.float32)

    def fit(
        self,
        sign_recency: T_SIGN,
        **kwargs,
    ) -> None:
        self._sign_recency = sign_recency

        sp_matrix_timestamp_scores = sign_recency * self._uim_timestamp
        self._sp_matrix_timestamp_scores = check_matrix(X=sp_matrix_timestamp_scores, dtype=np.float32, format="csr")

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

        arr_timestamp_users: np.ndarray = self._sp_matrix_timestamp_scores[user_id_array, :].toarray()
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

    def save_model(
        self,
        folder_path: str,
        file_name: Optional[str] = None,
    ) -> None:
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        DataIO.s_save_data(
            folder_path=folder_path,
            file_name=file_name,
            data_dict_to_save={
                "_sign_recency": self._sign_recency,
                "_sp_matrix_timestamp_scores": self._sp_matrix_timestamp_scores,
            }
        )

    def validate_load_trained_recommender(
        self, *args, **kwargs,
    ) -> None:
        assert hasattr(self, "_sign_recency")
        assert hasattr(self, "_sp_matrix_timestamp_scores")

        self._sp_matrix_timestamp_scores = check_matrix(
            X=self._sp_matrix_timestamp_scores,
            format="csr",
            dtype=np.float32,
        )
