import abc
from typing import Optional, Union

import attrs
import numpy as np
import scipy.sparse as sp
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender, \
    BaseUserSimilarityMatrixRecommender
from Recommenders.Recommender_utils import check_matrix

from recsys_framework_extensions.data.io import DataIO
from recsys_framework_extensions.recommenders.base import SearchHyperParametersBaseRecommender
from skopt.space import Real


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersWeightedUserProfileRecommender(SearchHyperParametersBaseRecommender):
    reg_urm: Real = attrs.field(
        default=Real(
            low=1e-2,
            high=1e2,
            prior="uniform",
            base=10,
        )
    )
    reg_uim: Real = attrs.field(
        default=Real(
            low=1e-2,
            high=1e2,
            prior="uniform",
            base=10,
        )
    )


class BaseWeightedUserProfileRecommender(BaseRecommender, abc.ABC):
    RECOMMENDER_NAME = "BaseWeightedUserProfileRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        trained_recommender: Union[BaseItemSimilarityMatrixRecommender, BaseUserSimilarityMatrixRecommender],
    ):
        """
        Notes


        Parameters
        ----------
        urm_train: csr_matrix
            A sparse matrix of shape (M, N), where M = #Users, and N = #Items. The content of this matrix is the latest
            recorded implicit interaction for the user-item pair (u,i), i.e., urm_train[u,i] = 1

        uim_train: csr_matrix
            A sparse matrix of shape (M, N), where M = #Users, and N = #Items. The content of this matrix is the latest
            recorded impression for the user-item pair (u,i), i.e., uim_train[u,i] = 1

        trained_recommender: BaseItemSimilarityMatrixRecommender
            An instance of a trained `similarity` recommender, i.e., that has an attribute `W_sparse`. This recommender
            must be able to execute `trained_recommender._compute_item_scores` without raising exceptions.
        """
        super().__init__(
            URM_train=urm_train,
            verbose=True,
        )
        self._attr_name_w_sparse = "W_sparse"

        if not hasattr(trained_recommender, self._attr_name_w_sparse):
            raise AttributeError(
                f"Cannot weight user profiles on the recommender {trained_recommender} as it has not been trained (it "
                f"lacks the attribute '{self._attr_name_w_sparse}'."
            )

        self._sparse_user_profile: sp.csr_matrix = sp.csr_matrix([], dtype=np.float32)
        self._sparse_similarity: sp.csr_matrix = sp.csr_matrix([], dtype=np.float32)

        self._uim_train: sp.csr_matrix = check_matrix(X=uim_train, format="csr", dtype=np.float32)
        self._reg_urm: float = 0.0
        self._reg_uim: float = 0.0

        self.trained_recommender = trained_recommender

    def fit(
        self,
        reg_urm: float,
        reg_uim: float,
        **kwargs,
    ) -> None:
        self._reg_urm = reg_urm
        self._reg_uim = reg_uim

        sparse_user_profile = (
            (self._reg_urm * self.URM_train)
            + (self._reg_uim * self._uim_train)
        )
        self._sparse_user_profile = check_matrix(X=sparse_user_profile, format="csr", dtype=np.float32)

        sparse_similarity = getattr(self.trained_recommender, self._attr_name_w_sparse)
        self._sparse_similarity = check_matrix(X=sparse_similarity, format="csr", dtype=np.float32)

    def save_model(
        self,
        folder_path: str,
        file_name: Optional[str] = None,
    ):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        DataIO.s_save_data(
            folder_path=folder_path,
            file_name=file_name,
            data_dict_to_save={
                "_sparse_user_profile": self._sparse_user_profile,
                "_sparse_similarity": self._sparse_similarity,
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

        assert hasattr(self, "_sparse_similarity") and self._sparse_similarity.nnz > 0
        assert hasattr(self, "_sparse_user_profile") and self._sparse_user_profile.nnz > 0

        self._sparse_similarity = check_matrix(X=self._sparse_similarity, format="csr", dtype=np.float32)
        self._sparse_user_profile = check_matrix(X=self._sparse_user_profile, format="csr", dtype=np.float32)


class ItemWeightedUserProfileRecommender(BaseWeightedUserProfileRecommender):
    RECOMMENDER_NAME = "ItemWeightedUserProfileRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        trained_recommender: BaseItemSimilarityMatrixRecommender,
    ):
        super().__init__(
            urm_train=urm_train,
            uim_train=uim_train,
            trained_recommender=trained_recommender,
        )

        if not isinstance(self.trained_recommender, BaseItemSimilarityMatrixRecommender):
            raise AttributeError(
                f"Cannot weight user profiles on the recommender {trained_recommender} as it does not inherit from "
                f"the class 'BaseItemSimilarityMatrixRecommender'."
            )

        self.RECOMMENDER_NAME = f"ItemWeightedUserProfile_{trained_recommender.RECOMMENDER_NAME}"

    def _compute_item_score(
        self,
        user_id_array: list[int],
        items_to_compute: Optional[list[int]] = None,
    ):
        assert user_id_array is not None
        assert len(user_id_array) > 0
        assert (self.n_items, self.n_items) == self._sparse_similarity.shape
        assert (self.n_users, self.n_items) == self._sparse_user_profile.shape

        num_score_users: int = len(user_id_array)
        num_score_items: int = self.n_items

        # Create the scores only for the users inside `user_id_array`
        item_scores_all = self._sparse_user_profile[user_id_array, :].dot(
            self._sparse_similarity,
        ).toarray()
        assert (num_score_users, num_score_items) == item_scores_all.shape

        # In case we're asked to compute the similarity only on a subset of items, then, we create a matrix of -inf
        # and only set the scores of the items inside `items_to_compute`.
        if items_to_compute is not None:
            item_scores = np.NINF * np.ones_like(
                item_scores_all,
                dtype=np.float32,
            )
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = item_scores_all

        assert (num_score_users, num_score_items) == item_scores.shape

        return item_scores


class UserWeightedUserProfileRecommender(BaseWeightedUserProfileRecommender):
    RECOMMENDER_NAME = "UserWeightedUserProfileRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        trained_recommender: BaseUserSimilarityMatrixRecommender,
    ):
        super().__init__(
            urm_train=urm_train,
            uim_train=uim_train,
            trained_recommender=trained_recommender,
        )

        if not isinstance(self.trained_recommender, BaseUserSimilarityMatrixRecommender):
            raise AttributeError(
                f"Cannot weight user profiles on the recommender {trained_recommender} as it does not inherit from "
                f"the class 'BaseUserSimilarityMatrixRecommender'."
            )

        self.RECOMMENDER_NAME = f"UserWeightedUserProfile_{trained_recommender.RECOMMENDER_NAME}"

    def _compute_item_score(
        self,
        user_id_array: list[int],
        items_to_compute: Optional[list[int]] = None,
    ):
        assert user_id_array is not None
        assert len(user_id_array) > 0
        assert (self.n_users, self.n_users) == self._sparse_similarity.shape
        assert (self.n_users, self.n_items) == self._sparse_user_profile.shape

        num_score_users: int = len(user_id_array)
        num_score_items: int = self.n_items

        # Create the scores only for the users inside `user_id_array`
        item_scores_all = self._sparse_similarity[user_id_array, :].dot(
            self._sparse_user_profile
        ).toarray()
        assert (num_score_users, num_score_items) == item_scores_all.shape

        # In case we're asked to compute the similarity only on a subset of items, then, we create a matrix of -inf
        # and only set the scores of the items inside `items_to_compute`.
        if items_to_compute is not None:
            item_scores = np.NINF * np.ones_like(
                item_scores_all,
                dtype=np.float32,
            )
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = item_scores_all

        assert (num_score_users, num_score_items) == item_scores.shape

        return item_scores
