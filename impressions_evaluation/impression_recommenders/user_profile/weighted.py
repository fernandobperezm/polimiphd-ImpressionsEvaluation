import enum
from abc import ABC
from typing import Optional

import attrs
import numpy as np
import scipy.sparse as sp
from Recommenders.BaseSimilarityMatrixRecommender import (
    BaseSimilarityMatrixRecommender,
    BaseItemSimilarityMatrixRecommender,
    BaseUserSimilarityMatrixRecommender,
)
from Recommenders.Recommender_utils import check_matrix
from recsys_framework_extensions.data.io import DataIO, attach_to_extended_json_decoder
import logging
from recsys_framework_extensions.recommenders.base import SearchHyperParametersBaseRecommender, \
    AbstractExtendedBaseRecommender
from skopt.space import Real, Categorical

logger = logging.getLogger(__name__)


@attach_to_extended_json_decoder
class EWeightedUserProfileType(enum.Enum):
    ONLY_IMPRESSIONS = "ONLY_IMPRESSIONS"
    INTERACTIONS_AND_IMPRESSIONS = "INTERACTIONS_AND_IMPRESSIONS"


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersWeightedUserProfileRecommender(SearchHyperParametersBaseRecommender):
    alpha: Real = attrs.field(
        default=Real(
            low=1e-5,
            high=1,
            prior="log-uniform",
            base=10,
        ),
    )
    sign: Categorical = attrs.field(
        default=Categorical(
            categories=[-1, 1],
        )
    )
    weighted_user_profile_type: Categorical = attrs.field(
        default=Categorical(
            categories=[EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS.value],
        )
    )


DICT_SEARCH_CONFIGS = {
    "REPRODUCIBILITY_ORIGINAL_PAPER": SearchHyperParametersWeightedUserProfileRecommender(
        alpha=Categorical(categories=[0.]),
        sign=Categorical(categories=[1]),
        weighted_user_profile_type=Categorical(categories=[EWeightedUserProfileType.ONLY_IMPRESSIONS.value]),
    ),
}


def compute_difference_between_impressions_and_interactions(
    uim: sp.csr_matrix,
    urm: sp.csr_matrix,
) -> sp.csr_matrix:
    """
    This function computes the set difference between `uim` and `urm`, i.e. it keeps the values inside `uim` that
    does not exist in `urm`. This function assumes that both matrices are binary, with values being either 1 or 0.
    """

    # The `-` operator yields a matrix with three possible outcomes on each cell of the matrix.
    # Value: 1. Only the `uim` had data.
    # Value: 0. Both matrices were zero or both matrices had data.
    # Value: -1. Only the `urm` had data.
    sp_difference_uim_urm: sp.coo_matrix = (uim - urm).tocoo()

    # Given that this is a set difference between `uim` and `urm` and based on the resulting values above,
    # we only keep those positions in which the value is 1.
    indices_only_on_impressions = sp_difference_uim_urm.data == 1.
    arr_rows_only_on_impressions = sp_difference_uim_urm.row[indices_only_on_impressions]
    arr_cols_only_on_impressions = sp_difference_uim_urm.col[indices_only_on_impressions]
    arr_data_only_on_impressions = sp_difference_uim_urm.data[indices_only_on_impressions]

    sp_impressions_profile: sp.csr_matrix = sp.csr_matrix(
        (
            arr_data_only_on_impressions,
            (arr_rows_only_on_impressions, arr_cols_only_on_impressions)
        ),
        dtype=np.float32,
        shape=sp_difference_uim_urm.shape,
    )
    assert np.all(sp_impressions_profile.data == 1.)

    return sp_impressions_profile


class BaseWeightedUserProfileRecommender(AbstractExtendedBaseRecommender, ABC):
    RECOMMENDER_NAME = "BaseWeightedUserProfileRecommender"
    ATTR_NAME_W_SPARSE = "W_sparse"

    def __init__(
        self,
        *,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        trained_recommender: BaseSimilarityMatrixRecommender,
        **kwargs,
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
            urm_train=urm_train,
        )

        if not hasattr(trained_recommender, self.ATTR_NAME_W_SPARSE):
            raise AttributeError(
                f"Cannot weight user profiles on the recommender {trained_recommender} as it has not been trained (it "
                f"lacks the attribute '{self.ATTR_NAME_W_SPARSE}'."
            )

        self._sparse_user_profile: sp.csr_matrix = sp.csr_matrix([], dtype=np.float32)
        self._sparse_similarity: sp.csr_matrix = sp.csr_matrix([], dtype=np.float32)

        self._uim_train: sp.csr_matrix = check_matrix(X=uim_train, format="csr", dtype=np.float32)
        self._alpha: float = 0.0
        self._sign: int = 1
        self._weighted_user_profile_type: EWeightedUserProfileType = EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS

        self.trained_recommender = trained_recommender

    def fit(
        self,
        alpha: float,
        sign: int,
        weighted_user_profile_type: str,
        **kwargs,
    ) -> None:
        assert sign == 1 or sign == -1
        assert alpha >= 0.

        self._alpha = alpha
        self._sign = sign
        self._weighted_user_profile_type = EWeightedUserProfileType(weighted_user_profile_type)

        if EWeightedUserProfileType.ONLY_IMPRESSIONS == self._weighted_user_profile_type:
            sparse_user_profile: sp.csr_matrix = self._uim_train.copy()
            sparse_user_profile.eliminate_zeros()

        elif EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS == self._weighted_user_profile_type:
            sp_impressions_profile = compute_difference_between_impressions_and_interactions(
                uim=self._uim_train,
                urm=self.URM_train,
            )

            sparse_user_profile = (
                self.URM_train + (self._sign * self._alpha * sp_impressions_profile)
            )
            sparse_user_profile.eliminate_zeros()

        else:
            raise ValueError(f"Invalid {weighted_user_profile_type}. Valid values are {list(EWeightedUserProfileType)}.")

        sp_similarity = getattr(self.trained_recommender, self.ATTR_NAME_W_SPARSE)

        format = "csr" if sp.issparse(sp_similarity) else "npy"
        self._sparse_similarity = check_matrix(X=sp_similarity, format=format, dtype=np.float32)
        self._sparse_user_profile = check_matrix(X=sparse_user_profile, format="csr", dtype=np.float32)

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
                "_alpha": self._alpha,
                "_sign": self._sign,
                "_weighted_user_profile_type": self._weighted_user_profile_type,
                "_sparse_user_profile": self._sparse_user_profile,
                "_sparse_similarity": self._sparse_similarity,
            }
        )

    def validate_load_trained_recommender(self, *args, **kwargs) -> None:
        assert hasattr(self, "_alpha") and self._alpha > 0.
        assert hasattr(self, "_sign") and (self._sign == -1 or self._sign == 1)
        assert hasattr(self, "_weighted_user_profile_type")
        assert hasattr(self, "_sparse_similarity") and self._sparse_similarity.nnz > 0
        assert hasattr(self, "_sparse_user_profile") and self._sparse_user_profile.nnz > 0

        self._sparse_similarity = check_matrix(
            X=self._sparse_similarity,
            format="csr",
            dtype=np.float32
        )
        self._sparse_user_profile = check_matrix(
            X=self._sparse_user_profile,
            format="csr",
            dtype=np.float32
        )


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

        self.RECOMMENDER_NAME = f"ItemWeightedUserProfileRecommender_{trained_recommender.RECOMMENDER_NAME}"

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

    @classmethod
    def _validate_load_trained_recommender(cls, *args, **kwargs) -> bool:
        if "trained_recommender" not in kwargs:
            return False

        trained_recommender = kwargs["trained_recommender"]

        instance_has_item_similarity = (
            isinstance(trained_recommender, BaseItemSimilarityMatrixRecommender)
            and hasattr(trained_recommender, cls.ATTR_NAME_W_SPARSE)
        )

        return instance_has_item_similarity


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

        self.RECOMMENDER_NAME = f"UserWeightedUserProfileRecommender_{trained_recommender.RECOMMENDER_NAME}"

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

    @classmethod
    def _validate_load_trained_recommender(cls, *args, **kwargs) -> bool:
        if "trained_recommender" not in kwargs:
            return False

        trained_recommender = kwargs["trained_recommender"]

        instance_has_user_similarity = (
            isinstance(trained_recommender, BaseUserSimilarityMatrixRecommender)
            and hasattr(trained_recommender, cls.ATTR_NAME_W_SPARSE)
        )

        return instance_has_user_similarity
