import logging
import sys

import attrs
import numpy as np
import scipy.sparse as sp
from Recommenders.BaseMatrixFactorizationRecommender import (
    BaseMatrixFactorizationRecommender,
)
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.BaseSimilarityMatrixRecommender import (
    BaseItemSimilarityMatrixRecommender,
)
from Recommenders.MatrixFactorization.PureSVDRecommender import (
    compute_W_sparse_from_item_latent_factors,
)
from Recommenders.Recommender_utils import check_matrix
from recsys_framework_extensions.recommenders.base import (
    SearchHyperParametersBaseRecommender,
    AbstractExtendedBaseRecommender,
)
from skopt.space import Real

logger = logging.getLogger(__name__)


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersFoldedMatrixFactorizationRecommender(
    SearchHyperParametersBaseRecommender
):
    # top_k: Integer = attrs.field(
    #     default=Integer(
    #         low=5,
    #         high=1000,
    #         prior="uniform",
    #         base=10,
    #     )
    # )
    ratio_items: Real = attrs.field(
        default=Real(
            low=1e-2,  # 1% of items
            high=1,  # 100% of items.
            prior="uniform",
            base=10,
        )
    )


class FoldedMatrixFactorizationRecommender(
    AbstractExtendedBaseRecommender, BaseItemSimilarityMatrixRecommender
):
    RECOMMENDER_NAME = f"FoldedMatrixFactorizationRecommender"
    ATTR_NAME_ITEM_FACTORS = "ITEM_factors"
    ATTR_NAME_W_SPARSE = "W_sparse"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        trained_recommender: BaseMatrixFactorizationRecommender,
    ):
        """

        Notes
        -----
        Useful to fold-in the following recommenders:
            - `PureSVDRecommender`,
            - `NMFRecommender`,
            - `IALSRecommender`,
            - `MatrixFactorization_BPR_Cython`
            - `MatrixFactorization_FunkSVD_Cython`

        Parameters
        ----------
        urm_train: csr_matrix
            A sparse matrix of shape (M, N), where M = #Users, and N = #Items. The content of this matrix is the latest
            recorded implicit interaction for the user-item pair (u,i), i.e., urm_train[u,i] = 1

        trained_recommender: BaseRecommender
            An instance of a trained recommender. This recommender must be able to execute
            `trained_recommender._compute_item_scores` without raising exceptions.
        """
        super().__init__(urm_train=urm_train)

        if not self.can_recommender_be_folded(recommender_instance=trained_recommender):
            raise AttributeError(
                f"Cannot fold-in the recommender {trained_recommender} as it has not been trained (it lacks the "
                f"attribute '{self.ATTR_NAME_ITEM_FACTORS}'."
            )

        self.RECOMMENDER_NAME = (
            f"FoldedMatrixFactorization_{trained_recommender.RECOMMENDER_NAME}"
        )

        self.trained_recommender = trained_recommender
        self.W_sparse: sp.csr_matrix = sp.csr_matrix([])

        self.ratio_items: float = 0.0

        # Take at most 25 GB of RAM = 25_000_000_000 memory
        self.maximum_memory_bytes = int(25e9)

    def fit(
        self,
        # top_k: Optional[int] = None,
        *,
        ratio_items: float,
        # *args,
        **kwargs,
    ) -> None:
        if not (0.0 <= ratio_items <= 1.0):
            raise ValueError(
                f"Invalid value for hyper-parameter `ratio_items`. Values are float between 0 and 1. Received value: {ratio_items}."
            )
        self.ratio_items = float(ratio_items)

        item_factors: np.ndarray = getattr(
            self.trained_recommender,
            self.ATTR_NAME_ITEM_FACTORS,
        )
        # if top_k is None:
        #     top_k = self.n_items

        top_k = int(self.ratio_items * self.n_items)

        # Ensure at least top-1
        if top_k < 1:
            top_k = 1

        # Ensure at least top-n_items
        if top_k > self.n_items:
            top_k = self.n_items

        # Numpy array memory usage (in bytes) for 2d array is (dim1 * dim2 * 4) for float32 or int32 and (dim1 * dim2 * 8) for float64 or int64.
        memory_required = self.n_items * top_k * 4
        logger.debug(
            f"\n\t*{self.ratio_items=}"
            f"\n\t*{top_k=}"
            f"\n\t*Maximum GiB: {self.maximum_memory_bytes / 1e9:.2f} GiB"
            f"\n\t*Required GiB: {memory_required / 1e9:.2f} GiB"
            f"\n\t*Overpass memory?: {memory_required > self.maximum_memory_bytes}"
        )
        if memory_required > self.maximum_memory_bytes:
            raise ValueError(
                f"This ratio value takes more memory than permitted. "
                f"Ratio value: {self.ratio_items}. "
                f"Memory required: {memory_required / 1e9} GB. "
                f"Maximum Memory: {self.maximum_memory_bytes / 1e9} GB."
            )

        sparse_similarity_matrix = compute_W_sparse_from_item_latent_factors(
            ITEM_factors=item_factors,
            topK=top_k,
        )

        self.W_sparse = check_matrix(
            X=sparse_similarity_matrix,
            format="csr",
            dtype=np.float32,
            # dtype=np.float64,
        )

        memory_required_w_sparse = (
            self.W_sparse.data.nbytes
            + self.W_sparse.indices.nbytes
            + self.W_sparse.indptr.nbytes
        )
        logger.debug(
            f"\n\t*Memory required by W_sparse."
            f"\n\t*{self.W_sparse.shape=}"
            f"\n\t*{self.W_sparse.dtype=}"
            f"\n\t*{memory_required_w_sparse=}-{memory_required_w_sparse / 1e9:.2f} GiB."
            f"\n\t*{sys.getsizeof(self.W_sparse)=}-{sys.getsizeof(self.W_sparse) / 1e9:.2f} GiB"
            f"\n\t*"
            f"{self.W_sparse.data.shape=}-"
            f"{self.W_sparse.data.nbytes=}-"
            f"{self.W_sparse.data.nbytes / 1e9:.2f} GiB"
            f"\n\t*"
            f"{self.W_sparse.indices.shape=}-"
            f"{self.W_sparse.indices.nbytes=}-"
            f"{self.W_sparse.indices.nbytes / 1e9:.2f} GiB"
            f"\n\t*"
            f"{self.W_sparse.indptr.shape=}-"
            f"{self.W_sparse.indptr.nbytes=}-"
            f"{self.W_sparse.indptr.nbytes / 1e9:.2f} GiB"
        )

    def validate_load_trained_recommender(self, *args, **kwargs) -> None:
        assert hasattr(self, self.ATTR_NAME_W_SPARSE)

        self.W_sparse = check_matrix(
            X=self.W_sparse,
            format="csr",
            dtype=np.float64,
        )

    @staticmethod
    def can_recommender_be_folded(recommender_instance: BaseRecommender) -> bool:
        return hasattr(
            recommender_instance,
            FoldedMatrixFactorizationRecommender.ATTR_NAME_ITEM_FACTORS,
        )
