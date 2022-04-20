import numpy as np
import scipy.sparse as sp
from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import compute_W_sparse_from_item_latent_factors

from Recommenders.Recommender_utils import check_matrix


class FoldedMatrixFactorizationRecommender(BaseItemSimilarityMatrixRecommender):
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
        super().__init__(
            URM_train=urm_train,
            verbose=True,
        )

        self._attr_name_item_factors = "ITEM_factors"
        if not hasattr(trained_recommender, self._attr_name_item_factors):
            raise AttributeError(
                f"Cannot fold-in the recommender {trained_recommender} as it has not been trained (it lacks the "
                f"attribute '{self._attr_name_item_factors}'."
            )

        self.RECOMMENDER_NAME = f"FoldedMatrixFactorization_{trained_recommender.RECOMMENDER_NAME}"

        self.trained_recommender = trained_recommender
        self.W_sparse: sp.csr_matrix = sp.csr_matrix([])

    def fit(self, top_k: int = None) -> None:
        item_factors: np.ndarray = getattr(self.trained_recommender, self._attr_name_item_factors)

        if top_k is None:
            top_k = self.n_items

        self.W_sparse = compute_W_sparse_from_item_latent_factors(
            ITEM_factors=item_factors,
            topK=top_k
        )

        self.W_sparse = check_matrix(
            X=self.W_sparse,
            format='csr',
        )
