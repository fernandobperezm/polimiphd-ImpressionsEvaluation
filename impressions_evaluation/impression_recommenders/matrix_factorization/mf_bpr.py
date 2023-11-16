"""
"""
from typing import Optional

from Recommenders.BaseMatrixFactorizationRecommender import (
    BaseMatrixFactorizationRecommender,
)
from Recommenders.Incremental_Training_Early_Stopping import (
    Incremental_Training_Early_Stopping,
)
from Recommenders.Recommender_utils import check_matrix

from impressions_evaluation.impression_recommenders.matrix_factorization.cython.mf import (
    SampleImpressionsNegativeMatrixFactorizationCythonBPRModel as MatrixFactorizationModel,
)


import sys
import numpy as np
import scipy.sparse as sp


class MatrixFactorizationBPRImpressionsNegatives(
    BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping
):
    RECOMMENDER_NAME = "MatrixFactorizationBPRImpressionsNegatives"

    def __init__(self, urm_train: sp.csr_matrix, uim_train: sp.csr_matrix, **kwargs):
        super().__init__(
            URM_train=urm_train,
            **kwargs,
        )
        self.uim_train = check_matrix(uim_train, dtype=np.float32, format="csr")
        self.uim_train.eliminate_zeros()

        self.matrix_factorization_model: Optional[MatrixFactorizationModel] = None

        self.epochs: int = 300
        self.batch_size: int = 300
        self.num_factors: int = 10
        self.positive_threshold_BPR: Optional[int] = None

        self.use_embeddings: bool = True
        self.WARP_neg_item_attempts: int = 10

        self.learning_rate: float = 0.001
        self.use_bias: bool = True
        self.sgd_mode: str = "sgd"

        self.impression_sampling_mode: str = "inside"
        self.impression_sampling_inside_ratio: float = 0.5
        self.negative_interactions_quota: float = 0.0
        self.dropout_quota: Optional[float] = 0.0

        self.init_mean: float = 0.0
        self.init_std_dev: float = 0.1
        self.user_reg: float = 0.0
        self.item_reg: float = 0.0
        self.bias_reg: float = 0.0
        self.positive_reg: float = 0.0
        self.negative_reg: float = 0.0

        self.random_seed: Optional[int] = None

        self.USER_factors: np.ndarray = np.array([], dtype=np.float32)
        self.ITEM_factors: np.ndarray = np.array([], dtype=np.float32)

        self.USER_factors_best: np.ndarray = np.array([], dtype=np.float32)
        self.ITEM_factors_best: np.ndarray = np.array([], dtype=np.float32)

    def fit(
        self,
        epochs=300,
        batch_size=1000,
        num_factors=10,
        positive_threshold_BPR=None,
        learning_rate=0.001,
        use_bias=True,
        use_embeddings=True,
        WARP_neg_item_attempts=10,
        sgd_mode="sgd",
        impression_sampling_mode="inside",
        impression_sampling_inside_ratio=0.5,
        negative_interactions_quota=0.0,
        dropout_quota=None,
        init_mean=0.0,
        init_std_dev=0.1,
        user_reg=0.0,
        item_reg=0.0,
        bias_reg=0.0,
        positive_reg=0.0,
        negative_reg=0.0,
        random_seed=None,
        **early_stopping_kwargs,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_factors = num_factors
        self.positive_threshold_BPR = positive_threshold_BPR

        self.learning_rate = learning_rate
        self.sgd_mode = sgd_mode
        self.use_bias = use_bias
        self.use_embeddings = use_embeddings

        self.impression_sampling_mode = impression_sampling_mode
        self.impression_sampling_inside_ratio = impression_sampling_inside_ratio
        self.negative_interactions_quota = negative_interactions_quota
        self.dropout_quota = dropout_quota

        self.WARP_neg_item_attempts = WARP_neg_item_attempts

        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.user_reg = user_reg
        self.item_reg = item_reg
        self.bias_reg = bias_reg
        self.positive_reg = positive_reg
        self.negative_reg = negative_reg

        self.random_seed = random_seed

        if not (0.0 <= negative_interactions_quota < 1.0):
            raise ValueError(
                f"The parameter `negative_interactions_quota` must be be a float value >=0 and < 1.0, provided was {negative_interactions_quota}"
            )

        # Select only positive interactions
        URM_train_positive = self.URM_train.copy()

        if self.positive_threshold_BPR is not None:
            URM_train_positive.data = (
                URM_train_positive.data >= self.positive_threshold_BPR
            )
            URM_train_positive.eliminate_zeros()

            if URM_train_positive.nnz <= 0:
                raise ValueError(
                    f"MatrixFactorizationBPRImpressionsNegatives: `URM_train_positive` is empty, positive threshold ({self.positive_threshold_BPR=}) is too high"
                )

        self.matrix_factorization_model = MatrixFactorizationModel(
            URM_train=URM_train_positive,
            UIM_train=self.uim_train,
            n_factors=self.num_factors,
            batch_size=self.batch_size,
            dropout_quota=self.dropout_quota,
            WARP_neg_item_attempts=self.WARP_neg_item_attempts,
            learning_rate=self.learning_rate,
            use_bias=self.use_bias,
            use_embeddings=self.use_embeddings,
            user_reg=self.user_reg,
            positive_reg=self.positive_reg,
            negative_reg=self.negative_reg,
            sgd_mode=self.sgd_mode,
            init_mean=self.init_mean,
            init_std_dev=self.init_std_dev,
            verbose=self.verbose,
            random_seed=self.random_seed,
        )
        self._prepare_model_for_validation()
        self._update_best_model()

        self._train_with_early_stopping(
            self.epochs,
            **early_stopping_kwargs,
        )

        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best

        sys.stdout.flush()

    def _prepare_model_for_validation(self):
        assert self.matrix_factorization_model is not None

        self.USER_factors = self.matrix_factorization_model.get_USER_factors()
        self.ITEM_factors = self.matrix_factorization_model.get_ITEM_factors()

    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()

    def _run_epoch(self, num_epoch):
        assert self.matrix_factorization_model is not None
        self.matrix_factorization_model.epochIteration_Cython()
