import numpy as np
import scipy.sparse as sp
from Recommenders.Recommender_utils import check_matrix

from recsys_framework_extensions.recommenders.graph_based.rp3_beta import (
    ExtendedRP3BetaRecommender,
)
from sklearn.preprocessing import normalize


class ImpressionsProfileRP3BetaRecommender(ExtendedRP3BetaRecommender):
    RECOMMENDER_NAME = "ImpressionsProfileRP3BetaRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            urm_train=urm_train,
            verbose=verbose,
        )
        self.uim_train = check_matrix(uim_train.copy(), "csr", dtype=np.float32)
        self.uim_train.eliminate_zeros()

    def __str__(self) -> str:
        return (
            f"ImpressionsProfileRP3Beta("
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"top_k={self.top_k}, "
            f"normalize_similarity={self.normalize_similarity}"
            f")"
        )

    def create_degree_array(
        self,
    ):
        _, num_items = self.uim_train.shape

        X_bool = self.uim_train.transpose(
            copy=True,
        )
        X_bool.data = np.ones(
            shape=X_bool.data.size,
            dtype=np.float32,
        )
        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()
        non_zero_mask = X_bool_sum != 0.0
        arr_degree = np.zeros(
            shape=num_items,
            dtype=np.float32,
        )
        arr_degree[non_zero_mask] = np.power(
            X_bool_sum[non_zero_mask],
            -self.beta,
        )

        self.arr_degree = arr_degree

    def create_pui_and_piu(
        self,
    ):
        # Pui is the row-normalized urm
        Pui = normalize(
            self.uim_train,
            norm="l1",
            axis=1,
        )

        # Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.uim_train.transpose(
            copy=True,
        )
        X_bool.data = np.ones(
            X_bool.data.size,
            dtype=np.float32,
        )
        # ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(
            X_bool,
            norm="l1",
            axis=1,
        )

        # Alfa power
        if self.alpha != 1.0:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        self.p_ui = Pui
        self.p_iu = Piu


class ImpressionsDirectedRP3BetaRecommender(ExtendedRP3BetaRecommender):
    RECOMMENDER_NAME = "ImpressionsDirectedRP3BetaRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            urm_train=urm_train,
            verbose=verbose,
        )
        self.uim_train = check_matrix(uim_train.copy(), "csr", dtype=np.float32)
        self.uim_train.eliminate_zeros()

    def __str__(self) -> str:
        return (
            f"ImpressionsDirectedRP3Beta("
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"top_k={self.top_k}, "
            f"normalize_similarity={self.normalize_similarity}"
            f")"
        )

    def create_degree_array(
        self,
    ):
        _, num_items = self.uim_train.shape

        X_bool = self.uim_train.transpose(
            copy=True,
        )
        X_bool.data = np.ones(
            shape=X_bool.data.size,
            dtype=np.float32,
        )
        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()
        non_zero_mask = X_bool_sum != 0.0
        arr_degree = np.zeros(
            shape=num_items,
            dtype=np.float32,
        )
        arr_degree[non_zero_mask] = np.power(
            X_bool_sum[non_zero_mask],
            -self.beta,
        )

        self.arr_degree = arr_degree

    def create_pui_and_piu(
        self,
    ):
        # Pui is the row-normalized urm
        Pui = normalize(
            self.URM_train,
            norm="l1",
            axis=1,
        )

        # Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.uim_train.transpose(
            copy=True,
        )
        X_bool.data = np.ones(
            X_bool.data.size,
            dtype=np.float32,
        )
        # ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(
            X_bool,
            norm="l1",
            axis=1,
        )

        # Alfa power
        if self.alpha != 1.0:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        self.p_ui = Pui
        self.p_iu = Piu
