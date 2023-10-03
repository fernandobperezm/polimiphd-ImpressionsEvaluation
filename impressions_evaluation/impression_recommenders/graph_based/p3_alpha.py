import numpy as np
import scipy.sparse as sp
from Recommenders.Recommender_utils import check_matrix

from recsys_framework_extensions.recommenders.graph_based.p3_alpha import (
    ExtendedP3AlphaRecommender,
)
from sklearn.preprocessing import normalize


class ImpressionsProfileP3AlphaRecommender(ExtendedP3AlphaRecommender):
    RECOMMENDER_NAME = "ImpressionsProfileP3AlphaRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        verbose: bool = False,
        **kwargs,
    ):
        assert urm_train.shape == uim_train.shape

        super().__init__(
            urm_train=urm_train,
            verbose=verbose,
        )
        self.uim_train = check_matrix(uim_train.copy(), "csr", dtype=np.float32)
        self.uim_train.eliminate_zeros()

    def __str__(self) -> str:
        return (
            f"ImpressionsProfileP3Alpha("
            f"alpha={self.alpha}, "
            f"top_k={self.top_k}, "
            f"normalize_similarity={self.normalize_similarity}"
            f")"
        )

    def create_adjacency_matrix(
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


class ImpressionsProfileWithFrequencyP3AlphaRecommender(ExtendedP3AlphaRecommender):
    RECOMMENDER_NAME = "ImpressionsProfileWithFrequencyP3AlphaRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        verbose: bool = False,
        **kwargs,
    ):
        assert urm_train.shape == uim_train.shape

        super().__init__(
            urm_train=urm_train,
            verbose=verbose,
        )
        self.uim_train = check_matrix(uim_train.copy(), "csr", dtype=np.float32)
        self.uim_train.eliminate_zeros()

        self.uim_frequency = check_matrix(uim_frequency.copy(), "csr", dtype=np.float32)
        self.uim_frequency.eliminate_zeros()

    def __str__(self) -> str:
        return (
            f"ImpressionsProfileWithFrequencyP3Alpha("
            f"alpha={self.alpha}, "
            f"top_k={self.top_k}, "
            f"normalize_similarity={self.normalize_similarity}"
            f")"
        )

    def create_adjacency_matrix(
        self,
    ):
        # Pui is the row-normalized urm
        Pui = normalize(
            self.uim_frequency,
            norm="l1",
            axis=1,
        )

        # Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.uim_frequency.transpose(
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


class ImpressionsDirectedP3AlphaRecommender(ExtendedP3AlphaRecommender):
    RECOMMENDER_NAME = "ImpressionsDirectedP3AlphaRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        verbose: bool = False,
        **kwargs,
    ):
        assert urm_train.shape == uim_train.shape

        super().__init__(
            urm_train=urm_train,
            verbose=verbose,
        )
        self.uim_train = check_matrix(uim_train.copy(), "csr", dtype=np.float32)
        self.uim_train.eliminate_zeros()

    def __str__(self) -> str:
        return (
            f"ImpressionsDirectedP3Alpha("
            f"alpha={self.alpha}, "
            f"top_k={self.top_k}, "
            f"normalize_similarity={self.normalize_similarity}"
            f")"
        )

    def create_adjacency_matrix(
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


class ImpressionsDirectedWithFrequencyP3AlphaRecommender(ExtendedP3AlphaRecommender):
    RECOMMENDER_NAME = "ImpressionsDirectedWithFrequencyP3AlphaRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        verbose: bool = False,
        **kwargs,
    ):
        assert urm_train.shape == uim_train.shape

        super().__init__(
            urm_train=urm_train,
            verbose=verbose,
        )
        self.uim_train = check_matrix(uim_train.copy(), "csr", dtype=np.float32)
        self.uim_train.eliminate_zeros()

        self.uim_frequency = check_matrix(uim_frequency.copy(), "csr", dtype=np.float32)
        self.uim_frequency.eliminate_zeros()

    def __str__(self) -> str:
        return (
            f"ImpressionsDirectedWithFrequencyP3Alpha("
            f"alpha={self.alpha}, "
            f"top_k={self.top_k}, "
            f"normalize_similarity={self.normalize_similarity}"
            f")"
        )

    def create_adjacency_matrix(
        self,
    ):
        # Pui is the row-normalized urm
        Pui = normalize(
            self.URM_train,
            norm="l1",
            axis=1,
        )

        # Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.uim_frequency.transpose(
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
