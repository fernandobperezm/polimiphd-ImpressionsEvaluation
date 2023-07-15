import numpy as np
import scipy.sparse as sp
import sklearn

from recsys_framework_extensions.recommenders.graph_based.p3_alpha import (
    ExtendedP3AlphaRecommender,
    create_pui_and_piu,
)


def create_pui_and_piu_using_impressions(
    urm_train: sp.csr_matrix,
    uim_train: sp.csr_matrix,
    alpha: float,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    # Pui is the row-normalized urm
    Pui = sklearn.preprocessing.normalize(
        urm_train,
        norm="l1",
        axis=1,
    )

    # Piu is the column-normalized, "boolean" urm transposed
    X_bool = uim_train.transpose(
        copy=True,
    )
    X_bool.data = np.ones(
        X_bool.data.size,
        dtype=np.float32,
    )
    # ATTENTION: axis is still 1 because i transposed before the normalization
    Piu = sklearn.preprocessing.normalize(
        X_bool,
        norm="l1",
        axis=1,
    )

    # Alfa power
    if alpha != 1.0:
        Pui = Pui.power(alpha)
        Piu = Piu.power(alpha)

    return Pui, Piu


class ImpressionsProfileP3AlphaRecommender(ExtendedP3AlphaRecommender):
    RECOMMENDER_NAME = "ImpressionsProfileP3AlphaRecommender"

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
        self.uim_train = uim_train

    def __str__(self) -> str:
        return (
            f"ImpressionsProfileP3Alpha("
            f"alpha={self.alpha}, "
            f"top_k={self.top_k}, "
            f"normalize_similarity={self.normalize_similarity}"
            f")"
        )

    def fit(
        self,
        *,
        top_k: int = 100,
        alpha: float = 1.0,
        normalize_similarity: bool = False,
        **kwargs,
    ) -> None:
        self.top_k = top_k
        self.alpha = alpha
        self.normalize_similarity = normalize_similarity

        self.p_ui, self.p_iu = create_pui_and_piu(
            urm_train=self.uim_train,
            alpha=self.alpha,
        )

        self.create_similarity_matrix()


class ImpressionsDirectedP3AlphaRecommender(ExtendedP3AlphaRecommender):
    RECOMMENDER_NAME = "ImpressionsDirectedP3AlphaRecommender"

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
        self.uim_train = uim_train

    def __str__(self) -> str:
        return (
            f"ImpressionsDirectedP3Alpha("
            f"alpha={self.alpha}, "
            f"top_k={self.top_k}, "
            f"normalize_similarity={self.normalize_similarity}"
            f")"
        )

    def fit(
        self,
        *,
        top_k: int = 100,
        alpha: float = 1.0,
        normalize_similarity: bool = False,
        **kwargs,
    ) -> None:
        self.top_k = top_k
        self.alpha = alpha
        self.normalize_similarity = normalize_similarity

        self.p_ui, self.p_iu = create_pui_and_piu_using_impressions(
            urm_train=self.urm_train,
            uim_train=self.uim_train,
            alpha=self.alpha,
        )

        self.create_similarity_matrix()
