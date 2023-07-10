import numpy as np
import scipy.sparse as sp
from recsys_framework_extensions.recommenders.graph_based.light_gcn import (
    ExtendedLightGCNRecommender,
    create_normalized_adjacency_matrix_from_urm,
)

__all__ = [
    "ImpressionsProfileLightGCNRecommender",
    "ImpressionsDirectedLightGCNRecommender",
]


def create_normalized_directed_adjacency_matrix_interactions_and_impressions(
    urm: sp.csr_matrix,
    uim: sp.csr_matrix,
    add_self_connection: bool = False,
) -> sp.coo_matrix:
    """
    This method creates an adjacency matrix where users and items are nodes and edges represent interactions or impressions. Particularly, all edges are directed ones: an edge is created from a user to an item when the user interacted with the item. An edge is created from an item to a user when the item has been impressed to the user.

    """
    assert urm.shape == uim.shape

    n_users, n_items = urm.shape

    arr_empty_users = sp.csr_matrix((n_users, n_users))
    arr_empty_items = sp.csr_matrix((n_items, n_items))

    # This is the adjacency matrix.
    A = sp.bmat(
        [
            [arr_empty_users, urm],
            [uim.T, arr_empty_items],
        ],
        format="csr",
    )

    if add_self_connection:
        A = A + sp.eye(A.shape[0])

    D_inv = 1 / (
        np.sqrt(
            np.array(
                A.sum(
                    axis=1,
                )
            ).squeeze()
        )
        + 1e-6
    )
    A_tilde = (
        sp.diags(D_inv)
        .dot(A)
        .dot(sp.diags(D_inv))
        .astype(
            np.float32,
        )
    )

    return sp.coo_matrix(A_tilde)


class ImpressionsProfileLightGCNRecommender(ExtendedLightGCNRecommender):
    RECOMMENDER_NAME = "ImpressionsProfileLightGCNRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        use_cython_sampler: bool = True,
        use_gpu: bool = False,
        verbose: bool = True,
    ):
        assert urm_train.shape == uim_train.shape

        super().__init__(
            urm_train=urm_train,
            use_cython_sampler=use_cython_sampler,
            use_gpu=use_gpu,
            verbose=verbose,
        )

        self.uim_train = uim_train
        self.adjacency_matrix = create_normalized_adjacency_matrix_from_urm(
            urm=uim_train,
            add_self_connection=False,
        )


class ImpressionsDirectedLightGCNRecommender(ExtendedLightGCNRecommender):
    RECOMMENDER_NAME = "ImpressionsDirectedLightGCNRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        use_cython_sampler: bool = True,
        use_gpu: bool = False,
        verbose: bool = True,
    ):
        assert urm_train.shape == uim_train.shape

        super().__init__(
            urm_train=urm_train,
            use_cython_sampler=use_cython_sampler,
            use_gpu=use_gpu,
            verbose=verbose,
        )

        self.uim_train = uim_train
        self.adjacency_matrix = (
            create_normalized_directed_adjacency_matrix_interactions_and_impressions(
                urm=urm_train,
                uim=uim_train,
                add_self_connection=False,
            )
        )
