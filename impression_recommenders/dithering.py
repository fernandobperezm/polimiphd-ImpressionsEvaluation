from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.stats as st

from Recommenders.BaseRecommender import BaseRecommender


T_RANK_METHOD = Literal["average", "min", "max", "dense", "ordinal"]


class DitheringRecommender(BaseRecommender):
    """
    Dithering [1]_ is a plug-in recommender that re-ranks recommendations by adding random noise to the original ranks
    in the recommendation list.

    The formula for dithering is:

    .. math::
     r^{*}_{u,i} = log(rank(r_{u,i})) + N(0, log(\\epsilon))

     \\epsilon = \\Delta rank(r_{u,i}) / rank(r_{u,i})

    Where :math:`r_{u,i}` is a function that returns the rank assigned to the item :math:`i` for user :math:`u`. Ranks
    returned by this function are higher if the relevance :math:`r_{u,i}` is higher.

    References
    ----------
    .. [1] Maya Hristakeva, Daniel Kershaw, Marco Rossetti, Petr Knoth, Benjamin Pettit, Saúl Vargas, and Kris Jack.
       2017. Building recommender systems for scholarly information. In Proceedings of the 1st Workshop on
       Scholarly Web Mining (SWM '17). Association for Computing Machinery, New York, NY, USA, 25–32.
       DOI: https://doi.org/10.1145/3057148.3057152

    """

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        trained_recommender: BaseRecommender,
        seed: int,
    ):
        super().__init__(
            URM_train=urm_train,
            verbose=True,
        )

        self._trained_recommender = trained_recommender
        self._rank_method: T_RANK_METHOD = "average"

        self._rng = np.random.default_rng(seed=seed)
        self._rng_func_mean = 0.0
        self._rng_func_variance = 1.0

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """

        :class:`BaseRecommender` ranks items in a further step by their score. Items with the highest scores are
        positioned first in the list.

        :class:`DitheringRecommender` computes the item scores as the

        """

        item_scores_trained_recommender = self._trained_recommender._compute_item_score(
            user_id_array=user_id_array,
            items_to_compute=items_to_compute,
        )

        # `st.rankdata` computes the rankings from 1 to n. It gives 1 to the highest score and n to the lowest score.
        # Dithering requires that 1 is given to the lowest score and n to the highest score.
        # Therefore, we apply `st.rankdata` to -item_scores_trained_recommender (notice the - sign) so the highest
        # score become the lowest and vice-versa. This ensures that the originally-highest-score gets the highest
        # ranking (n) while the originally-lowest-score gets the lowest ranking (1).
        ranks_item_scores_trained_recommender = st.rankdata(
            a=-item_scores_trained_recommender,
            method=self._rank_method,
            axis=1,
        )

        log_ranks_item_scores = np.log(
            ranks_item_scores_trained_recommender
        )

        random_noise_item_scores = self._rng.normal(
            loc=self._rng_func_mean,
            scale=self._rng_func_variance,
            size=ranks_item_scores_trained_recommender.shape,
        )

        new_item_scores = log_ranks_item_scores + random_noise_item_scores

        assert item_scores_trained_recommender.shape == ranks_item_scores_trained_recommender.shape
        assert item_scores_trained_recommender.shape == log_ranks_item_scores.shape
        assert item_scores_trained_recommender.shape == random_noise_item_scores.shape
        assert item_scores_trained_recommender.shape == new_item_scores.shape

        return new_item_scores

    def fit(
        self,
        rank_method: T_RANK_METHOD,
        epsilon: float,
        **kwargs,
    ):
        self._rank_method = rank_method
        self._rng_func_variance = np.log(epsilon)

    def save_model(self, folder_path, file_name=None):
        pass
