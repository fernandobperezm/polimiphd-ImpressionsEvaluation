from typing import Optional

import attrs
import numpy as np
import scipy.sparse as sp
import scipy.stats as st

from Recommenders.BaseRecommender import BaseRecommender
from recsys_framework_extensions.recommenders.base import SearchHyperParametersBaseRecommender
from skopt.space import Real

from impression_recommenders.constants import ERankMethod


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersDitheringRecommender(SearchHyperParametersBaseRecommender):
    epsilon: Real = attrs.field(
        default=Real(
            low=1e-5,
            high=1e2,
            prior="uniform",
            base=10,
        )
    )


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
    RECOMMENDER_NAME = "DitheringRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        trained_recommender: BaseRecommender,
        seed: int,
        **kwargs,
    ):
        super().__init__(
            URM_train=urm_train,
            verbose=True,
        )

        self._trained_recommender = trained_recommender
        self._rank_method: ERankMethod = ERankMethod.MIN

        self._rng = np.random.default_rng(seed=seed)
        self._rng_func_mean = 0.0
        self._rng_func_variance = 1.0

    def _compute_item_score(
        self,
        user_id_array: list[int],
        items_to_compute: Optional[list[int]] = None
    ):
        """

        :class:`BaseRecommender` ranks items in a further step by their score. Items with the highest scores are
        positioned first in the list.

        :class:`DitheringRecommender` computes the item scores as the

        """
        assert user_id_array is not None
        assert len(user_id_array) > 0

        item_scores_trained_recommender = self._trained_recommender._compute_item_score(
            user_id_array=user_id_array,
            items_to_compute=items_to_compute,
        )

        # `st.rankdata` computes the rankings from 1 to n.
        # It gives 1 to the lowest score and n to the lowest score.
        # Dithering requires the same: 1 is given to the lowest score and n to the highest score.
        ranks_item_scores_trained_recommender = st.rankdata(
            a=item_scores_trained_recommender,
            method=self._rank_method.value,
            axis=1,
        ).astype(
            np.float32,
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

        # If we are computing scores to a specific set of items, then we must set items outside this set to np.NINF,
        # so they are not
        if items_to_compute is not None:
            arr_mask_items = np.zeros_like(new_item_scores, dtype=np.bool8)
            arr_mask_items[:, items_to_compute] = True

            # If the item is in `items_to_compute`, then keep the value from `new_item_scores`.
            # Else, set to -inf.
            new_item_scores = np.where(
                arr_mask_items,
                new_item_scores,
                np.NINF,
            )

        assert item_scores_trained_recommender.shape == ranks_item_scores_trained_recommender.shape
        assert item_scores_trained_recommender.shape == log_ranks_item_scores.shape
        assert item_scores_trained_recommender.shape == random_noise_item_scores.shape
        assert item_scores_trained_recommender.shape == new_item_scores.shape

        return new_item_scores

    def fit(
        self,
        epsilon: float,
        **kwargs,
    ):
        self._rng_func_variance = np.log(epsilon)

    def save_model(self, folder_path, file_name=None):
        pass
