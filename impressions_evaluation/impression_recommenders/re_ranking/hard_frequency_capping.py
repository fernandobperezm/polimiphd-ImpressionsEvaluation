from typing import Optional, Literal

import attrs
import numpy as np
import scipy.sparse as sp
from Recommenders.BaseRecommender import BaseRecommender
from recsys_framework_extensions.data.io import DataIO
from recsys_framework_extensions.recommenders.base import (
    SearchHyperParametersBaseRecommender,
    AbstractExtendedBaseRecommender,
)
from skopt.space import Integer, Categorical


T_MODE = Literal["leq", "geq"]


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersHardFrequencyCappingRecommender(
    SearchHyperParametersBaseRecommender
):
    threshold: Integer = attrs.field(
        default=Integer(
            low=1,
            high=50,
            prior="uniform",
            base=10,
        )
    )
    mode: Categorical = attrs.field(
        default=Categorical(
            categories=["leq", "geq"],
        )
    )


HARD_FREQUENCY_CAPPING_HYPER_PARAMETER_SEARCH_CONFIGURATIONS = {
    "ORIGINAL": SearchHyperParametersHardFrequencyCappingRecommender(),
    "REPRODUCIBILITY_ORIGINAL_PAPER": SearchHyperParametersHardFrequencyCappingRecommender(
        mode=Categorical(categories=["leq"]),
    ),
    "SIGNAL_ANALYSIS_LESS_OR_EQUAL_THRESHOLD": SearchHyperParametersHardFrequencyCappingRecommender(
        mode=Categorical(categories=["leq"]),
    ),
    "SIGNAL_ANALYSIS_GREAT_OR_EQUAL_THRESHOLD": SearchHyperParametersHardFrequencyCappingRecommender(
        mode=Categorical(categories=["geq"]),
    ),
}


class HardFrequencyCappingRecommender(AbstractExtendedBaseRecommender):
    RECOMMENDER_NAME = "HardFrequencyCappingRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        trained_recommender: BaseRecommender,
        **kwargs,
    ):
        super().__init__(
            urm_train=urm_train,
        )

        self._trained_recommender = trained_recommender
        self._uim_frequency = uim_frequency
        self._matrix_presentation_scores = sp.csr_matrix(np.array([], dtype=np.float32))
        self._hfc_frequency_threshold: int = 1
        self._hfc_mode: T_MODE = "geq"

        self.RECOMMENDER_NAME = (
            f"HardFrequencyCappingRecommender_{trained_recommender.RECOMMENDER_NAME}"
        )

    def _compute_item_score(
        self,
        user_id_array: list[int],
        items_to_compute: Optional[list[int]] = None,
    ) -> np.ndarray:
        """
        This function computes the item scores using the definition of hard frequency capping.



        Hard Frequency Capping holds two arrays `arr_scores_presentation` and `arr_scores_relevance`. The first tells how many times
        each item (columns) has been impressed to the users (rows). The second is the relevance score given by the
        trained recommender to each user-item pair (users in rows, items in columns).

        The new relevance score is computed by assigning the rank (higher is more relevant) to each user-item pair. To
        assign this rank for each user, items are sorted first by their presentation score `arr_scores_presentation`
        in ascending order and then by their relevance score `arr_scores_relevance` in ascending order as well.

        HardFrequencyCapping implies that items with fewer impressions will get low rank scores (therefore, highly unlikely to be
        recommended)

        This method assigns ranks (and does return sorted item indices) to comply with the `recommend` function in
        BaseRecommender, i.e., the `recommend` function expects that each user-item pair holds a relevance score,
        where items with the highest scores are recommended.

        Returns
        -------
        np.ndarray
            A (M, N) numpy array that contains the score for each user-item pair.

        """
        assert user_id_array is not None
        assert len(user_id_array) > 0

        num_score_users: int = len(user_id_array)
        num_score_items: int = self.URM_train.shape[1]

        # Dense array of shape (M,N) where M is len(user_id_array) and N is the total number of users in the dataset.
        arr_scores_relevance: np.ndarray = (
            self._trained_recommender._compute_item_score(
                user_id_array=user_id_array,
                items_to_compute=items_to_compute,
            )
        )

        arr_frequency_users_items: np.ndarray = self._uim_frequency[
            user_id_array, :
        ].todense()

        assert (num_score_users, num_score_items) == arr_scores_relevance.shape
        assert arr_scores_relevance.shape == arr_frequency_users_items.shape

        # The modes work as follow.
        # We create a Mask array where a True value will leave the predicted score as generated by the other recommended.
        # However, a False value will set those scores as np.NINF, causing the recommender to not predict those items.
        # The "leq" mode assigns True values to those items which frequency is lower or equal than the threshold.
        # The "geq" mode assigns True values to those items which frequency is higher or equal than the threshold.
        if self._hfc_mode == "leq":
            arr_frequency_mask = (
                arr_frequency_users_items <= self._hfc_frequency_threshold
            )
        elif self._hfc_mode == "geq":
            arr_frequency_mask = (
                arr_frequency_users_items >= self._hfc_frequency_threshold
            )
        else:
            raise ValueError(
                f"HFC MODE is not valid. Valid values are 'leq' and 'geq' but received {self._hfc_mode}"
            )

        # If we are computing scores to a specific set of items, then we must set items outside this set to np.NINF,
        # so they are not recommended in the next step.
        # To achieve this, we create a boolean mask where a True value indicates to use the predicted score and a False value indicates to use np.NINF.
        if items_to_compute is None:
            arr_mask_items = np.ones_like(arr_scores_relevance, dtype=np.bool8)
        else:
            arr_mask_items = np.zeros_like(arr_scores_relevance, dtype=np.bool8)
            arr_mask_items[:, items_to_compute] = True

        # We join both masks inside an AND operator, because we want to perform the update using both criteria.
        arr_scores_relevance = np.where(
            arr_frequency_mask & arr_mask_items,
            arr_scores_relevance,
            np.NINF,
        )

        assert (num_score_users, num_score_items) == arr_scores_relevance.shape

        return arr_scores_relevance

    def fit(
        self,
        threshold: int,
        mode: T_MODE,
        **kwargs,
    ):
        assert threshold > 0
        assert mode == "leq" or mode == "geq"

        self._hfc_frequency_threshold = threshold
        self._hfc_mode = mode

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
                "_hfc_frequency_threshold": self._hfc_frequency_threshold,
                "_hfc_mode": self._hfc_mode,
            },
        )

    def validate_load_trained_recommender(
        self,
        *args,
        **kwargs,
    ) -> None:
        assert hasattr(self, "_hfc_frequency_threshold")
        assert hasattr(self, "_hfc_mode")