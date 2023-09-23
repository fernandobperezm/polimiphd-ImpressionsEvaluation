from typing import Optional

import numpy as np
import pytest
import scipy.sparse as sp
from Recommenders.BaseRecommender import BaseRecommender
from mock import patch

from impressions_evaluation.impression_recommenders.re_ranking.hard_frequency_capping import (
    HardFrequencyCappingRecommender,
    T_MODE,
)
from tests.conftest import seed

test_trained_recommender_compute_item_score_all_users: np.ndarray = np.array(
    [
        [1, 6, 3, 2, 3, 5, 4],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1],
        [9, 7, 5, 4, 8, 7, 3],
        [1, 6, 3, 2, 3, 5, 4],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1],
        [9, 7, 5, 4, 8, 7, 3],
    ],
    dtype=np.float32,
)


test_trained_recommender_compute_item_score_some_users: np.ndarray = np.array(
    [
        [1, 6, 3, 2, 3, 5, 4],
        [1, 1, 1, 1, 1, 1, 1],
        [7, 6, 5, 4, 3, 2, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1],
        [9, 7, 5, 4, 8, 7, 3],
    ],
    dtype=np.float32,
)


def compute_expected_item_scores(
    arr_item_scores: np.ndarray,
    threshold: int,
    mode: T_MODE,
    uim_frequency: sp.csr_matrix,
    list_test_users: list[int],
    list_test_items: Optional[list[int]],
    num_items: int,
) -> np.ndarray:
    num_rows, num_cols = arr_item_scores.shape
    arr_expected_scores = arr_item_scores.copy().astype(
        np.float32,
    )

    arr_all_items = np.arange(
        start=0,
        stop=num_items,
    )

    for user_id, row_idx in zip(list_test_users, range(num_rows)):
        for item_id, col_idx in zip(arr_all_items, range(num_cols)):
            val_score = arr_expected_scores[row_idx, col_idx]
            val_frequency = uim_frequency[user_id, item_id]

            val_new_score = val_score
            if mode == "leq" and val_frequency > threshold:
                val_new_score = np.NINF

            elif mode == "geq" and val_frequency < threshold:
                val_new_score = np.NINF

            elif list_test_items is not None and item_id not in list_test_items:
                val_new_score = np.NINF

            arr_expected_scores[row_idx, col_idx] = val_new_score

    return arr_expected_scores


@pytest.mark.parametrize("test_threshold", [1, 2, 4, 100])
@pytest.mark.parametrize("test_mode", ["leq", "geq"])
class TestHardFrequencyCappingRecommender:
    def test_fit_function_saves_values(
        self,
        urm: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        test_threshold: int,
        test_mode: T_MODE,
    ):
        # arrange
        test_recommender = BaseRecommender(URM_train=urm)

        rec = HardFrequencyCappingRecommender(
            urm_train=urm,
            uim_frequency=uim_frequency,
            trained_recommender=test_recommender,
            seed=seed,
        )

        rec.fit(
            threshold=test_threshold,
            mode=test_mode,
        )

        # assert
        assert rec._hfc_frequency_threshold == test_threshold
        assert rec._hfc_mode == test_mode

    # Test two cases of users: ALL USERS, SOME USERS
    @pytest.mark.parametrize(
        "test_users,test_arr_item_scores",
        [
            (
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                test_trained_recommender_compute_item_score_all_users,
            ),
            (
                [0, 1, 3, 6, 7, 8, 9],
                test_trained_recommender_compute_item_score_some_users,
            ),
        ],
    )
    # Test three cases of items: ALL ITEMS (None), SOME ITEMS, ALL ITEMS (LISTED)
    @pytest.mark.parametrize(
        "test_items",
        [None, [1, 2, 5], [0, 1, 2, 3, 4, 5, 6]],
    )
    def test_correct_computation_of_user_item_scores(
        self,
        urm: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        num_users: int,
        num_items: int,
        test_threshold: int,
        test_mode: T_MODE,
        test_users: list[int],
        test_arr_item_scores: np.ndarray,
        test_items: Optional[list[int]],
    ):
        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            "_compute_item_score",
            return_value=test_arr_item_scores,
        ) as _:
            # arrange
            test_cutoff = 3
            arr_expected_item_scores = compute_expected_item_scores(
                arr_item_scores=test_arr_item_scores,
                threshold=test_threshold,
                mode=test_mode,
                uim_frequency=uim_frequency,
                list_test_users=test_users,
                list_test_items=test_items,
                num_items=num_items,
            )

            rec = HardFrequencyCappingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(
                threshold=test_threshold,
                mode=test_mode,
            )
            recommendations, scores = rec.recommend(
                user_id_array=test_users,
                items_to_compute=test_items,
                cutoff=test_cutoff,
                remove_seen_flag=False,
                remove_top_pop_flag=False,
                remove_custom_items_flag=False,
                return_scores=True,
            )

            # assert
            assert np.allclose(arr_expected_item_scores, scores)
