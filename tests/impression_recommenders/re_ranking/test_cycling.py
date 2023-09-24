from typing import Literal, Optional

import pytest
from mock import patch

import numpy as np
import scipy.sparse as sp

from impressions_evaluation.impression_recommenders.re_ranking.cycling import (
    CyclingRecommender,
    T_SIGN,
    compute_cycling_recommender_score,
)
from tests.conftest import seed
from Recommenders.BaseRecommender import BaseRecommender


TEST_TRAINED_RECOMMENDER_COMPUTE_ITEM_SCORES_ALL_USERS = np.asarray(
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
    dtype=np.int32,
)

TEST_TRAINED_RECOMMENDER_COMPUTE_ITEM_SCORES_SOME_USERS = np.asarray(
    [
        [1, 6, 3, 2, 3, 5, 4],
        [1, 1, 1, 1, 1, 1, 1],
        [7, 6, 5, 4, 3, 2, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1],
        [9, 7, 5, 4, 8, 7, 3],
    ],
    dtype=np.int32,
)


TEST_RECOMMENDATIONS_RANK_ALL_USERS = [
    [6, 4, 0, 1, 5, 2],
    [5, 2, 1, 6, 4, 3],
    [5, 4, 2, 1, 6, 3],
    [0, 1, 2, 3, 4, 5],
    [4, 0, 5, 1, 3, 2],
    [6, 1, 4, 0, 3, 5],
    [6, 4, 3, 1, 5, 2],
    [6, 0, 4, 1, 5, 2],
    [0, 1, 2, 3, 4, 5],
    [0, 5, 2, 3, 4, 1],
]


TEST_RECOMMENDATIONS_NORM_ALL_USERS = [
    [6, 4, 0, 1, 5, 2],
    [5, 2, 1, 6, 4, 0],
    [5, 4, 2, 1, 6, 3],
    [0, 1, 2, 3, 4, 5],
    [4, 0, 5, 1, 3, 2],
    [6, 1, 4, 0, 3, 5],
    [4, 3, 6, 1, 5, 2],
    [6, 0, 4, 1, 5, 2],
    [0, 1, 2, 3, 4, 5],
    [0, 5, 2, 3, 4, 1],
]


class TestCyclingRecommender:
    def test_weight(
        self,
        urm: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
    ):
        # arrange
        test_recommender = BaseRecommender(URM_train=urm)
        test_weight_values = [1, 2, 4, 100]
        test_sign: Literal[1] = 1

        dict_expected_presentation_scores = {
            1: uim_frequency,
            2: np.array(
                [
                    [1, 1, 1, 1, 0, 1, 0],
                    [1, 0, 0, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 1, 1],
                    [1, 1, 2, 1, 1, 2, 0],
                    [1, 0, 1, 0, 0, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 1],
                ],
                dtype=np.float32,
            ),
            4: np.array(
                [
                    [1, 2, 2, 2, 0, 2, 0],
                    [2, 0, 0, 2, 1, 0, 1],
                    [1, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 2, 1, 0, 1, 3],
                    [2, 2, 1, 3, 2, 1, 0],
                    [1, 0, 1, 0, 0, 1, 0],
                    [0, 1, 2, 3, 1, 2, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 1],
                ],
                dtype=np.float32,
            ),
            100: uim_frequency,
        }

        rec = CyclingRecommender(
            urm_train=urm,
            uim_frequency=uim_frequency,
            trained_recommender=test_recommender,
            seed=seed,
        )

        # act
        for test_weight in test_weight_values:
            expected_presentation_scores = sp.csr_matrix(
                dict_expected_presentation_scores[test_weight]
            )

            rec.fit(weight=test_weight, sign=test_sign)

            # assert
            assert np.array_equal(
                rec._matrix_presentation_scores.indptr,
                expected_presentation_scores.indptr,
            )
            assert np.array_equal(
                rec._matrix_presentation_scores.indices,
                expected_presentation_scores.indices,
            )
            assert np.allclose(
                rec._matrix_presentation_scores.data,
                expected_presentation_scores.data,
            )

    def test_sign(self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix):
        # arrange
        test_recommender = BaseRecommender(URM_train=urm)
        test_weight = 2
        test_sign_values: list[T_SIGN] = [-1, 1]

        base_expected_presentation_scores = sp.csr_matrix(
            np.array(
                [
                    [1, 1, 1, 1, 0, 1, 0],
                    [1, 0, 0, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 1, 1],
                    [1, 1, 2, 1, 1, 2, 0],
                    [1, 0, 1, 0, 0, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 1],
                ],
                dtype=np.float32,
            ),
            shape=uim_frequency.shape,
        )

        rec = CyclingRecommender(
            urm_train=urm,
            uim_frequency=uim_frequency,
            trained_recommender=test_recommender,
            seed=seed,
        )

        # act
        test_sign: T_SIGN
        for test_sign in test_sign_values:
            expected_presentation_scores = base_expected_presentation_scores * test_sign

            rec.fit(weight=test_weight, sign=test_sign)

            # assert
            # For this particular recommender, we cannot test recommendations, as there might be several ties (same
            # timestamp for two impressions) and the .recommend handles ties in a non-deterministic way.
            assert np.array_equal(
                rec._matrix_presentation_scores.indptr,
                expected_presentation_scores.indptr,
            )
            assert np.array_equal(
                rec._matrix_presentation_scores.indices,
                expected_presentation_scores.indices,
            )
            assert np.allclose(
                rec._matrix_presentation_scores.data,
                expected_presentation_scores.data,
            )

    @pytest.mark.parametrize(
        "test_users,test_recommender_compute_item_scores",
        [
            (
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                TEST_TRAINED_RECOMMENDER_COMPUTE_ITEM_SCORES_ALL_USERS,
            ),
            (
                [0, 1, 3, 6, 7, 8, 9],
                TEST_TRAINED_RECOMMENDER_COMPUTE_ITEM_SCORES_SOME_USERS,
            ),
        ],
    )
    # Test three cases of items: ALL ITEMS (None), SOME ITEMS, ALL ITEMS (LISTED)
    @pytest.mark.parametrize(
        "test_items",
        [None, [1, 2, 5], [0, 1, 2, 3, 4, 5, 6]],
    )
    def test_scores_norm(
        self,
        num_items: int,
        urm: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        test_users: list[int],
        test_items: Optional[list[int]],
        test_recommender_compute_item_scores: np.ndarray,
    ):
        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            "_compute_item_score",
            return_value=test_recommender_compute_item_scores,
        ) as _:
            # arrange
            # test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # test_items = None
            test_cutoff = num_items
            test_weight = (
                1  # weight as 1 to have presentation score = frequency of impressions.
            )
            test_sign: Literal[-1] = -1

            rec = CyclingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(
                weight=test_weight,
                sign=test_sign,
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
            arr_expected_items_scores = compute_cycling_recommender_score(
                list_user_ids=test_users,
                list_item_ids=test_items,
                num_score_users=len(test_users),
                num_score_items=test_recommender_compute_item_scores.shape[1],
                arr_recommender_scores=test_recommender_compute_item_scores,
                matrix_presentation_scores=rec._matrix_presentation_scores,
            )
            # print(repr(recommendations))
            # print(repr(scores))
            # For this particular recommender, we cannot test recommendations, as there might be several ties (same
            # timestamp for two impressions) and the .recommend handles ties in a non-deterministic way.
            assert np.allclose(arr_expected_items_scores, scores)
            # assert np.array_equal(TEST_RECOMMENDATIONS_ALL_USERS, recommendations)

    @pytest.mark.parametrize(
        "test_users,test_recommender_compute_item_scores",
        [
            (
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                TEST_TRAINED_RECOMMENDER_COMPUTE_ITEM_SCORES_ALL_USERS,
            ),
            (
                [0, 1, 3, 6, 7, 8, 9],
                TEST_TRAINED_RECOMMENDER_COMPUTE_ITEM_SCORES_SOME_USERS,
            ),
        ],
    )
    # Test three cases of items: ALL ITEMS (None), SOME ITEMS, ALL ITEMS (LISTED)
    @pytest.mark.parametrize(
        "test_items",
        [None, [1, 2, 5], [0, 1, 2, 3, 4, 5, 6]],
    )
    @pytest.mark.xfail(
        reason="The Cycling recommender does not rank pairs of scores anymore (presentation and recommender). Now, instead, it uses a real number (the combination of the presentation and recommender scores) as recommender score."
    )
    def test_scores_rank(
        self,
        num_items: int,
        urm: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        test_users: list[int],
        test_items: Optional[list[int]],
        test_recommender_compute_item_scores: np.ndarray,
    ):
        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            "_compute_item_score",
            return_value=test_recommender_compute_item_scores,
        ) as _:
            # arrange
            # test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # test_items = None
            test_cutoff = num_items
            test_weight = (
                1  # weight as 1 to have presentation score = frequency of impressions.
            )
            test_sign: Literal[-1] = -1

            rec = CyclingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                trained_recommender=mock_base_recommender,
                seed=seed,
            )

            # act
            rec.fit(
                weight=test_weight,
                sign=test_sign,
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
            arr_expected_items_scores = compute_cycling_recommender_score(
                list_user_ids=test_users,
                list_item_ids=test_items,
                num_score_users=len(test_users),
                num_score_items=test_recommender_compute_item_scores.shape[1],
                arr_recommender_scores=test_recommender_compute_item_scores,
                matrix_presentation_scores=rec._matrix_presentation_scores,
            )

            # For this particular recommender, we cannot test recommendations, as there might be several ties (same
            # timestamp for two impressions) and the .recommend handles ties in a non-deterministic way.
            assert np.allclose(arr_expected_items_scores, scores)
