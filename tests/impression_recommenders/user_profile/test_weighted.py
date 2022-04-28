import pytest
import numpy as np
import scipy.sparse as sp
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender, \
    BaseUserSimilarityMatrixRecommender

from impression_recommenders.user_profile.weighted import ItemWeightedUserProfileRecommender, \
    UserWeightedUserProfileRecommender, EWeightedUserProfileType


class TestBaseWeightedUserProfileRecommender:
    def test_sign(self, urm: sp.csr_matrix, uim: sp.csr_matrix):
        # arrange
        mock_base_recommender = BaseItemSimilarityMatrixRecommender(URM_train=urm)
        mock_base_recommender.W_sparse = sp.csr_matrix(
            np.array(
                [
                    [1, 2, 2, 3, 1, 1, 1],
                    [2, 2, 2, 2, 1, 2, 1],
                    [1, 2, 1, 2, 1, 2, 1],
                    [1, 2, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [2, 2, 1, 1, 1, 3, 1],
                    [1, 1, 1, 1, 1, 1, 1],

                ],
                dtype=np.float32,
            )
        )

        test_alpha = .5
        test_sign_values = [-1, 1]
        test_user_weight_type = EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS

        expected_user_profile = sp.csr_matrix(
            np.array([
                [0.5, 0.5, 0., 0., 0., 0., 0.],
                [0.5, 0., 0., 0., 0., 0., 0.5],
                [0., 0., 0., 0.5, 0., 0., 0.5],
                [0., 0., 0., 0., 0., 0., 0.],
                [0.5, 0., 0.5, 0.5, 0., 0., 0.],
                [0., 0.5, 0., 0., 0.5, 0., 0.],
                [0.5, 0., 0.5, 0., 0., 0., 0.],
                [0., 0.5, 0.5, 0., 0.5, 0.5, 0.],
                [0., 0., 0., 0., 0., 0., 0.],
                [0., 0.5, 0., 0., 0.5, 0., 0.]
            ], dtype=np.float64),
            shape=uim.shape,
        )

        rec = ItemWeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        for test_sign in test_sign_values:
            test_expected_user_profile: sp.csr_matrix = urm + (test_sign * expected_user_profile)

            rec.fit(alpha=test_alpha, sign=test_sign, weighted_user_profile_type=test_user_weight_type)

            # assert
            # For this particular recommender, we cannot test recommendations, as there might be several ties (same
            # timestamp for two impressions) and the .recommend handles ties in a non-deterministic way.
            assert np.array_equal(
                test_expected_user_profile.indptr,
                rec._sparse_user_profile.indptr,
            )
            assert np.array_equal(
                test_expected_user_profile.indices,
                rec._sparse_user_profile.indices,
            )
            assert np.array_equal(
                test_expected_user_profile.data,
                rec._sparse_user_profile.data,
            )

    def test_alpha(self, urm: sp.csr_matrix, uim: sp.csr_matrix):
        # arrange
        mock_base_recommender = BaseItemSimilarityMatrixRecommender(URM_train=urm)
        mock_base_recommender.W_sparse = sp.csr_matrix(
            np.array(
                [
                    [1, 2, 2, 3, 1, 1, 1],
                    [2, 2, 2, 2, 1, 2, 1],
                    [1, 2, 1, 2, 1, 2, 1],
                    [1, 2, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [2, 2, 1, 1, 1, 3, 1],
                    [1, 1, 1, 1, 1, 1, 1],

                ],
                dtype=np.float32,
            )
        )

        test_alpha_values = [0, 0.3, 0.5, 1]
        test_sign = 1
        test_user_weight_type = EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS

        expected_user_profile = sp.csr_matrix(
            np.array([
                [1., 1., 0., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 1., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0.],
                [1., 0., 1., 1., 0., 0., 0.],
                [0., 1., 0., 0., 1., 0., 0.],
                [1., 0., 1., 0., 0., 0., 0.],
                [0., 1., 1., 0., 1., 1., 0.],
                [0., 0., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 1., 0., 0.]
            ], dtype=np.float64),
            shape=uim.shape,
        )

        rec = ItemWeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        for test_alpha in test_alpha_values:
            test_expected_user_profile: sp.csr_matrix = urm + (test_alpha * expected_user_profile)

            rec.fit(alpha=test_alpha, sign=test_sign, weighted_user_profile_type=test_user_weight_type)

            # assert
            # For this particular recommender, we cannot test recommendations, as there might be several ties (same
            # timestamp for two impressions) and the .recommend handles ties in a non-deterministic way.
            assert np.array_equal(
                test_expected_user_profile.indptr,
                rec._sparse_user_profile.indptr,
            )
            assert np.array_equal(
                test_expected_user_profile.indices,
                rec._sparse_user_profile.indices,
            )
            assert np.allclose(
                test_expected_user_profile.data,
                rec._sparse_user_profile.data,
            )

    def test_weighted_user_profile_type(self, urm: sp.csr_matrix, uim: sp.csr_matrix):
        # arrange
        mock_base_recommender = BaseItemSimilarityMatrixRecommender(URM_train=urm)
        mock_base_recommender.W_sparse = sp.csr_matrix([])

        test_alpha = 0
        test_sign = 1

        rec = ItemWeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        for test_user_weight_type in list(EWeightedUserProfileType):
            if EWeightedUserProfileType.ONLY_IMPRESSIONS == test_user_weight_type:
                test_expected_user_profile = uim
            else:
                test_expected_user_profile = urm

            rec.fit(alpha=test_alpha, sign=test_sign, weighted_user_profile_type=test_user_weight_type)

            # assert
            assert np.array_equal(
                test_expected_user_profile.indptr,
                rec._sparse_user_profile.indptr,
            )
            assert np.array_equal(
                test_expected_user_profile.indices,
                rec._sparse_user_profile.indices,
            )
            assert np.allclose(
                test_expected_user_profile.data,
                rec._sparse_user_profile.data,
            )


class TestItemWeightedUserProfileRecommender:
    def test_all_users_no_items(
        self, urm: sp.csr_matrix, uim: sp.csr_matrix,
    ):
        # arrange
        mock_base_recommender = BaseItemSimilarityMatrixRecommender(URM_train=urm)
        mock_base_recommender.W_sparse = sp.csr_matrix(
            np.array(
                [
                    [1, 2, 2, 3, 1, 1, 1],
                    [2, 2, 2, 2, 1, 2, 1],
                    [1, 2, 1, 2, 1, 2, 1],
                    [1, 2, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [2, 2, 1, 1, 1, 3, 1],
                    [1, 1, 1, 1, 1, 1, 1],

                ],
                dtype=np.float32,
            )
        )

        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = None
        test_cutoff = 3

        test_alpha = 0.
        test_sign = 1
        test_user_weight_type = EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS
        expected_item_scores = np.array([
            [4., 6., 3., 4., 3., 6., 3.],
            [2., 3., 2., 2., 2., 2., 2.],
            [1., 2., 2., 3., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0.],
            [5., 5., 4., 4., 3., 6., 3.],
            [5., 8., 5., 7., 4., 7., 4.],
            [2., 2., 1., 1., 1., 3., 1.],
            [1., 2., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1.],
        ], dtype=np.float64)

        rec = ItemWeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(alpha=test_alpha, sign=test_sign, weighted_user_profile_type=test_user_weight_type)
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
        assert np.allclose(expected_item_scores, scores)

    def test_all_users_some_items(
        self, urm: sp.csr_matrix, uim: sp.csr_matrix,
    ):
        # arrange
        mock_base_recommender = BaseItemSimilarityMatrixRecommender(URM_train=urm)
        mock_base_recommender.W_sparse = sp.csr_matrix(
            np.array(
                [
                    [1, 2, 2, 3, 1, 1, 1],
                    [2, 2, 2, 2, 1, 2, 1],
                    [1, 2, 1, 2, 1, 2, 1],
                    [1, 2, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [2, 2, 1, 1, 1, 3, 1],
                    [1, 1, 1, 1, 1, 1, 1],

                ],
                dtype=np.float32,
            )
        )

        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = [1, 2, 5]
        test_cutoff = 3

        test_alpha = 0.
        test_sign = 1
        test_user_weight_type = EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS
        expected_item_scores = np.array([
            [np.NINF, 6., 3., np.NINF, np.NINF, 6., np.NINF],
            [np.NINF, 3., 2., np.NINF, np.NINF, 2., np.NINF],
            [np.NINF, 2., 2., np.NINF, np.NINF, 1., np.NINF],
            [np.NINF, 0., 0., np.NINF, np.NINF, 0., np.NINF],
            [np.NINF, 5., 4., np.NINF, np.NINF, 6., np.NINF],
            [np.NINF, 8., 5., np.NINF, np.NINF, 7., np.NINF],
            [np.NINF, 2., 1., np.NINF, np.NINF, 3., np.NINF],
            [np.NINF, 2., 1., np.NINF, np.NINF, 1., np.NINF],
            [np.NINF, 0., 0., np.NINF, np.NINF, 0., np.NINF],
            [np.NINF, 1., 1., np.NINF, np.NINF, 1., np.NINF],
        ], dtype=np.float64)

        rec = ItemWeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(alpha=test_alpha, sign=test_sign, weighted_user_profile_type=test_user_weight_type)
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
        for row in range(expected_item_scores.shape[0]):
            for col in range(expected_item_scores.shape[1]):
                assert expected_item_scores[row, col] == scores[row, col]

        assert np.allclose(expected_item_scores, scores)

    def test_all_users_all_items(
        self, urm: sp.csr_matrix, uim: sp.csr_matrix,
    ):
        mock_base_recommender = BaseItemSimilarityMatrixRecommender(URM_train=urm)
        mock_base_recommender.W_sparse = sp.csr_matrix(
            np.array(
                [
                    [1, 2, 2, 3, 1, 1, 1],
                    [2, 2, 2, 2, 1, 2, 1],
                    [1, 2, 1, 2, 1, 2, 1],
                    [1, 2, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [2, 2, 1, 1, 1, 3, 1],
                    [1, 1, 1, 1, 1, 1, 1],

                ],
                dtype=np.float32,
            )
        )

        # arrange
        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = [0, 1, 2, 3, 4, 5, 6]
        test_cutoff = 3

        test_alpha = 0.
        test_sign = 1
        test_user_weight_type = EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS
        expected_item_scores = np.array([
            [4., 6., 3., 4., 3., 6., 3.],
            [2., 3., 2., 2., 2., 2., 2.],
            [1., 2., 2., 3., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0.],
            [5., 5., 4., 4., 3., 6., 3.],
            [5., 8., 5., 7., 4., 7., 4.],
            [2., 2., 1., 1., 1., 3., 1.],
            [1., 2., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1.],
        ], dtype=np.float64)

        rec = ItemWeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(alpha=test_alpha, sign=test_sign, weighted_user_profile_type=test_user_weight_type)
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
        assert np.allclose(expected_item_scores, scores)

    def test_some_users_no_items(
        self, urm: sp.csr_matrix, uim: sp.csr_matrix,
    ):
        # arrange
        mock_base_recommender = BaseItemSimilarityMatrixRecommender(URM_train=urm)
        mock_base_recommender.W_sparse = sp.csr_matrix(
            np.array(
                [
                    [1, 2, 2, 3, 1, 1, 1],
                    [2, 2, 2, 2, 1, 2, 1],
                    [1, 2, 1, 2, 1, 2, 1],
                    [1, 2, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [2, 2, 1, 1, 1, 3, 1],
                    [1, 1, 1, 1, 1, 1, 1],

                ],
                dtype=np.float32,
            )
        )

        test_users = [0, 1, 3, 6, 7, 8, 9]
        test_items = None
        test_cutoff = 3

        test_alpha = 0.
        test_sign = 1
        test_user_weight_type = EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS
        expected_item_scores = np.array([
            [4., 6., 3., 4., 3., 6., 3.],
            [2., 3., 2., 2., 2., 2., 2.],
            [0., 0., 0., 0., 0., 0., 0.],
            [2., 2., 1., 1., 1., 3., 1.],
            [1., 2., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1.],
        ], dtype=np.float64)

        rec = ItemWeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(alpha=test_alpha, sign=test_sign, weighted_user_profile_type=test_user_weight_type)
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
        assert np.array_equal(expected_item_scores, scores)

    def test_some_users_some_items(
        self, urm: sp.csr_matrix, uim: sp.csr_matrix,
    ):
        # arrange
        mock_base_recommender = BaseItemSimilarityMatrixRecommender(URM_train=urm)
        mock_base_recommender.W_sparse = sp.csr_matrix(
            np.array(
                [
                    [1, 2, 2, 3, 1, 1, 1],
                    [2, 2, 2, 2, 1, 2, 1],
                    [1, 2, 1, 2, 1, 2, 1],
                    [1, 2, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [2, 2, 1, 1, 1, 3, 1],
                    [1, 1, 1, 1, 1, 1, 1],

                ],
                dtype=np.float32,
            )
        )

        test_users = [0, 1, 3, 6, 7, 8, 9]
        test_items = [1, 2, 5]
        test_cutoff = 3

        test_alpha = 0.
        test_sign = 1
        test_user_weight_type = EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS
        expected_item_scores = np.array([
            [np.NINF, 6., 3., np.NINF, np.NINF, 6., np.NINF],
            [np.NINF, 3., 2., np.NINF, np.NINF, 2., np.NINF],
            [np.NINF, 0., 0., np.NINF, np.NINF, 0., np.NINF],
            [np.NINF, 2., 1., np.NINF, np.NINF, 3., np.NINF],
            [np.NINF, 2., 1., np.NINF, np.NINF, 1., np.NINF],
            [np.NINF, 0., 0., np.NINF, np.NINF, 0., np.NINF],
            [np.NINF, 1., 1., np.NINF, np.NINF, 1., np.NINF],
        ], dtype=np.float64)

        rec = ItemWeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(alpha=test_alpha, sign=test_sign, weighted_user_profile_type=test_user_weight_type)
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
        assert np.allclose(expected_item_scores, scores)

    def test_some_users_all_items(
        self, urm: sp.csr_matrix, uim: sp.csr_matrix,
    ):
        # arrange
        mock_base_recommender = BaseItemSimilarityMatrixRecommender(URM_train=urm)
        mock_base_recommender.W_sparse = sp.csr_matrix(
            np.array(
                [
                    [1, 2, 2, 3, 1, 1, 1],
                    [2, 2, 2, 2, 1, 2, 1],
                    [1, 2, 1, 2, 1, 2, 1],
                    [1, 2, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [2, 2, 1, 1, 1, 3, 1],
                    [1, 1, 1, 1, 1, 1, 1],

                ],
                dtype=np.float32,
            )
        )
        test_users = [0, 1, 3, 6, 7, 8, 9]
        test_items = [0, 1, 2, 3, 4, 5, 6]
        test_cutoff = 3

        test_alpha = 0.
        test_sign = 1
        test_user_weight_type = EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS
        expected_item_scores = np.array([
            [4., 6., 3., 4., 3., 6., 3.],
            [2., 3., 2., 2., 2., 2., 2.],
            [0., 0., 0., 0., 0., 0., 0.],
            [2., 2., 1., 1., 1., 3., 1.],
            [1., 2., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1.],
        ], dtype=np.float64)

        rec = ItemWeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(alpha=test_alpha, sign=test_sign, weighted_user_profile_type=test_user_weight_type)
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
        assert np.allclose(expected_item_scores, scores)


@pytest.mark.skip
class TestUserWeightedUserProfileRecommender:
    def test_all_users_no_items(
        self, urm: sp.csr_matrix, uim: sp.csr_matrix,
    ):
        # arrange
        mock_base_recommender = BaseUserSimilarityMatrixRecommender(URM_train=urm)
        mock_base_recommender.W_sparse = sp.csr_matrix(
            np.array(
                [
                    [1, 2, 2, 3, 1, 1, 1, 1, 2, 1],
                    [2, 2, 2, 2, 1, 2, 1, 4, 2, 3],
                    [1, 2, 1, 2, 1, 2, 1, 8, 2, 4],
                    [1, 2, 1, 1, 1, 1, 1, 1, 1, 4],
                    [1, 1, 1, 1, 1, 1, 1, 1, 6, 7],
                    [2, 2, 1, 1, 1, 3, 1, 8, 2, 1],
                    [1, 2, 1, 2, 1, 2, 1, 8, 2, 4],
                    [1, 1, 1, 1, 1, 1, 1, 0, 2, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 2, 1],
                    [1, 2, 1, 1, 1, 1, 1, 1, 1, 4],
                ],
                dtype=np.float32,
            )
        )

        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = None
        test_cutoff = 3

        test_alpha = 0.
        test_sign = 1
        test_user_weight_type = EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS
        expected_item_scores = np.array([
            [52., 29., 33., 60., 33., 41., 38.],
            [66., 64., 66., 105., 63., 74., 56.],
            [52., 84., 77., 127., 88., 85., 60.],
            [43., 44., 33., 55., 48., 41., 60.],
            [38., 59., 33., 46., 54., 41., 82.],
            [66., 79., 95., 145., 78., 103., 33.],
            [52., 84., 77., 127., 88., 85., 60.],
            [38., 24., 28., 37., 19., 36., 28.],
            [38., 24., 28., 37., 19., 36., 28.],
            [43., 44., 33., 55., 48., 41., 60.],
        ], dtype=np.float64)

        rec = UserWeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(alpha=test_alpha, sign=test_sign, weighted_user_profile_type=test_user_weight_type)
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
        # For this particular recommender, we cannot test recommendations, as there might be several ties (same
        # timestamp for two impressions) and the .recommend handles ties in a non-deterministic way.
        for row in range(expected_item_scores.shape[0]):
            for col in range(expected_item_scores.shape[1]):
                assert expected_item_scores[row, col] == scores[row, col]

        assert np.array_equal(expected_item_scores, scores)

    def test_all_users_some_items(
        self, urm: sp.csr_matrix, uim: sp.csr_matrix,
    ):
        # arrange
        mock_base_recommender = BaseUserSimilarityMatrixRecommender(URM_train=urm)
        mock_base_recommender.W_sparse = sp.csr_matrix(
            np.array(
                [
                    [1, 2, 2, 3, 1, 1, 1, 1, 2, 1],
                    [2, 2, 2, 2, 1, 2, 1, 4, 2, 3],
                    [1, 2, 1, 2, 1, 2, 1, 8, 2, 4],
                    [1, 2, 1, 1, 1, 1, 1, 1, 1, 4],
                    [1, 1, 1, 1, 1, 1, 1, 1, 6, 7],
                    [2, 2, 1, 1, 1, 3, 1, 8, 2, 1],
                    [1, 2, 1, 2, 1, 2, 1, 8, 2, 4],
                    [1, 1, 1, 1, 1, 1, 1, 0, 2, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 2, 1],
                    [1, 2, 1, 1, 1, 1, 1, 1, 1, 4],
                ],
                dtype=np.float32,
            )
        )

        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = [1, 2, 5]
        test_cutoff = 3

        test_alpha = 0.
        test_sign = 1
        test_user_weight_type = EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS
        expected_item_scores = np.array([
            [np.NINF, 29., 33., np.NINF, np.NINF, 41., np.NINF],
            [np.NINF, 64., 66., np.NINF, np.NINF, 74., np.NINF],
            [np.NINF, 84., 77., np.NINF, np.NINF, 85., np.NINF],
            [np.NINF, 44., 33., np.NINF, np.NINF, 41., np.NINF],
            [np.NINF, 59., 33., np.NINF, np.NINF, 41., np.NINF],
            [np.NINF, 79., 95., np.NINF, np.NINF, 103., np.NINF],
            [np.NINF, 84., 77., np.NINF, np.NINF, 85., np.NINF],
            [np.NINF, 24., 28., np.NINF, np.NINF, 36., np.NINF],
            [np.NINF, 24., 28., np.NINF, np.NINF, 36., np.NINF],
            [np.NINF, 44., 33., np.NINF, np.NINF, 41., np.NINF],
        ], dtype=np.float64)

        rec = UserWeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(alpha=test_alpha, sign=test_sign, weighted_user_profile_type=test_user_weight_type)
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
        for row in range(expected_item_scores.shape[0]):
            for col in range(expected_item_scores.shape[1]):
                assert expected_item_scores[row, col] == scores[row, col]

        assert np.allclose(expected_item_scores, scores)

    def test_all_users_all_items(
        self, urm: sp.csr_matrix, uim: sp.csr_matrix,
    ):
        mock_base_recommender = BaseUserSimilarityMatrixRecommender(URM_train=urm)
        mock_base_recommender.W_sparse = sp.csr_matrix(
            np.array(
                [
                    [1, 2, 2, 3, 1, 1, 1, 1, 2, 1],
                    [2, 2, 2, 2, 1, 2, 1, 4, 2, 3],
                    [1, 2, 1, 2, 1, 2, 1, 8, 2, 4],
                    [1, 2, 1, 1, 1, 1, 1, 1, 1, 4],
                    [1, 1, 1, 1, 1, 1, 1, 1, 6, 7],
                    [2, 2, 1, 1, 1, 3, 1, 8, 2, 1],
                    [1, 2, 1, 2, 1, 2, 1, 8, 2, 4],
                    [1, 1, 1, 1, 1, 1, 1, 0, 2, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 2, 1],
                    [1, 2, 1, 1, 1, 1, 1, 1, 1, 4],
                ],
                dtype=np.float32,
            )
        )

        # arrange
        test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_items = [0, 1, 2, 3, 4, 5, 6]
        test_cutoff = 3

        test_alpha = 0.
        test_sign = 1
        test_user_weight_type = EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS
        expected_item_scores = np.array([
            [52., 29., 33., 60., 33., 41., 38.],
            [66., 64., 66., 105., 63., 74., 56.],
            [52., 84., 77., 127., 88., 85., 60.],
            [43., 44., 33., 55., 48., 41., 60.],
            [38., 59., 33., 46., 54., 41., 82.],
            [66., 79., 95., 145., 78., 103., 33.],
            [52., 84., 77., 127., 88., 85., 60.],
            [38., 24., 28., 37., 19., 36., 28.],
            [38., 24., 28., 37., 19., 36., 28.],
            [43., 44., 33., 55., 48., 41., 60.],
        ], dtype=np.float64)

        rec = UserWeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(alpha=test_alpha, sign=test_sign, weighted_user_profile_type=test_user_weight_type)
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
        assert np.allclose(expected_item_scores, scores)

    def test_some_users_no_items(
        self, urm: sp.csr_matrix, uim: sp.csr_matrix,
    ):
        # arrange
        mock_base_recommender = BaseUserSimilarityMatrixRecommender(URM_train=urm)
        mock_base_recommender.W_sparse = sp.csr_matrix(
            np.array(
                [
                    [1, 2, 2, 3, 1, 1, 1, 1, 2, 1],
                    [2, 2, 2, 2, 1, 2, 1, 4, 2, 3],
                    [1, 2, 1, 2, 1, 2, 1, 8, 2, 4],
                    [1, 2, 1, 1, 1, 1, 1, 1, 1, 4],
                    [1, 1, 1, 1, 1, 1, 1, 1, 6, 7],
                    [2, 2, 1, 1, 1, 3, 1, 8, 2, 1],
                    [1, 2, 1, 2, 1, 2, 1, 8, 2, 4],
                    [1, 1, 1, 1, 1, 1, 1, 0, 2, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 2, 1],
                    [1, 2, 1, 1, 1, 1, 1, 1, 1, 4],
                ],
                dtype=np.float32,
            )
        )

        test_users = [0, 1, 3, 6, 7, 8, 9]
        test_items = None
        test_cutoff = 3

        test_alpha = 0.
        test_sign = 1
        test_user_weight_type = EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS
        expected_item_scores = np.array([
            [52., 29., 33., 60., 33., 41., 38.],
            [66., 64., 66., 105., 63., 74., 56.],
            [43., 44., 33., 55., 48., 41., 60.],
            [52., 84., 77., 127., 88., 85., 60.],
            [38., 24., 28., 37., 19., 36., 28.],
            [38., 24., 28., 37., 19., 36., 28.],
            [43., 44., 33., 55., 48., 41., 60.],
        ], dtype=np.float64)

        rec = UserWeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(alpha=test_alpha, sign=test_sign, weighted_user_profile_type=test_user_weight_type)
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
        assert np.array_equal(expected_item_scores, scores)

    def test_some_users_some_items(
        self, urm: sp.csr_matrix, uim: sp.csr_matrix,
    ):
        # arrange
        mock_base_recommender = BaseUserSimilarityMatrixRecommender(URM_train=urm)
        mock_base_recommender.W_sparse = sp.csr_matrix(
            np.array(
                [
                    [1, 2, 2, 3, 1, 1, 1, 1, 2, 1],
                    [2, 2, 2, 2, 1, 2, 1, 4, 2, 3],
                    [1, 2, 1, 2, 1, 2, 1, 8, 2, 4],
                    [1, 2, 1, 1, 1, 1, 1, 1, 1, 4],
                    [1, 1, 1, 1, 1, 1, 1, 1, 6, 7],
                    [2, 2, 1, 1, 1, 3, 1, 8, 2, 1],
                    [1, 2, 1, 2, 1, 2, 1, 8, 2, 4],
                    [1, 1, 1, 1, 1, 1, 1, 0, 2, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 2, 1],
                    [1, 2, 1, 1, 1, 1, 1, 1, 1, 4],
                ],
                dtype=np.float32,
            )
        )

        test_users = [0, 1, 3, 6, 7, 8, 9]
        test_items = [1, 2, 5]
        test_cutoff = 3

        test_alpha = 0.
        test_sign = 1
        test_user_weight_type = EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS
        expected_item_scores = np.array([
            [np.NINF, 29., 33., np.NINF, np.NINF, 41., np.NINF],
            [np.NINF, 64., 66., np.NINF, np.NINF, 74., np.NINF],
            [np.NINF, 44., 33., np.NINF, np.NINF, 41., np.NINF],
            [np.NINF, 84., 77., np.NINF, np.NINF, 85., np.NINF],
            [np.NINF, 24., 28., np.NINF, np.NINF, 36., np.NINF],
            [np.NINF, 24., 28., np.NINF, np.NINF, 36., np.NINF],
            [np.NINF, 44., 33., np.NINF, np.NINF, 41., np.NINF],
        ], dtype=np.float64)

        rec = UserWeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(alpha=test_alpha, sign=test_sign, weighted_user_profile_type=test_user_weight_type)
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
        assert np.allclose(expected_item_scores, scores)

    def test_some_users_all_items(
        self, urm: sp.csr_matrix, uim: sp.csr_matrix,
    ):
        # arrange
        mock_base_recommender = BaseUserSimilarityMatrixRecommender(URM_train=urm)
        mock_base_recommender.W_sparse = sp.csr_matrix(
            np.array(
                [
                    [1, 2, 2, 3, 1, 1, 1, 1, 2, 1],
                    [2, 2, 2, 2, 1, 2, 1, 4, 2, 3],
                    [1, 2, 1, 2, 1, 2, 1, 8, 2, 4],
                    [1, 2, 1, 1, 1, 1, 1, 1, 1, 4],
                    [1, 1, 1, 1, 1, 1, 1, 1, 6, 7],
                    [2, 2, 1, 1, 1, 3, 1, 8, 2, 1],
                    [1, 2, 1, 2, 1, 2, 1, 8, 2, 4],
                    [1, 1, 1, 1, 1, 1, 1, 0, 2, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 2, 1],
                    [1, 2, 1, 1, 1, 1, 1, 1, 1, 4],
                ],
                dtype=np.float32,
            )
        )
        test_users = [0, 1, 3, 6, 7, 8, 9]
        test_items = [0, 1, 2, 3, 4, 5, 6]
        test_cutoff = 3

        test_alpha = 1
        test_sign = 1
        test_user_weight_type = EWeightedUserProfileType.INTERACTIONS_AND_IMPRESSIONS
        expected_item_scores = np.array([
            [52., 29., 33., 60., 33., 41., 38.],
            [66., 64., 66., 105., 63., 74., 56.],
            [43., 44., 33., 55., 48., 41., 60.],
            [52., 84., 77., 127., 88., 85., 60.],
            [38., 24., 28., 37., 19., 36., 28.],
            [38., 24., 28., 37., 19., 36., 28.],
            [43., 44., 33., 55., 48., 41., 60.],
        ], dtype=np.float64)

        rec = UserWeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(alpha=test_alpha, sign=test_sign, weighted_user_profile_type=test_user_weight_type)
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
        assert np.allclose(expected_item_scores, scores)
