import numpy as np
import scipy.sparse as sp
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

from impression_recommenders.user_profile.weighted import WeightedUserProfileRecommender


class TestWeightedUserProfileRecommender:
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

        test_reg_urm = 4.
        test_reg_uim = 5.

        expected_item_scores = np.array([
            [51., 74., 47., 61., 37., 69., 37.],
            [28., 42., 33., 38., 28., 28., 28.],
            [19., 33., 28., 37., 19., 19., 19.],
            [0., 0., 0., 0., 0., 0., 0.],
            [60., 75., 56., 66., 42., 74., 42.],
            [60., 87., 60., 78., 46., 78., 46.],
            [28., 38., 24., 34., 19., 42., 19.],
            [39., 53., 34., 39., 29., 49., 29.],
            [0., 0., 0., 0., 0., 0., 0.],
            [24., 24., 24., 24., 19., 24., 19.]
        ], dtype=np.float64)

        rec = WeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(
            reg_urm=test_reg_urm,
            reg_uim=test_reg_uim,
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

        test_reg_urm = 4.
        test_reg_uim = 5.

        expected_item_scores = np.array([
            [np.NINF, 74., 47., np.NINF, np.NINF, 69., np.NINF],
            [np.NINF, 42., 33., np.NINF, np.NINF, 28., np.NINF],
            [np.NINF, 33., 28., np.NINF, np.NINF, 19., np.NINF],
            [np.NINF, 0., 0., np.NINF, np.NINF, 0., np.NINF],
            [np.NINF, 75., 56., np.NINF, np.NINF, 74., np.NINF],
            [np.NINF, 87., 60., np.NINF, np.NINF, 78., np.NINF],
            [np.NINF, 38., 24., np.NINF, np.NINF, 42., np.NINF],
            [np.NINF, 53., 34., np.NINF, np.NINF, 49., np.NINF],
            [np.NINF, 0., 0., np.NINF, np.NINF, 0., np.NINF],
            [np.NINF, 24., 24., np.NINF, np.NINF, 24., np.NINF],
        ], dtype=np.float64)

        rec = WeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(
            reg_urm=test_reg_urm,
            reg_uim=test_reg_uim,
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

        test_reg_urm = 4.
        test_reg_uim = 5.

        expected_item_scores = np.array([
            [51., 74., 47., 61., 37., 69., 37.],
            [28., 42., 33., 38., 28., 28., 28.],
            [19., 33., 28., 37., 19., 19., 19.],
            [0., 0., 0., 0., 0., 0., 0.],
            [60., 75., 56., 66., 42., 74., 42.],
            [60., 87., 60., 78., 46., 78., 46.],
            [28., 38., 24., 34., 19., 42., 19.],
            [39., 53., 34., 39., 29., 49., 29.],
            [0., 0., 0., 0., 0., 0., 0.],
            [24., 24., 24., 24., 19., 24., 19.]
        ], dtype=np.float64)

        rec = WeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(
            reg_urm=test_reg_urm,
            reg_uim=test_reg_uim,
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

        test_reg_urm = 4.
        test_reg_uim = 5.

        expected_item_scores = np.array([
            [51., 74., 47., 61., 37., 69., 37.],
            [28., 42., 33., 38., 28., 28., 28.],
            [0., 0., 0., 0., 0., 0., 0.],
            [28., 38., 24., 34., 19., 42., 19.],
            [39., 53., 34., 39., 29., 49., 29.],
            [0., 0., 0., 0., 0., 0., 0.],
            [24., 24., 24., 24., 19., 24., 19.]
        ], dtype=np.float64)

        rec = WeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(
            reg_urm=test_reg_urm,
            reg_uim=test_reg_uim,
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

        test_reg_urm = 4.
        test_reg_uim = 5.

        expected_item_scores = np.array([
            [np.NINF, 74., 47., np.NINF, np.NINF, 69., np.NINF],
            [np.NINF, 42., 33., np.NINF, np.NINF, 28., np.NINF],
            [np.NINF, 0., 0., np.NINF, np.NINF, 0., np.NINF],
            [np.NINF, 38., 24., np.NINF, np.NINF, 42., np.NINF],
            [np.NINF, 53., 34., np.NINF, np.NINF, 49., np.NINF],
            [np.NINF, 0., 0., np.NINF, np.NINF, 0., np.NINF],
            [np.NINF, 24., 24., np.NINF, np.NINF, 24., np.NINF],
        ], dtype=np.float64)

        rec = WeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(
            reg_urm=test_reg_urm,
            reg_uim=test_reg_uim,
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

        test_reg_urm = 4.
        test_reg_uim = 5.

        expected_item_scores = np.array([
            [51., 74., 47., 61., 37., 69., 37.],
            [28., 42., 33., 38., 28., 28., 28.],
            [0., 0., 0., 0., 0., 0., 0.],
            [28., 38., 24., 34., 19., 42., 19.],
            [39., 53., 34., 39., 29., 49., 29.],
            [0., 0., 0., 0., 0., 0., 0.],
            [24., 24., 24., 24., 19., 24., 19.]
        ], dtype=np.float64)

        rec = WeightedUserProfileRecommender(
            urm_train=urm,
            uim_train=uim,
            trained_recommender=mock_base_recommender,
        )

        # act
        rec.fit(
            reg_urm=test_reg_urm,
            reg_uim=test_reg_uim,
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
        assert np.allclose(expected_item_scores, scores)
