import numpy as np
import scipy.sparse as sp
from Recommenders.BaseRecommender import BaseRecommender
from mock import patch

from impression_recommenders.re_ranking.impressions_discounting import (
    ImpressionsDiscountingRecommender,
    EImpressionsDiscountingFunctions,
    _func_exponential,
)


class TestImpressionsDiscountingRecommender:
    def test_exponential_overflow_warning(self):
        # np.exp() overflows for values >= 709.8 on 64 bits, >= 88.8 on 32 bits
        # np.exp() + 1 overflows for values >= 709.8 on 64 bits, >= 88.8 on 32 bits
        # In this test we expect that overflows are transformed to np.exp(88.7) + 1
        test_array = np.array([0, 1, 2, 3, 88, 88.7, 88.8, 90, 1000, 2000], dtype=np.float64)
        expected_array = (
            np.exp(
                np.array([0, 1, 2, 3, 88, 88.7, 88.7, 88.7, 88.7, 88.7], dtype=np.float32)
            )
            + np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
        ).astype(
            dtype=np.float64,
        )

        obtained_array = _func_exponential(x=test_array)

        assert np.array_equal(obtained_array, expected_array)
        assert obtained_array.dtype == expected_array.dtype and obtained_array.dtype == np.float64

    def test_all_linear(
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix, uim_position: sp.csr_matrix,
        uim_last_seen: sp.csr_matrix,
    ):
        test_trained_recommender_compute_item_score = np.array(
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

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_items = None
            test_cutoff = 3
            test_signs = dict(
                sign_user_frequency=1,
                sign_uim_frequency=1,
                sign_uim_position=1,
                sign_uim_last_seen=1,
            )
            test_regs = dict(
                reg_user_frequency=1.0,
                reg_uim_frequency=1.0,
                reg_uim_position=1.0,
                reg_uim_last_seen=1.0,
            )
            test_funcs = dict(
                func_user_frequency=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_frequency=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_position=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_last_seen=EImpressionsDiscountingFunctions.LINEAR,
            )

            expected_item_scores = np.array([
                [  5., 162.,  75.,  64.,  12.,  55.,  16.],
                [ 27.,   3.,   3.,  28.,   6.,   3.,   4.],
                [  3.,   4.,   6.,  16.,  10.,  12.,  35.],
                [  7.,   6.,   5.,   4.,   3.,   2.,   1.],
                [ 45.,  35., 190.,  20.,  32.,  35.,  69.],
                [ 29.,  72., 102.,  58., 144.,  70.,  20.],
                [  4.,   2.,   3.,   2.,   2.,   5.,   2.],
                [  2.,   6.,  99.,  80.,  15., 186.,  14.],
                [  7.,   6.,   5.,   4.,   3.,   2.,   1.],
                [ 18.,  21.,  10.,   8.,  40.,  14.,  12.],
                ],
                dtype=np.float32
            )

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_signs, **test_regs, **test_funcs)  # type: ignore
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
            assert np.allclose(expected_item_scores, scores)

    def test_all_inverse(
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix, uim_position: sp.csr_matrix,
        uim_last_seen: sp.csr_matrix,
    ):
        test_trained_recommender_compute_item_score = np.array(
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

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_items = None
            test_cutoff = 3
            test_signs = dict(
                sign_user_frequency=1,
                sign_uim_frequency=1,
                sign_uim_position=1,
                sign_uim_last_seen=1,
            )
            test_regs = dict(
                reg_user_frequency=1.0,
                reg_uim_frequency=1.0,
                reg_uim_position=1.0,
                reg_uim_last_seen=1.0,
            )
            test_funcs = dict(
                func_user_frequency=EImpressionsDiscountingFunctions.INVERSE,
                func_uim_frequency=EImpressionsDiscountingFunctions.INVERSE,
                func_uim_position=EImpressionsDiscountingFunctions.INVERSE,
                func_uim_last_seen=EImpressionsDiscountingFunctions.INVERSE,
            )

            expected_item_scores = np.array([
                [2.33333333, 14.31578947, 5.65789474, 5.74666667, 4., 10.16666667, 5.33333333],
                [2.04545455, 1.5, 1.5, 3.04545455, 3., 1.5, 2.5],
                [3., 4., 6., 16., 10., 12., 24.5],
                [7., 6., 5., 4., 3., 2., 1.],
                [21., 16.33333333, 11.83333333, 9.33333333, 10.66666667, 16.33333333, 8.2],
                [1.79545455, 11.7, 4.475, 3.26190476, 8.325, 11.66666667, 5.],
                [4., 2., 3., 2., 2., 3.5, 2.],
                [2., 6., 9.11111111, 13.61904762, 15., 15.22222222, 14.],
                [7., 6., 5., 4., 3., 2., 1.],
                [18., 21., 10., 8., 28., 14., 12.]
            ],
                dtype=np.float32,
            )

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_signs, **test_regs, **test_funcs)  # type: ignore
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
            assert np.allclose(expected_item_scores, scores)

    def test_all_exponential(
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix, uim_position: sp.csr_matrix,
        uim_last_seen: sp.csr_matrix,
    ):
        test_trained_recommender_compute_item_score = np.array(
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

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_items = None
            test_cutoff = 3
            test_signs = dict(
                sign_user_frequency=1,
                sign_uim_frequency=1,
                sign_uim_position=1,
                sign_uim_last_seen=1,
            )
            test_regs = dict(
                reg_user_frequency=1.0,
                reg_uim_frequency=1.0,
                reg_uim_position=1.0,
                reg_uim_last_seen=1.0,
            )
            test_funcs = dict(
                func_user_frequency=EImpressionsDiscountingFunctions.EXPONENTIAL,
                func_uim_frequency=EImpressionsDiscountingFunctions.EXPONENTIAL,
                func_uim_position=EImpressionsDiscountingFunctions.EXPONENTIAL,
                func_uim_last_seen=EImpressionsDiscountingFunctions.EXPONENTIAL,
            )

            expected_item_scores = np.array([
                [2.38038188e+01, 1.07089402e+09, 5.35446991e+08, 1.44009799e+11, 6.32566108e+01, 8.84438761e+02, 8.43421477e+01],
                [3.58491286e+09, 8.38905610e+00, 8.38905610e+00, 3.58491286e+09, 1.84963940e+01, 8.38905610e+00, 1.11073379e+01],
                [7.43656366e+00, 7.43656366e+00, 1.11548455e+01, 3.66193819e+01, 1.85914091e+01, 2.23096910e+01, 9.67793383e+01],
                [1.40000000e+01, 1.20000000e+01, 1.00000000e+01, 8.00000000e+00, 6.00000000e+00, 4.00000000e+00, 2.00000000e+00],
                [2.14234369e+02, 1.73626731e+02, 5.34323729e+13, 9.52152750e+01, 1.68684295e+02, 1.66626731e+02, 9.80718379e+06],
                [3.58491291e+09, 1.26840219e+03, 7.94673670e+10, 2.63763162e+09, 7.06155801e+17, 6.88354465e+02, 2.22392600e+02],
                [9.15484549e+00, 3.71828183e+00, 7.43656366e+00, 3.71828183e+00, 3.71828183e+00, 1.38256198e+01, 3.71828183e+00],
                [3.71828183e+00, 1.28731273e+01, 1.59614472e+12, 4.81052323e+06, 3.21828183e+01, 3.19228944e+12, 2.60279728e+01],
                [1.40000000e+01, 1.20000000e+01, 1.00000000e+01, 8.00000000e+00, 6.00000000e+00, 4.00000000e+00, 2.00000000e+00],
                [3.34645365e+01, 5.20559456e+01, 1.85914091e+01, 1.48731273e+01, 1.10604958e+02, 2.60279728e+01, 2.74645365e+01],
                ],
                dtype=np.float32
            )

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_signs, **test_regs, **test_funcs)  # type: ignore
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
            assert np.allclose(expected_item_scores, scores)

    def test_all_quadratic(
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix, uim_position: sp.csr_matrix,
        uim_last_seen: sp.csr_matrix,
    ):
        test_trained_recommender_compute_item_score = np.array(
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

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_items = None
            test_cutoff = 3
            test_signs = dict(
                sign_user_frequency=1,
                sign_uim_frequency=1,
                sign_uim_position=1,
                sign_uim_last_seen=1,
            )
            test_regs = dict(
                reg_user_frequency=1.0,
                reg_uim_frequency=1.0,
                reg_uim_position=1.0,
                reg_uim_last_seen=1.0,
            )
            test_funcs = dict(
                func_user_frequency=EImpressionsDiscountingFunctions.QUADRATIC,
                func_uim_frequency=EImpressionsDiscountingFunctions.QUADRATIC,
                func_uim_position=EImpressionsDiscountingFunctions.QUADRATIC,
                func_uim_last_seen=EImpressionsDiscountingFunctions.QUADRATIC,
            )

            expected_item_scores = np.array([
                [1.100e+01, 2.274e+03, 1.125e+03, 1.280e+03, 3.000e+01, 1.950e+02, 4.000e+01],
                [4.930e+02, 5.000e+00, 5.000e+00, 4.940e+02, 1.000e+01, 5.000e+00, 6.000e+00],
                [3.000e+00, 4.000e+00, 6.000e+00, 1.600e+01, 1.000e+01, 1.200e+01, 4.900e+01],
                [7.000e+00, 6.000e+00, 5.000e+00, 4.000e+00, 3.000e+00, 2.000e+00, 1.000e+00],
                [9.900e+01, 7.700e+01, 4.590e+03, 4.400e+01, 8.000e+01, 7.700e+01, 7.350e+02],
                [5.050e+02, 2.760e+02, 1.854e+03, 9.340e+02, 4.866e+03, 2.300e+02, 6.800e+01],
                [4.000e+00, 2.000e+00, 3.000e+00, 2.000e+00, 2.000e+00, 7.000e+00, 2.000e+00],
                [2.000e+00, 6.000e+00, 2.217e+03, 8.320e+02, 1.500e+01, 4.410e+03, 1.400e+01],
                [7.000e+00, 6.000e+00, 5.000e+00, 4.000e+00, 3.000e+00, 2.000e+00, 1.000e+00],
                [1.800e+01, 2.100e+01, 1.000e+01, 8.000e+00, 5.600e+01, 1.400e+01, 1.200e+01],
                ],
                dtype=np.float32
            )

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_signs, **test_regs, **test_funcs)  # type: ignore
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
            assert np.allclose(expected_item_scores, scores)

    def test_all_users_no_items(
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix, uim_position: sp.csr_matrix,
        uim_last_seen: sp.csr_matrix,
    ):

        test_trained_recommender_compute_item_score = np.array(
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

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_items = None
            test_cutoff = 3
            test_signs = dict(
                sign_user_frequency=1,
                sign_uim_frequency=1,
                sign_uim_position=1,
                sign_uim_last_seen=1,
            )
            test_regs = dict(
                reg_user_frequency=1.0,
                reg_uim_frequency=1.0,
                reg_uim_position=1.0,
                reg_uim_last_seen=1.0,
            )
            test_funcs = dict(
                func_user_frequency=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_frequency=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_position=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_last_seen=EImpressionsDiscountingFunctions.LINEAR,
            )

            expected_item_scores = np.array([
                [5., 162., 75., 64., 12., 55., 16.],
                [27., 3., 3., 28., 6., 3., 4.],
                [3., 4., 6., 16., 10., 12., 35.],
                [7., 6., 5., 4., 3., 2., 1.],
                [45., 35., 190., 20., 32., 35., 69.],
                [29., 72., 102., 58., 144., 70., 20.],
                [4., 2., 3., 2., 2., 5., 2.],
                [2., 6., 99., 80., 15., 186., 14.],
                [7., 6., 5., 4., 3., 2., 1.],
                [18., 21., 10., 8., 40., 14., 12.],
            ], dtype=np.float32)

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_signs, **test_regs, **test_funcs)  # type: ignore
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
            assert np.allclose(expected_item_scores, scores)

    def test_all_users_some_items(
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix, uim_position: sp.csr_matrix,
        uim_last_seen: sp.csr_matrix,
    ):
        test_trained_recommender_compute_item_score = np.array(
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

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_items = [1, 2, 5]
            test_cutoff = 3
            test_signs = dict(
                sign_user_frequency=1,
                sign_uim_frequency=1,
                sign_uim_position=1,
                sign_uim_last_seen=1,
            )
            test_regs = dict(
                reg_user_frequency=1.0,
                reg_uim_frequency=1.0,
                reg_uim_position=1.0,
                reg_uim_last_seen=1.0,
            )
            test_funcs = dict(
                func_user_frequency=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_frequency=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_position=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_last_seen=EImpressionsDiscountingFunctions.LINEAR,
            )

            expected_item_scores = np.array([
                [np.NINF, 162., 75., np.NINF, np.NINF, 55., np.NINF],
                [np.NINF, 3., 3., np.NINF, np.NINF, 3., np.NINF],
                [np.NINF, 4., 6., np.NINF, np.NINF, 12., np.NINF],
                [np.NINF, 6., 5., np.NINF, np.NINF, 2., np.NINF],
                [np.NINF, 35., 190., np.NINF, np.NINF, 35., np.NINF],
                [np.NINF, 72., 102., np.NINF, np.NINF, 70., np.NINF],
                [np.NINF, 2., 3., np.NINF, np.NINF, 5., np.NINF],
                [np.NINF, 6., 99., np.NINF, np.NINF, 186., np.NINF],
                [np.NINF, 6., 5., np.NINF, np.NINF, 2., np.NINF],
                [np.NINF, 21., 10., np.NINF, np.NINF, 14., np.NINF],
            ], dtype=np.float32)

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_signs, **test_regs, **test_funcs)  # type: ignore
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

    def test_all_users_all_items(
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix, uim_position: sp.csr_matrix,
        uim_last_seen: sp.csr_matrix,
    ):
        test_trained_recommender_compute_item_score = np.array(
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

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_items = [0, 1, 2, 3, 4, 5, 6]
            test_cutoff = 3
            test_signs = dict(
                sign_user_frequency=1,
                sign_uim_frequency=1,
                sign_uim_position=1,
                sign_uim_last_seen=1,
            )
            test_regs = dict(
                reg_user_frequency=1.0,
                reg_uim_frequency=1.0,
                reg_uim_position=1.0,
                reg_uim_last_seen=1.0,
            )
            test_funcs = dict(
                func_user_frequency=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_frequency=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_position=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_last_seen=EImpressionsDiscountingFunctions.LINEAR,
            )

            expected_item_scores = np.array([
                [5., 162., 75., 64., 12., 55., 16.],
                [27., 3., 3., 28., 6., 3., 4.],
                [3., 4., 6., 16., 10., 12., 35.],
                [7., 6., 5., 4., 3., 2., 1.],
                [45., 35., 190., 20., 32., 35., 69.],
                [29., 72., 102., 58., 144., 70., 20.],
                [4., 2., 3., 2., 2., 5., 2.],
                [2., 6., 99., 80., 15., 186., 14.],
                [7., 6., 5., 4., 3., 2., 1.],
                [18., 21., 10., 8., 40., 14., 12.],
            ], dtype=np.float32)

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_signs, **test_regs, **test_funcs)  # type: ignore
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
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix, uim_position: sp.csr_matrix,
        uim_last_seen: sp.csr_matrix,
    ):
        test_trained_recommender_compute_item_score = np.array(
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

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 3, 6, 7, 8, 9]
            test_items = None
            test_cutoff = 3
            test_signs = dict(
                sign_user_frequency=1,
                sign_uim_frequency=1,
                sign_uim_position=1,
                sign_uim_last_seen=1,
            )
            test_regs = dict(
                reg_user_frequency=1.0,
                reg_uim_frequency=1.0,
                reg_uim_position=1.0,
                reg_uim_last_seen=1.0,
            )
            test_funcs = dict(
                func_user_frequency=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_frequency=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_position=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_last_seen=EImpressionsDiscountingFunctions.LINEAR,
            )

            expected_item_scores = np.array([
                [5., 162., 75., 64., 12., 55., 16.],
                [27., 3., 3., 28., 6., 3., 4.],
                [7., 6., 5., 4., 3., 2., 1.],
                [4., 2., 3., 2., 2., 5., 2.],
                [2., 6., 99., 80., 15., 186., 14.],
                [7., 6., 5., 4., 3., 2., 1.],
                [18., 21., 10., 8., 40., 14., 12.],
            ], dtype=np.float32)

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_signs, **test_regs, **test_funcs)  # type: ignore
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

    def test_some_users_some_items(
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix, uim_position: sp.csr_matrix,
        uim_last_seen: sp.csr_matrix,
    ):

        test_trained_recommender_compute_item_score = np.array(
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

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 3, 6, 7, 8, 9]
            test_items = [1, 2, 5]
            test_cutoff = 3
            test_signs = dict(
                sign_user_frequency=1,
                sign_uim_frequency=1,
                sign_uim_position=1,
                sign_uim_last_seen=1,
            )
            test_regs = dict(
                reg_user_frequency=1.0,
                reg_uim_frequency=1.0,
                reg_uim_position=1.0,
                reg_uim_last_seen=1.0,
            )
            test_funcs = dict(
                func_user_frequency=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_frequency=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_position=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_last_seen=EImpressionsDiscountingFunctions.LINEAR,
            )

            expected_item_scores = np.array([
                [np.NINF, 162., 75., np.NINF, np.NINF, 55., np.NINF],
                [np.NINF, 3., 3., np.NINF, np.NINF, 3., np.NINF],
                [np.NINF, 6., 5., np.NINF, np.NINF, 2., np.NINF],
                [np.NINF, 2., 3., np.NINF, np.NINF, 5., np.NINF],
                [np.NINF, 6., 99., np.NINF, np.NINF, 186., np.NINF],
                [np.NINF, 6., 5., np.NINF, np.NINF, 2., np.NINF],
                [np.NINF, 21., 10., np.NINF, np.NINF, 14., np.NINF],
            ], dtype=np.float32)

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_signs, **test_regs, **test_funcs)  # type: ignore
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
        self, urm: sp.csr_matrix, uim_frequency: sp.csr_matrix, uim_position: sp.csr_matrix,
        uim_last_seen: sp.csr_matrix,
    ):
        test_trained_recommender_compute_item_score = np.array(
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

        mock_base_recommender = BaseRecommender(URM_train=urm)
        with patch.object(
            mock_base_recommender,
            '_compute_item_score',
            return_value=test_trained_recommender_compute_item_score
        ) as _:
            # arrange
            test_users = [0, 1, 3, 6, 7, 8, 9]
            test_items = [0, 1, 2, 3, 4, 5, 6]
            test_cutoff = 3
            test_signs = dict(
                sign_user_frequency=1,
                sign_uim_frequency=1,
                sign_uim_position=1,
                sign_uim_last_seen=1,
            )
            test_regs = dict(
                reg_user_frequency=1.0,
                reg_uim_frequency=1.0,
                reg_uim_position=1.0,
                reg_uim_last_seen=1.0,
            )
            test_funcs = dict(
                func_user_frequency=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_frequency=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_position=EImpressionsDiscountingFunctions.LINEAR,
                func_uim_last_seen=EImpressionsDiscountingFunctions.LINEAR,
            )

            expected_item_scores = np.array([
                [5., 162., 75., 64., 12., 55., 16.],
                [27., 3., 3., 28., 6., 3., 4.],
                [7., 6., 5., 4., 3., 2., 1.],
                [4., 2., 3., 2., 2., 5., 2.],
                [2., 6., 99., 80., 15., 186., 14.],
                [7., 6., 5., 4., 3., 2., 1.],
                [18., 21., 10., 8., 40., 14., 12.],
            ], dtype=np.float32)

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_signs, **test_regs, **test_funcs)  # type: ignore
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
