import numpy as np
import scipy.sparse as sp
from Recommenders.BaseRecommender import BaseRecommender
from mock import patch

from impression_recommenders.re_ranking.impressions_discounting import ImpressionsDiscountingRecommender, \
    EImpressionsDiscountingFunctions


class TestImpressionsDiscountingRecommender:
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
                [0.08510638, 3.31914887, 1.53191486, 1.31914891, 0.19148936, 1.06382976, 0.25531914],
                [0.55319148, 0.04255319, 0.04255319, 0.57446807, 0.10638298, 0.04255319, 0.06382979],
                [0.04255319, 0.04255319, 0.06382979, 0.25531914, 0.10638298, 0.12765957, 0.59574467],
                [0., 0., 0., 0., 0., 0., 0.],
                [0.76595743, 0.59574467, 3.93617013, 0.34042552, 0.51063829, 0.59574467, 1.40425529],
                [0.59574467, 1.40425529, 2.10638293, 1.19148934, 2.99999994, 1.38297869, 0.34042552],
                [0.06382979, 0.0212766, 0.04255319, 0.0212766, 0.0212766, 0.08510638, 0.0212766],
                [0.0212766, 0.08510638, 2.04255315, 1.61702124, 0.21276595, 3.82978715, 0.14893617],
                [0., 0., 0., 0., 0., 0., 0.],
                [0.19148936, 0.29787233, 0.10638298, 0.08510638, 0.68085105, 0.14893617, 0.19148936],
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
            rec.fit(**test_regs, **test_funcs)  # type: ignore
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
                [0.44444433, 2.77192909, 0.88596466, 1.24888849, 0.33333325, 1.72222182, 0.44444433],
                [0.34848472, 0.16666661, 0.16666661, 0.68181795, 0.66666645, 0.16666661, 0.49999985],
                [0.66666645, 0.66666645, 0.99999967, 3.99999881, 1.66666612, 1.99999934, 5.83333153],
                [0., 0., 0., 0., 0., 0., 0.],
                [3.99999896, 3.1111103, 2.27777705, 1.77777731, 0.88888866, 3.1111103, 1.73333287],
                [0.26515144, 1.89999944, 0.49166648, 0.42063481, 1.77499938, 2.22222164, 0.33333322],
                [0.9999997, 0.33333322, 0.66666645, 0.33333322, 0.33333322, 0.83333308, 0.33333322],
                [0.33333322, 1.3333329, 2.03703654, 3.20634842, 3.33333224, 3.07407296, 2.33333257],
                [0., 0., 0., 0., 0., 0., 0.],
                [2.99999902, 4.66666514, 1.66666612, 1.3333329, 6.6666646, 2.33333257, 2.99999911]],
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
            rec.fit(**test_regs, **test_funcs)  # type: ignore
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
                [9.68787117e-17, 4.54953741e-09, 2.27476871e-09, 6.11804637e-07, 2.55991449e-16, 3.73616783e-15,
                 3.41321932e-16],
                [1.52299808e-08, 3.13913289e-17, 3.13913289e-17, 1.52299808e-08, 7.43308800e-17, 3.13913289e-17,
                 4.29395544e-17],
                [2.73448060e-17, 2.30964510e-17, 3.46446766e-17, 1.38578706e-16, 5.77411276e-17, 6.92893531e-17,
                 3.81414459e-16],
                [2.97384818e-17, 2.54901273e-17, 2.12417727e-17, 1.69934182e-17, 1.27450636e-17, 8.49670909e-18,
                 4.24835454e-18],
                [8.71908406e-16, 7.07889455e-16, 2.26999655e-04, 3.87514847e-16, 6.82643863e-16, 6.78150982e-16,
                 4.16643777e-11],
                [1.52299808e-08, 5.36313204e-15, 3.37605528e-07, 1.12055947e-08, 3.00000000e+00, 2.90313184e-15,
                 9.27809185e-16],
                [3.46446766e-17, 1.15482255e-17, 2.73448060e-17, 1.15482255e-17, 1.15482255e-17, 5.44877799e-17,
                 1.15482255e-17],
                [1.15482255e-17, 4.61929021e-17, 6.78098831e-06, 2.04367929e-11, 1.15482255e-16, 1.35619766e-05,
                 8.08375786e-17],
                [2.97384818e-17, 2.54901273e-17, 2.12417727e-17, 1.69934182e-17, 1.27450636e-17, 8.49670909e-18,
                 4.24835454e-18],
                [1.03934030e-16, 1.91413642e-16, 5.77411276e-17, 4.61929021e-17, 4.35902239e-16, 8.08375786e-17,
                 1.03934030e-16]
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
            rec.fit(**test_regs, **test_funcs)  # type: ignore
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
                [6.16903137e-03, 1.39913633e+00, 6.92165315e-01, 7.88402200e-01, 1.66563853e-02, 1.17211593e-01,
                 2.22085137e-02],
                [3.03516358e-01, 2.46761250e-03, 2.46761250e-03, 3.04133236e-01, 5.55212842e-03, 2.46761250e-03,
                 3.08451569e-03],
                [1.23380625e-03, 1.23380625e-03, 1.85070938e-03, 7.40283774e-03, 3.08451563e-03, 3.70141875e-03,
                 2.59099321e-02],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                 0.00000000e+00],
                [5.55212824e-02, 4.31832196e-02, 2.82850087e+00, 2.46761255e-02, 4.44170274e-02, 4.31832196e-02,
                 4.51573089e-01],
                [3.10919195e-01, 1.66563850e-01, 1.14188772e+00, 5.74953735e-01, 3.00000000e+00, 1.38803208e-01,
                 3.94818000e-02],
                [1.85070944e-03, 6.16903126e-04, 1.23380625e-03, 6.16903126e-04, 6.16903126e-04, 3.70141887e-03,
                 6.16903126e-04],
                [6.16903126e-04, 2.46761250e-03, 1.36582357e+00, 5.10795832e-01, 6.16903126e-03, 2.71684152e+00,
                 4.31832188e-03],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                 0.00000000e+00],
                [5.55212813e-03, 8.63664376e-03, 3.08451563e-03, 2.46761250e-03, 2.96113510e-02, 4.31832188e-03,
                 5.55212831e-03]
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
            rec.fit(**test_regs, **test_funcs)  # type: ignore
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
            print(repr(scores))
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
                [0.08510638, 3.31914887, 1.53191486, 1.31914891, 0.19148936, 1.06382976, 0.25531914],
                [0.55319148, 0.04255319, 0.04255319, 0.57446807, 0.10638298, 0.04255319, 0.06382979],
                [0.04255319, 0.04255319, 0.06382979, 0.25531914, 0.10638298, 0.12765957, 0.59574467],
                [0., 0., 0., 0., 0., 0., 0.],
                [0.76595743, 0.59574467, 3.93617013, 0.34042552, 0.51063829, 0.59574467, 1.40425529],
                [0.59574467, 1.40425529, 2.10638293, 1.19148934, 2.99999994, 1.38297869, 0.34042552],
                [0.06382979, 0.0212766, 0.04255319, 0.0212766, 0.0212766, 0.08510638, 0.0212766],
                [0.0212766, 0.08510638, 2.04255315, 1.61702124, 0.21276595, 3.82978715, 0.14893617],
                [0., 0., 0., 0., 0., 0., 0.],
                [0.19148936, 0.29787233, 0.10638298, 0.08510638, 0.68085105, 0.14893617, 0.19148936],
            ], dtype=np.float32)

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_regs, **test_funcs)  # type: ignore
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
                [np.NINF, 3.31914887, 1.53191486, np.NINF, np.NINF, 1.06382976, np.NINF],
                [np.NINF, 0.04255319, 0.04255319, np.NINF, np.NINF, 0.04255319, np.NINF],
                [np.NINF, 0.04255319, 0.06382979, np.NINF, np.NINF, 0.12765957, np.NINF],
                [np.NINF, 0., 0., np.NINF, np.NINF, 0., np.NINF],
                [np.NINF, 0.59574467, 3.93617013, np.NINF, np.NINF, 0.59574467, np.NINF],
                [np.NINF, 1.40425529, 2.10638293, np.NINF, np.NINF, 1.38297869, np.NINF],
                [np.NINF, 0.0212766, 0.04255319, np.NINF, np.NINF, 0.08510638, np.NINF],
                [np.NINF, 0.08510638, 2.04255315, np.NINF, np.NINF, 3.82978715, np.NINF],
                [np.NINF, 0., 0., np.NINF, np.NINF, 0., np.NINF],
                [np.NINF, 0.29787233, 0.10638298, np.NINF, np.NINF, 0.14893617, np.NINF],
            ], dtype=np.float32)

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_regs, **test_funcs)  # type: ignore
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
                [0.08510638, 3.31914887, 1.53191486, 1.31914891, 0.19148936, 1.06382976, 0.25531914],
                [0.55319148, 0.04255319, 0.04255319, 0.57446807, 0.10638298, 0.04255319, 0.06382979],
                [0.04255319, 0.04255319, 0.06382979, 0.25531914, 0.10638298, 0.12765957, 0.59574467],
                [0., 0., 0., 0., 0., 0., 0.],
                [0.76595743, 0.59574467, 3.93617013, 0.34042552, 0.51063829, 0.59574467, 1.40425529],
                [0.59574467, 1.40425529, 2.10638293, 1.19148934, 2.99999994, 1.38297869, 0.34042552],
                [0.06382979, 0.0212766, 0.04255319, 0.0212766, 0.0212766, 0.08510638, 0.0212766],
                [0.0212766, 0.08510638, 2.04255315, 1.61702124, 0.21276595, 3.82978715, 0.14893617],
                [0., 0., 0., 0., 0., 0., 0.],
                [0.19148936, 0.29787233, 0.10638298, 0.08510638, 0.68085105, 0.14893617, 0.19148936],
            ], dtype=np.float32)

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_regs, **test_funcs)  # type: ignore
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
                [0.08510638, 3.31914887, 1.53191486, 1.31914891, 0.19148936, 1.06382976, 0.25531914],
                [0.55319148, 0.04255319, 0.04255319, 0.57446807, 0.10638298, 0.04255319, 0.06382979],
                [0., 0., 0., 0., 0., 0., 0.],
                [0.06382979, 0.0212766, 0.04255319, 0.0212766, 0.0212766, 0.08510638, 0.0212766],
                [0.0212766, 0.08510638, 2.04255315, 1.61702124, 0.21276595, 3.82978715, 0.14893617],
                [0., 0., 0., 0., 0., 0., 0.],
                [0.19148936, 0.29787233, 0.10638298, 0.08510638, 0.68085105, 0.14893617, 0.19148936],
            ], dtype=np.float32)

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_regs, **test_funcs)  # type: ignore
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
                [np.NINF, 3.31914887, 1.53191486, np.NINF, np.NINF, 1.06382976, np.NINF],
                [np.NINF, 0.04255319, 0.04255319, np.NINF, np.NINF, 0.04255319, np.NINF],
                [np.NINF, 0., 0., np.NINF, np.NINF, 0., np.NINF],
                [np.NINF, 0.0212766, 0.04255319, np.NINF, np.NINF, 0.08510638, np.NINF],
                [np.NINF, 0.08510638, 2.04255315, np.NINF, np.NINF, 3.82978715, np.NINF],
                [np.NINF, 0., 0., np.NINF, np.NINF, 0., np.NINF],
                [np.NINF, 0.29787233, 0.10638298, np.NINF, np.NINF, 0.14893617, np.NINF],
            ], dtype=np.float32)

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_regs, **test_funcs)  # type: ignore
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
            print(repr(scores))
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
                [0.08510638, 3.31914887, 1.53191486, 1.31914891, 0.19148936, 1.06382976, 0.25531914],
                [0.55319148, 0.04255319, 0.04255319, 0.57446807, 0.10638298, 0.04255319, 0.06382979],
                [0., 0., 0., 0., 0., 0., 0.],
                [0.06382979, 0.0212766, 0.04255319, 0.0212766, 0.0212766, 0.08510638, 0.0212766],
                [0.0212766, 0.08510638, 2.04255315, 1.61702124, 0.21276595, 3.82978715, 0.14893617],
                [0., 0., 0., 0., 0., 0., 0.],
                [0.19148936, 0.29787233, 0.10638298, 0.08510638, 0.68085105, 0.14893617, 0.19148936],
            ], dtype=np.float32)

            rec = ImpressionsDiscountingRecommender(
                urm_train=urm,
                uim_frequency=uim_frequency,
                uim_position=uim_position,
                uim_last_seen=uim_last_seen,
                trained_recommender=mock_base_recommender,
            )

            # act
            rec.fit(**test_regs, **test_funcs)  # type: ignore
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
