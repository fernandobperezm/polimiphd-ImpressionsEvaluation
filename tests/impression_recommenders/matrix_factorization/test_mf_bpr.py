import pytest
import numpy as np
import scipy.sparse as sp
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import (
    MatrixFactorization_BPR_Cython,
)

from impressions_evaluation.impression_recommenders.matrix_factorization.jax.model_mfbpr import (
    MFBPRModel,
)
from impressions_evaluation.impression_recommenders.matrix_factorization.mf_bpr import (
    MatrixFactorizationBPRImpressionsNegatives,
)
from impressions_evaluation.impression_recommenders.matrix_factorization.mf_bpr_jax import (
    MatrixFactorizationBPRJAX,
)


@pytest.mark.skip
class TestMFBPR:
    def test_recommender_framework_and_extension_is_equivalent(
        self,
        urm: sp.csr_matrix,
        uim: sp.csr_matrix,
        num_users: int,
        num_items: int,
    ):
        # Arrange
        epochs: int = 10
        batch_size: int = 100
        num_factors: int = 30
        learning_rate: float = 1e-2
        init_mean: float = 1.0
        init_std_dev: float = 1.0
        user_reg: float = 1e-2
        item_reg: float = 1e-3
        bias_reg: float = 1e-4
        positive_reg: float = 1e-1
        negative_reg: float = 1e-2
        use_bias: bool = False
        algorithm_name: str = "MF_BPR"
        random_seed: int = 1234567890

        test_users_array = np.arange(
            start=0,
            stop=num_users,
            dtype=np.int32,
        )
        test_items_array = np.arange(
            start=0,
            stop=num_items,
            dtype=np.int32,
        )

        # Act
        recommender_framework = MatrixFactorization_BPR_Cython(
            URM_train=urm,
        )
        recommender_extended = MatrixFactorizationBPRImpressionsNegatives(
            urm_train=urm,
            uim_train=uim,
        )

        recommender_framework.fit(
            epochs=epochs,
            use_bias=use_bias,
            batch_size=batch_size,
            num_factors=num_factors,
            learning_rate=learning_rate,
            init_mean=init_mean,
            init_std_dev=init_std_dev,
            user_reg=user_reg,
            item_reg=item_reg,
            bias_reg=bias_reg,
            positive_reg=positive_reg,
            negative_reg=negative_reg,
            random_seed=random_seed,
        )
        recommender_extended.fit(
            epochs=epochs,
            use_bias=use_bias,
            batch_size=batch_size,
            num_factors=num_factors,
            learning_rate=learning_rate,
            init_mean=init_mean,
            init_std_dev=init_std_dev,
            user_reg=user_reg,
            item_reg=item_reg,
            bias_reg=bias_reg,
            positive_reg=positive_reg,
            negative_reg=negative_reg,
            random_seed=random_seed,
        )

        framework_arr_user_factors: np.ndarray = (
            recommender_framework.USER_factors.copy().astype(np.float32)
        )
        framework_arr_item_factors: np.ndarray = (
            recommender_framework.ITEM_factors.copy().astype(np.float32)
        )

        extended_arr_user_factors: np.ndarray = (
            recommender_extended.USER_factors.copy().astype(np.float32)
        )
        extended_arr_item_factors: np.ndarray = (
            recommender_extended.ITEM_factors.copy().astype(np.float32)
        )

        score_framework = recommender_framework._compute_item_score(
            user_id_array=test_users_array,
            items_to_compute=test_items_array,
        )
        score_extended = recommender_extended._compute_item_score(
            user_id_array=test_users_array,
            items_to_compute=test_items_array,
        )

        score_framework = score_framework.astype(np.float64)
        score_extended = score_extended.astype(np.float64)

        # Assert
        assert np.array_equal(
            framework_arr_user_factors,
            extended_arr_user_factors,
            equal_nan=True,
        )
        assert np.array_equal(
            framework_arr_item_factors,
            extended_arr_item_factors,
            equal_nan=True,
        )
        assert np.array_equal(
            score_framework,
            score_extended,
            equal_nan=True,
        )

    def test_recommender_framework_and_extension_time_is_close(
        self,
        urm: sp.csr_matrix,
        uim: sp.csr_matrix,
        num_users: int,
        num_items: int,
    ):
        # Arrange
        epochs: int = 10
        batch_size: int = 100
        num_factors: int = 30
        learning_rate: float = 1e-2
        init_mean: float = 1.0
        init_std_dev: float = 1.0
        user_reg: float = 1e-2
        item_reg: float = 1e-3
        bias_reg: float = 1e-4
        positive_reg: float = 1e-1
        negative_reg: float = 1e-2
        use_bias: bool = False
        random_seed: int = 1234567890

        # Act
        recommender_framework = MatrixFactorization_BPR_Cython(
            URM_train=urm,
        )
        recommender_extended = MatrixFactorizationBPRImpressionsNegatives(
            urm_train=urm,
            uim_train=uim,
        )

        import timeit

        results_framework = timeit.repeat(
            stmt=lambda: recommender_framework.fit(
                epochs=epochs,
                use_bias=use_bias,
                batch_size=batch_size,
                num_factors=num_factors,
                learning_rate=learning_rate,
                init_mean=init_mean,
                init_std_dev=init_std_dev,
                user_reg=user_reg,
                item_reg=item_reg,
                bias_reg=bias_reg,
                positive_reg=positive_reg,
                negative_reg=negative_reg,
                random_seed=random_seed,
            ),
            repeat=5,
            number=1000,
            globals=globals(),
        )
        results_extended = timeit.repeat(
            stmt=lambda: recommender_extended.fit(
                epochs=epochs,
                use_bias=use_bias,
                batch_size=batch_size,
                num_factors=num_factors,
                learning_rate=learning_rate,
                init_mean=init_mean,
                init_std_dev=init_std_dev,
                user_reg=user_reg,
                item_reg=item_reg,
                bias_reg=bias_reg,
                positive_reg=positive_reg,
                negative_reg=negative_reg,
                random_seed=random_seed,
            ),
            repeat=5,
            number=1000,
            globals=globals(),
        )

        # Assert
        print(results_framework)
        print(results_extended)


class TestMFBPRJax:
    @pytest.mark.skip
    def test_recommender_framework_and_extension_time_is_close(
        self,
        urm: sp.csr_matrix,
        uim: sp.csr_matrix,
        num_users: int,
        num_items: int,
    ):
        # Arrange
        from dotenv import load_dotenv

        load_dotenv()

        epochs: int = 10
        batch_size: int = 100
        num_factors: int = 30
        learning_rate: float = 1e-2
        init_mean: float = 1.0
        init_std_dev: float = 1.0
        user_reg: float = 1e-2
        item_reg: float = 1e-3
        bias_reg: float = 1e-4
        positive_reg: float = 1e-1
        negative_reg: float = 1e-2
        use_bias: bool = False
        random_seed: int = 1234567890

        # Act
        recommender_framework = MatrixFactorization_BPR_Cython(
            URM_train=urm,
        )
        recommender_extended = MatrixFactorizationBPRJAX(
            urm_train=urm,
            use_gpu=False,
        )

        import timeit

        results_framework = timeit.repeat(
            stmt=lambda: recommender_framework.fit(
                epochs=epochs,
                use_bias=use_bias,
                batch_size=batch_size,
                num_factors=num_factors,
                learning_rate=learning_rate,
                init_mean=init_mean,
                init_std_dev=init_std_dev,
                user_reg=user_reg,
                item_reg=item_reg,
                bias_reg=bias_reg,
                positive_reg=positive_reg,
                negative_reg=negative_reg,
                random_seed=random_seed,
            ),
            repeat=1,
            number=1,
            globals=globals(),
        )
        results_extended = timeit.repeat(
            stmt=lambda: recommender_extended.fit(
                epochs=epochs,
                use_bias=use_bias,
                batch_size=batch_size,
                num_factors=num_factors,
                learning_rate=learning_rate,
                init_mean=init_mean,
                init_std_dev=init_std_dev,
                user_reg=user_reg,
                item_reg=item_reg,
                bias_reg=bias_reg,
                positive_reg=positive_reg,
                negative_reg=negative_reg,
                random_seed=random_seed,
            ),
            repeat=1,
            number=1,
            globals=globals(),
        )

        # Assert
        print(results_framework)
        print(results_extended)
        import pdb

        pdb.set_trace()
        print(results_framework, results_extended)

    def test_model_framework_and_extension_time_is_close(
        self,
        urm: sp.csr_matrix,
        uim: sp.csr_matrix,
        num_users: int,
        num_items: int,
    ):
        # Arrange
        from dotenv import load_dotenv

        load_dotenv()

        epochs: int = 10
        batch_size: int = 100
        num_factors: int = 30
        learning_rate: float = 1e-2
        init_mean: float = 1.0
        init_std_dev: float = 1.0
        user_reg: float = 1e-2
        positive_reg: float = 1e-1
        negative_reg: float = 1e-2
        random_seed: int = 1234567890
        sgd_mode = "sgd"

        from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython_Epoch import (
            MatrixFactorization_Cython_Epoch,
        )

        # Act
        model_framework = MatrixFactorization_Cython_Epoch(
            URM_train=urm,
            n_factors=num_factors,
            algorithm_name="MF_BPR",
            batch_size=batch_size,
            negative_interactions_quota=0.0,
            dropout_quota=None,
            WARP_neg_item_attempts=10,
            learning_rate=learning_rate,
            use_bias=False,
            use_embeddings=True,
            user_reg=user_reg,
            item_reg=0.0,
            bias_reg=0.0,
            positive_reg=positive_reg,
            negative_reg=negative_reg,
            init_mean=init_mean,
            init_std_dev=init_std_dev,
            sgd_mode=sgd_mode,
            random_seed=random_seed,
        )
        model_extended = MFBPRModel(
            urm_train=urm,
            num_users=num_users,
            num_items=num_items,
            num_factors=num_factors,
            batch_size=batch_size,
            learning_rate=learning_rate,
            user_reg=user_reg,
            positive_reg=positive_reg,
            negative_reg=negative_reg,
            init_mean=init_mean,
            init_std_dev=init_std_dev,
            sgd_mode=sgd_mode,
            use_gpu=False,
            seed=random_seed,
        )

        model_framework.epochIteration_Cython()
        model_extended.run_epoch()

        import timeit

        results_framework = timeit.repeat(
            stmt=lambda: model_framework.epochIteration_Cython(),
            repeat=3,
            number=100,
            globals=globals(),
        )
        results_extended = timeit.repeat(
            stmt=lambda: model_extended.run_epoch(),
            repeat=3,
            number=100,
            globals=globals(),
        )

        # Assert
        print(results_framework)
        print(results_extended)
        import pdb

        pdb.set_trace()
        print(results_framework, results_extended)
