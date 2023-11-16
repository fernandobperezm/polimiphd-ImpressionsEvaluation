import numpy as np
import numba as nb
import timeit

from impressions_evaluation.impression_recommenders.matrix_factorization.SFC import (
    _compute_soft_frequency_capping_score,
    _compute_soft_frequency_capping_score_for_loop,
)


class TestTimeNumba:
    def test_time(self):
        num_factors = 50
        num_users = 100
        num_items = 100_000

        np.random.seed(1234)

        arr_user_ids = np.arange(start=0, stop=num_users)
        arr_item_ids = np.arange(start=0, stop=num_items)
        arr_global_bias = np.asarray([[-0.01234512]], dtype=np.float32)
        arr_user_factors = np.random.normal(
            loc=0,
            scale=1.0,
            size=num_users * num_factors,
        ).reshape((num_users, num_factors))
        arr_item_factors = np.random.normal(
            loc=0,
            scale=1.0,
            size=num_items * num_factors,
        ).reshape((num_items, num_factors))
        arr_frequency_factors = np.random.normal(
            loc=0,
            scale=1.0,
            size=num_users * num_items,
        ).reshape((num_users, num_items))
        arr_item_scores_py_vector = np.zeros(
            (num_users, num_items),
            dtype=np.float32,
        )
        arr_item_scores_py_for = arr_item_scores_py_vector.copy()
        arr_item_scores_nb_vector = arr_item_scores_py_vector.copy()
        arr_item_scores_nb_for = arr_item_scores_py_vector.copy()

        import timeit
        import numba as nb

        func_py_vector = _compute_soft_frequency_capping_score
        func_py_loop = _compute_soft_frequency_capping_score_for_loop

        func_nb_vector = nb.njit(_compute_soft_frequency_capping_score)
        func_nb_loop = nb.njit(_compute_soft_frequency_capping_score_for_loop)

        time_nb_vector = timeit.timeit(
            stmt=lambda: func_nb_vector(
                arr_item_scores=arr_item_scores_nb_vector,
                arr_user_ids=arr_user_ids,
                arr_item_ids=arr_item_ids,
                arr_global_bias=arr_global_bias,
                arr_user_factors=arr_user_factors,
                arr_item_factors=arr_item_factors,
                arr_frequency_factors=arr_frequency_factors,
            ),
            globals=globals(),
            number=50,
        )
        time_nb_for = timeit.timeit(
            stmt=lambda: func_nb_loop(
                arr_item_scores=arr_item_scores_nb_for,
                arr_user_ids=arr_user_ids,
                arr_item_ids=arr_item_ids,
                arr_global_bias=arr_global_bias,
                arr_user_factors=arr_user_factors,
                arr_item_factors=arr_item_factors,
                arr_frequency_factors=arr_frequency_factors,
            ),
            globals=globals(),
            number=50,
        )

        # print(f"\t* {time_py_vector=}")
        # print(f"\t* {time_py_for=}")
        print(f"\t* {time_nb_vector=}")
        print(f"\t* {time_nb_for=}")
