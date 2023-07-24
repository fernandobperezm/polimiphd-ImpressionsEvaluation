import logging

import jax
import jax.numpy as jnp
import numba as nb
import numpy as np
import optax
import scipy.sparse as sp
from Recommenders.Recommender_utils import check_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def sigmoid(x_uij: jax.Array):
    return 1 / (1 + jnp.exp(x_uij))


def bpr_loss(x_uij: jax.Array):
    return x_uij**2


def predict_and_gradient(
    params,
    data,
    hyper_params,
    grads,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    embedding_user: jax.Array
    embedding_item: jax.Array
    embedding_user, embedding_item = params

    user: int
    pos_item: int
    neg_item: int
    user, pos_item, neg_item = data

    user_reg: float
    pos_reg: float
    neg_reg: float
    user_reg, pos_reg, neg_reg = hyper_params

    grad_user: jax.Array
    grad_item: jax.Array
    grad_user, grad_item = grads

    num_factors = embedding_user.shape[1]

    # compute embedding
    # x_uij = 0.0
    # for factor_index in range(num_factors):
    #     x_uij += embedding_user[user, factor_index] * (
    #         embedding_item[pos_item, factor_index]
    #         - embedding_item[neg_item, factor_index]
    #     )
    x_uij = (
        embedding_user[user, :]
        @ (embedding_item[pos_item, :] - embedding_item[neg_item, :]).transpose()
    )

    bpr_l = bpr_loss(x_uij=x_uij)

    sigmoid_item = sigmoid(x_uij=x_uij)
    sigmoid_user = sigmoid_item

    # for factor_index in range(num_factors):
    #     W_u = embedding_user[user, factor_index]
    #     H_i = embedding_item[pos_item, factor_index]
    #     H_j = embedding_item[neg_item, factor_index]
    #
    #     # Compute gradients
    #     local_gradient_u = sigmoid_user * (H_i - H_j) - user_reg * W_u
    #     local_gradient_pos_item = sigmoid_item * (W_u) - pos_reg * H_i
    #     local_gradient_neg_item = sigmoid_item * (-W_u) - neg_reg * H_j
    #
    #     grad_user = grad_user.at[user, factor_index].add(local_gradient_u)
    #     grad_item = grad_item.at[pos_item, factor_index].add(local_gradient_pos_item)
    #     grad_item = grad_item.at[neg_item, factor_index].add(local_gradient_neg_item)

    W_u = embedding_user[user, :]
    H_i = embedding_item[pos_item, :]
    H_j = embedding_item[neg_item, :]

    # Compute gradients
    local_gradient_u = sigmoid_user * (H_i - H_j) - user_reg * W_u
    local_gradient_pos_item = sigmoid_item * (W_u) - pos_reg * H_i
    local_gradient_neg_item = sigmoid_item * (-W_u) - neg_reg * H_j

    grad_user = grad_user.at[user, :].add(local_gradient_u)
    grad_item = grad_item.at[pos_item, :].add(local_gradient_pos_item)
    grad_item = grad_item.at[neg_item, :].add(local_gradient_neg_item)

    return bpr_l, grad_user, grad_item


batched_predict = jax.vmap(
    predict_and_gradient,
    in_axes=(None, (0, 0, 0), None, None),
    out_axes=(0, 0, 0),
)


@jax.jit
def update(
    params: optax.Params,
    data,
    hyper_params,
):
    grads = (
        jnp.zeros_like(params[0], dtype=jnp.float64),
        jnp.zeros_like(params[1], dtype=jnp.float64),
    )
    loss, grads_user, grads_item = batched_predict(
        params,
        data,
        hyper_params,
        grads,
    )

    return loss.sum(axis=0), (grads_user.sum(axis=0), grads_item.sum(axis=0))


@nb.njit
def sample_bpr(
    num_users: int,
    num_items: int,
    urm_indices: np.ndarray,
    urm_indptr: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    start_pos_seen_items = -1
    end_pos_seen_items = -1

    user_id = -1
    pos_item = -1
    neg_item = -1

    num_seen_items = 0
    while num_seen_items == 0 or num_seen_items == num_items:
        user_id = rng.integers(
            low=0,
            high=num_users,
            dtype=np.int32,
            endpoint=False,
        )

        start_pos_seen_items = urm_indptr[user_id]
        end_pos_seen_items = urm_indptr[user_id + 1]

        num_seen_items = end_pos_seen_items - start_pos_seen_items

    index = rng.integers(
        low=0,
        high=num_seen_items,
        dtype=np.int32,
        endpoint=False,
    )

    pos_item = urm_indices[start_pos_seen_items + index]

    neg_item_selected = False

    # It's faster to just try again then to build a mapping of the non-seen items
    # for every user
    while not neg_item_selected:
        neg_item = rng.integers(
            low=0,
            high=num_items,
            dtype=np.int32,
            endpoint=False,
        )

        index = 0
        # Indices data is sorted, so I don't need to go to the end of the current row
        while (
            index < num_seen_items
            and urm_indices[start_pos_seen_items + index] < neg_item
        ):
            index += 1

        # If the positive item in position 'index' is == sample.neg_item, negative not selected
        # If the positive item in position 'index' is > sample.neg_item or index == n_seen_items, negative selected
        if (
            index == num_seen_items
            or urm_indices[start_pos_seen_items + index] > neg_item
        ):
            neg_item_selected = True

    return user_id, pos_item, neg_item


class MFBPRModel:
    def __init__(
        self,
        urm_train: sp.csr_matrix,
        num_users: int,
        num_items: int,
        num_factors: int,
        batch_size: int,
        learning_rate: float,
        user_reg: float,
        positive_reg: float,
        negative_reg: float,
        init_mean: float = 0.0,
        init_std_dev: float = 0.0,
        sgd_mode: str = "sgd",
        gamma: float = 0.995,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        use_gpu: bool = False,
        seed: int = 1234567890,
    ):
        super().__init__()

        self.urm_train = urm_train

        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors

        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.user_reg = user_reg
        self.positive_reg = positive_reg
        self.negative_reg = negative_reg

        self.init_mean = init_mean
        self.init_std_dev = init_std_dev

        self.sgd_mode = sgd_mode

        self.gamma = gamma
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.use_gpu = use_gpu

        jax.clear_caches()
        if self.use_gpu:
            logging.debug("SFCModel: running computations on GPU")
            self.device = jax.devices("gpu")[0]  # This will fail if no gpu is present.
        else:
            logging.debug("SFCModel: running computations on CPU")
            self.device = jax.devices("cpu")[0]

        init_normal: jax.nn.initializers.Initializer = jax.nn.initializers.normal(
            stddev=self.init_std_dev,
            dtype=jnp.float64,
        )
        key = jax.random.PRNGKey(
            seed=seed,
        )

        key, subkey = jax.random.split(key)
        self.embedding_user: jax.Array = init_normal(
            key=subkey,
            shape=(self.num_users, self.num_factors),
        )

        key, subkey = jax.random.split(key)
        self.embedding_item: jax.Array = init_normal(
            key=subkey,
            shape=(self.num_items, self.num_factors),
        )

        self.params = (
            self.embedding_user,
            self.embedding_item,
        )

        if "sgd" == self.sgd_mode:
            optimizer = optax.sgd(
                learning_rate=self.learning_rate,
            )
        elif "adagrad" == self.sgd_mode:
            optimizer = optax.adagrad(
                learning_rate=self.learning_rate,
                initial_accumulator_value=0.0,
            )
        elif "rmsprop" == self.sgd_mode:
            optimizer = optax.rmsprop(
                learning_rate=self.learning_rate,
                decay=self.gamma,
                initial_scale=0.0,
            )
        elif "adam" == self.sgd_mode:
            optimizer = optax.adam(
                learning_rate=self.learning_rate,
                b1=self.beta_1,
                b2=self.beta_2,
            )
        else:
            raise ValueError(
                f"Received an invalid optimizer value {self.sgd_mode}. Valid values are {['sgd', 'adam', 'adagrad', 'rmsprop']}"
            )

        self.dataset = MFBPRDataset(
            urm_train=urm_train,
            seed=seed,
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.optimizer = optimizer
        self.opt_state = self.optimizer.init(self.params)

    def run_epoch(self) -> float:
        params = self.params
        opt_state = self.opt_state
        epoch_loss = 0.0

        for batch_users, batch_pos_items, batch_neg_items in self.dataloader:
            batch_users = jax.device_put(
                jnp.ravel(jnp.array(batch_users, dtype=jnp.int32)),
                device=self.device,
            )
            batch_pos_items = jax.device_put(
                jnp.ravel(jnp.array(batch_pos_items, dtype=jnp.int32)),
                device=self.device,
            )
            batch_neg_items = jax.device_put(
                jnp.ravel(jnp.array(batch_neg_items, dtype=jnp.int32)),
                device=self.device,
            )

            l, grads = update(
                params,
                data=(
                    batch_users,
                    batch_pos_items,
                    batch_neg_items,
                ),
                hyper_params=(
                    self.user_reg,
                    self.positive_reg,
                    self.negative_reg,
                ),
            )

            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            epoch_loss += l

        self.embedding_user = params[0].block_until_ready()
        self.embedding_item = params[1].block_until_ready()

        self.params = (
            self.embedding_user,
            self.embedding_item,
        )

        return epoch_loss


class MFBPRDataset(Dataset):
    def __init__(
        self,
        urm_train: sp.csr_matrix,
        seed: int = 1234567890,
    ):
        super().__init__()

        self.urm_train = check_matrix(
            urm_train,
            format="csr",
            dtype=np.float32,
        )
        self.urm_train.eliminate_zeros()
        self.num_users, self.num_items = self.urm_train.shape
        self.nnz: int = self.urm_train.nnz

        self.rng = np.random.default_rng(
            seed=seed,
        )
        self.key = jax.random.PRNGKey(
            seed=seed,
        )
        self.subkey = self.key

    def __len__(self) -> int:
        return self.nnz

    def __getitem__(self, _: int) -> tuple[int, int, int]:
        """

        This method must return (user_id, pos_item_id, neg_item_id).
        """
        bpr_sample = sample_bpr(
            num_users=self.num_users,
            num_items=self.num_items,
            urm_indices=self.urm_train.indices,
            urm_indptr=self.urm_train.indptr,
            rng=self.rng,
        )
        return bpr_sample
