import logging
from typing import Literal

import jax
import jax.numpy as jnp
import numba as nb
import numpy as np
import optax
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@jax.jit
def learning_rate_scheduler(
    initial_lr: float,
    epoch_loss: float,
    alpha: float,
    beta: float,
) -> float:
    return initial_lr * (1 / (alpha + (epoch_loss**beta)))


def loss(params, data, labels):
    preds = batched_predict(params, data).ravel()
    l_optax = optax.sigmoid_binary_cross_entropy(
        logits=preds,
        labels=labels,
    )

    return l_optax.sum()


def predict(
    params,
    data,
) -> jax.Array:
    embedding_bias: jax.Array = params[0]
    embedding_user: jax.Array = params[1]
    embedding_item: jax.Array = params[2]
    embedding_frequencies: jax.Array = params[3]

    user: int = data[0]
    item: int = data[1]
    freq: int = data[2]
    idx_emb: int = data[3]

    # compute embedding
    bias_embedding: jax.Array = embedding_bias[0].ravel()  # shape=(1,)
    users_embeddings: jax.Array = embedding_user[user, :].ravel()  # shape=(n, )
    items_embeddings: jax.Array = embedding_item[item, :].ravel()  # shape=(n, )
    freq_embeddings: jax.Array = embedding_frequencies[
        idx_emb, freq
    ].ravel()  # shape=(1,)

    # shape = (1,)
    pred_rating = (
        bias_embedding
        + users_embeddings @ items_embeddings.transpose()
        + freq_embeddings
    ).ravel()

    return pred_rating


batched_predict = jax.vmap(
    predict,
    in_axes=(None, 0),
)


@jax.jit
def update(
    params: optax.Params,
    data: jax.Array,
    labels: jax.Array,
):
    l, grads = jax.value_and_grad(loss)(params, data, labels)
    return l, grads


FREQUENCY_MODE = Literal["global", "user", "item"]


@nb.njit
def create_dict_binned_frequency(
    uim_frequency_coo_row: np.ndarray,
    uim_frequency_coo_col: np.ndarray,
    uim_frequency_coo_data: np.ndarray,
    num_bins: int,
) -> dict[tuple[int, int], int]:
    return {
        (user, item): val if val < num_bins else num_bins - 1
        for user, item, val in zip(
            uim_frequency_coo_row,
            uim_frequency_coo_col,
            uim_frequency_coo_data,
        )
    }


class SFCModel:
    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        num_users: int,
        num_items: int,
        batch_size: int,
        learning_rate: float,
        scheduler_alpha: float,
        scheduler_beta: float,
        frequency_mode: FREQUENCY_MODE,
        frequency_num_bins: int,
        frequency_num_embeddings: int,
        embedding_size: int,
        use_gpu: bool = False,
        seed: int = 1234567890,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.frequency_mode = frequency_mode
        self.frequency_num_bins = frequency_num_bins
        self.frequency_num_embeddings = frequency_num_embeddings
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scheduler_alpha = scheduler_alpha
        self.scheduler_beta = scheduler_beta
        self.use_gpu = use_gpu

        jax.clear_caches()
        if self.use_gpu:
            logger.debug("SFCModel: running computations on GPU")
            self.device = jax.devices("gpu")[0]  # This will fail if no gpu is present.
        else:
            logger.debug("SFCModel: running computations on CPU")
            self.device = jax.devices("cpu")[0]

        self.dataset = SFCDataset(
            urm_train=urm_train,
            uim_train=uim_train,
            uim_frequency=uim_frequency,
            frequency_mode=self.frequency_mode,
            num_frequency_bins=self.frequency_num_bins,
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        init_normal: jax.nn.initializers.Initializer = jax.nn.initializers.normal(
            stddev=0.1,
            dtype=jnp.float32,
        )
        key = jax.random.PRNGKey(
            seed=seed,
        )

        key, subkey = jax.random.split(key)
        self.embedding_bias: jax.Array = init_normal(
            key=subkey,
            shape=(1,),
        )

        key, subkey = jax.random.split(key)
        self.embedding_user: jax.Array = init_normal(
            key=subkey,
            shape=(self.num_users, self.embedding_size),
        )

        key, subkey = jax.random.split(key)
        self.embedding_item: jax.Array = init_normal(
            key=subkey,
            shape=(self.num_items, self.embedding_size),
        )

        self.embedding_frequencies: jax.Array = jnp.zeros(
            shape=(self.frequency_num_embeddings, frequency_num_bins),
            dtype=jnp.float32,
        )

        self.params = (
            self.embedding_bias,
            self.embedding_user,
            self.embedding_item,
            self.embedding_frequencies,
        )
        # It is mandatory the `optax.inject_hyperparams` call when initializing the Adagrad optimizer because we need to change the learning rate after every batch/epoch according to the paper.
        self.optimizer = optax.inject_hyperparams(optax.adagrad)(
            learning_rate=self.learning_rate,
        )
        self.opt_state = self.optimizer.init(self.params)
        self.opt_update_fn = jax.jit(self.optimizer.update)
        self.apply_updates_fn = jax.jit(optax.apply_updates)

    def run_epoch(self) -> float:
        params = self.params
        opt_state = self.opt_state
        epoch_loss = 0.0

        for batch in tqdm(self.dataloader):
            data, labels = batch

            data[0] = jax.device_put(
                jnp.ravel(jnp.array(data[0], dtype=jnp.int32)),
                device=self.device,
            )
            data[1] = jax.device_put(
                jnp.ravel(jnp.array(data[1], dtype=jnp.int32)),
                device=self.device,
            )
            data[2] = jax.device_put(
                jnp.ravel(jnp.array(data[2], dtype=jnp.int32)),
                device=self.device,
            )
            data[3] = jax.device_put(
                jnp.ravel(jnp.array(data[3], dtype=jnp.int32)),
                device=self.device,
            )
            labels = jax.device_put(
                jnp.ravel(jnp.array(labels, dtype=jnp.int32)),
                device=self.device,
            )

            l, grads = update(params, data, labels)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            epoch_loss += l

        # The paper requires the learning rate to change after every epoch (they call it steps) by a factor depending on the initial learning rate, two hyper-parameters (alpha and beta) and the epoch loss.
        # See https://github.com/deepmind/optax/discussions/262#discussioncomment-1822204 for a discussion about how to change the learning rate of the optimizer.
        new_learning_rate = learning_rate_scheduler(
            initial_lr=self.learning_rate,
            alpha=self.scheduler_alpha,
            beta=self.scheduler_beta,
            epoch_loss=epoch_loss,
        )
        opt_state.hyperparams["learning_rate"] = new_learning_rate
        self.optimizer.update(params, opt_state)

        # epoch_loss = epoch_loss.block_until_ready()
        self.embedding_bias = params[0].block_until_ready()
        self.embedding_user = params[1].block_until_ready()
        self.embedding_item = params[2].block_until_ready()
        self.embedding_frequencies = params[3].block_until_ready()

        self.params = (
            self.embedding_bias,
            self.embedding_user,
            self.embedding_item,
            self.embedding_frequencies,
        )

        return epoch_loss


class SFCDataset(Dataset):
    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        frequency_mode: FREQUENCY_MODE,
        num_frequency_bins: int,
    ):
        super().__init__()

        self.frequency_mode = frequency_mode
        self.num_frequency_bins = num_frequency_bins

        urm_train = urm_train.copy()
        uim_train = uim_train.copy()

        uim_train = uim_train - urm_train

        urm_train_coo: sp.coo_matrix = sp.coo_matrix(urm_train, copy=True)
        uim_train_coo: sp.coo_matrix = sp.coo_matrix(uim_train, copy=True)
        uim_frequency_coo: sp.coo_matrix = sp.coo_matrix(uim_frequency, copy=True)

        self.arr_users = np.concatenate(
            (
                urm_train_coo.row,
                uim_train_coo.row,
            ),
            dtype=np.int32,
            axis=None,
        )

        self.arr_items = np.concatenate(
            (
                urm_train_coo.col,
                uim_train_coo.col,
            ),
            dtype=np.int32,
            axis=None,
        )
        self.arr_labels = np.concatenate(
            (
                np.ones_like(urm_train_coo.data),
                np.zeros_like(uim_train_coo.data),
            ),
            dtype=np.float32,
            axis=None,
        )
        self.dict_binned_frequency = create_dict_binned_frequency(
            uim_frequency_coo_row=uim_frequency_coo.row,
            uim_frequency_coo_col=uim_frequency_coo.col,
            uim_frequency_coo_data=uim_frequency_coo.data,
            num_bins=self.num_frequency_bins,
        )

        assert self.arr_users.shape == self.arr_items.shape
        assert self.arr_users.shape == self.arr_labels.shape

    def __len__(self) -> int:
        return len(self.arr_users)

    def __getitem__(self, idx: int) -> tuple[tuple[int, int, int, int], int]:
        """

        This method must return (user_id, item_id, freq_ui, idx_emb, label, 0) where:
        * user is the user id
        * item is the item id
        * freq_ui is the frequency of the user-item pair;
        * idx_emb is the index of the embedding vector based on the type of model.
        * label is the label (interacted or non-interacted impression)
        """
        user = self.arr_users[idx]
        item = self.arr_items[idx]
        label = self.arr_labels[idx]
        freq_ui = self.dict_binned_frequency.get((user, item), 0)
        idx_emb = (
            0
            if self.frequency_mode == "global"
            else (user if self.frequency_mode == "user" else item)
        )

        return (
            (user, item, freq_ui, idx_emb),
            label,
        )
