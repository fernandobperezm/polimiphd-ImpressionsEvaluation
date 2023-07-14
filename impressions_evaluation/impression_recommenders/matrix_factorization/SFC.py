import copy
from typing import Optional, Literal

import attrs
import numba as nb
import numpy as np
import scipy.sparse as sp
import torch
from Recommenders.BaseMatrixFactorizationRecommender import (
    BaseMatrixFactorizationRecommender,
)
from Recommenders.Incremental_Training_Early_Stopping import (
    Incremental_Training_Early_Stopping,
)
from Utils.PyTorch.utils import clone_pytorch_model_to_numpy_dict
from recsys_framework_extensions.data.io import DataIO
from recsys_framework_extensions.recommenders.base import (
    SearchHyperParametersBaseRecommender,
)
from skopt.space import Integer, Categorical, Real
from torch.utils.data import DataLoader, Dataset


@attrs.define(kw_only=True, frozen=True, slots=False)
class SearchHyperParametersSFCRecommender(SearchHyperParametersBaseRecommender):
    epochs: Categorical = attrs.field(
        default=Categorical(  # The paper does not specify.
            [500],
        )
    )
    frequency_num_bins: Categorical = attrs.field(
        default=Categorical(  # The paper uses 26
            [26],
        )
    )
    frequency_mode: Categorical = attrs.field(
        default=Categorical(  # The paper uses global or two content-based variants of item.
            ["global", "user", "item"],
        )
    )
    batch_size: Integer = attrs.field(
        default=Integer(  # The paper uses 1.
            low=2**0,
            high=2**14,  # 32768
            prior="uniform",
        ),
    )
    embedding_size: Integer = attrs.field(
        default=Integer(  # The paper does not specify ranges.
            low=2**0,
            high=2**10,
            prior="uniform",
        ),
    )
    learning_rate: Real = attrs.field(
        default=Real(  # The paper does not specify ranges.
            low=1e-5,
            high=1e-2,
            prior="log-uniform",
        )
    )
    l2_reg: Real = attrs.field(
        default=Real(  # The paper does not specify ranges.
            low=1e-5,
            high=1e-2,
            prior="log-uniform",
        )
    )
    scheduler_alpha: Real = attrs.field(
        default=Real(  # The paper does not specify ranges.
            low=1e-5,
            high=1e-2,
            prior="log-uniform",
        )
    )
    scheduler_beta: Real = attrs.field(
        default=Real(  # The paper does not specify ranges.
            low=1e-5,
            high=1e-2,
            prior="log-uniform",
        )
    )


FREQUENCY_MODE = Literal["global", "user", "item"]


def learning_rate_scheduler(
    initial_lr: float,
    epoch_loss: float,
    alpha: float,
    beta: float,
) -> float:
    return initial_lr * (1 / (alpha + (epoch_loss**beta)))


@nb.njit
def _create_dict_binned_frequency(
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


@nb.njit
def _get_frequency_embedding_index(
    frequency_mode: FREQUENCY_MODE,
    user_id: int,
    item_id: int,
) -> int:
    if frequency_mode == "global":
        return 0
    elif frequency_mode == "user":
        return user_id
    elif frequency_mode == "item":
        return item_id

    # return value needed for numba
    return 0  # type: ignore


@nb.njit
def _get_frequency_num_embeddings(
    frequency_mode: FREQUENCY_MODE,
    num_users: int,
    num_items: int,
) -> int:
    if frequency_mode == "global":
        return 1
    elif frequency_mode == "user":
        return num_users
    elif frequency_mode == "item":
        return num_items

    # return value needed for numba
    return 1  # type: ignore


@nb.njit
def _compute_array_frequencies(
    arr_user_ids: np.ndarray,
    arr_item_ids: np.ndarray,
    arr_frequency_factors: np.ndarray,
    dict_frequency: dict[tuple[int, int], int],
    num_items: int,
    frequency_mode: FREQUENCY_MODE,
) -> np.ndarray:
    num_users = len(arr_user_ids)

    arr_frequencies = np.zeros(
        (num_users, num_items),
        dtype=np.float32,
    )

    for idx_user, user in enumerate(arr_user_ids):
        for item in arr_item_ids:
            freq = (
                dict_frequency[(user, item)]
                if (user, item) in dict_frequency
                else int(0)
            )

            idx = _get_frequency_embedding_index(
                frequency_mode=frequency_mode,
                user_id=user,
                item_id=item,
            )

            factor = arr_frequency_factors[idx, freq]

            arr_frequencies[idx_user, item] = factor

    return arr_frequencies


@nb.njit
def _compute_soft_frequency_capping_score(
    arr_item_scores: np.ndarray,
    arr_user_ids: np.ndarray,
    arr_item_ids: np.ndarray,
    arr_global_bias: np.ndarray,
    arr_user_factors: np.ndarray,
    arr_item_factors: np.ndarray,
    arr_frequency_factors: np.ndarray,
) -> np.ndarray:
    assert arr_global_bias.shape == (1,)
    assert arr_user_factors.shape[1] == arr_item_factors.shape[1]
    assert arr_frequency_factors.shape == arr_item_scores.shape

    arr_item_scores[:, arr_item_ids] = (
        arr_global_bias[0]
        + np.dot(arr_user_factors[arr_user_ids, :], arr_item_factors.T)
        + arr_frequency_factors
    )[:, arr_item_ids]

    return arr_item_scores


class SFCDataset(Dataset):
    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        dict_binned_frequency: dict[tuple[int, int], int],
        frequency_mode: FREQUENCY_MODE,
    ):
        urm_train = urm_train.copy()
        uim_train = uim_train.copy()

        uim_train = uim_train - urm_train

        urm_train_coo = sp.coo_matrix(urm_train)
        uim_train_coo = sp.coo_matrix(uim_train)

        self.arr_users: np.ndarray = np.concatenate(
            (
                urm_train_coo.row,
                uim_train_coo.row,
            ),
            axis=None,
        )

        self.arr_items: np.ndarray = np.concatenate(
            (
                urm_train_coo.col,
                uim_train_coo.col,
            ),
            axis=None,
        )
        self.arr_labels = np.concatenate(
            (
                np.ones_like(urm_train_coo.data),
                np.zeros_like(uim_train_coo.data),
            ),
            axis=None,
            dtype=np.float32,
        )
        self.dict_binned_frequency = dict_binned_frequency
        self.frequency_mode = frequency_mode

        assert self.arr_users.shape == self.arr_items.shape
        assert self.arr_users.shape == self.arr_labels.shape

    def __len__(self) -> int:
        return len(self.arr_users)

    def __getitem__(self, idx: int) -> tuple[int, int, int, int, int, int]:
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
        return user, item, freq_ui, idx_emb, label, 0


class SFCModel(torch.nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        frequency_mode: FREQUENCY_MODE,
        frequency_num_bins: int,
        frequency_num_embeddings: int,
        embedding_size: int,
        device: torch.device,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.frequency_mode = frequency_mode
        self.frequency_num_bins = frequency_num_bins
        self.frequency_num_embeddings = frequency_num_embeddings
        self.embedding_size = embedding_size

        self.embedding_bias = torch.nn.Embedding(
            num_embeddings=1,
            embedding_dim=1,
            device=device,
        )
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users,
            embedding_dim=self.embedding_size,
            device=device,
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.embedding_size,
            device=device,
        )
        self.embedding_frequencies = torch.nn.ModuleList(
            [
                torch.nn.Embedding(
                    num_embeddings=self.frequency_num_bins,
                    embedding_dim=1,
                    device=device,
                )
                for _ in range(self.frequency_num_embeddings)
            ]
        )
        # NOTE: Uncomment this when changing the loss from :torch.nn.BCEWithLogitsLoss to another that does not include the sigmoid layer.
        # self.sigmoid_layer = torch.nn.Sigmoid()

        torch.nn.init.normal_(
            self.embedding_bias.weight,
            mean=0.0,
            std=0.1,
        )
        torch.nn.init.normal_(
            self.embedding_user.weight,
            mean=0.0,
            std=0.1,
        )
        torch.nn.init.normal_(
            self.embedding_item.weight,
            mean=0.0,
            std=0.1,
        )
        [
            torch.nn.init.zeros_(  # The paper sets the tensor as zeros. See Algorithm , page 7.
                emb.weight,
            )
            for emb in self.embedding_frequencies
        ]

    def forward(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        freq: torch.Tensor,
        idx_emb: torch.Tensor,
        zero: torch.Tensor,
    ):
        # compute embedding
        bias_embedding: torch.Tensor = self.embedding_bias(zero)
        users_embeddings: torch.Tensor = self.embedding_user(user)
        items_embeddings: torch.Tensor = self.embedding_item(item)
        freq_embeddings: torch.Tensor = torch.stack(
            [self.embedding_frequencies[idx](f) for idx, f in zip(idx_emb, freq)]
        )

        # METHOD 1: Easy to understand but non-efficient memory-wise: it computes the product of two [b,m] matrices to then get the diagonal of them (pytorch does not seep to optimize).
        # the diagonal contains the predicted rating of every user-item pair in the batch.
        # pred_rating = torch.diag(
        #     bias_embedding + users_embeddings @ items_embeddings.T + freq_embeddings
        # )

        # METHOD 2: More scalable as it computes the equivalent of the diagonal of the previous example. Faster in CPU and more memory efficient.
        # :torch.bmm is a function to compute the batch-wise product between two tensors M and Q with shapes [b, m, n] and [b, n, p] resulting in a tensor with shape [b, m, p].
        # Originally, `users_embedding` and `items_embeddings` have shape [b, m] and we want a tensor with shape [b, 1].
        # Hence, we use `users_embeddings.unsqueeze(1)` to have a tensor with the shape [b, 1, m]
        # and we use `items_embeddings.unsqueeze(2)` to have a tensor with shape [b, m, 1]
        # Consequently, our resulting tensor has shape [b, 1, 1], to that tensor, we apply `squeeze(1)` so we end up with a tensor of shape [b, 1].
        # The remaining tensors have shape [b, 1], so we apply `squeeze(1)` one more time to the resulting tensor to end up with a tensor with shape [b].
        pred_rating = (
            bias_embedding
            + torch.bmm(
                users_embeddings.unsqueeze(1), items_embeddings.unsqueeze(2)
            ).squeeze(1)
            + freq_embeddings
        ).squeeze(1)

        # NOTE: Uncomment this when changing the loss from :torch.nn.BCEWithLogitsLoss to another not including the sigmoid layer.
        # sigmoid_pred_rating = self.sigmoid_layer(pred_rating)

        return pred_rating


class SoftFrequencyCappingRecommender(
    BaseMatrixFactorizationRecommender,
    Incremental_Training_Early_Stopping,
):
    """"""

    RECOMMENDER_NAME = "SFCRecommender"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        uim_train: sp.csr_matrix,
        uim_frequency: sp.csr_matrix,
        use_gpu: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            URM_train=urm_train,
            verbose=verbose,
        )

        self.urm_train: sp.csr_matrix = urm_train.copy()
        self.uim_train: sp.csr_matrix = uim_train.copy()
        self.uim_frequency_coo: sp.coo_matrix = uim_frequency.astype(
            dtype=np.int32,
        ).tocoo()
        self.dict_binned_frequency: dict[tuple[int, int], int] = {}

        self.dataset: Optional[SFCDataset] = None
        self.dataloader: Optional[torch.utils.data.DataLoader] = None

        self.BIAS_factors: np.ndarray = np.array([])
        self.USER_factors: np.ndarray = np.array([])
        self.ITEM_factors: np.ndarray = np.array([])
        self.FREQUENCY_factors: np.ndarray = np.array([])

        self.BIAS_factors_best: np.ndarray = np.array([])
        self.USER_factors_best: np.ndarray = np.array([])
        self.ITEM_factors_best: np.ndarray = np.array([])
        self.FREQUENCY_factors_best: np.ndarray = np.array([])

        self.model_state: dict[str, np.ndarray] = {}
        self.model_state_best: dict[str, np.ndarray] = {}

        self.criterion: Optional[torch.nn.BCEWithLogitsLoss] = None
        self.model: Optional[SFCModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.frequency_mode: Optional[FREQUENCY_MODE] = None
        self.frequency_num_bins: Optional[int] = None
        self.frequency_num_embeddings: Optional[int] = None

        self.epochs: Optional[int] = None
        self.batch_size: Optional[int] = None
        self.embedding_size: Optional[int] = None

        self.learning_rate: Optional[float] = None
        self.l2_reg: Optional[float] = None
        self.initial_step_size: Optional[float] = None
        self.scheduler_alpha: Optional[float] = None
        self.scheduler_beta: Optional[float] = None

        if use_gpu:
            assert torch.cuda.is_available(), "GPU is requested but not available"
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu:0")

    def _compute_item_score(
        self,
        user_id_array: np.ndarray,
        items_to_compute: Optional[np.ndarray] = None,
    ):
        """
        USER_factors is n_users x n_factors
        ITEM_factors is n_items x n_factors

        The prediction for cold users will always be -inf for ALL items

        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        assert (
            self.USER_factors.shape[1] == self.ITEM_factors.shape[1]
        ), "{}: User and Item factors have inconsistent shape".format(
            self.RECOMMENDER_NAME
        )

        assert self.USER_factors.shape[0] > np.max(
            user_id_array
        ), "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
            self.RECOMMENDER_NAME, self.USER_factors.shape[0], np.max(user_id_array)
        )

        assert self.frequency_mode is not None
        assert self.frequency_num_bins is not None

        arr_item_scores = (
            np.ones(
                (len(user_id_array), self.n_items),
                dtype=np.float32,
            )
            * np.NINF
        )

        arr_user_ids = np.asarray(user_id_array, dtype=np.int32)
        arr_item_ids = (
            np.asarray(items_to_compute, dtype=np.int32)
            if items_to_compute is not None
            else np.arange(start=0, stop=self.n_items, dtype=np.int32)
        )

        arr_frequency_factors = _compute_array_frequencies(
            arr_user_ids=arr_user_ids,
            arr_item_ids=arr_item_ids,
            arr_frequency_factors=self.FREQUENCY_factors,
            dict_frequency=self.dict_binned_frequency,
            num_items=self.n_items,
            frequency_mode=self.frequency_mode,
        )

        arr_item_scores = _compute_soft_frequency_capping_score(
            arr_item_scores=arr_item_scores,
            arr_user_ids=arr_user_ids,
            arr_item_ids=arr_item_ids,
            arr_global_bias=self.BIAS_factors,
            arr_user_factors=self.USER_factors,
            arr_item_factors=self.ITEM_factors,
            arr_frequency_factors=arr_frequency_factors,
        )

        return arr_item_scores

    def fit(
        self,
        *,
        epochs: int,
        frequency_mode: FREQUENCY_MODE,
        frequency_num_bins: int,
        batch_size: int,
        embedding_size: int,
        learning_rate: float,
        l2_reg: float,
        scheduler_alpha: float,
        scheduler_beta: float,
        **earlystopping_kwargs,
    ):
        self.epochs = epochs

        self.frequency_mode = frequency_mode
        self.frequency_num_bins = frequency_num_bins
        self.frequency_num_embeddings = _get_frequency_num_embeddings(
            frequency_mode=self.frequency_mode,
            num_users=self.n_users,
            num_items=self.n_items,
        )

        self.batch_size = int(batch_size)
        self.embedding_size = embedding_size

        self.learning_rate = learning_rate
        self.l2_reg = l2_reg

        self.scheduler_alpha = scheduler_alpha
        self.scheduler_beta = scheduler_beta

        self.dict_binned_frequency = _create_dict_binned_frequency(
            uim_frequency_coo_row=self.uim_frequency_coo.row,
            uim_frequency_coo_col=self.uim_frequency_coo.col,
            uim_frequency_coo_data=self.uim_frequency_coo.data,
            num_bins=self.frequency_num_bins,
        )
        self.dataset = SFCDataset(
            urm_train=self.urm_train,
            uim_train=self.uim_train,
            dict_binned_frequency=self.dict_binned_frequency,
            frequency_mode=self.frequency_mode,
        )
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        torch.cuda.empty_cache()

        self.model = SFCModel(
            num_users=self.n_users,
            num_items=self.n_items,
            frequency_mode=self.frequency_mode,
            frequency_num_bins=self.frequency_num_bins,
            frequency_num_embeddings=self.frequency_num_embeddings,
            embedding_size=self.embedding_size,
            device=self.device,
        )
        self.optimizer = torch.optim.Adagrad(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=l2_reg,
        )
        # Loss used in the paper (Binary Cross-Entropy).
        # This criterion applies a sigmoid to the output (y) as required by the paper.
        # When changing the loss, the layer should be applied to the output of the model.
        # TODO: Future works should include a weight to the classes (interaction and non-interacted impression). This is because the number of non-interacted impressions is much higher than the number of interactions.
        self.criterion = torch.nn.BCEWithLogitsLoss(
            reduction="none" if self.batch_size == 1 else "sum",
        )

        ############################################################
        ### This is a standard training with early stopping part ###
        ############################################################

        # Initializing for epoch 0
        self._prepare_model_for_validation()
        self._update_best_model()
        self._train_with_early_stopping(
            epochs,
            algorithm_name=self.RECOMMENDER_NAME,
            **earlystopping_kwargs,
        )
        self._print("Training complete")

        self.BIAS_factors = self.BIAS_factors_best.copy()
        self.USER_factors = self.USER_factors_best.copy()
        self.ITEM_factors = self.ITEM_factors_best.copy()
        self.FREQUENCY_factors = self.FREQUENCY_factors_best.copy()
        self.model_state = self.model_state_best.copy()

    def _prepare_model_for_validation(self):
        assert self.model is not None
        assert self.frequency_num_embeddings is not None

        with torch.no_grad():
            self.model.eval()

            # Expected shape: (1, )
            self.BIAS_factors = (
                self.model.embedding_bias.weight.detach().cpu().numpy().squeeze(axis=1)
            )
            # Expected shape: (num_users, num_factors)
            self.USER_factors = self.model.embedding_user.weight.detach().cpu().numpy()
            # Expected shape: (num_items, num_factors)
            self.ITEM_factors = self.model.embedding_item.weight.detach().cpu().numpy()
            # expected shape: (frequency_num_embeddings, frequency_num_bins)
            self.FREQUENCY_factors = np.vstack(
                [
                    emb.weight.detach().cpu().numpy().reshape(1, -1)
                    for emb in self.model.embedding_frequencies
                ]
            )

            self.model_state = clone_pytorch_model_to_numpy_dict(
                self.model,
            )
            self.model.train()

    def _update_best_model(self):
        self.BIAS_factors_best = self.BIAS_factors.copy()
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self.FREQUENCY_factors_best = self.FREQUENCY_factors.copy()
        self.model_state_best = copy.deepcopy(self.model_state)

    def _run_epoch(self, num_epoch: int):
        assert self.model is not None
        assert self.optimizer is not None
        assert self.criterion is not None
        assert self.dataloader is not None
        assert self.learning_rate is not None
        assert self.scheduler_alpha is not None
        assert self.scheduler_beta is not None

        self.model.train()

        epoch_loss = 0
        batch: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        for batch in self.dataloader:
            self.optimizer.zero_grad()
            (
                user_batch,
                item_batch,
                freq_batch,
                idx_emb_batch,
                label_batch,
                zero_batch,
            ) = batch

            user_batch = user_batch.to(self.device)
            item_batch = item_batch.to(self.device)
            freq_batch = freq_batch.to(self.device)
            idx_emb_batch = idx_emb_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            zero_batch = zero_batch.to(self.device)

            pred_label_batch = self.model(
                user_batch,
                item_batch,
                freq_batch,
                idx_emb_batch,
                zero_batch,
            )

            loss = self.criterion(
                pred_label_batch,
                label_batch,
            )

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        # INSTEAD OF RE-CREATING THE OPTIMIZER ON EVERY EPOCH TO UPDATE THE LEARNING RATE, we modify its learning rate when needed.
        # This should be equivalent to using a :torch:optim.lr_scheduler, however, there is no class that can accommodate the learning rate change used in the paper.
        # Hence, to successfully replicate the algorithm, we must do it by hand.
        # This code is based on how the :torch:optim.lr_scheduler class does it, see: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
        # Another reference is: https://stackoverflow.com/a/58302852
        new_learning_rate = learning_rate_scheduler(
            initial_lr=self.learning_rate,
            alpha=self.scheduler_alpha,
            beta=self.scheduler_beta,
            epoch_loss=epoch_loss,
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_learning_rate

        self._print("Loss {:.2E} - New LR {:.2E}".format(epoch_loss, new_learning_rate))

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {
            "BIAS_factors": self.BIAS_factors,
            "USER_factors": self.USER_factors,
            "ITEM_factors": self.ITEM_factors,
            "FREQUENCY_factors": self.FREQUENCY_factors,
            "model_state": self.model_state,
        }

        DataIO.s_save_data(
            folder_path=folder_path,
            file_name=file_name,
            data_dict_to_save=data_dict_to_save,
        )

        self._print("Saving complete")


__all__ = [
    "SoftFrequencyCappingRecommender",
    "SearchHyperParametersSFCRecommender",
]
