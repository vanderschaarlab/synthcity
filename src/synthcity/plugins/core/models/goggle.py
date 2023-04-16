# stdlib
import random
from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union

# third party
import dgl
import numpy as np
import torch
from dgl.nn import GraphConv, SAGEConv
from pydantic import validate_arguments
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results

# synthcity relative
from .factory import get_nonlin
from .layers import MultiActivationHead
from .RGCNConv import RGCNConv


class Goggle(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        input_dim: int,
        n_iter: int = 1000,
        encoder_dim: int = 64,
        encoder_l: int = 2,
        het_encoding: bool = True,
        decoder_dim: int = 64,
        decoder_l: int = 2,
        threshold: float = 0.1,
        decoder_arch: str = "gcn",
        graph_prior: Any = None,
        prior_mask: Any = None,
        batch_size: int = 32,
        decoder_nonlin: str = "relu",
        encoder_nonlin: str = "relu",
        decoder_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        loss: Any = None,
        learning_rate: float = 5e-3,
        iter_opt: bool = True,
        dataloader_sampler: Any = None,
        logging_epoch: int = 100,
        patience: int = 50,
        device: Union[str, torch.device] = DEVICE,
        random_state: int = 0,
    ) -> None:
        super().__init__()
        self.loss = loss
        self.iter_opt = iter_opt
        self.input_dim = input_dim
        self.n_iter = n_iter
        self.decoder_l = decoder_l
        self.batch_size = batch_size
        self.device = device
        self.dataloader_sampler = dataloader_sampler
        self.logging_epoch = logging_epoch
        self.patience = patience
        self.random_state = random_state

        enable_reproducible_results(self.random_state)

        if decoder_nonlin_out is None:
            decoder_nonlin_out = [("none", input_dim)]
        self.decoder_nonlin_out = decoder_nonlin_out

        self.learned_graph = LearnedGraph(
            input_dim, graph_prior, prior_mask, threshold, device=self.device
        )
        self.encoder = Encoder(input_dim, encoder_dim, encoder_l, encoder_nonlin)
        if decoder_arch == "het":
            n_edge_types = input_dim * input_dim
            self.graph_processor = GraphInputProcessorHet(
                input_dim,
                decoder_dim,
                n_edge_types,
                het_encoding,
                device=self.device,
            )
            self.decoder = GraphDecoderHet(
                decoder_dim,
                self.decoder_l,
                n_edge_types,
                n_units_out=input_dim,
                decoder_nonlin=decoder_nonlin,
                decoder_nonlin_out=self.decoder_nonlin_out,
                device=self.device,
            )
        else:
            self.graph_processor = GraphInputProcessorHomo(
                input_dim,
                decoder_dim,
                het_encoding,
                device=self.device,
            )
            self.decoder = GraphDecoderHomo(
                decoder_dim,
                self.decoder_l,
                decoder_arch,
                n_units_out=input_dim,
                decoder_nonlin=decoder_nonlin,
                decoder_nonlin_out=self.decoder_nonlin_out,
                device=self.device,
            )

    def fit(
        self,
        X: np.ndarray,
        optimiser_gl: Any,
        optimiser_ga: Any,
        optimiser: Any,
    ) -> None:
        clear_cache()
        self.optimiser_gl = optimiser_gl
        self.optimiser_ga = optimiser_ga
        self.optimiser = optimiser

        X = self._check_tensor(X).float()

        X, X_val = self._train_test_split(X)

        # Load Dataset
        train_loader: TorchDataLoader = self.get_dataloader(X)
        val_loader: TorchDataLoader = self.get_dataloader(X_val)

        # Training loop
        best_loss = np.inf
        best_model_state = self.state_dict()
        for epoch in tqdm(range(self.n_iter)):
            train_loss, num_samples = 0.0, 0
            data: Any = None
            for i, data in enumerate(train_loader):
                if self.iter_opt:
                    if i % 2 == 0:
                        self.train()
                        self.optimiser_ga.zero_grad()

                        data = data[0].to(self.device)

                        x_hat, adj, mu_z, logvar_z = self(data, epoch)
                        loss, _, _, _ = self.loss(x_hat, data, mu_z, logvar_z, adj)

                        loss.backward(retain_graph=True)
                        self.optimiser_ga.step()

                        train_loss += loss.item()
                        num_samples += data.shape[0]
                    else:
                        self.train()
                        self.optimiser_gl.zero_grad()

                        data = data[0].to(self.device)

                        x_hat, adj, mu_z, logvar_z = self(data, epoch)
                        loss, _, _, _ = self.loss(x_hat, data, mu_z, logvar_z, adj)

                        loss.backward(retain_graph=True)
                        self.optimiser_gl.step()

                        train_loss += loss.item()
                        num_samples += data.shape[0]

                else:
                    data = data[0].to(self.device)
                    self.train()

                    self.optimiser.zero_grad()

                    x_hat, adj, mu_z, logvar_z = self(data, epoch)
                    loss, _, _, _ = self.loss(x_hat, data, mu_z, logvar_z, adj)

                    loss.backward(retain_graph=True)
                    self.optimiser.step()

                    train_loss += loss.item()
                    num_samples += data.shape[0]

            train_loss /= num_samples

            val_loss = self.evaluate(val_loader, epoch)

            if val_loss[1] < best_loss:
                best_loss = val_loss[1]
                patience = 0
                best_model_state = deepcopy(self.state_dict())
            else:
                patience += 1

            if (epoch + 1) % self.logging_epoch == 0:
                log.debug(
                    f"[Epoch {(epoch+1):3}/{self.n_iter}, patience {patience:2}] train: {train_loss:.3f}, val: {val_loss[0]:.3f}"
                )

            if patience == self.patience:
                log.debug(f"Training terminated after {epoch} epochs")
                self.load_state_dict(best_model_state)
                break

    def forward(
        self,
        x: torch.Tensor,
        iter: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, (mu_z, logvar_z) = self.encoder(x)
        b_size, _ = z.shape
        adj = self.learned_graph(iter)
        graph_input = self.graph_processor(z, adj)  # .to(self.device)
        x_hat = self.decoder(graph_input, b_size)

        return x_hat, adj, mu_z, logvar_z

    def evaluate(
        self, data_loader: TorchDataLoader, epoch: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            eval_loss, rec_loss, kld_loss, graph_loss = 0.0, 0.0, 0.0, 0.0
            num_samples = 0
            for _, data in enumerate(data_loader):
                self.eval()
                data = data[0].to(self.device)
                x_hat, adj, mu_z, logvar_z = self(data, epoch)
                loss, loss_rec, loss_kld, loss_graph = self.loss(
                    x_hat, data, mu_z, logvar_z, adj
                )

                eval_loss += loss.item()
                rec_loss += loss_rec.item()
                kld_loss += loss_kld.item()
                graph_loss += loss_graph.item() * data.shape[0]
                num_samples += data.shape[0]

            eval_loss /= num_samples
            rec_loss /= num_samples
            kld_loss /= num_samples
            graph_loss /= num_samples

            return eval_loss, rec_loss, kld_loss, graph_loss

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: int,
    ) -> np.ndarray:
        with torch.no_grad():
            mu = torch.zeros(self.input_dim)
            sigma = torch.ones(self.input_dim)
            q = torch.distributions.Normal(mu, sigma)
            z = q.rsample(sample_shape=torch.Size([count])).squeeze().to(self.device)

            self.learned_graph.eval()
            self.graph_processor.eval()
            self.decoder.eval()

            adj = self.learned_graph(100)
            graph_input = self.graph_processor(z, adj)
            synth_x = self.decoder(graph_input, count)

        return synth_x.detach().cpu().numpy()

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)

    def _train_test_split(
        self,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total = np.arange(0, len(X))
        np.random.shuffle(total)
        split = int(len(total) * 0.8)
        train_idx, val_idx = total[:split], total[split:]

        X_train, X_val = X[train_idx], X[val_idx]
        return X_train, X_val

    def seed_worker(self) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_dataloader(
        self,
        X: torch.Tensor,
    ) -> DataLoader:
        dataset = TensorDataset(X)
        g = torch.Generator()
        g.manual_seed(self.random_state)
        return TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=self.dataloader_sampler,
            pin_memory=False,
            worker_init_fn=self.seed_worker,
        )


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_dim: int,
        encoder_l: int,
        encoder_nonlin: str,
    ) -> None:
        super().__init__()
        encoder = nn.ModuleList([nn.Linear(input_dim, encoder_dim), nn.ReLU()])
        for _ in range(encoder_l - 2):
            encoder_dim_ = int(encoder_dim / 2)
            encoder.append(nn.Linear(encoder_dim, encoder_dim_))
            encoder.append(nn.ReLU())
            encoder_dim = encoder_dim_
        encoder.append(get_nonlin(encoder_nonlin))
        self.encoder = nn.Sequential(*encoder)
        self.encode_mu = nn.Linear(encoder_dim, input_dim)
        self.encode_logvar = nn.Linear(encoder_dim, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        h = self.encoder(x)
        mu_z, logvar_z = self.encode_mu(h), self.encode_logvar(h)
        z = self.reparameterize(mu_z, logvar_z)
        return z, (mu_z, logvar_z)


class GraphDecoderHomo(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        decoder_dim: int,
        decoder_l: int,
        decoder_arch: str,
        n_units_out: int,
        decoder_nonlin: str,
        decoder_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        device: Union[str, torch.device] = DEVICE,
    ) -> None:
        super().__init__()
        decoder = nn.ModuleList([])
        nonlin_layers = nn.ModuleList([])

        if decoder_arch == "gcn":
            for i in range(decoder_l):
                if i == decoder_l - 1:
                    decoder.append(
                        GraphConv(decoder_dim, 1, norm="both", weight=True, bias=True)
                    )
                else:
                    decoder_dim_ = int(decoder_dim / 2)
                    decoder.append(
                        GraphConv(
                            decoder_dim,
                            decoder_dim_,
                            norm="both",
                            weight=True,
                            bias=True,
                            activation=get_nonlin(decoder_nonlin),
                            # activation=nn.Tanh(),
                        )
                    )
                    decoder_dim = decoder_dim_
        elif decoder_arch == "sage":
            for i in range(decoder_l):
                if i == decoder_l - 1:
                    decoder.append(
                        SAGEConv(decoder_dim, 1, aggregator_type="mean", bias=True)
                    )
                else:
                    decoder_dim_ = int(decoder_dim / 2)
                    decoder.append(
                        SAGEConv(
                            decoder_dim,
                            decoder_dim_,
                            aggregator_type="mean",
                            bias=True,
                            activation=get_nonlin(decoder_nonlin),
                            # activation=nn.Tanh(),
                        )
                    )
                    decoder_dim = decoder_dim_
        else:
            raise Exception("decoder can only be {het|gcn|sage}")

        if decoder_nonlin_out is not None:
            total_nonlin_len = 0
            activations = []
            for nonlin, nonlin_len in decoder_nonlin_out:
                total_nonlin_len += nonlin_len
                activations.append((get_nonlin(nonlin), nonlin_len))

            if total_nonlin_len != n_units_out:
                raise RuntimeError(
                    f"Shape mismatch for the output layer. Expected length {n_units_out}, but got {decoder_nonlin_out} with length {total_nonlin_len}"
                )
            nonlin_layers.append(MultiActivationHead(activations, device=device))
        self.decoder = nn.Sequential(*decoder)
        self.nonlin_layers = nn.Sequential(*nonlin_layers)

    def forward(self, graph_input: torch.Tensor, b_size: int) -> torch.Tensor:
        b_z, b_adj, b_edge_weight = graph_input

        for layer in self.decoder:
            b_z = layer(b_adj, feat=b_z, edge_weight=b_edge_weight)
        x_hat = b_z.reshape(b_size, -1)
        for layer in self.nonlin_layers:
            x_hat = layer(x_hat)

        return x_hat


class GraphDecoderHet(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        decoder_dim: int,
        decoder_l: int,
        n_edge_types: int,
        n_units_out: int,
        decoder_nonlin: str,
        decoder_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        device: Union[str, torch.device] = DEVICE,
    ) -> None:
        super().__init__()
        decoder = nn.ModuleList([])
        nonlin_layers = nn.ModuleList([])

        for i in range(decoder_l):
            if i == decoder_l - 1:
                decoder.append(
                    RGCNConv(
                        decoder_dim,
                        1,
                        num_relations=n_edge_types + 1,
                        # num_blocks=1,
                        root_weight=False,
                    )
                )
            else:
                decoder_dim_ = int(decoder_dim / 2)
                decoder.append(
                    RGCNConv(
                        decoder_dim,
                        decoder_dim_,
                        num_relations=n_edge_types + 1,
                        # num_blocks=1,
                        root_weight=False,
                    )
                )
                decoder.append(nn.ReLU())
                decoder_dim = decoder_dim_
        decoder.append(get_nonlin(decoder_nonlin))

        if decoder_nonlin_out is not None:
            total_nonlin_len = 0
            activations = []
            for nonlin, nonlin_len in decoder_nonlin_out:
                total_nonlin_len += nonlin_len
                activations.append((get_nonlin(nonlin), nonlin_len))

            if total_nonlin_len != n_units_out:
                raise RuntimeError(
                    f"Shape mismatch for the output layer. Expected length {n_units_out}, but got {decoder_nonlin_out} with length {total_nonlin_len}"
                )
            nonlin_layers.append(MultiActivationHead(activations, device=device))
        self.decoder = nn.Sequential(*decoder)
        self.nonlin_layers = nn.Sequential(*nonlin_layers)

    def forward(self, graph_input: torch.Tensor, b_size: int) -> torch.Tensor:
        b_z, b_edge_index, b_edge_weights, b_edge_types = graph_input

        h = b_z
        for layer in self.decoder:
            if isinstance(
                layer,
                MessagePassing,
            ):
                h = layer(h, b_edge_index, b_edge_types, b_edge_weights)
            else:
                h = layer(h)

        x_hat = h.reshape(b_size, -1)
        for layer in self.nonlin_layers:
            x_hat = layer(x_hat)

        return x_hat


class GraphInputProcessorHomo(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        input_dim: int,
        decoder_dim: int,
        het_encoding: bool,
        device: Union[str, torch.device] = DEVICE,
    ) -> None:
        super().__init__()
        self.device = device
        self.het_encoding = het_encoding

        if het_encoding:
            feat_dim = input_dim + 1
        else:
            feat_dim = 1

        self.embedding_functions = []
        for _ in range(input_dim):
            self.embedding_functions.append(
                nn.Sequential(nn.Linear(feat_dim, decoder_dim), nn.Tanh()).to(device)
            )

    def forward(
        self, z: torch.Tensor, adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepares embeddings for graph decoding
            Parameters:
                z (Tensor): feature embeddings
                adj (Tensor): adjacency matrix
                iter (int): training iteration
            Returns:
                b_z (Tensor): dense feature matrix, shape = (b_size*n_nodes, n_feats)
                b_adj (Tensor): batched adjacency matrix
                b_edge_weight (Sparse Tensor): sparse edge weights, shape = (n_edges)
        """
        b_z = z.unsqueeze(-1)
        b_size, n_nodes, _ = b_z.shape

        if self.het_encoding:
            one_hot_encoding = torch.eye(n_nodes).to(self.device)
            b_encoding = torch.stack([one_hot_encoding for _ in range(b_size)], dim=0)
            b_z = torch.cat([b_z, b_encoding], dim=-1)

        b_z = [f(b_z[:, i]) for i, f in enumerate(self.embedding_functions)]
        b_z = torch.stack(b_z, dim=1)
        b_z = torch.flatten(b_z, start_dim=0, end_dim=1)

        edge_index = adj.nonzero().t()
        row, col = edge_index
        edge_weight = adj[row, col]

        g = dgl.graph((edge_index[0], edge_index[1]))
        b_adj = dgl.batch([g] * b_size)
        b_edge_weight = edge_weight.repeat(b_size)

        return (b_z, b_adj, b_edge_weight)


class GraphInputProcessorHet(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        input_dim: int,
        decoder_dim: int,
        n_edge_types: int,
        het_encoding: bool,
        device: Union[str, torch.device] = DEVICE,
    ) -> None:
        super().__init__()
        self.n_edge_types = n_edge_types
        self.device = device
        self.het_encoding = het_encoding

        if het_encoding:
            feat_dim = input_dim + 1
        else:
            feat_dim = 1

        self.embedding_functions = []
        for _ in range(input_dim):
            self.embedding_functions.append(
                nn.Sequential(nn.Linear(feat_dim, decoder_dim), nn.Tanh()).to(device)
            )

    def forward(
        self, z: torch.Tensor, adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepares embeddings for graph decoding
            Parameters:
                z (Tensor): feature embeddings
                adj (Tensor): adjacency matrix
                het_encoding (bool): use of heterogeneous encoding
            Returns:
                b_z (Tensor): dense feature matrix, shape = (b_size*n_nodes, n_feats)
                b_adj (Tensor): batched adjacency matrix, shape = (b_size, n_nodes, n_nodes)
                b_edge_index (Sparse Tensor): sparse edge index, shape = (2, n_edges)
                b_edge_weights (Sparse Tensor): sparse edge weights, shape = (n_edges)
                b_edge_types (Sparse Tensor): sparse edge type, shape = (n_edges)
        """
        b_size, n_nodes = z.shape

        b_z = z.unsqueeze(-1)

        if self.het_encoding:
            one_hot_encoding = torch.eye(n_nodes).to(self.device)
            b_encoding = torch.stack([one_hot_encoding for _ in range(b_size)], dim=0)
            b_z = torch.cat([b_z, b_encoding], dim=-1)

        b_z = [f(b_z[:, i]) for i, f in enumerate(self.embedding_functions)]
        b_z = torch.stack(b_z, dim=1)
        b_size, n_nodes, n_feats = b_z.shape

        n_edge_types = self.n_edge_types
        edge_types = (
            torch.arange(1, n_edge_types + 1, 1)
            .reshape(n_nodes, n_nodes)
            .to(self.device)
        )

        b_adj = torch.stack([adj for _ in range(b_size)], dim=0)

        b_edge_index, b_edge_weights = dense_to_sparse(b_adj)
        r, c = b_edge_index.to(self.device)

        b_edge_types = edge_types[r % n_nodes, c % n_nodes]
        b_z = b_z.reshape(b_size * n_nodes, n_feats)

        return (b_z, b_edge_index, b_edge_weights, b_edge_types)


class LearnedGraph(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        input_dim: int,
        graph_prior: Any,
        prior_mask: Any,
        threshold: float,
        device: Union[str, torch.device] = DEVICE,
    ) -> None:
        super().__init__()

        self.graph = nn.Parameter(
            torch.zeros(input_dim, input_dim, requires_grad=True, device=device)
        )

        if all(i is not None for i in [graph_prior, prior_mask]):
            self.graph_prior = (
                graph_prior.detach().clone().requires_grad_(False).to(device)
            )
            self.prior_mask = (
                prior_mask.detach().clone().requires_grad_(False).to(device)
            )
            self.use_prior = True
        else:
            self.use_prior = False

        self.act = nn.Sigmoid()
        self.threshold = nn.Threshold(threshold, 0)
        self.device = device

    def forward(self, iter: int) -> torch.Tensor:
        if self.use_prior:
            graph = (
                self.prior_mask * self.graph_prior + (1 - self.prior_mask) * self.graph
            )
        else:
            graph = self.graph

        graph = self.act(graph)
        graph = graph.clone()
        graph = graph * (
            torch.ones(graph.shape[0]).to(self.device)
            - torch.eye(graph.shape[0]).to(self.device)
        ) + torch.eye(graph.shape[0]).to(self.device)

        if iter > 50:
            graph = self.threshold(graph)
        else:
            graph = graph
        return graph


class GoggleLoss(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        graph_prior: Any = None,
        device: Union[str, torch.device] = DEVICE,
    ) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.device = device
        self.alpha = alpha
        self.beta = beta
        if graph_prior is not None:
            self.use_prior = True
            self.graph_prior = (
                torch.Tensor(graph_prior).requires_grad_(False).to(device)
            )
        else:
            self.use_prior = False

    def forward(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        graph: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_mse = self.mse_loss(x_recon, x)
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        if self.use_prior:
            loss_graph = (graph - self.graph_prior).norm(p=1) / torch.numel(graph)
        else:
            loss_graph = graph.norm(p=1) / torch.numel(graph)

        loss = loss_mse + self.alpha * loss_kld + self.beta * loss_graph

        return loss, loss_mse, loss_kld, loss_graph
