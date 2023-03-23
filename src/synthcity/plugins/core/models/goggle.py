# Standard
# stdlib
from typing import Any, Tuple

# third party
import dgl  # remove dependency

# 3rd party
import numpy as np
import torch
from dgl.nn import GraphConv, SAGEConv  # remove dependency
from pydantic import validate_arguments
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.utils import dense_to_sparse  # remove dependency
from tqdm import tqdm

# synthcity absolute
import synthcity.logger as log
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.models.RGCNConv import RGCNConv
from synthcity.utils.constants import DEVICE
from synthcity.utils.reproducibility import clear_cache


class Goggle(nn.Module):
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
        loss: Any = None,
        learning_rate: float = 5e-3,
        iter_opt: bool = True,
        dataloader_sampler: Any = None,
        logging_epoch: int = 100,
        patience: int = 50,
        device: str = DEVICE,
        random_state: int = 0,
    ) -> None:
        super().__init__()
        self.loss = loss
        self.iter_opt = iter_opt
        self.n_iter = n_iter
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.device = device
        self.learning_rate = learning_rate
        self.dataloader_sampler = dataloader_sampler
        self.logging_epoch = logging_epoch
        self.patience = patience
        self.random_state = random_state
        torch.manual_seed(self.random_state)

        self.learned_graph = LearnedGraph(
            input_dim, graph_prior, prior_mask, threshold, device
        )
        self.encoder = Encoder(input_dim, encoder_dim, encoder_l, device)
        if decoder_arch == "het":
            n_edge_types = input_dim * input_dim
            self.graph_processor = GraphInputProcessorHet(
                input_dim, decoder_dim, n_edge_types, het_encoding, device
            )
            self.decoder = GraphDecoderHet(decoder_dim, decoder_l, n_edge_types, device)
        else:
            self.graph_processor = GraphInputProcessorHomo(
                input_dim, decoder_dim, het_encoding, device
            )
            self.decoder = GraphDecoderHomo(
                decoder_dim, decoder_l, decoder_arch, device
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
        for epoch in tqdm(range(self.n_iter)):
            train_loss, num_samples = 0.0, 0
            data: Any = None  # work-around for mypy - Need type annotation for "data"
            for i, data in enumerate(train_loader):
                # if epoch == 0:
                #     log.debug(f"data[0].shape: {data[0].shape}")
                #     log.debug(f"data: {data}")
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
            else:
                patience += 1

            if (epoch + 1) % self.logging_epoch == 0:
                log.debug(
                    f"[Epoch {(epoch+1):3}/{self.n_iter}, patience {patience:2}] train: {train_loss:.3f}, val: {val_loss[0]:.3f}"
                )

            if patience == self.patience:
                log.debug(f"Training terminated after {epoch} epochs")
                break

    def forward(
        self, x: torch.Tensor, iter: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, (mu_z, logvar_z) = self.encoder(x)
        b_size, _ = z.shape
        adj = self.learned_graph(iter)
        graph_input = self.graph_processor(z, adj)
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

        return synth_x

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
        train_idx, test_idx = total[:split], total[split:]

        X_train, X_val = X[train_idx], X[test_idx]
        return X_train, X_val

    def get_dataloader(
        self,
        X: torch.Tensor,
    ) -> DataLoader:
        dataset = TensorDataset(X)
        return TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=self.dataloader_sampler,
            pin_memory=False,
        )


class Encoder(nn.Module):
    def __init__(
        self, input_dim: int, encoder_dim: int, encoder_l: int, device: str
    ) -> None:
        super().__init__()
        encoder = nn.ModuleList([nn.Linear(input_dim, encoder_dim), nn.ReLU()])
        for _ in range(encoder_l - 2):
            encoder_dim_ = int(encoder_dim / 2)
            encoder.append(nn.Linear(encoder_dim, encoder_dim_))
            encoder.append(nn.ReLU())
            encoder_dim = encoder_dim_
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
    def __init__(
        self, decoder_dim: int, decoder_l: int, decoder_arch: str, device: str
    ) -> None:
        super().__init__()
        decoder = nn.ModuleList([])

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
                            activation=nn.Tanh(),
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
                            activation=nn.Tanh(),
                        )
                    )
                    decoder_dim = decoder_dim_
        else:
            raise Exception("decoder can only be {het|gcn|sage}")

        self.decoder = nn.Sequential(*decoder)

    def forward(self, graph_input: torch.Tensor, b_size: int) -> torch.Tensor:
        b_z, b_adj, b_edge_weight = graph_input

        for layer in self.decoder:
            b_z = layer(b_adj, feat=b_z, edge_weight=b_edge_weight)

        x_hat = b_z.reshape(b_size, -1)

        return x_hat


class GraphDecoderHet(nn.Module):
    def __init__(
        self, decoder_dim: int, decoder_l: int, n_edge_types: int, device: str
    ) -> None:
        super().__init__()
        decoder = nn.ModuleList([])

        for i in range(decoder_l):
            if i == decoder_l - 1:
                decoder.append(
                    RGCNConv(
                        decoder_dim,
                        1,
                        num_relations=n_edge_types + 1,
                        num_blocks=1,
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
                        num_blocks=1,
                        root_weight=False,
                    )
                )
                decoder.append(nn.ReLU())
                decoder_dim = decoder_dim_

        self.decoder = nn.Sequential(*decoder)

    def forward(self, graph_input: torch.Tensor, b_size: int) -> torch.Tensor:
        b_z, b_edge_index, b_edge_weights, b_edge_types = graph_input

        h = b_z
        for layer in self.decoder:
            if not isinstance(layer, nn.ReLU):
                h = layer(h, b_edge_index, b_edge_types, b_edge_weights)
            else:
                h = layer(h)

        x_hat = h.reshape(b_size, -1)

        return x_hat


class GraphInputProcessorHomo(nn.Module):
    def __init__(
        self, input_dim: int, decoder_dim: int, het_encoding: bool, device: str
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
    def __init__(
        self,
        input_dim: int,
        decoder_dim: int,
        n_edge_types: int,
        het_encoding: bool,
        device: str,
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
        edge_types = torch.arange(1, n_edge_types + 1, 1).reshape(n_nodes, n_nodes)

        b_adj = torch.stack([adj for _ in range(b_size)], dim=0)

        b_edge_index, b_edge_weights = dense_to_sparse(b_adj)
        r, c = b_edge_index
        b_edge_types = edge_types[r % n_nodes, c % n_nodes]
        b_z = b_z.reshape(b_size * n_nodes, n_feats)

        return (b_z, b_edge_index, b_edge_weights, b_edge_types)


class LearnedGraph(nn.Module):
    def __init__(
        self,
        input_dim: int,
        graph_prior: Any,
        prior_mask: Any,
        threshold: float,
        device: str,
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
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        graph_prior: Any = None,
        device: str = "cpu",
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
