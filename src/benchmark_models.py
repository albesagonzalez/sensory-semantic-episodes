import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset


CLS_network_parameters = {}

######################
# memory parameters
######################

CLS_network_parameters["ctx_dim"] = 100
CLS_network_parameters["hopfield_beta"] = 8.0
CLS_network_parameters["hopfield_replay_steps"] = 3
CLS_network_parameters["hopfield_normalize"] = False

######################
# replay parameters
######################

CLS_network_parameters["replay_samples"] = 4096
CLS_network_parameters["replay_noise_type"] = "bernoulli"
CLS_network_parameters["replay_noise_scale"] = 1.0

######################
# decoder/training parameters
######################

CLS_network_parameters["mask_fraction"] = 0.3
CLS_network_parameters["decoder_hidden_dim"] = 256
CLS_network_parameters["decoder_epochs"] = 100
CLS_network_parameters["decoder_batch_size"] = 128
CLS_network_parameters["decoder_learning_rate"] = 1e-3
CLS_network_parameters["decoder_weight_decay"] = 0.0

######################
# runtime parameters
######################

CLS_network_parameters["seed"] = 42
CLS_network_parameters["device"] = "cpu"


class ModernHopfieldMemory(nn.Module):
    def __init__(self, ctx_dim: int, beta: float = 8.0, normalize: bool = False):
        super().__init__()
        self.ctx_dim = int(ctx_dim)
        self.beta = float(beta)
        self.normalize = bool(normalize)
        self.register_buffer("memory", torch.empty(0, self.ctx_dim))

    def store(self, ctx_patterns: torch.Tensor) -> None:
        if ctx_patterns.ndim != 2:
            raise ValueError("ctx_patterns must have shape (n_patterns, ctx_dim).")
        if ctx_patterns.shape[1] != self.ctx_dim:
            raise ValueError(
                f"ctx_patterns.shape[1]={ctx_patterns.shape[1]} does not match ctx_dim={self.ctx_dim}."
            )
        memory = ctx_patterns.detach().float()
        if self.normalize:
            memory = F.normalize(memory, dim=-1)
        self.memory = memory

    def retrieve(self, query: torch.Tensor, steps: int = 1) -> torch.Tensor:
        if self.memory.numel() == 0:
            raise RuntimeError("Memory is empty. Call store(...) before retrieve(...).")
        x = query.float()
        for _ in range(int(steps)):
            q = F.normalize(x, dim=-1) if self.normalize else x
            attn_logits = self.beta * (q @ self.memory.T)
            attn = F.softmax(attn_logits, dim=-1)
            x = attn @ self.memory
            x = torch.sigmoid(x)
        return x


class SigmoidDecoder(nn.Module):
    def __init__(self, ctx_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(ctx_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, ctx_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class CLSNetwork(nn.Module):
    def __init__(self, cls_network_parameters: Optional[Dict] = None):
        super().__init__()
        self.cls_network_parameters = deepcopy(CLS_network_parameters)
        if cls_network_parameters is not None:
            self.cls_network_parameters.update(cls_network_parameters)

        self.ctx_dim = int(self.cls_network_parameters["ctx_dim"])
        self.mask_fraction = float(self.cls_network_parameters["mask_fraction"])
        self.hopfield_replay_steps = int(self.cls_network_parameters["hopfield_replay_steps"])
        self.replay_noise_type = str(self.cls_network_parameters["replay_noise_type"])
        self.replay_noise_scale = float(self.cls_network_parameters["replay_noise_scale"])

        self.hopfield = ModernHopfieldMemory(
            ctx_dim=self.ctx_dim,
            beta=float(self.cls_network_parameters["hopfield_beta"]),
            normalize=bool(self.cls_network_parameters["hopfield_normalize"]),
        )
        self.decoder = SigmoidDecoder(
            ctx_dim=self.ctx_dim,
            hidden_dim=int(self.cls_network_parameters["decoder_hidden_dim"]),
        )

    def fit_memory(self, ctx_patterns: torch.Tensor) -> None:
        self.hopfield.store(ctx_patterns)

    def sample_replay(
        self,
        num_samples: Optional[int] = None,
        replay_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        samples = int(
            self.cls_network_parameters["replay_samples"] if num_samples is None else num_samples
        )
        steps = int(self.hopfield_replay_steps if replay_steps is None else replay_steps)
        device = next(self.parameters()).device if device is None else device

        noise = self._sample_noise(samples, device)
        replay = self.hopfield.retrieve(noise, steps=steps)
        return replay

    def mask_ctx(
        self, ctx: torch.Tensor, mask_fraction: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frac = self.mask_fraction if mask_fraction is None else float(mask_fraction)
        frac = min(max(frac, 0.0), 1.0)
        keep_mask = (torch.rand_like(ctx) > frac).float()
        masked_ctx = ctx * keep_mask
        return masked_ctx, keep_mask

    def forward(self, masked_ctx: torch.Tensor) -> torch.Tensor:
        return self.decoder(masked_ctx)

    def _sample_noise(self, num_samples: int, device: torch.device) -> torch.Tensor:
        if self.replay_noise_type.lower() == "bernoulli":
            p = 0.5 * self.replay_noise_scale
            p = min(max(p, 0.0), 1.0)
            return torch.bernoulli(torch.full((num_samples, self.ctx_dim), p, device=device))
        if self.replay_noise_type.lower() == "gaussian":
            return torch.randn(num_samples, self.ctx_dim, device=device) * self.replay_noise_scale
        raise ValueError(
            "Unsupported replay_noise_type. Use 'bernoulli' or 'gaussian'."
        )


def train_cls_network(
    model: CLSNetwork,
    ctx_patterns: torch.Tensor,
    cls_network_parameters: Optional[Dict] = None,
    device: Optional[str] = None,
) -> Dict[str, list]:
    params = deepcopy(model.cls_network_parameters)
    if cls_network_parameters is not None:
        params.update(cls_network_parameters)

    if ctx_patterns.ndim != 2:
        raise ValueError("ctx_patterns must have shape (n_patterns, ctx_dim).")
    if int(params["ctx_dim"]) != int(ctx_patterns.shape[1]):
        raise ValueError(
            f"ctx_dim={params['ctx_dim']} but ctx_patterns.shape[1]={ctx_patterns.shape[1]}."
        )
    if model.decoder.fc1.in_features != int(params["ctx_dim"]):
        raise ValueError(
            "Model decoder input size does not match ctx_dim in parameters. "
            "Reinitialize CLSNetwork with matching ctx_dim."
        )
    if model.decoder.fc1.out_features != int(params["decoder_hidden_dim"]):
        raise ValueError(
            "Model decoder hidden size does not match decoder_hidden_dim in parameters. "
            "Reinitialize CLSNetwork with matching decoder_hidden_dim."
        )

    seed = int(params["seed"])
    torch.manual_seed(seed)

    resolved_device = torch.device(params["device"] if device is None else device)
    model.to(resolved_device)
    model.cls_network_parameters.update(params)
    model.mask_fraction = float(params["mask_fraction"])
    model.hopfield_replay_steps = int(params["hopfield_replay_steps"])
    model.replay_noise_type = str(params["replay_noise_type"])
    model.replay_noise_scale = float(params["replay_noise_scale"])

    ctx_patterns = ctx_patterns.float().to(resolved_device)
    model.fit_memory(ctx_patterns)

    replay_targets = model.sample_replay(
        num_samples=int(params["replay_samples"]),
        replay_steps=int(params["hopfield_replay_steps"]),
        device=resolved_device,
    ).detach()

    dataloader = DataLoader(
        TensorDataset(replay_targets),
        batch_size=int(params["decoder_batch_size"]),
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        model.decoder.parameters(),
        lr=float(params["decoder_learning_rate"]),
        weight_decay=float(params["decoder_weight_decay"]),
    )

    history = {"decoder_bce": []}
    epochs = int(params["decoder_epochs"])
    mask_fraction = float(params["mask_fraction"])

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for (target_ctx,) in dataloader:
            masked_ctx, _ = model.mask_ctx(target_ctx, mask_fraction=mask_fraction)
            pred_ctx = model(masked_ctx)
            loss = F.binary_cross_entropy(pred_ctx, target_ctx)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        history["decoder_bce"].append(epoch_loss / max(num_batches, 1))

    return history
