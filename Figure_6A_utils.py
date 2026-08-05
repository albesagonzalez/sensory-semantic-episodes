from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import SSCNetwork
from src.utils.episode_generation_protocol import (
    LatentSpace,
    make_input,
)
from src.utils.general import get_cos_sim_torch, get_sample_from_num_swaps, seed_everything


class SparseHopfieldNetwork(nn.Module):
    def __init__(self, net_params, rec_params=None):
        super().__init__()

        for key, value in net_params.items():
            setattr(self, key, value)

        if not hasattr(self, "sparse_hopfield_threshold_scale"):
            self.sparse_hopfield_threshold_scale = 0.5

        self.mtl_sensory_size = int(torch.sum(self.mtl_sensory_size_subregions).item())
        self.mtl_sensory = torch.zeros(self.mtl_sensory_size)
        self.mtl_sensory_mtl_sensory = torch.zeros(
            (self.mtl_sensory_size, self.mtl_sensory_size)
        )
        self.num_patterns_stored = 0

    def _topk_project_mtl_sensory(self, x):
        num_active = int(self.mtl_sensory_size * float(self.mtl_sensory_sparsity[0]))
        num_active = max(num_active, 1)
        top_indices = torch.topk(x, num_active).indices
        h = torch.zeros_like(x)
        h[top_indices] = 1.0
        return h

    def forward(self, input, debug=False):
        del debug

        patterns = input.float()
        activity = float(self.mtl_sensory_sparsity[0])
        norm = patterns.shape[1] * activity * (1 - activity)
        self.num_patterns_stored = int(patterns.shape[0])

        weights = torch.zeros_like(self.mtl_sensory_mtl_sensory)

        for pattern in patterns:
            pattern_centered = pattern - activity
            weights += torch.outer(pattern_centered, pattern_centered)
        self.mtl_sensory_mtl_sensory = weights / norm
        
        self.mtl_sensory_mtl_sensory.fill_diagonal_(0)

    def pattern_complete(
        self,
        region,
        h_0=None,
        h_conditioned=None,
        subregion_index=None,
        sleep=False,
        num_iterations=None,
        sparsity=None,
    ):
        del region, h_conditioned, subregion_index, sleep, sparsity

        num_iterations = (
            self.mtl_sensory_pattern_complete_iterations
            if num_iterations is None
            else num_iterations
        )
        activity = float(self.mtl_sensory_sparsity[0])
        theta = (
            float(self.sparse_hopfield_threshold_scale)
            * float(self.num_patterns_stored)
            / float(self.mtl_sensory_size)
        )
        h = h_0.clone() if h_0 is not None else self.mtl_sensory.clone()

        for _ in range(int(num_iterations)):
            field = F.linear(h - activity, self.mtl_sensory_mtl_sensory)
            h = (field > theta).float()

        return self._topk_project_mtl_sensory(h + torch.rand_like(h) * 1e-6)  # Add small noise to break ties
        #return h

def sample_random_mtl_sensory_patterns(num_patterns, pattern_size, pattern_sparsity):
    num_active = int(pattern_size * float(pattern_sparsity))
    patterns = torch.zeros((int(num_patterns), int(pattern_size)))
    for pattern_idx in range(int(num_patterns)):
        active_idx = torch.randperm(pattern_size)[:num_active]
        patterns[pattern_idx, active_idx] = 1
    return patterns


def count_unique_presented_concepts(input_latents, dims=None):
    """Count distinct latent-factor values encountered in a sequence."""
    input_latents = torch.as_tensor(input_latents).reshape(-1, input_latents.shape[-1])
    unique_count = sum(
        int(torch.unique(input_latents[:, latent_idx]).numel())
        for latent_idx in range(input_latents.shape[1])
    )
    if dims is not None:
        unique_count = min(unique_count, int(sum(dims)))
    return unique_count


def sample_random_mtl_patterns(num_patterns, size_subregions, sparsity_subregions):
    """Sample fixed-sparsity random patterns across all MTL subregions."""
    size_subregions = [int(size) for size in size_subregions]
    sparsity_subregions = [float(sparsity) for sparsity in sparsity_subregions]
    patterns = torch.zeros((int(num_patterns), int(sum(size_subregions))))

    start = 0
    for subregion_size, subregion_sparsity in zip(
        size_subregions, sparsity_subregions
    ):
        num_active = int(subregion_size * subregion_sparsity)
        for pattern_idx in range(int(num_patterns)):
            active_idx = torch.randperm(subregion_size)[:num_active] + start
            patterns[pattern_idx, active_idx] = 1
        start += subregion_size

    return patterns


def get_capacity_recall(
    net_params,
    num_patterns,
    seed,
    network_architecture="ssc",
    recall_region="mtl_sensory",
    network_mode="semantics_present",
    input_generation="random",
    cue_num_swaps=4,
):
    """Store one set of patterns and quantify recall in the Figure 6A assay."""
    seed_everything(seed)

    record_structured_activity = (
        isinstance(input_generation, dict) and network_architecture == "ssc"
    )
    rec_params = {
        "regions": ["mtl_sensory", "mtl_semantic"]
        if record_structured_activity
        else [],
        "rate_activity": 1 if record_structured_activity else np.inf,
        "connections": [],
        "rate_connectivity": np.inf,
    }

    if network_architecture == "ssc":
        if isinstance(net_params, str):
            network = torch.load(net_params, weights_only=False)
            network.init_recordings(rec_params)
        elif isinstance(net_params, SSCNetwork):
            network = deepcopy(net_params)
            network.init_recordings(rec_params)
        else:
            network = SSCNetwork(deepcopy(net_params), rec_params)

        effective_recall_region = recall_region
        if network_mode == "semantics_absent":
            effective_recall_region = "mtl_sensory"
        elif network_mode == "semantics_random":
            network.lesioned = {"mtl_semantic"}
        elif network_mode != "semantics_present":
            raise ValueError(f"Unknown network_mode: {network_mode!r}")

        network.frozen = False
        if hasattr(network, "activity_recordings_rate"):
            network.activity_recordings_rate = rec_params["rate_activity"]
        if hasattr(network, "connectivity_recordings_rate"):
            network.connectivity_recordings_rate = rec_params["rate_connectivity"]
    else:
        network = SparseHopfieldNetwork(deepcopy(net_params), rec_params)
        effective_recall_region = recall_region

    with torch.no_grad():
        semantic_patterns = None
        num_unique_concepts_shown = np.nan

        if input_generation == "random":
            if (
                network_architecture == "ssc"
                and effective_recall_region == "mtl_sensory"
            ):
                patterns = sample_random_mtl_sensory_patterns(
                    num_patterns=num_patterns,
                    pattern_size=network.mtl_sensory_size,
                    pattern_sparsity=network.mtl_sensory_sparsity[0],
                )
                network.mtl_sensory_mtl_sensory.zero_()
                network(patterns)
                network.mtl_sensory_mtl_sensory.fill_diagonal_(0)
            elif network_architecture == "ssc" and effective_recall_region == "mtl":
                patterns = sample_random_mtl_patterns(
                    num_patterns=num_patterns,
                    size_subregions=network.mtl_size_subregions,
                    sparsity_subregions=network.mtl_sparsity,
                )
                network.mtl_mtl.zero_()
                network.lesioned = {"mtl_semantic"}
                for pattern in patterns:
                    network.mtl = pattern.clone()
                    network.hebbian("mtl", "mtl")
                    network.homeostasis("mtl", "mtl")
                network.mtl_mtl.fill_diagonal_(0)
            else:
                patterns = sample_random_mtl_sensory_patterns(
                    num_patterns=num_patterns,
                    pattern_size=network.mtl_sensory_size,
                    pattern_sparsity=network.mtl_sensory_sparsity[0],
                )
                network.mtl_sensory_mtl_sensory.zero_()
                network(patterns)
                network.mtl_sensory_mtl_sensory.fill_diagonal_(0)
        elif isinstance(input_generation, dict):
            if network_architecture != "ssc":
                raise ValueError(
                    "Structured input_generation is only supported for "
                    "network_architecture='ssc'."
                )

            input_params = deepcopy(input_generation)
            latent_specs = input_params.pop("latent_specs", None)
            if latent_specs is not None and "latent_space" not in input_params:
                input_params["latent_space"] = LatentSpace(**deepcopy(latent_specs))

            input_params.setdefault("num_days", 1)
            if "day_length" not in input_params:
                input_params["day_length"] = int(
                    input_params.get("mean_duration", 1) * num_patterns
                )

            input_tensor, _, input_latents = make_input(**input_params)
            network(input_tensor[0], debug=False)
            num_unique_concepts_shown = count_unique_presented_concepts(
                input_latents,
                dims=latent_specs["dims"] if latent_specs is not None else None,
            )
            sensory_patterns = torch.stack(
                network.activity_recordings["mtl_sensory"], dim=0
            )[network.awake_indices].clone()
            semantic_patterns = torch.stack(
                network.activity_recordings["mtl_semantic"], dim=0
            )[network.awake_indices].clone()
            if network_mode == "semantics_absent":
                semantic_patterns = None

            if effective_recall_region == "mtl_sensory":
                patterns = sensory_patterns
                network.mtl_sensory_mtl_sensory.fill_diagonal_(0)
            else:
                patterns = torch.cat((sensory_patterns, semantic_patterns), dim=1)
                network.mtl_mtl.fill_diagonal_(0)
        else:
            raise ValueError(
                "input_generation must be 'random' or a dictionary of "
                "input-generation parameters."
            )

        mtl_sensory_recalls = []
        mtl_semantic_recalls = []
        mtl_recalls = []
        for pattern_idx, pattern in enumerate(patterns):
            cue = get_sample_from_num_swaps(
                pattern.clone(), num_swaps=int(cue_num_swaps)
            )

            if network_architecture == "sparse_hopfield":
                recalled_sensory = network.pattern_complete(
                    "mtl_sensory",
                    h_0=cue,
                    num_iterations=network.mtl_sensory_pattern_complete_iterations,
                )
                mtl_sensory_recalls.append(
                    get_cos_sim_torch(recalled_sensory, pattern).item()
                )
            elif effective_recall_region == "mtl_sensory":
                recalled_sensory = network.pattern_complete(
                    "mtl_sensory",
                    h_0=cue,
                    num_iterations=network.mtl_sensory_pattern_complete_iterations,
                )
                mtl_sensory_recalls.append(
                    get_cos_sim_torch(recalled_sensory, pattern).item()
                )
            else:
                if input_generation == "random":
                    mtl_0 = cue
                else:
                    mtl_0 = torch.zeros(network.mtl_size)
                    mtl_0[: network.mtl_sensory_size] = cue[
                        : network.mtl_sensory_size
                    ]

                recalled = network.pattern_complete(
                    "mtl",
                    h_0=mtl_0,
                    num_iterations=network.mtl_pattern_complete_iterations,
                )
                recalled_sensory = recalled[: network.mtl_sensory_size]
                mtl_sensory_recalls.append(
                    get_cos_sim_torch(
                        recalled_sensory, pattern[: network.mtl_sensory_size]
                    ).item()
                )
                mtl_recalls.append(get_cos_sim_torch(recalled, pattern).item())

                if semantic_patterns is not None:
                    recalled_semantic = recalled[network.mtl_sensory_size :]
                    mtl_semantic_recalls.append(
                        get_cos_sim_torch(
                            recalled_semantic, semantic_patterns[pattern_idx]
                        ).item()
                    )

    return {
        "recalled_mtl_sensory_cosine_mean": float(
            torch.tensor(mtl_sensory_recalls).nanmean().item()
        ),
        "recalled_mtl_semantic_cosine_mean": (
            float(torch.tensor(mtl_semantic_recalls).nanmean().item())
            if mtl_semantic_recalls
            else np.nan
        ),
        "recalled_mtl_cosine_mean": (
            float(torch.tensor(mtl_recalls).nanmean().item())
            if mtl_recalls
            else np.nan
        ),
        "num_unique_concepts_shown": num_unique_concepts_shown,
    }
