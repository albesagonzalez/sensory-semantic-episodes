import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import SSCNetwork
from src.utils.general import (
    LatentSpace,
    get_cos_sim_torch,
    get_sample_from_num_swaps,
    make_input,
)


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

        return h


def sample_random_mtl_sensory_patterns(num_patterns, pattern_size, pattern_sparsity):
    num_active = int(pattern_size * float(pattern_sparsity))
    patterns = torch.zeros((int(num_patterns), int(pattern_size)))
    for pattern_idx in range(int(num_patterns)):
        active_idx = torch.randperm(pattern_size)[:num_active]
        patterns[pattern_idx, active_idx] = 1
    return patterns


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _apply_network_mode(network, network_mode):
    if network_mode == "semantics_present":
        return network
    if network_mode == "semantics_absent":
        network.sensory_replay_only = True
        return network
    if network_mode == "semantics_random":
        network.lesioned = {"mtl_semantic"}
        return network
    raise ValueError(f"Unknown network_mode: {network_mode!r}")


def _instantiate_capacity_network(
    network_source,
    network_type,
    network_mode,
    rec_params,
):
    if network_type == "ssc":
        if isinstance(network_source, str):
            network = torch.load(network_source, weights_only=False)
            network.init_recordings(rec_params)
        elif isinstance(network_source, SSCNetwork):
            network = deepcopy(network_source)
            network.init_recordings(rec_params)
        else:
            network = SSCNetwork(deepcopy(network_source), rec_params)

        network = _apply_network_mode(network, network_mode)
        network.frozen = False
        if hasattr(network, "activity_recordings_rate"):
            network.activity_recordings_rate = rec_params.get("rate_activity", np.inf)
        if hasattr(network, "connectivity_recordings_rate"):
            network.connectivity_recordings_rate = rec_params.get("rate_connectivity", np.inf)
        return network

    if network_type == "sparse_hopfield":
        return SparseHopfieldNetwork(deepcopy(network_source), rec_params)

    raise ValueError(f"Unknown network_type: {network_type!r}")


def _build_structured_input_params(input_generation, num_patterns):
    input_params = deepcopy(input_generation)
    latent_specs = input_params.pop("latent_specs", None)

    if latent_specs is not None and "latent_space" not in input_params:
        input_params["latent_space"] = LatentSpace(**deepcopy(latent_specs))

    input_params["num_days"] = 1
    if "day_length" not in input_params:
        mean_duration = int(input_params.get("mean_duration", 1))
        input_params["day_length"] = int(mean_duration * num_patterns)

    return input_params


def get_capacity_recall(
    network_source,
    num_patterns,
    seed,
    network_type="ssc",
    network_mode="semantics_present",
    input_generation="random",
):
    _seed_everything(seed)

    record_mtl_sensory = isinstance(input_generation, dict) and network_type == "ssc"
    rec_params = {
        "regions": ["mtl_sensory"] if record_mtl_sensory else [],
        "rate_activity": 1 if record_mtl_sensory else np.inf,
        "connections": [],
        "rate_connectivity": np.inf,
    }

    network = _instantiate_capacity_network(
        network_source=network_source,
        network_type=network_type,
        network_mode=network_mode,
        rec_params=rec_params,
    )

    with torch.no_grad():
        if input_generation == "random":
            patterns = sample_random_mtl_sensory_patterns(
                num_patterns=num_patterns,
                pattern_size=network.mtl_sensory_size,
                pattern_sparsity=network.mtl_sensory_sparsity[0],
            )
            network.mtl_sensory_mtl_sensory.zero_()
            network(patterns)
            network.mtl_sensory_mtl_sensory.fill_diagonal_(0)
        elif isinstance(input_generation, dict):
            if network_type != "ssc":
                raise ValueError(
                    "Structured input_generation is only supported for network_type='ssc'."
                )

            input_params = _build_structured_input_params(
                input_generation=input_generation,
                num_patterns=num_patterns,
            )
            input_tensor, _, _ = make_input(**input_params)
            network(input_tensor[0], debug=False)
            if input_params.get("sleep_after_day", True):
                network.sleep()
            patterns = torch.stack(network.activity_recordings["mtl_sensory"], dim=0)[
                network.awake_indices
            ].clone()
            network.mtl_sensory_mtl_sensory.fill_diagonal_(0)
        else:
            raise ValueError(
                "input_generation must be 'random' or a dictionary of input-generation parameters."
            )

        recalls = []
        for pattern in patterns:
            recalled = network.pattern_complete(
                "mtl_sensory",
                h_0=get_sample_from_num_swaps(pattern.clone(), num_swaps=0),
                num_iterations=network.mtl_sensory_pattern_complete_iterations,
            )
            recalls.append(get_cos_sim_torch(recalled, pattern).item())

    return float(torch.tensor(recalls).nanmean().item())
