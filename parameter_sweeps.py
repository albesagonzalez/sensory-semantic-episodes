from copy import deepcopy
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.model import SSCNetwork
from src.utils.episode_generation_protocol import LatentSpace
from src.utils.general import get_ordered_indices, test_network

from src.network_parameters import network_parameters


DEFAULT_TRAINING_RECORDING_PARAMETERS = {
    "regions": [],
    "rate_activity": np.inf,
    "connections": [],
    "rate_connectivity": np.inf,
}


DEFAULT_EVAL_RECORDING_PARAMETERS = {
    "regions": ["mtl_semantic", "ctx"],
    "rate_activity": 1,
    "connections": [],
    "rate_connectivity": np.inf,
}


DEFAULT_INPUT_PARAMS = {
    "num_days": 600,
    "day_length": 80,
    "mean_duration": 5,
    "fixed_duration": True,
    "num_swaps": 4,
}


DEFAULT_LATENT_SPECS = {
    "num": 2,
    "total_sizes": [50, 50],
    "act_sizes": [10, 10],
    "dims": [5, 5],
    "prob_list": [0.5 / 5 if i == j else 0.5 / 20 for i in range(5) for j in range(5)],
}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _mean_max_selectivity(selectivity):
    return float(torch.as_tensor(selectivity).max(dim=1)[0].mean().item())


def run_default_600_day_selectivity(seed=0, print_rate=50, return_network=False):
    seed_everything(seed)

    training_recording_parameters = deepcopy(DEFAULT_TRAINING_RECORDING_PARAMETERS)
    eval_recording_parameters = deepcopy(DEFAULT_EVAL_RECORDING_PARAMETERS)
    training_input_params = deepcopy(DEFAULT_INPUT_PARAMS)
    latent_specs = deepcopy(DEFAULT_LATENT_SPECS)
    training_input_params["latent_space"] = LatentSpace(**latent_specs)

    net_params = deepcopy(network_parameters)
    net_params["duration_phase_A"] = 200
    net_params["duration_phase_B"] = 400
    net_params["max_semantic_load_replay"] = 2

    network = SSCNetwork(net_params, training_recording_parameters)
    _, _, _, network = test_network(
        network,
        training_input_params,
        sleep=True,
        print_rate=print_rate,
    )

    eval_input_params = deepcopy(DEFAULT_INPUT_PARAMS)
    eval_input_params["num_days"] = 100
    eval_input_params["latent_space"] = LatentSpace(**latent_specs)

    network.init_recordings(eval_recording_parameters)
    network.frozen = True
    network.activity_recordings_rate = 1
    network.connectivity_recordings_rate = np.inf

    _, eval_input_episodes, eval_input_latents, network = test_network(
        network,
        eval_input_params,
        sleep=False,
        print_rate=np.inf,
    )

    X_ctx = torch.stack(network.activity_recordings["ctx"], dim=0)[network.awake_indices]
    X_mtl_semantic = torch.stack(network.activity_recordings["mtl_semantic"], dim=0)[
        network.awake_indices
    ]

    X_latent_A = F.one_hot(
        eval_input_latents[:, :, 0].long(),
        num_classes=latent_specs["dims"][0],
    )
    X_latent_B = F.one_hot(
        eval_input_latents[:, :, 1].long(),
        num_classes=latent_specs["dims"][1],
    )
    X_latent_AB = torch.cat((X_latent_A, X_latent_B), dim=2)
    X_episodes = F.one_hot(
        eval_input_episodes.long(),
        num_classes=int(np.prod(latent_specs["dims"])),
    )

    selectivity_ctx_simple, _ = get_ordered_indices(
        X_ctx[:, :100],
        X_latent_AB,
        assembly_size=10,
    )
    selectivity_mtl_semantic, _ = get_ordered_indices(
        X_mtl_semantic,
        X_latent_AB,
        assembly_size=10,
    )
    selectivity_ctx_complex, ordered_indices_ctx_complex = get_ordered_indices(
        X_ctx[:, 100:],
        X_episodes,
        assembly_size=10,
    )

    ctx_simple_mean = _mean_max_selectivity(selectivity_ctx_simple)
    ctx_complex_mean = float(
        selectivity_ctx_complex[ordered_indices_ctx_complex].max(dim=1)[0][:250].mean().item()
    )
    mtl_semantic_simple_mean = _mean_max_selectivity(selectivity_mtl_semantic)

    results = {
        "ctx_simple_mean_selectivity": ctx_simple_mean,
        "ctx_complex_mean_selectivity": ctx_complex_mean,
        "mtl_semantic_simple_mean_selectivity": mtl_semantic_simple_mean,
    }
    if return_network:
        results["network"] = network
    return results
