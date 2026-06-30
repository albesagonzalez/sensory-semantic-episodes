from copy import deepcopy
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.model import SSCNetwork
from src.utils.episode_generation_protocol import (
    LatentSpace,
    make_input,
)
from src.utils.general import (
    get_max_overlap,
    get_ordered_indices,
    get_signal_to_noise_ratio,
)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_test_network_silent(net, input_params, sleep=True):
    input_tensor, input_episodes, input_latents = make_input(**input_params)
    with torch.no_grad():
        for day in range(int(input_params["num_days"])):
            net(input_tensor[day], debug=False)
            if sleep:
                net.sleep()
    return input_tensor, input_episodes, input_latents, net


def _make_recording_parameters():
    return {
        "regions": ["mtl_sensory", "ctx"],
        "rate_activity": 1,
        "connections": [],
        "rate_connectivity": np.inf,
    }


def _make_input_parameters(num_days, day_length, mean_duration, num_swaps):
    return {
        "num_days": int(num_days),
        "day_length": int(day_length),
        "mean_duration": int(mean_duration),
        "fixed_duration": True,
        "num_swaps": int(num_swaps),
    }


def _make_latent_specs():
    return {
        "num": 2,
        "total_sizes": [50, 50],
        "act_sizes": [10, 10],
        "dims": [5, 5],
    }


def _set_a_only_probabilities(latent_specs, a_index):
    latent_specs = deepcopy(latent_specs)
    latent_specs["prob_list"] = [
        0.2 if i == int(a_index) else 0
        for i in range(latent_specs["dims"][0])
        for j in range(latent_specs["dims"][1])
    ]
    return latent_specs


def _set_b_only_probabilities(latent_specs, b_index):
    latent_specs = deepcopy(latent_specs)
    latent_specs["prob_list"] = [
        0.2 if j == int(b_index) else 0
        for i in range(latent_specs["dims"][0])
        for j in range(latent_specs["dims"][1])
    ]
    return latent_specs


def _set_joint_probe_probabilities(latent_specs):
    latent_specs = deepcopy(latent_specs)
    latent_specs["prob_list"] = [
        0.5 / 5 if i == j else 0.5 / 20
        for i in range(latent_specs["dims"][0])
        for j in range(latent_specs["dims"][1])
    ]
    return latent_specs


def _set_uniform_joint_probabilities(latent_specs):
    latent_specs = deepcopy(latent_specs)
    latent_specs["prob_list"] = [
        1 / (latent_specs["dims"][0] * latent_specs["dims"][1])
        for i in range(latent_specs["dims"][0])
        for j in range(latent_specs["dims"][1])
    ]
    return latent_specs


def _get_block_prototype_and_replays(network):
    mtl_sensory_recordings = torch.stack(
        network.activity_recordings["mtl_sensory"], dim=0
    )
    replayed = mtl_sensory_recordings[network.sleep_indices_A].detach().cpu().clone()
    prototype = network.activation(
        mtl_sensory_recordings[network.awake_indices].mean(dim=0),
        "mtl_sensory",
        sleep=True,
    )[0].detach().cpu().clone()
    network.awake_indices = []
    network.sleep_indices_A = []
    return prototype, replayed


def _run_selectivity_probe(network, num_swaps):
    input_params = _make_input_parameters(
        num_days=200,
        day_length=80,
        mean_duration=5,
        num_swaps=num_swaps,
    )
    latent_specs = _set_joint_probe_probabilities(_make_latent_specs())
    input_params["latent_space"] = LatentSpace(**latent_specs)
    _, _, input_latents, network = _run_test_network_silent(
        network,
        input_params,
        sleep=False,
    )

    ctx_awake = torch.stack(network.activity_recordings["ctx"], dim=0)[
        network.awake_indices
    ][-100 * input_params["day_length"] :]
    latent_a = F.one_hot(
        input_latents[-100:, :, 0].long(),
        num_classes=latent_specs["dims"][0],
    )
    latent_b = F.one_hot(
        input_latents[-100:, :, 1].long(),
        num_classes=latent_specs["dims"][1],
    )
    latent_ab = torch.cat((latent_a, latent_b), dim=2)

    selectivity_ctx, ordered_indices_ctx = get_ordered_indices(
        ctx_awake,
        latent_ab,
        assembly_size=10,
    )
    network.selectivity_ctx = selectivity_ctx
    network.ordered_indices_ctx = ordered_indices_ctx
    return network, selectivity_ctx.max(dim=1)[0].detach().cpu()


def run_blocked_interleaved_noise_point(seed, num_swaps, network_parameters):
    seed_everything(seed)

    latent_specs_base = _make_latent_specs()
    recording_parameters = _make_recording_parameters()
    blocked_input_params = _make_input_parameters(
        num_days=10,
        day_length=50,
        mean_duration=1,
        num_swaps=num_swaps,
    )

    blocked_network = SSCNetwork(deepcopy(network_parameters), recording_parameters)

    prototype_a = []
    replayed_a = []
    prototype_b = []
    replayed_b = []

    for a_index in range(latent_specs_base["dims"][0]):
        latent_specs = _set_a_only_probabilities(latent_specs_base, a_index)
        blocked_input_params["latent_space"] = LatentSpace(**latent_specs)
        _, _, _, blocked_network = _run_test_network_silent(
            blocked_network,
            blocked_input_params,
            sleep=True,
        )
        prototype, replayed = _get_block_prototype_and_replays(blocked_network)
        prototype_a.append(prototype)
        replayed_a.append(replayed)

    for b_index in range(latent_specs_base["dims"][1]):
        latent_specs = _set_b_only_probabilities(latent_specs_base, b_index)
        blocked_input_params["latent_space"] = LatentSpace(**latent_specs)
        _, _, _, blocked_network = _run_test_network_silent(
            blocked_network,
            blocked_input_params,
            sleep=True,
        )
        prototype, replayed = _get_block_prototype_and_replays(blocked_network)
        prototype_b.append(prototype)
        replayed_b.append(replayed)

    prototypes = torch.stack(prototype_a + prototype_b, dim=0)
    blocked_overlap_tensors = [
        get_max_overlap(replayed, prototypes).detach().cpu()
        for replayed in replayed_a + replayed_b
    ]
    max_overlaps_blocked = torch.cat(blocked_overlap_tensors, dim=0)

    blocked_network, sel_blocked = _run_selectivity_probe(blocked_network, num_swaps)

    interleaved_input_params = _make_input_parameters(
        num_days=100,
        day_length=50,
        mean_duration=1,
        num_swaps=num_swaps,
    )
    interleaved_latent_specs = _set_uniform_joint_probabilities(_make_latent_specs())
    interleaved_input_params["latent_space"] = LatentSpace(**interleaved_latent_specs)

    interleaved_network = SSCNetwork(deepcopy(network_parameters), recording_parameters)
    _, _, _, interleaved_network = _run_test_network_silent(
        interleaved_network,
        interleaved_input_params,
        sleep=True,
    )

    replayed_interleaved = torch.stack(
        interleaved_network.activity_recordings["mtl_sensory"], dim=0
    )[interleaved_network.sleep_indices_A].detach().cpu()
    max_overlaps_interleaved = get_max_overlap(
        replayed_interleaved,
        prototypes,
    ).detach().cpu()

    interleaved_network, sel_interleaved = _run_selectivity_probe(
        interleaved_network,
        num_swaps,
    )

    return {
        "seed": int(seed),
        "num_swaps": int(num_swaps),
        "snr": float(
            get_signal_to_noise_ratio(num_swaps, blocked_network, region="mtl_sensory")
        ),
        "max_overlaps_blocked": max_overlaps_blocked.tolist(),
        "max_overlaps_interleaved": max_overlaps_interleaved.tolist(),
        "sel_blocked": sel_blocked.tolist(),
        "sel_interleaved": sel_interleaved.tolist(),
        "mean_overlap_blocked": float(max_overlaps_blocked.mean().item()),
        "mean_overlap_interleaved": float(max_overlaps_interleaved.mean().item()),
        "mean_selectivity_blocked": float(sel_blocked[:100].mean().item()),
        "mean_selectivity_interleaved": float(sel_interleaved[:100].mean().item()),
    }
