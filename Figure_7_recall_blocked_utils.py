from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from src.model import SSCNetwork
from src.input_parameters import (
    input_params as default_input_params,
    latent_specs as default_latent_specs,
)
from src.utils.episode_generation_protocol import LatentSpace
from src.utils.general import (
    get_accuracy,
    get_ordered_indices,
    seed_everything,
    train_network,
)


def configure_blocked_interleaved_worker():
    """Prevent nested PyTorch threading inside the multiprocessing pool."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # The inter-op pool may already be initialized when using fork.
        pass


def _make_recording_parameters():
    return {
        "regions": ["ctx"],
        "rate_activity": 1,
        "connections": [],
        "rate_connectivity": np.inf,
    }


def _make_no_recording_parameters():
    return {
        "regions": [],
        "rate_activity": np.inf,
        "connections": [],
        "rate_connectivity": np.inf,
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


def _run_accuracy_probe(network, num_swaps):
    # Training activity is not used by the metric.  Reset recording immediately
    # before the probe so only the required CTX responses are retained.
    network.init_recordings(_make_recording_parameters())
    input_params = deepcopy(default_input_params)
    input_params["num_days"] = 200
    input_params["num_swaps"] = int(num_swaps)
    latent_specs = _set_joint_probe_probabilities(deepcopy(default_latent_specs))
    input_params["latent_space"] = LatentSpace(**latent_specs)
    probe_input, probe_episodes, input_latents, network = train_network(
        network,
        input_params,
        sleep=False,
        print_rate=np.inf,
    )
    del probe_input, probe_episodes

    ctx_awake = torch.stack(network.activity_recordings["ctx"], dim=0)[
        network.awake_indices
    ][-100 * input_params["day_length"] :]
    eval_latents = input_latents[-100:].reshape(-1, input_latents.shape[-1])
    latent_a = F.one_hot(
        input_latents[-100:, :, 0].long(),
        num_classes=latent_specs["dims"][0],
    )
    latent_b = F.one_hot(
        input_latents[-100:, :, 1].long(),
        num_classes=latent_specs["dims"][1],
    )
    latent_ab = torch.cat((latent_a, latent_b), dim=2).reshape(-1, latent_a.shape[-1] + latent_b.shape[-1])
    fit_num_samples = ctx_awake.shape[0] // 2
    if fit_num_samples < 1 or fit_num_samples >= ctx_awake.shape[0]:
        raise ValueError(
            "Accuracy probe requires at least two recorded awake samples "
            f"to split fit/test data, got {ctx_awake.shape[0]}."
        )

    _, ordered_indices_ctx = get_ordered_indices(
        ctx_awake[:fit_num_samples],
        latent_ab[:fit_num_samples],
        assembly_size=10,
    )
    ctx_accuracy = get_accuracy(
        ctx_awake[fit_num_samples:, ordered_indices_ctx[:100]],
        eval_latents[fit_num_samples:],
        assembly_size=10,
    )
    ctx_accuracy = ctx_accuracy.detach().cpu()

    # Each worker handles many jobs.  Release the probe recordings before it
    # begins the next condition/job rather than retaining them in the PyTorch
    # allocator for the lifetime of the worker.
    del ctx_awake, eval_latents, latent_a, latent_b, latent_ab, ordered_indices_ctx
    network.activity_recordings.clear()
    network.activity_recordings_time.clear()
    network.awake_indices.clear()
    network.sleep_indices_A.clear()
    network.sleep_indices_B.clear()
    return ctx_accuracy


def run_blocked_interleaved_noise_point(seed, num_swaps, network_parameters):
    seed_everything(seed)

    latent_specs_base = deepcopy(default_latent_specs)
    training_recording_parameters = _make_no_recording_parameters()
    blocked_input_params = deepcopy(default_input_params)
    blocked_input_params["num_days"] = 10
    blocked_input_params["day_length"] = 80
    blocked_input_params["mean_duration"] = 5
    blocked_input_params["num_swaps"] = int(num_swaps)

    blocked_network = SSCNetwork(deepcopy(network_parameters), training_recording_parameters)

    for a_index in range(latent_specs_base["dims"][0]):
        latent_specs = _set_a_only_probabilities(latent_specs_base, a_index)
        blocked_input_params["latent_space"] = LatentSpace(**latent_specs)
        training_output = train_network(
            blocked_network,
            blocked_input_params,
            sleep=True,
            print_rate=np.inf,
        )
        blocked_network = training_output[-1]
        del training_output

    for b_index in range(latent_specs_base["dims"][1]):
        latent_specs = _set_b_only_probabilities(latent_specs_base, b_index)
        blocked_input_params["latent_space"] = LatentSpace(**latent_specs)
        training_output = train_network(
            blocked_network,
            blocked_input_params,
            sleep=True,
            print_rate=np.inf,
        )
        blocked_network = training_output[-1]
        del training_output

    acc_blocked = _run_accuracy_probe(blocked_network, num_swaps)
    del blocked_network

    interleaved_input_params = deepcopy(default_input_params)
    interleaved_input_params["num_days"] = 100
    interleaved_input_params["day_length"] = 80
    interleaved_input_params["mean_duration"] = 1
    interleaved_input_params["num_swaps"] = int(num_swaps)
    interleaved_latent_specs = _set_uniform_joint_probabilities(
        deepcopy(default_latent_specs)
    )
    interleaved_input_params["latent_space"] = LatentSpace(**interleaved_latent_specs)

    interleaved_network = SSCNetwork(
        deepcopy(network_parameters), training_recording_parameters
    )
    training_output = train_network(
        interleaved_network,
        interleaved_input_params,
        sleep=True,
        print_rate=np.inf,
    )
    interleaved_network = training_output[-1]
    del training_output

    acc_interleaved = _run_accuracy_probe(
        interleaved_network,
        num_swaps,
    )
    del interleaved_network

    return {
        "seed": int(seed),
        "num_swaps": int(num_swaps),
        "acc_blocked": acc_blocked.tolist(),
        "acc_interleaved": acc_interleaved.tolist(),
        "mean_accuracy_blocked": float(acc_blocked.mean().item()),
        "mean_accuracy_interleaved": float(acc_interleaved.mean().item()),
    }
