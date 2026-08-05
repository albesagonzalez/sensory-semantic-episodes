from copy import deepcopy
from itertools import product
import multiprocessing

import numpy as np
import torch
import torch.nn.functional as F

from src.model import SSCNetwork
from src.model_outgoing_ff import SSCNetwork as SSCNetworkOutgoingFF
from src.utils.episode_generation_protocol import LatentSpace
from src.utils.general import get_ordered_indices, seed_everything, train_network

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


def _mean_max_selectivity(selectivity):
    return float(torch.as_tensor(selectivity).max(dim=1)[0].mean().item())


def _as_list(values):
    if isinstance(values, (list, tuple)):
        return list(values)
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    return [values]


def run_default_600_day_selectivity(
    seed=0,
    print_rate=50,
    return_network=False,
    network_parameter_overrides=None,
    input_parameter_overrides=None,
    network_class=SSCNetwork,
):
    seed_everything(seed)

    training_recording_parameters = deepcopy(DEFAULT_TRAINING_RECORDING_PARAMETERS)
    eval_recording_parameters = deepcopy(DEFAULT_EVAL_RECORDING_PARAMETERS)
    effective_input_overrides = (
        {} if input_parameter_overrides is None else deepcopy(input_parameter_overrides)
    )
    latent_specs = deepcopy(effective_input_overrides.pop("latent_space_specs", DEFAULT_LATENT_SPECS))
    effective_input_overrides.pop("latent_space", None)
    training_input_params = deepcopy(DEFAULT_INPUT_PARAMS)
    training_input_params.update(effective_input_overrides)
    training_input_params["latent_space"] = LatentSpace(**latent_specs)

    net_params = deepcopy(network_parameters)
    net_params["duration_phase_A"] = 200
    net_params["duration_phase_B"] = 400
    net_params["max_semantic_load_replay"] = 2
    if network_parameter_overrides is not None:
        net_params.update(deepcopy(network_parameter_overrides))

    network = network_class(net_params, training_recording_parameters)
    _, _, _, network = train_network(
        network,
        training_input_params,
        sleep=True,
        print_rate=print_rate,
    )

    eval_input_params = deepcopy(DEFAULT_INPUT_PARAMS)
    eval_input_params["num_days"] = 100
    eval_input_params.update(effective_input_overrides)
    eval_input_params["num_days"] = 100
    eval_input_params["latent_space"] = LatentSpace(**latent_specs)

    network.init_recordings(eval_recording_parameters)
    network.frozen = True
    network.activity_recordings_rate = 1
    network.connectivity_recordings_rate = np.inf

    _, eval_input_episodes, eval_input_latents, network = train_network(
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
    ctx_simple_indices = network.ctx_subregions[0]
    ctx_complex_indices = network.ctx_subregions[1]
    assembly_size = 10
    num_episode_neurons = int(np.prod(latent_specs["dims"])) * assembly_size

    selectivity_ctx_simple, _ = get_ordered_indices(
        X_ctx[:, ctx_simple_indices],
        X_latent_AB,
        assembly_size=assembly_size,
    )
    selectivity_mtl_semantic, _ = get_ordered_indices(
        X_mtl_semantic,
        X_latent_AB,
        assembly_size=assembly_size,
    )
    selectivity_ctx_complex, ordered_indices_ctx_complex = get_ordered_indices(
        X_ctx[:, ctx_complex_indices],
        X_episodes,
        assembly_size=assembly_size,
    )

    ctx_simple_mean = _mean_max_selectivity(selectivity_ctx_simple)
    ctx_complex_mean = float(
        selectivity_ctx_complex[ordered_indices_ctx_complex]
        .max(dim=1)[0][:num_episode_neurons]
        .mean()
        .item()
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


def _run_default_600_day_selectivity_job(
    index_tuple,
    network_keys,
    input_keys,
    combination,
    seed,
    print_rate,
    network_class,
):
    network_overrides = {
        key: deepcopy(value)
        for key, value in zip(network_keys, combination[: len(network_keys)])
    }
    input_overrides = {
        key: deepcopy(value)
        for key, value in zip(input_keys, combination[len(network_keys) :])
    }
    summary = run_default_600_day_selectivity(
        seed=seed,
        print_rate=print_rate,
        network_parameter_overrides=network_overrides,
        input_parameter_overrides=input_overrides,
        network_class=network_class,
    )
    return index_tuple, summary


def sweep_default_600_day_selectivity(
    network_parameter_grid=None,
    input_parameter_grid=None,
    seed=0,
    print_rate=50,
    num_cpu=12,
    network_class=SSCNetwork,
):
    network_parameter_grid = (
        {} if network_parameter_grid is None else deepcopy(network_parameter_grid)
    )
    input_parameter_grid = (
        {} if input_parameter_grid is None else deepcopy(input_parameter_grid)
    )

    network_keys = list(network_parameter_grid.keys())
    input_keys = list(input_parameter_grid.keys())
    sweep_keys = network_keys + input_keys
    sweep_values = [
        _as_list(network_parameter_grid[key]) for key in network_keys
    ] + [
        _as_list(input_parameter_grid[key]) for key in input_keys
    ]

    measure_names = [
        "ctx_simple_mean_selectivity",
        "ctx_complex_mean_selectivity",
        "mtl_semantic_simple_mean_selectivity",
    ]

    if len(sweep_values) == 0:
        summary = run_default_600_day_selectivity(
            seed=seed,
            print_rate=print_rate,
            network_class=network_class,
        )
        values = np.array(
            [[summary[name] for name in measure_names]],
            dtype=float,
        )
        return {
            "values": values.reshape((1, len(measure_names))),
            "measure_names": measure_names,
            "sweep_keys": [],
            "sweep_values": [],
            "network_keys": network_keys,
            "input_keys": input_keys,
        }

    grid_shape = tuple(len(values) for values in sweep_values)
    values = np.zeros(grid_shape + (len(measure_names),), dtype=float)
    jobs = [
        (
            index_tuple,
            network_keys,
            input_keys,
            combination,
            seed,
            print_rate,
            network_class,
        )
        for index_tuple, combination in zip(
            np.ndindex(*grid_shape),
            product(*sweep_values),
        )
    ]

    if int(num_cpu) == 1:
        job_results = [
            _run_default_600_day_selectivity_job(*job)
            for job in jobs
        ]
    else:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=int(num_cpu)) as pool:
            job_results = pool.starmap(
                _run_default_600_day_selectivity_job,
                jobs,
            )

    for index_tuple, summary in job_results:
        values[index_tuple] = [summary[name] for name in measure_names]

    return {
        "values": values,
        "measure_names": measure_names,
        "sweep_keys": sweep_keys,
        "sweep_values": sweep_values,
        "network_keys": network_keys,
        "input_keys": input_keys,
        "num_cpu": int(num_cpu),
    }
