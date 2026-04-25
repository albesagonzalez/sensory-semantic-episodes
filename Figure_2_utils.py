import multiprocessing
import random

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

#from src.model import SSCNetwork
from src.model_engrams import SSCNetwork


from src.utils.general import LatentSpace, test_network


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_concept_names(latent_space):
    concept_names = []
    latent_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for latent_idx, latent_dim in enumerate(latent_space.dims):
        prefix = latent_labels[latent_idx] if latent_idx < len(latent_labels) else f"L{latent_idx}"
        for sub_idx in range(latent_dim):
            concept_names.append(f"{prefix}{sub_idx + 1}")
    return concept_names


def _sample_latent_with_focal_probability(latent_dim, focal_probability, concentration=1.0):
    focal_index = int(np.random.randint(latent_dim))
    latent_probs = torch.zeros(latent_dim, dtype=torch.float32)
    latent_probs[focal_index] = float(focal_probability)

    if latent_dim > 1:
        remaining_mass = 1.0 - float(focal_probability)
        remaining_probs = torch.distributions.Dirichlet(
            concentration * torch.ones(latent_dim - 1, dtype=torch.float32)
        ).sample()
        remaining_indices = [idx for idx in range(latent_dim) if idx != focal_index]
        latent_probs[remaining_indices] = remaining_mass * remaining_probs

    return latent_probs, focal_index


def _make_independent_episode_probabilities(latent_probabilities):
    prob_matrix = latent_probabilities[0][:, None] * latent_probabilities[1][None, :]
    return prob_matrix.reshape(-1).tolist()


def _summarize_receptive_fields(network, latent_space):
    mature_mask = network.ctx_IM == 0
    mature_ctx_indices = torch.where(mature_mask)[0]
    w_ctx_mtl_mature = network.ctx_mtl[mature_mask, : network.mtl_sensory_size]

    concept_names = _get_concept_names(latent_space)
    concept_indices = latent_space.sub_index_to_neuron_index
    concept_marginals = list(latent_space.sub_index_to_marginal)

    if w_ctx_mtl_mature.shape[0] > 0:
        concept_scores = torch.stack(
            [w_ctx_mtl_mature[:, neuron_indices].mean(dim=1) for neuron_indices in concept_indices],
            dim=1,
        )
        winning_concepts = concept_scores.argmax(dim=1)
        neuron_to_concept = {
            int(ctx_idx): concept_names[int(concept_idx)]
            for ctx_idx, concept_idx in zip(mature_ctx_indices.tolist(), winning_concepts.tolist())
        }
    else:
        concept_scores = torch.empty((0, len(concept_names)))
        winning_concepts = torch.empty((0,), dtype=torch.long)
        neuron_to_concept = {}

    formed_concepts = sorted(set(neuron_to_concept.values()))
    concept_was_formed = {
        concept_name: int(concept_name in formed_concepts) for concept_name in concept_names
    }
    concept_formation_pairs = [
        (float(concept_marginal), concept_was_formed[concept_name])
        for concept_name, concept_marginal in zip(concept_names, concept_marginals)
    ]

    return {
        "mature_mask": mature_mask.clone(),
        "mature_ctx_indices": mature_ctx_indices.clone(),
        "num_mature": int(mature_mask.sum().item()),
        "concept_names": concept_names,
        "concept_marginals": concept_marginals,
        "concept_was_formed": concept_was_formed,
        "concept_formation_pairs": concept_formation_pairs,
        "concept_scores": concept_scores.clone(),
        "winning_concepts": winning_concepts.clone(),
        "neuron_to_concept": neuron_to_concept,
        "formed_concepts": formed_concepts,
        "w_ctx_mtl_mature": w_ctx_mtl_mature.clone(),
    }


def analyze_dirichlet_sleep_receptive_fields(
    network_parameters,
    recording_parameters,
    input_params,
    latent_specs,
    seed=42,
    dirichlet_alpha=None,
    return_dynamics=False,
    get_network=False,
):
    print(f"starting simulation - {seed}")

    seed_everything(seed)

    net_params = deepcopy(network_parameters)
    rec_params = deepcopy(recording_parameters)
    input_params_local = deepcopy(input_params)
    latent_specs_local = deepcopy(latent_specs)

    num_labels = int(np.prod(latent_specs_local["dims"]))
    alpha = torch.ones(num_labels) if dirichlet_alpha is None else torch.as_tensor(dirichlet_alpha, dtype=torch.float32)
    latent_specs_local["prob_list"] = torch.distributions.Dirichlet(alpha).sample().tolist()
    input_params_local["latent_space"] = LatentSpace(**latent_specs_local)

    network = SSCNetwork(net_params, rec_params)
    input, input_episodes, input_latents, network = test_network(
        network, input_params_local, sleep=True, print_rate=np.inf
    )

    results = {
        "seed": int(seed),
        "dirichlet_probs": deepcopy(latent_specs_local["prob_list"]),
    }
    results.update(_summarize_receptive_fields(network, input_params_local["latent_space"]))

    if return_dynamics:
        ctx_mtl_recordings = torch.stack(network.connectivity_recordings["ctx_mtl"], dim=0)
        sleep_indices_A = list(network.sleep_indices_A)
        results["sleep_indices_A"] = sleep_indices_A
        results["ctx_mtl_sleep"] = ctx_mtl_recordings[sleep_indices_A][:, results["mature_mask"], : network.mtl_sensory_size].clone()
        results["ctx_mtl_before_sleep"] = (
            ctx_mtl_recordings[sleep_indices_A[0] - 1][results["mature_mask"], : network.mtl_sensory_size].clone()
            if len(sleep_indices_A) > 0
            else torch.empty((0, network.mtl_sensory_size))
        )

    if get_network:
        results["network"] = network
        results["input"] = input
        results["input_episodes"] = input_episodes
        results["input_latents"] = input_latents

    return results


def analyze_focal_concept_sleep_receptive_fields(
    network_parameters,
    recording_parameters,
    input_params,
    latent_specs,
    seed=42,
    focal_probability_grid=None,
    concentration=1.0,
    return_dynamics=False,
    get_network=False,
):
    print(f"starting simulation - {seed}")

    seed_everything(seed)

    net_params = deepcopy(network_parameters)
    rec_params = deepcopy(recording_parameters)
    input_params_local = deepcopy(input_params)
    latent_specs_local = deepcopy(latent_specs)

    if focal_probability_grid is None:
        focal_probability_grid = np.linspace(0.05, 0.95, 19)
    focal_probability_grid = np.asarray(focal_probability_grid, dtype=float)

    latent_probabilities = []
    focal_concept_indices = []
    focal_concept_probabilities = []

    for latent_dim in latent_specs_local["dims"]:
        focal_probability = float(np.random.choice(focal_probability_grid))
        latent_probs, focal_index = _sample_latent_with_focal_probability(
            latent_dim,
            focal_probability,
            concentration=concentration,
        )
        latent_probabilities.append(latent_probs)
        focal_concept_indices.append(int(focal_index))
        focal_concept_probabilities.append(float(focal_probability))

    latent_specs_local["prob_list"] = _make_independent_episode_probabilities(latent_probabilities)
    input_params_local["latent_space"] = LatentSpace(**latent_specs_local)

    network = SSCNetwork(net_params, rec_params)
    input, input_episodes, input_latents, network = test_network(
        network, input_params_local, sleep=True, print_rate=np.inf
    )

    results = {
        "seed": int(seed),
        "episode_probs": deepcopy(latent_specs_local["prob_list"]),
        "latent_probabilities": [latent_probs.clone() for latent_probs in latent_probabilities],
        "focal_concept_indices": focal_concept_indices,
        "focal_concept_probabilities": focal_concept_probabilities,
        "focal_probability_grid": focal_probability_grid.tolist(),
        "sampling_scheme": "focal_concept_grid",
    }
    results.update(_summarize_receptive_fields(network, input_params_local["latent_space"]))

    focal_concept_names = []
    focal_concept_pairs = []
    concept_offset = 0
    for latent_idx, (latent_dim, focal_index) in enumerate(zip(latent_specs_local["dims"], focal_concept_indices)):
        focal_name = results["concept_names"][concept_offset + focal_index]
        focal_marginal = float(focal_concept_probabilities[latent_idx])
        focal_was_formed = int(results["concept_was_formed"][focal_name])
        focal_concept_names.append(focal_name)
        focal_concept_pairs.append((focal_marginal, focal_was_formed))
        concept_offset += latent_dim

    results["focal_concept_names"] = focal_concept_names
    results["focal_concept_pairs"] = focal_concept_pairs

    if return_dynamics:
        ctx_mtl_recordings = torch.stack(network.connectivity_recordings["ctx_mtl"], dim=0)
        sleep_indices_A = list(network.sleep_indices_A)
        results["sleep_indices_A"] = sleep_indices_A
        results["ctx_mtl_sleep"] = ctx_mtl_recordings[sleep_indices_A][:, results["mature_mask"], : network.mtl_sensory_size].clone()
        results["ctx_mtl_before_sleep"] = (
            ctx_mtl_recordings[sleep_indices_A[0] - 1][results["mature_mask"], : network.mtl_sensory_size].clone()
            if len(sleep_indices_A) > 0
            else torch.empty((0, network.mtl_sensory_size))
        )

    if get_network:
        results["network"] = network
        results["input"] = input
        results["input_episodes"] = input_episodes
        results["input_latents"] = input_latents

    return results


def analyze_dirichlet_sleep_receptive_fields_many_seeds(
    network_parameters,
    recording_parameters,
    input_params,
    latent_specs,
    seeds,
    dirichlet_alpha=None,
    num_cpu=None,
    return_dynamics=False,
    start_method="fork",
):
    seeds = [int(seed) for seed in np.asarray(seeds).ravel().tolist()]
    experiment_params = [
        (
            network_parameters,
            recording_parameters,
            input_params,
            latent_specs,
            seed,
            dirichlet_alpha,
            return_dynamics,
            False,
        )
        for seed in seeds
    ]

    if num_cpu == 1:
        results_list = [
            analyze_dirichlet_sleep_receptive_fields(*params) for params in experiment_params
        ]
    else:
        ctx = multiprocessing.get_context(start_method) if start_method is not None else multiprocessing
        with ctx.Pool(processes=num_cpu) as pool:
            results_list = pool.starmap(analyze_dirichlet_sleep_receptive_fields, experiment_params)

    concept_names = results_list[0]["concept_names"] if len(results_list) > 0 else []
    concept_frequency = {
        concept_name: sum(concept_name in result["formed_concepts"] for result in results_list)
        for concept_name in concept_names
    }


def analyze_focal_concept_sleep_receptive_fields_many_seeds(
    network_parameters,
    recording_parameters,
    input_params,
    latent_specs,
    seeds,
    focal_probability_grid=None,
    concentration=1.0,
    num_cpu=None,
    return_dynamics=False,
    start_method="fork",
):
    seeds = [int(seed) for seed in np.asarray(seeds).ravel().tolist()]
    experiment_params = [
        (
            network_parameters,
            recording_parameters,
            input_params,
            latent_specs,
            seed,
            focal_probability_grid,
            concentration,
            return_dynamics,
            False,
        )
        for seed in seeds
    ]

    if num_cpu == 1:
        results_list = [
            analyze_focal_concept_sleep_receptive_fields(*params) for params in experiment_params
        ]
    else:
        ctx = multiprocessing.get_context(start_method) if start_method is not None else multiprocessing
        with ctx.Pool(processes=num_cpu) as pool:
            results_list = pool.starmap(analyze_focal_concept_sleep_receptive_fields, experiment_params)

    all_focal_concept_pairs = [
        pair for result in results_list for pair in result["focal_concept_pairs"]
    ]

    return {
        "seed": [result["seed"] for result in results_list],
        "focal_concept_names": [result["focal_concept_names"] for result in results_list],
        "focal_concept_probabilities": [result["focal_concept_probabilities"] for result in results_list],
        "focal_concept_pairs": [result["focal_concept_pairs"] for result in results_list],
        "all_focal_concept_pairs": all_focal_concept_pairs,
        "num_mature": [result["num_mature"] for result in results_list],
        "formed_concepts": [result["formed_concepts"] for result in results_list],
        "results_list": results_list,
    }
    all_concept_formation_pairs = [
        pair for result in results_list for pair in result["concept_formation_pairs"]
    ]

    return {
        "seed": [result["seed"] for result in results_list],
        "dirichlet_probs": [result["dirichlet_probs"] for result in results_list],
        "num_mature": [result["num_mature"] for result in results_list],
        "formed_concepts": [result["formed_concepts"] for result in results_list],
        "neuron_to_concept": [result["neuron_to_concept"] for result in results_list],
        "concept_formation_pairs": [result["concept_formation_pairs"] for result in results_list],
        "all_concept_formation_pairs": all_concept_formation_pairs,
        "concept_frequency": concept_frequency,
        "formed_concepts_union": sorted(
            set().union(*(set(result["formed_concepts"]) for result in results_list))
        ),
        "results_list": results_list,
    }


def get_binned_concept_formation_probability(concept_formation_pairs, num_bins=10, bin_edges=None):
    concept_formation_pairs = list(concept_formation_pairs)
    if len(concept_formation_pairs) == 0:
        return {
            "bin_centers": np.array([]),
            "bin_left_edges": np.array([]),
            "bin_right_edges": np.array([]),
            "formed_probability": np.array([]),
            "counts": np.array([]),
        }

    probabilities = np.array([pair[0] for pair in concept_formation_pairs], dtype=float)
    formed = np.array([pair[1] for pair in concept_formation_pairs], dtype=float)

    if bin_edges is None:
        p_min = probabilities.min()
        p_max = probabilities.max()
        if np.isclose(p_min, p_max):
            p_min = max(0.0, p_min - 1e-6)
            p_max = p_max + 1e-6
        bin_edges = np.linspace(p_min, p_max, num_bins + 1)
    else:
        bin_edges = np.asarray(bin_edges, dtype=float)

    bin_indices = np.digitize(probabilities, bin_edges[1:-1], right=False)
    num_effective_bins = len(bin_edges) - 1

    formed_probability = np.full(num_effective_bins, np.nan)
    counts = np.zeros(num_effective_bins, dtype=int)
    for bin_idx in range(num_effective_bins):
        in_bin = bin_indices == bin_idx
        counts[bin_idx] = int(in_bin.sum())
        if counts[bin_idx] > 0:
            formed_probability[bin_idx] = formed[in_bin].mean()

    return {
        "bin_centers": 0.5 * (bin_edges[:-1] + bin_edges[1:]),
        "bin_left_edges": bin_edges[:-1],
        "bin_right_edges": bin_edges[1:],
        "formed_probability": formed_probability,
        "counts": counts,
    }


def get_discrete_concept_formation_probability(concept_formation_pairs, round_decimals=8):
    concept_formation_pairs = list(concept_formation_pairs)
    if len(concept_formation_pairs) == 0:
        return {
            "probabilities": np.array([]),
            "formed_probability": np.array([]),
            "counts": np.array([]),
        }

    grouped = {}
    for probability, formed in concept_formation_pairs:
        probability = round(float(probability), round_decimals)
        grouped.setdefault(probability, []).append(float(formed))

    probabilities = np.array(sorted(grouped.keys()), dtype=float)
    formed_probability = np.array([np.mean(grouped[p]) for p in probabilities], dtype=float)
    counts = np.array([len(grouped[p]) for p in probabilities], dtype=int)

    return {
        "probabilities": probabilities,
        "formed_probability": formed_probability,
        "counts": counts,
    }


def plot_binned_concept_formation_probability(
    concept_formation_pairs,
    num_bins=10,
    bin_edges=None,
    ax=None,
    scatter=True,
):
    binned_results = get_binned_concept_formation_probability(
        concept_formation_pairs,
        num_bins=num_bins,
        bin_edges=bin_edges,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    if scatter:
        probabilities = [pair[0] for pair in concept_formation_pairs]
        formed = [pair[1] for pair in concept_formation_pairs]
        ax.scatter(probabilities, formed, s=10, alpha=0.15, color="black")

    valid_bins = ~np.isnan(binned_results["formed_probability"])
    ax.plot(
        binned_results["bin_centers"][valid_bins],
        binned_results["formed_probability"][valid_bins],
        marker="o",
        color="tab:red",
    )
    ax.set_xlabel("Marginal concept probability")
    ax.set_ylabel("Empirical RF formation probability")

    return ax, binned_results
