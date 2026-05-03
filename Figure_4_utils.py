

import multiprocessing
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from src.model import SSCNetwork
from src.utils.general import (
    make_input,
    LatentSpace,
    get_ordered_indices,
    get_accuracy,
    test_network,
    get_prototypes,
    get_cos_sim_matrix_torch,
)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)




def higher_order_selectivity(mode, seed, recording_parameters, input_params, latent_specs, initial_network_path, get_network=False):

    seed_everything(seed)

    network = torch.load(initial_network_path, weights_only=False)

    network.max_semantic_charge_replay = 2
    network.init_recordings(recording_parameters)
    network.frozen = False
    network.activity_recordings_rate = 1
    network.connectivity_recordings_rate = np.inf


    if mode == 'scrambled':
        '''
        # Generate independent permutations for each row
        perms = torch.argsort(torch.rand_like(network.mtl_semantic_ctx), dim=1)
        # Apply the permutations
        network.mtl_semantic_ctx = torch.gather(network.mtl_semantic_ctx, dim=1, index=perms)
        # Freeze ctx to mtl semantic connections
        network.mtl_semantic_ctx_lmbda = 0
        network.mtl_semantic_b[:] = -1
        '''
        #network.lesioned = {"mtl_semantic"}
        network.sensory_replay_only = True
        

    network.max_semantic_charge_replay = 2
    network.init_recordings(recording_parameters)
    network.frozen = False
    network.activity_recordings_rate = 1
    network.connectivity_recordings_rate = np.inf


    input_params["num_days"] = 1000
    input_params["day_length"] = 80
    input_params["mean_duration"] = 5
    input_params["latent_space"] = LatentSpace(**latent_specs)



    input, input_episodes, input_latents = make_input(**input_params)


    sleep = True
    print_rate = 50
    with torch.no_grad():
        for day in range(input_params["num_days"]):
            if day%print_rate == 0:
                print(day)
            network(input[day], debug=False)
            if sleep:
                network.sleep()


    X_ctx = torch.stack(network.activity_recordings["ctx"], dim=0)[network.awake_indices][-100*input_params["day_length"]:]
    X_mtl_semantic = torch.stack(network.activity_recordings["mtl_semantic"], dim=0)[network.awake_indices][-100*input_params["day_length"]:]
    X_mtl_sensory = torch.stack(network.activity_recordings["mtl_sensory"], dim=0)[network.awake_indices][-100*input_params["day_length"]:]

    X_latent_A = F.one_hot(input_latents[-100:, :, 0].long(), num_classes=latent_specs["dims"][0])
    X_latent_B = F.one_hot(input_latents[-100:, :, 1].long(), num_classes=latent_specs["dims"][1])
    X_latent_AB = torch.cat((X_latent_A, X_latent_B), axis=2)

    X_episodes = F.one_hot(input_episodes[-100:].long(), num_classes=np.prod(latent_specs["dims"]))


    network.selectivity_ctx, network.ordered_indices_ctx = get_ordered_indices(X_ctx[:, :100], X_latent_AB, assembly_size=10)
    network.selectivity_mtl_semantic, network.ordered_indices_mtl_semantic = get_ordered_indices(X_mtl_semantic, X_latent_AB, assembly_size=10)
    network.selectivity_mtl_sensory, network.ordered_indices_mtl_sensory = get_ordered_indices(X_mtl_sensory, X_latent_AB, assembly_size=10)
    network.selectivity_ctx_episodes, network.ordered_indices_ctx_episodes = get_ordered_indices(X_ctx[:, 100:], X_episodes, assembly_size=10)
    network.ordered_indices_ctx_episodes = network.ordered_indices_ctx_episodes + 100
    network.input_latents_higher_order = input_latents.clone()
    network.input_episodes_higher_order = input_episodes.clone()
    network.input_params_higher_order = {
        "num_days": input_params["num_days"],
        "day_length": input_params["day_length"],
        "mean_duration": input_params["mean_duration"],
        "fixed_duration": input_params["fixed_duration"],
        "num_swaps": input_params["num_swaps"],
        "latent_space": input_params["latent_space"],
    }

    if get_network:
        return network
    
    else:
        return torch.mean(network.selectivity_ctx_episodes[network.ordered_indices_ctx_episodes - 100].max(axis=1)[0][:250])


def _get_episode_names(latent_space):
    return [f"A{label[0] + 1}B{label[1] + 1}" for label in latent_space.index_to_label]


def _make_focal_episode_probabilities(num_episodes, focal_index, focal_probability):
    probs = torch.full((num_episodes,), 0.0, dtype=torch.float32)
    if num_episodes > 1 and float(focal_probability) < 1.0:
        probs[:] = (1.0 - float(focal_probability)) / float(num_episodes - 1)
    probs[int(focal_index)] = float(focal_probability)
    return probs.tolist()


def _summarize_complex_receptive_fields(
    network,
    latent_space,
    assembly_size=10,
    formation_threshold=0.5,
):
    episode_names = _get_episode_names(latent_space)
    episode_probabilities = [
        float(latent_space.label_to_probs[label]) for label in latent_space.index_to_label
    ]
    episode_prototypes = get_prototypes(latent_space, semantic_charge=2)
    ctx_subregion1_rf = network.ctx_mtl[
        network.ctx_subregions[1]
    ][:, network.mtl_sensory_size + network.ordered_indices_mtl_semantic].detach()
    episode_cosine = get_cos_sim_matrix_torch(
        episode_prototypes,
        ctx_subregion1_rf,
    )
    max_episode_cosine = episode_cosine.max(dim=1).values

    episode_scores = [float(score.item()) for score in max_episode_cosine]
    episode_was_formed = {
        episode_name: int(score >= float(formation_threshold))
        for episode_name, score in zip(episode_names, episode_scores)
    }

    episode_formation_pairs = [
        (probability, episode_was_formed[episode_name])
        for episode_name, probability in zip(episode_names, episode_probabilities)
    ]

    return {
        "episode_names": episode_names,
        "episode_probabilities": episode_probabilities,
        "episode_scores": episode_scores,
        "episode_was_formed": episode_was_formed,
        "episode_formation_pairs": episode_formation_pairs,
        "episode_cosine_matrix": episode_cosine.clone(),
    }


def analyze_focal_episode_higher_order_selectivity(
    mode,
    recording_parameters,
    input_params,
    latent_specs,
    initial_network_path,
    seed=42,
    focal_probability_grid=None,
    formation_threshold=0.5,
    assembly_size=10,
    get_network=False,
):
    seed_everything(seed)

    rec_params = deepcopy(recording_parameters)
    input_params_local = deepcopy(input_params)
    latent_specs_local = deepcopy(latent_specs)
    if focal_probability_grid is None:
        focal_probability_grid = np.linspace(0.0, 1.0, 11)
    focal_probability_grid = np.asarray(focal_probability_grid, dtype=float)

    num_episodes = int(np.prod(latent_specs_local["dims"]))
    focal_probability = float(np.random.choice(focal_probability_grid))
    focal_episode_index = int(np.random.randint(num_episodes))

    latent_specs_local["prob_list"] = _make_focal_episode_probabilities(
        num_episodes=num_episodes,
        focal_index=focal_episode_index,
        focal_probability=focal_probability,
    )
    input_params_local["num_days"] = 1
    input_params_local["day_length"] = 80
    input_params_local["mean_duration"] = 5
    input_params_local["fixed_duration"] = True
    input_params_local["latent_space"] = LatentSpace(**latent_specs_local)

    network = torch.load(initial_network_path, weights_only=False)
    network.max_semantic_charge_replay = 2
    network.init_recordings(rec_params)
    network.frozen = False
    network.activity_recordings_rate = 1
    network.connectivity_recordings_rate = np.inf

    if mode == "scrambled":
        network.sensory_replay_only = True

    input, input_episodes, input_latents, network = test_network(
        network,
        input_params_local,
        sleep=True,
        print_rate=np.inf,
    )

    X_ctx = torch.stack(network.activity_recordings["ctx"], dim=0)[network.awake_indices]
    X_episodes = F.one_hot(
        input_episodes.long(),
        num_classes=np.prod(latent_specs_local["dims"]),
    ).float()
    network.selectivity_ctx_episodes, network.ordered_indices_ctx_episodes = get_ordered_indices(
        X_ctx[:, 100:],
        X_episodes,
        assembly_size=assembly_size,
    )
    network.ordered_indices_ctx_episodes = network.ordered_indices_ctx_episodes + 100

    latent_space = LatentSpace(**latent_specs_local)
    results = {
        "seed": int(seed),
        "focal_episode_index": focal_episode_index,
        "focal_episode_probability": focal_probability,
        "focal_probability_grid": focal_probability_grid.tolist(),
        "formation_threshold": float(formation_threshold),
        "assembly_size": int(assembly_size),
    }
    results.update(
        _summarize_complex_receptive_fields(
            network=network,
            latent_space=latent_space,
            assembly_size=assembly_size,
            formation_threshold=formation_threshold,
        )
    )

    focal_episode_name = results["episode_names"][focal_episode_index]
    results["focal_episode_name"] = focal_episode_name
    results["focal_episode_pair"] = (
        float(focal_probability),
        int(results["episode_was_formed"][focal_episode_name]),
    )

    if get_network:
        results["network"] = network
        results["input"] = input
        results["input_episodes"] = input_episodes
        results["input_latents"] = input_latents

    return results


def analyze_focal_episode_higher_order_selectivity_many_seeds(
    mode,
    recording_parameters,
    input_params,
    latent_specs,
    initial_network_path,
    seeds,
    focal_probability_grid=None,
    formation_threshold=0.5,
    assembly_size=10,
    num_cpu=None,
    start_method="fork",
):
    seeds = [int(seed) for seed in np.asarray(seeds).ravel().tolist()]
    experiment_params = [
        (
            mode,
            recording_parameters,
            input_params,
            latent_specs,
            initial_network_path,
            seed,
            focal_probability_grid,
            formation_threshold,
            assembly_size,
            False,
        )
        for seed in seeds
    ]

    if num_cpu == 1:
        results_list = [
            analyze_focal_episode_higher_order_selectivity(*params)
            for params in experiment_params
        ]
    else:
        ctx = multiprocessing.get_context(start_method) if start_method is not None else multiprocessing
        with ctx.Pool(processes=num_cpu) as pool:
            results_list = pool.starmap(
                analyze_focal_episode_higher_order_selectivity,
                experiment_params,
            )

    all_focal_episode_pairs = [
        result["focal_episode_pair"] for result in results_list
    ]

    return {
        "seed": [result["seed"] for result in results_list],
        "focal_episode_name": [result["focal_episode_name"] for result in results_list],
        "focal_episode_index": [result["focal_episode_index"] for result in results_list],
        "focal_episode_probability": [result["focal_episode_probability"] for result in results_list],
        "focal_episode_pair": [result["focal_episode_pair"] for result in results_list],
        "all_focal_episode_pairs": all_focal_episode_pairs,
        "episode_scores": [result["episode_scores"] for result in results_list],
        "results_list": results_list,
    }
