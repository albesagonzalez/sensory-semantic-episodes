

import torch
import numpy as np
import torch.nn.functional as F


from copy import deepcopy
import itertools
import random
from collections import OrderedDict


from src.model import SSCNetwork
from src.utils.general import make_input, LatentSpace, get_ordered_indices, get_accuracy, test_network


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
        # Generate independent permutations for each row
        perms = torch.argsort(torch.rand_like(network.mtl_semantic_ctx), dim=1)
        # Apply the permutations
        network.mtl_semantic_ctx = torch.gather(network.mtl_semantic_ctx, dim=1, index=perms)
        # Freeze ctx to mtl semantic connections
        network.mtl_semantic_ctx_lmbda = 0
        network.mtl_semantic_b[:] = -1
        

    network.max_semantic_charge_replay = 2
    network.init_recordings(recording_parameters)
    network.frozen = False
    network.activity_recordings_rate = 1
    network.connectivity_recordings_rate = np.inf


    input_params["num_days"] = 500
    input_params["day_length"] = 80
    input_params["mean_duration"] = 5
    input_params["num_swaps"] = 2
    latent_specs["prob_list"] = [0.5/5 if i==j else 0.5/20 for i in range(5) for j in range(5)]

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
    network.selectivity_mtl_semantic, network.ordered_indices_mtl_semantic = get_ordered_indices(X_mtl_semantic, X_latent_AB, assembly_size=5)
    network.selectivity_mtl_sensory, network.ordered_indices_mtl_sensory = get_ordered_indices(X_mtl_sensory, X_latent_AB, assembly_size=10)
    network.selectivity_ctx_episodes, network.ordered_indices_ctx_episodes = get_ordered_indices(X_ctx[:, 100:], X_episodes, assembly_size=10)
    network.ordered_indices_ctx_episodes = network.ordered_indices_ctx_episodes + 100

    if get_network:
        return network
    
    else:
        return torch.mean(network.selectivity_ctx_episodes[network.ordered_indices_ctx_episodes - 100].max(axis=1)[0][:250])

