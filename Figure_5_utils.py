

import torch
import numpy as np
import torch.nn.functional as F


from copy import deepcopy
import itertools
import random
from collections import OrderedDict


from model import SSCNetwork
from src.utils.general import make_input, LatentSpace, get_ordered_indices, get_accuracy, test_network
from src.utils.general_old import get_selectivity, get_cos_sim_torch, get_selectivity_accuracy


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def blocked_interleaved(network_parameters, recording_parameters, input_params, latent_specs, training='blocked', seed=42, get_network=False):
    print(f"starting simulation {training} - {seed}")


    seed_everything(seed)
    
    if training=='interleaved':


        original_prob_list = deepcopy(latent_specs["prob_list"])

        
        input_params["num_days"] = 100
        input_params["day_length"] = 200
        input_params["mean_duration"] = 1

        latent_specs["prob_list"] = [0.5/5 if i==j else 0.5/20 for i in range(5) for j in range(5)]
        input_params["latent_space"] = LatentSpace(**latent_specs)
        network = SSCNetwork(network_parameters, recording_parameters)
        input, input_episodes, input_latents, network = test_network(network, input_params, print_rate=50)


        input_params["num_days"] = 1400
        input_params["day_length"] = 80
        input_params["mean_duration"] = 5
        latent_specs["prob_list"] = original_prob_list
        input_params["latent_space"] = LatentSpace(**latent_specs)
        input, input_episodes, input_latents, network = test_network(network, input_params, print_rate=50)


        network.init_recordings(recording_parameters)
        network.frozen = True
        network.activity_recordings_rate = 1
        network.connectivity_recordings_rate = np.inf

        input_params["num_days"] = 100
        input, input_episodes, input_latents, network = test_network(network, input_params, sleep=False, print_rate=50)

    if training == 'blocked':

        original_prob_list = deepcopy(latent_specs["prob_list"])

        input_params["num_days"] = 10
        input_params["day_length"] = 200
        input_params["mean_duration"] = 1

        latent_specs["prob_list"] = [0.2 if i==0 else 0 for i in range(5) for j in range(5)]
        input_params["latent_space"] = LatentSpace(**latent_specs)

        network = SSCNetwork(network_parameters, recording_parameters)
        input, input_episodes, input_latents, network = test_network(network, input_params, print_rate=50)

        for k in range(1, 5):
            latent_specs["prob_list"] = [0.2 if i==k else 0 for i in range(5) for j in range(5)]
            input_params["latent_space"] = LatentSpace(**latent_specs)
            input, input_episodes, input_latents, network = test_network(network, input_params, print_rate=50)

        for k in range(5):
            latent_specs["prob_list"] = [0.2 if j==k else 0 for i in range(5) for j in range(5)]
            input_params["latent_space"] = LatentSpace(**latent_specs)
            input, input_episodes, input_latents, network = test_network(network, input_params, print_rate=50)


        input_params["num_days"] = 1400
        input_params["day_length"] = 80
        input_params["mean_duration"] = 5
        latent_specs["prob_list"] = original_prob_list
        input_params["latent_space"] = LatentSpace(**latent_specs)
        input, input_episodes, input_latents, network = test_network(network, input_params, print_rate=50)


        network.init_recordings(recording_parameters)
        network.frozen = True
        network.activity_recordings_rate = 1
        network.connectivity_recordings_rate = np.inf

        input_params["num_days"] = 100
        input, input_episodes, input_latents, network = test_network(network, input_params, sleep=False, print_rate=50)


    
    X_ctx = torch.stack(network.activity_recordings["ctx"], dim=0)[network.awake_indices][-100*input_params["day_length"]:]
    X_mtl_sparse = torch.stack(network.activity_recordings["mtl_sparse"], dim=0)[network.awake_indices][-100*input_params["day_length"]:]
    X_mtl_dense = torch.stack(network.activity_recordings["mtl_dense"], dim=0)[network.awake_indices][-100*input_params["day_length"]:]

    X_latent_A = F.one_hot(input_latents[-100:, :, 0].long(), num_classes=latent_specs["dims"][0])
    X_latent_B = F.one_hot(input_latents[-100:, :, 1].long(), num_classes=latent_specs["dims"][1])
    X_latent_AB = torch.cat((X_latent_A, X_latent_B), axis=2)


    selectivity_ctx, ordered_indices_ctx = get_ordered_indices(X_ctx, X_latent_AB, assembly_size=10)
    selectivity_mtl_sparse, ordered_indices_mtl_sparse = get_ordered_indices(X_mtl_sparse, X_latent_AB, assembly_size=5)
    selectivity_mtl_dense, ordered_indices_mtl_dense = get_ordered_indices(X_mtl_dense, X_latent_AB, assembly_size=10)


    accuracy = get_accuracy(X_mtl_sparse[:, ordered_indices_mtl_sparse[:50]], input_latents.reshape((-1, 2)), assembly_size=5)


    if get_network:
      return network, input, input_latents, input_episodes, ordered_indices_ctx, ordered_indices_mtl_dense, ordered_indices_mtl_sparse, selectivity_ctx, selectivity_mtl_dense, selectivity_mtl_sparse, accuracy
    else:     
      return selectivity_mtl_sparse, accuracy

