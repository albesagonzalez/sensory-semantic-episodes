import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset

import itertools
import random

from collections import OrderedDict


def get_sample_from_num_swaps(x_0, num_swaps, regions=None):
    if regions == None:
      x = x_0.clone().detach()
      #get on and off index
      on_index = x_0.nonzero().squeeze(1)
      off_index = (x_0 ==0).nonzero().squeeze(1)
      #choose at random num_flips indices
      flip_off = on_index[torch.randperm(len(on_index))[:int(num_swaps/2)]]
      flip_on = off_index[torch.randperm(len(off_index))[:int(num_swaps/2)]]
      #flip on to off and off to on
      x[flip_off] = 0
      x[flip_on] = 1
      return x
    
    else:
      x = x_0.clone().detach()
      total_size = sum([len(region) for region in regions])  # Total size of all regions

      for region in regions:
          # Get the size of the region
          region_size = len(region)

          # Determine the number of swaps for this region
          num_swaps_region = round(num_swaps * region_size / total_size)

          # Get on and off indices for this region
          on_index = region[x_0[region] == 1]
          off_index = region[x_0[region] == 0]

          # Choose at random num_swaps_region indices
          flip_off = on_index[torch.randperm(len(on_index))[:num_swaps_region // 2]]
          flip_on = off_index[torch.randperm(len(off_index))[:num_swaps_region // 2]]

          # Flip on to off and off to on within this region
          x[flip_off] = 0
          x[flip_on] = 1

      return x


def make_input(num_days, day_length, mean_duration, fixed_duration, num_swaps, latent_space, regions=None, satellite=False):

  #initialize input tensor
  input = torch.zeros((num_days, day_length, latent_space.total_size))
  input_episodes = torch.zeros((num_days, day_length), dtype=torch.int32)
  input_latents = torch.zeros((num_days, day_length, latent_space.num_latents), dtype=torch.int32)

  #create input from noisy patterns
  for day in range(num_days):
    day_timestep = 0
    while day_timestep < day_length:
      #pattern_duration = pattern_duration if (day_timestep + pattern_duration <= day_length) else day_length - day_timestep
      pattern_duration = mean_duration if fixed_duration else int(torch.poisson(mean_duration*torch.ones(1))[0])
      if satellite:
        latent_index, pattern = latent_space.sample()
        input_latents[day, day_timestep:day_timestep+pattern_duration] = latent_index
      else:
        label, pattern = latent_space.sample()
        input_episodes[day, day_timestep:day_timestep+pattern_duration] = latent_space.label_to_index[label]
        input_latents[day, day_timestep:day_timestep+pattern_duration, :] = torch.tensor(label, dtype=torch.int32)

      for pattern_timestep in range(pattern_duration):
        input[day, (day_timestep + pattern_timestep):(day_timestep+pattern_timestep + 1)] = get_sample_from_num_swaps(pattern, num_swaps, regions)
      day_timestep += pattern_duration

  return input, input_episodes, input_latents



class LatentSpace():
    def __init__(self, num, total_sizes, act_sizes, dims, prob_list, random_neurons=False):
      self.num_latents = num
      self.dims = dims
      self.total_sizes = total_sizes
      self.total_size = sum(total_sizes)
      self.act_sizes = act_sizes
      self.random_neurons = random_neurons
      self.latent_patterns = [[self.get_sub_latent(latent, sub_dim) for sub_dim in range(self.dims[latent])] for latent in range(self.num_latents)]
      self.sub_index_to_neuron_index = [self.latent_patterns[latent][sub_dim].nonzero().squeeze(1).detach().numpy() + sum(self.total_sizes[:latent]) for latent in range(self.num_latents) for sub_dim in range(self.dims[latent])]
      self.sub_index_to_latent_sub = [(latent, sub_dim) for latent in range(self.num_latents) for sub_dim in range(self.dims[latent])]
      self.index_to_label = list(itertools.product(*[[i for i in range(dim)] for dim in self.dims]))
      self.label_to_index = {label: index for index, label in enumerate(self.index_to_label)}
      self.label_to_neurons = OrderedDict({label: self.get_neurons_from_label(label) for label in self.index_to_label})
      self.label_to_probs = OrderedDict({label: prob for label, prob in zip(self.index_to_label, prob_list)})
      self.sub_index_to_marginal = [self.get_marginal(latent, sub) for (latent, sub) in self.sub_index_to_latent_sub]

    def get_sub_latent(self, latent, sub_dim):
      if self.random_neurons:
        sub_latent = torch.zeros((self.total_sizes[latent]))
        sub_latent[torch.randperm(self.total_sizes[latent])[:self.act_sizes[latent]]] = 1
      else:
        sub_latent = torch.zeros((self.total_sizes[latent]))
        sub_latent[sub_dim*self.act_sizes[latent]:(sub_dim+1)*self.act_sizes[latent]] = 1
      return sub_latent

    def get_neurons_from_label(self, label):
        return torch.cat(tuple(self.latent_patterns[latent][index] for latent, index in enumerate(label)))

    def get_marginal(self, latent, sub):
      marginal = 0
      for label, prob in self.label_to_probs.items():
        if label[latent] == sub:
          marginal += prob
      return marginal

    def sample(self):
      return random.choices(list(self.label_to_neurons.items()), weights=list(self.label_to_probs.values()))[0]
    







def get_selectivity(recordings, latents):
    # Flatten time
    _, num_neurons = recordings.shape


    days, day_length, num_latents = latents.shape

    recordings_flat = recordings.reshape(-1, num_neurons).float()    # shape: (days * day_length, num_neurons)
    latents_flat = latents.reshape(-1, num_latents).float()    # shape: (days * day_length, num_latents)


    # Normalize (zero mean, unit variance)
    latents_norm = (latents_flat - latents_flat.mean(dim=0)) / latents_flat.std(dim=0)
    recordings_norm = (recordings_flat - recordings_flat.mean(dim=0)) / recordings_flat.std(dim=0)


    # Compute correlation (selectivity): (num_neurons, num_latents)
    selectivity = recordings_norm.T @ latents_norm / latents_norm.shape[0]



    selectivity[torch.isnan(selectivity)] = 0


    return selectivity




def get_ordered_indices(recordings, latents, assembly_size, seed=None):
    """
    Constructs neuron assemblies ordered by selectivity. If a latent ends up with fewer than
    `assembly_size` neurons, fills the gap by randomly sampling from the discarded pool.

    Args:
        recordings: (T, N) activity tensor
        latents: (T, 2) latent labels
        assembly_size: number of neurons per latent group
        seed: random seed (optional)

    Returns:
        selectivity: (N, L) selectivity tensor
        flat_indices: (N,) long tensor with new neuron ordering
        assemblies: list of (latent_idx → list of (neuron_idx, selectivity))
    """
    if seed is not None:
        random.seed(seed)

    selectivity = get_selectivity(recordings, latents)  # (N, L)
    N, L = selectivity.shape

    # L assemblies + 1 for discarded
    ordered_indices = [[] for _ in range(L + 1)]
    assemblies = [[] for _ in range(L + 1)]

    for n in range(N):
        latent = torch.argmax(selectivity[n]).item()
        sel = selectivity[n, latent].item()
        current = assemblies[latent]

        if len(current) < assembly_size:
            current.append((n, sel))
        else:
            # Replace weakest if current is better
            min_idx, (min_n, min_sel) = min(enumerate(current), key=lambda x: x[1][1])
            if sel > min_sel:
                assemblies[latent][min_idx] = (n, sel)
                assemblies[-1].append((min_n, min_sel))
            else:
                assemblies[-1].append((n, sel))

    # Fill underfull assemblies with random discarded neurons
    discarded = assemblies[-1]
    discarded_pool = [n for n, _ in discarded]
    random.shuffle(discarded_pool)

    pool_ptr = 0
    for l in range(L):
        current = assemblies[l]
        while len(current) < assembly_size and pool_ptr < len(discarded_pool):
            n = discarded_pool[pool_ptr]
            current.append((n, float('nan')))  # placeholder selectivity if needed
            pool_ptr += 1

    # Build ordered list
    for l in range(L + 1):
        ordered_indices[l] = [n for n, _ in assemblies[l]]

    flat_indices = [n for group in ordered_indices for n in group]
    return selectivity, torch.tensor(flat_indices, dtype=torch.long)




def get_accuracy(recordings, latents, assembly_size):

    T, N = recordings.shape
    L = N//assembly_size
    
    recordings_grouped = recordings.view(T, L, assembly_size).mean(dim=2)
    
    pred_A = torch.argmax(recordings_grouped[:, :L//2], dim=1)
    pred_B = torch.argmax(recordings_grouped[:, L//2:], dim=1)

    # Compute accuracy for A and B
    acc_A = (pred_A == latents[:, 0]).float().mean()
    acc_B = (pred_B == latents[:, 1]).float().mean()

    accuracies = torch.tensor([acc_A, acc_B], device=recordings.device)

    return accuracies



def test_network(net, input_params, sleep=True, print_rate=1):
  input, input_episodes, input_latents = make_input(**input_params)
  with torch.no_grad():
    for day in range(input_params["num_days"]):
      if day%print_rate == 0:
        print(day)
      net(input[day], debug=False)
      if sleep:
        net.sleep()
  return input, input_episodes, input_latents, net



def get_cos_sim_torch(x1, x2):
  return torch.dot(x1, x2)/(torch.norm(x1)*torch.norm(x2))
def get_cos_sim_np(x1, x2):
  return np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))