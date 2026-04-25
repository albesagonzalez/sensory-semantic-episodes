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
      flip_off = on_index[torch.randperm(len(on_index))[:int(num_swaps)]]
      flip_on = off_index[torch.randperm(len(off_index))[:int(num_swaps)]]
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
          flip_off = on_index[torch.randperm(len(on_index))[:num_swaps_region]]
          flip_on = off_index[torch.randperm(len(off_index))[:num_swaps_region]]

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
    




def get_prototypes(
    latent_space,
    sub_pattern=None,
    semantic_charge=1,
    return_labels=False,
):
    """
    Return full neural sheets for latent prototypes.

    For ``semantic_charge=1``, each returned row contains exactly one sub-pattern
    (for example ``A1`` or ``B5``), with all other latent blocks set to zero.
    For ``semantic_charge=2``, each returned row contains one full episode
    prototype (for example ``A1_B5``).

    Args:
        latent_space: ``LatentSpace`` instance.
        sub_pattern: optional selector. If ``None``, returns all prototypes for
            the requested semantic charge. If provided:
            - for ``semantic_charge=1`` it can be a string like ``"A1"`` or
              ``"B5"``, or a tuple ``(latent_idx, sub_idx)``
            - for ``semantic_charge=2`` it can be a string like ``"A1_B5"``, or
              a full latent label tuple like ``(0, 4)``
        semantic_charge: ``1`` for sub-pattern prototypes, ``2`` for full
            episode prototypes.
        return_labels: if ``True``, also return the corresponding labels.

    Returns:
        ``prototypes`` or ``(prototypes, labels)`` where ``prototypes`` has
        shape ``(num_selected_subpatterns, latent_space.total_size)``.
    """
    def _label_for(latent_idx, sub_idx):
        if latent_idx < 26:
            return f"{chr(ord('A') + latent_idx)}{sub_idx + 1}"
        return f"L{latent_idx + 1}_{sub_idx + 1}"

    def _selector_from_subpattern_label(label):
        if not isinstance(label, str) or len(label) < 2:
            raise ValueError("sub_pattern must be a string like 'A1' or a tuple (latent_idx, sub_idx).")

        prefix = label[0].upper()
        if not ("A" <= prefix <= "Z"):
            raise ValueError("String sub_pattern must start with a letter, for example 'A1'.")

        try:
            sub_idx = int(label[1:]) - 1
        except ValueError as exc:
            raise ValueError("String sub_pattern must end with a positive integer, for example 'A1'.") from exc

        latent_idx = ord(prefix) - ord("A")
        return latent_idx, sub_idx

    def _parse_episode_label(label):
        if not isinstance(label, str):
            raise ValueError(
                "For semantic_charge=2, sub_pattern must be a string like 'A1_B5' or a tuple matching a full label."
            )

        tokens = label.replace("-", "_").split("_")
        if len(tokens) != latent_space.num_latents:
            raise ValueError(
                f"Expected {latent_space.num_latents} latent tokens in label {label!r}, got {len(tokens)}."
            )

        parsed = []
        for latent_idx, token in enumerate(tokens):
            expected_prefix = chr(ord("A") + latent_idx) if latent_idx < 26 else f"L{latent_idx + 1}"
            token_upper = token.upper()

            if latent_idx < 26:
                if not token_upper.startswith(expected_prefix):
                    raise ValueError(
                        f"Expected token {latent_idx + 1} in {label!r} to start with {expected_prefix!r}."
                    )
                token_value = token_upper[len(expected_prefix):]
            else:
                if not token_upper.startswith(expected_prefix.upper() + "_"):
                    raise ValueError(
                        f"Expected token {latent_idx + 1} in {label!r} to start with {expected_prefix + '_'}."
                    )
                token_value = token_upper[len(expected_prefix) + 1:]

            try:
                sub_idx = int(token_value) - 1
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse latent index from token {token!r} in episode label {label!r}."
                ) from exc

            parsed.append(sub_idx)

        return tuple(parsed)

    if semantic_charge == 1:
        if sub_pattern is None:
            selected = None
        elif isinstance(sub_pattern, tuple) and len(sub_pattern) == 2:
            selected = (int(sub_pattern[0]), int(sub_pattern[1]))
        else:
            selected = _selector_from_subpattern_label(sub_pattern)

        prototypes = []
        labels = []
        start_idx = 0

        for latent_idx, (latent_dim, latent_size) in enumerate(zip(latent_space.dims, latent_space.total_sizes)):
            end_idx = start_idx + latent_size
            for sub_idx in range(latent_dim):
                if selected is not None and (latent_idx, sub_idx) != selected:
                    continue

                prototype = torch.zeros(
                    latent_space.total_size,
                    dtype=latent_space.latent_patterns[latent_idx][sub_idx].dtype,
                    device=latent_space.latent_patterns[latent_idx][sub_idx].device,
                )
                prototype[start_idx:end_idx] = latent_space.latent_patterns[latent_idx][sub_idx]
                prototypes.append(prototype)
                labels.append(_label_for(latent_idx, sub_idx))

            start_idx = end_idx

    elif semantic_charge == 2:
        if sub_pattern is None:
            selected = None
        elif isinstance(sub_pattern, tuple) and len(sub_pattern) == latent_space.num_latents:
            selected = tuple(int(idx) for idx in sub_pattern)
        else:
            selected = _parse_episode_label(sub_pattern)

        prototypes = []
        labels = []
        for label in latent_space.index_to_label:
            if selected is not None and tuple(label) != selected:
                continue
            prototypes.append(latent_space.label_to_neurons[tuple(label)].clone())
            labels.append("_".join(_label_for(latent_idx, sub_idx) for latent_idx, sub_idx in enumerate(label)))

    else:
        raise ValueError("semantic_charge must be 1 or 2.")

    if len(prototypes) == 0:
        raise ValueError(
            f"Requested selector {sub_pattern!r} is not available for semantic_charge={semantic_charge} "
            f"and latent dims {tuple(latent_space.dims)}."
        )

    prototypes = torch.stack(prototypes, dim=0)
    if return_labels:
        return prototypes, labels
    return prototypes






def get_selectivity(recordings, latents):
    # Flatten time
    _, num_neurons = recordings.shape


    days, day_length, num_latents = latents.shape

    recordings_flat = recordings.reshape(-1, num_neurons).float()    # shape: (days * day_length, num_neurons)
    latents_flat = latents.reshape(-1, num_latents).float()    # shape: (days * day_length, num_latents)


    # Normalize (zero mean, unit variance)
    latents_norm = (latents_flat - latents_flat.mean(dim=0)) / latents_flat.std(dim=0)
    
    #recordings_norm = (recordings_flat - recordings_flat.mean(dim=0)) / recordings_flat.std(dim=0) if recordings_flat.std(dim=0) != 0 else (recordings_flat - recordings_flat.mean(dim=0))
    recordings_norm = (recordings_flat - recordings_flat.mean(dim=0)) / recordings_flat.std(dim=0)


    # Compute correlation (selectivity): (num_neurons, num_latents)
    selectivity = recordings_norm.T @ latents_norm / latents_norm.shape[0]



    selectivity[torch.isnan(selectivity)] = 0


    return selectivity


def _mutual_information_discrete(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples.")
    if x.shape[0] == 0:
        return 0.0

    x_values, x_inverse = np.unique(x, return_inverse=True)
    y_values, y_inverse = np.unique(y, return_inverse=True)

    joint = np.zeros((len(x_values), len(y_values)), dtype=float)
    np.add.at(joint, (x_inverse, y_inverse), 1.0)
    joint /= float(x.shape[0])

    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    independent = px @ py

    mask = joint > 0
    mi_bits = np.sum(joint[mask] * np.log2(joint[mask] / independent[mask]))
    return float(mi_bits)


def _entropy_discrete(x):
    x = np.asarray(x).reshape(-1)
    if x.shape[0] == 0:
        return 0.0
    values, counts = np.unique(x, return_counts=True)
    probs = counts.astype(float) / float(x.shape[0])
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def MI_trajectory_per_latent(recordings, latent_binary, window_size, stride, normalize=True):
    x = torch.as_tensor(recordings).detach().cpu().numpy()
    z = torch.as_tensor(latent_binary).detach().cpu().numpy().reshape(-1)

    if x.shape[0] != z.shape[0]:
        raise ValueError("recordings and latent_binary must have same length.")

    if normalize is True:
        normalize = "window"
    if normalize is False:
        normalize = None
    if normalize not in [None, "window", "global"]:
        raise ValueError("normalize must be True, False, 'window', or 'global'.")

    starts = range(0, x.shape[0] - int(window_size) + 1, int(stride))
    mi = np.full((x.shape[1], len(list(starts))), np.nan, dtype=np.float32)
    starts = range(0, x.shape[0] - int(window_size) + 1, int(stride))
    h_global = _entropy_discrete(z)

    for w, start in enumerate(starts):
        end = start + int(window_size)
        z_w = z[start:end]
        h = _entropy_discrete(z_w)
        if normalize == "window" and h == 0:
            continue
        if normalize == "global" and h_global == 0:
            continue
        for n in range(x.shape[1]):
            v = _mutual_information_discrete(x[start:end, n], z_w)
            if normalize == "window":
                mi[n, w] = v / h
            elif normalize == "global":
                mi[n, w] = v / h_global
            else:
                mi[n, w] = v

    return torch.tensor(mi, dtype=torch.float32)


def get_binary_latents(input_params, input_latents, semantic_load=1, input_episodes=None):
    dims = input_params["latent_space"].dims
    x = torch.as_tensor(input_latents)

    if x.dim() == 2:
        x = x.unsqueeze(0)

    x = x.reshape(-1, x.shape[-1])
    binary_latents = {}

    if semantic_load == 1:
        for i in range(dims[0]):
            binary_latents[f"A_{i}"] = (x[:, 0] == i).float()

        for j in range(dims[1]):
            binary_latents[f"B_{j}"] = (x[:, 1] == j).float()

    elif semantic_load == 2:
        if input_episodes is None:
            raise ValueError("input_episodes is required when semantic_load=2.")
        episodes = torch.as_tensor(input_episodes).reshape(-1)
        index_to_label = input_params["latent_space"].index_to_label
        for episode_idx, label in enumerate(index_to_label):
            binary_latents[f"A_{label[0]}_B_{label[1]}"] = (episodes == episode_idx).float()
    else:
        raise ValueError("semantic_load must be 1 or 2.")

    return binary_latents


def MI_learning_curve(recordings, ordered_indices, assembly_size, input_latents, input_params, window_size, stride, normalize=True, semantic_load=1, input_episodes=None):
    x = torch.as_tensor(recordings)
    ind = torch.as_tensor(ordered_indices).long()
    binary_latents = get_binary_latents(
        input_params,
        input_latents,
        semantic_load=semantic_load,
        input_episodes=input_episodes,
    )

    curves = []
    for k, latent_name in enumerate(binary_latents.keys()):
        latent_binary = binary_latents[latent_name]
        assembly = ind[k * int(assembly_size):(k + 1) * int(assembly_size)]
        mi = MI_trajectory_per_latent(x[:, assembly], latent_binary, window_size, stride, normalize=normalize)
        curves.append(mi.mean(dim=0))

    return torch.stack(curves, dim=0)


def get_mutual_information_most_selective_latent(recordings, latents, selectivity_threshold=0.75):
    """
    Compute MI (in bits) between each neuron's activity and the latent feature
    to which that neuron is maximally selective, keeping only neurons with
    max selectivity >= selectivity_threshold.

    Args:
        recordings: tensor/array shaped (T, N) or (D, T, N)
        latents: tensor/array shaped (T, L) or (D, T, L)
        selectivity_threshold: float threshold on per-neuron max selectivity

    Returns:
        dict with:
            - mutual_information: (K,) tensor of MI values for selected neurons
            - selected_neuron_indices: (K,) tensor of original neuron indices
            - selected_latent_indices: (K,) tensor of latent indices per neuron
            - selected_max_selectivity: (K,) tensor of max selectivity values
            - selectivity: (N, L) tensor of all selectivities
            - max_selectivity: (N,) tensor of max selectivities
    """
    recordings_tensor = torch.as_tensor(recordings)
    latents_tensor = torch.as_tensor(latents)

    if recordings_tensor.dim() == 3:
        recordings_for_selectivity = recordings_tensor.reshape(-1, recordings_tensor.shape[-1])
    elif recordings_tensor.dim() == 2:
        recordings_for_selectivity = recordings_tensor
    else:
        raise ValueError("recordings must be 2D or 3D.")

    if latents_tensor.dim() == 2:
        latents_for_selectivity = latents_tensor.unsqueeze(0)
    elif latents_tensor.dim() == 3:
        latents_for_selectivity = latents_tensor
    else:
        raise ValueError("latents must be 2D or 3D.")

    if recordings_for_selectivity.shape[0] == 0:
        raise ValueError("recordings must contain at least one sample.")

    num_recording_samples = int(recordings_for_selectivity.shape[0])
    num_latent_samples = int(latents_for_selectivity.shape[0] * latents_for_selectivity.shape[1])
    if num_recording_samples != num_latent_samples:
        raise ValueError(
            "recordings and latents must contain the same number of samples after flattening. "
            f"Got recordings={num_recording_samples}, latents={num_latent_samples}."
        )

    selectivity = get_selectivity(recordings_for_selectivity, latents_for_selectivity)
    max_selectivity, selected_latent_indices_all = torch.max(selectivity, dim=1)
    selected_neuron_mask = max_selectivity >= float(selectivity_threshold)
    selected_neuron_indices = torch.nonzero(selected_neuron_mask, as_tuple=True)[0]

    # Flatten sample axes as in get_selectivity.
    recordings_flat = recordings_for_selectivity.reshape(-1, recordings_for_selectivity.shape[-1])
    latents_flat = latents_for_selectivity.reshape(-1, latents_for_selectivity.shape[-1])

    # Neural activity is binary in this model family; keep robust binarization.
    recordings_binary = (recordings_flat > 0).int().cpu().numpy()
    latents_values = latents_flat.cpu().numpy()

    selected_latent_indices = selected_latent_indices_all[selected_neuron_indices]
    selected_max_selectivity = max_selectivity[selected_neuron_indices]

    mutual_information_values = []
    latent_entropy_values = []
    normalized_mutual_information_values = []
    for neuron_idx, latent_idx in zip(selected_neuron_indices.tolist(), selected_latent_indices.tolist()):
        x = recordings_binary[:, neuron_idx]
        y = latents_values[:, latent_idx]
        mi_bits = _mutual_information_discrete(x, y)
        h_y_bits = _entropy_discrete(y)
        normalized_mi = mi_bits / h_y_bits if h_y_bits > 0 else 0.0
        mutual_information_values.append(mi_bits)
        latent_entropy_values.append(h_y_bits)
        normalized_mutual_information_values.append(normalized_mi)

    if len(mutual_information_values) == 0:
        mutual_information = torch.zeros(0, dtype=torch.float32)
        latent_entropy = torch.zeros(0, dtype=torch.float32)
        normalized_mutual_information = torch.zeros(0, dtype=torch.float32)
    else:
        mutual_information = torch.tensor(mutual_information_values, dtype=torch.float32)
        latent_entropy = torch.tensor(latent_entropy_values, dtype=torch.float32)
        normalized_mutual_information = torch.tensor(
            normalized_mutual_information_values, dtype=torch.float32
        )

    return {
        "mutual_information": mutual_information,
        "latent_entropy": latent_entropy,
        "normalized_mutual_information": normalized_mutual_information,
        "selected_neuron_indices": selected_neuron_indices,
        "selected_latent_indices": selected_latent_indices,
        "selected_max_selectivity": selected_max_selectivity,
        "selectivity": selectivity,
        "max_selectivity": max_selectivity,
    }




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


def get_cos_sim_matrix_torch(A, B, eps=1e-12):
  """
  Compute the cosine similarity matrix between two sets of vectors.

  Args:
      A: tensor of shape ``(n, l)`` or ``(l,)``
      B: tensor of shape ``(m, l)`` or ``(l,)``
      eps: numerical stability constant for normalization

  Returns:
      Tensor of shape ``(n, m)`` with cosine similarities across all pairs.
  """
  A = torch.as_tensor(A).float()
  B = torch.as_tensor(B).float()

  if A.dim() == 1:
    A = A.unsqueeze(0)
  if B.dim() == 1:
    B = B.unsqueeze(0)

  if A.dim() != 2 or B.dim() != 2:
    raise ValueError("A and B must be 1D or 2D tensors.")
  if A.shape[1] != B.shape[1]:
    raise ValueError(
      f"A and B must have the same feature dimension. Got {A.shape[1]} and {B.shape[1]}."
    )

  A_norm = F.normalize(A, p=2, dim=1, eps=eps)
  B_norm = F.normalize(B, p=2, dim=1, eps=eps)
  return A_norm @ B_norm.T


def get_max_overlap(A, B, return_matrix=False, return_indices=False, eps=1e-12):
  """
  Compute row-wise maximum cosine overlap between ``A`` and ``B``.

  Args:
      A: tensor of shape ``(n, l)`` or ``(l,)``
      B: tensor of shape ``(m, l)`` or ``(l,)``
      return_matrix: if ``True``, return the full cosine similarity matrix
      return_indices: if ``True``, also return the argmax index in ``B`` for
          each row of ``A``
      eps: numerical stability constant for normalization

  Returns:
      If ``return_matrix`` is ``True``, returns the cosine similarity matrix of
      shape ``(n, m)``.

      Otherwise returns row-wise max cosine similarities of shape ``(n,)``.
      If the input ``A`` was 1D, returns a scalar tensor. When
      ``return_indices`` is also ``True``, returns ``(max_vals, max_indices)``.
  """
  A_tensor = torch.as_tensor(A)
  A_was_vector = A_tensor.dim() == 1
  cos_sim = get_cos_sim_matrix_torch(A, B, eps=eps)

  if return_matrix:
    return cos_sim

  max_vals, max_indices = cos_sim.max(dim=1)

  if A_was_vector:
    max_vals = max_vals.squeeze(0)
    max_indices = max_indices.squeeze(0)

  if return_indices:
    return max_vals, max_indices
  return max_vals



def get_cond_matrix(latent_space, weights, eta):
  num_subs = len(latent_space.sub_index_to_neuron_index)
  sim_cond_matrix = np.zeros((num_subs, num_subs))
  th_cond_matrix = np.zeros((num_subs, num_subs))
  for conditioned_sub_index, ((conditioned_latent, conditioned_sub), conditioned_neuron_index) in enumerate(zip(latent_space.sub_index_to_latent_sub, latent_space.sub_index_to_neuron_index)):
    for condition_sub_index, ((condition_latent, condition_sub), condition_neuron_index) in  enumerate(zip(latent_space.sub_index_to_latent_sub, latent_space.sub_index_to_neuron_index)):
      if conditioned_sub != condition_sub:
        sim_cond_matrix[conditioned_sub_index][condition_sub_index] = np.mean(weights[conditioned_neuron_index][:, condition_neuron_index])
      else:
        sim_cond_matrix[conditioned_sub_index][condition_sub_index] = np.mean(weights[conditioned_neuron_index][:, condition_neuron_index])
      if conditioned_latent != condition_latent:
        label = [0, 0]
        label[conditioned_latent] = conditioned_sub
        label[condition_latent] = condition_sub
        try:
          th_cond_matrix[conditioned_sub_index][condition_sub_index] = latent_space.label_to_probs[tuple(label)]/(eta*latent_space.sub_index_to_marginal[condition_sub_index] + (1 - eta)*latent_space.sub_index_to_marginal[conditioned_sub_index])
        except:
          th_cond_matrix[conditioned_sub_index][condition_sub_index] = 0

      elif conditioned_sub == condition_sub:
        th_cond_matrix[conditioned_sub_index][condition_sub_index] = 1

      else:
        th_cond_matrix[conditioned_sub_index][condition_sub_index] = 0
  return sim_cond_matrix, th_cond_matrix
