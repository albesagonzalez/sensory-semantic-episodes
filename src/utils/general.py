import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset

from src.utils.episode_generation_protocol import (
    LatentSpace,
    make_input,
)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_selectivity(recordings, latents, debug_label=None, chunk_size=64):
    recordings_tensor = torch.as_tensor(recordings).float()
    latents_tensor = torch.as_tensor(latents).float()

    recordings_flat = recordings_tensor.reshape(-1, recordings_tensor.shape[-1])
    latents_flat = latents_tensor.reshape(-1, latents_tensor.shape[-1])

    latents_centered = latents_flat - latents_flat.mean(dim=0, keepdim=True)
    recordings_centered = recordings_flat - recordings_flat.mean(dim=0, keepdim=True)

    eps = 1e-8
    latents_scale = torch.sqrt((latents_centered.pow(2).mean(dim=0, keepdim=True)).clamp_min(eps))
    recordings_scale = torch.sqrt((recordings_centered.pow(2).mean(dim=0, keepdim=True)).clamp_min(eps))
    latents_norm = latents_centered / latents_scale
    recordings_norm = recordings_centered / recordings_scale

    num_neurons = int(recordings_norm.shape[1])
    selectivity = torch.empty(
        (num_neurons, int(latents_norm.shape[1])),
        dtype=recordings_norm.dtype,
        device=recordings_norm.device,
    )
    for start in range(0, num_neurons, int(chunk_size)):
        end = min(start + int(chunk_size), num_neurons)
        selectivity[start:end] = (
            recordings_norm[:, start:end].T @ latents_norm / latents_norm.shape[0]
        )
    selectivity[torch.isnan(selectivity)] = 0

    return selectivity


def get_ordered_indices(
    recordings,
    latents,
    assembly_size,
    seed=None,
    debug_label=None,
    assemblies_only=True,
):
    selectivity = get_selectivity(recordings, latents, debug_label=debug_label)
    N, L = selectivity.shape
    if N < L * int(assembly_size):
        raise ValueError(
            f"Not enough neurons ({N}) to assign {assembly_size} neurons to each of {L} latents."
        )

    generator = None
    if seed is not None:
        generator = torch.Generator(device=selectivity.device)
        generator.manual_seed(int(seed))

    assemblies = [[] for _ in range(L)]
    available = torch.ones(N, dtype=torch.bool, device=selectivity.device)

    for round_idx in range(int(assembly_size)):
        latent_order = torch.randperm(L, generator=generator, device=selectivity.device)
        available_indices = torch.nonzero(available, as_tuple=False).flatten()
        if available_indices.numel() < L:
            raise RuntimeError(
                "Available neuron pool was exhausted before all assemblies were filled."
            )

        scores = selectivity[available_indices][:, latent_order]
        claimed_positions = []
        claimed_neurons = []

        for order_idx in range(L):
            latent_scores = scores[:, order_idx]
            if claimed_positions:
                latent_scores = latent_scores.clone()
                latent_scores[torch.tensor(claimed_positions, device=latent_scores.device)] = -torch.inf
            best_pos = torch.argmax(latent_scores)
            claimed_positions.append(int(best_pos.item()))
            claimed_neurons.append(int(available_indices[best_pos].item()))

        for latent_idx, neuron_idx in zip(latent_order.tolist(), claimed_neurons):
            assemblies[latent_idx].append(neuron_idx)
            available[neuron_idx] = False

    flat_indices = [neuron for assembly in assemblies for neuron in assembly]
    if not assemblies_only:
        leftover_neurons = torch.nonzero(available, as_tuple=False).flatten().tolist()
        flat_indices.extend(leftover_neurons)
    return selectivity, torch.tensor(flat_indices, dtype=torch.long)


def get_signal_to_noise_ratio(
    num_swaps,
    network,
    region: str = "mtl",
    return_per_subregion: bool = False,
    sleep: bool = False,
):
    num_subregions = int(getattr(network, f"{region}_num_subregions"))
    size_subregions = torch.as_tensor(getattr(network, f"{region}_size_subregions")).detach().cpu().float()
    sparsity_attr = f"{region}_sparsity_sleep" if sleep else f"{region}_sparsity"
    sparsity = torch.as_tensor(getattr(network, sparsity_attr)).detach().cpu().float()
    total_size = float(
        getattr(
            network,
            f"{region}_size",
            torch.as_tensor(size_subregions).detach().cpu().float().sum().item(),
        )
    )

    if isinstance(num_swaps, (list, tuple, np.ndarray, torch.Tensor)):
        num_swaps_per_subregion = [float(v) for v in list(num_swaps)]
        if len(num_swaps_per_subregion) != num_subregions:
            raise ValueError(
                f"Expected one swap count per {region} subregion "
                f"({num_subregions}), got {len(num_swaps_per_subregion)}."
            )
        explicit_per_subregion = True
    else:
        num_swaps_value = float(num_swaps)
        if num_swaps_value == 0:
            mean_snr = float("inf")
            if return_per_subregion:
                return mean_snr, [float("inf")] * num_subregions
            return mean_snr
        num_swaps_per_subregion = [
            float(torch.round(torch.tensor(num_swaps_value * N / total_size)).item())
            for N in size_subregions
        ]
        explicit_per_subregion = False

    snr_list = []
    for subregion_index in range(num_subregions):
        N = float(size_subregions[subregion_index].item())
        K = float((size_subregions[subregion_index] * sparsity[subregion_index]).item())
        num_swaps_region = float(num_swaps_per_subregion[subregion_index])

        if num_swaps_region == 0:
            snr_list.append(float("inf"))
            continue

        signal = (K - num_swaps_region) - (K ** 2 / N)
        noise = 2.0 * num_swaps_region
        snr = float((signal / noise) ** 2)
        snr_list.append(snr)

    if explicit_per_subregion:
        return snr_list

    mean_snr = float(np.mean(snr_list))
    if return_per_subregion:
        return mean_snr, snr_list
    return mean_snr


def get_sample_from_num_swaps(x_0, num_swaps):
    x = x_0.clone().detach()
    on_index = x_0.nonzero().squeeze(1)
    off_index = (x_0 == 0).nonzero().squeeze(1)
    num_swaps = int(num_swaps)
    flip_off = on_index[torch.randperm(len(on_index))[:num_swaps]]
    flip_on = off_index[torch.randperm(len(off_index))[:num_swaps]]
    x[flip_off] = 0
    x[flip_on] = 1
    return x



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
    """Compute full-recording NMI for each selected neuron's preferred concept.

    Selectivity determines the preferred concept for each neuron.  The NMI
    calculation itself reuses ``MI_trajectory_per_latent`` with one window that
    spans the complete recording, so this assay and the learning trajectories
    share the same empirical MI implementation.
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

    recordings_flat = recordings_for_selectivity.reshape(-1, recordings_for_selectivity.shape[-1])
    latents_flat = latents_for_selectivity.reshape(-1, latents_for_selectivity.shape[-1])

    recordings_binary = (recordings_flat > 0).int().cpu()
    latents_flat_cpu = latents_flat.detach().cpu()

    selected_latent_indices = selected_latent_indices_all[selected_neuron_indices]
    selected_max_selectivity = max_selectivity[selected_neuron_indices]

    if selected_neuron_indices.numel() == 0:
        mutual_information = torch.zeros(0, dtype=torch.float32)
        latent_entropy = torch.zeros(0, dtype=torch.float32)
        normalized_mutual_information = torch.zeros(0, dtype=torch.float32)
    else:
        selected_neuron_indices_cpu = selected_neuron_indices.detach().cpu()
        selected_latent_indices_cpu = selected_latent_indices.detach().cpu()
        num_selected = selected_neuron_indices_cpu.numel()
        mutual_information = torch.zeros(num_selected, dtype=torch.float32)
        latent_entropy = torch.zeros(num_selected, dtype=torch.float32)
        normalized_mutual_information = torch.zeros(num_selected, dtype=torch.float32)

        # Neurons can have different preferred concepts.  Grouping them by
        # concept lets each group share one call to the trajectory helper.
        for latent_idx in torch.unique(selected_latent_indices_cpu):
            positions = torch.nonzero(
                selected_latent_indices_cpu == latent_idx, as_tuple=True
            )[0]
            neuron_indices = selected_neuron_indices_cpu[positions]
            target = latents_flat_cpu[:, int(latent_idx.item())]
            target_entropy = _entropy_discrete(target.numpy())

            latent_entropy[positions] = target_entropy
            if target_entropy == 0:
                continue

            full_recording_nmi = MI_trajectory_per_latent(
                recordings_binary[:, neuron_indices],
                target,
                window_size=recordings_binary.shape[0],
                stride=recordings_binary.shape[0],
                normalize="global",
            )[:, 0]
            full_recording_nmi = torch.nan_to_num(full_recording_nmi, nan=0.0)
            normalized_mutual_information[positions] = full_recording_nmi
            mutual_information[positions] = full_recording_nmi * target_entropy

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




def get_accuracy(recordings, latents, assembly_size):

    T, N = recordings.shape
    L = N//assembly_size
    
    recordings_grouped = recordings.view(T, L, assembly_size).mean(dim=2)
    
    pred_A = torch.argmax(recordings_grouped[:, :L//2], dim=1)
    pred_B = torch.argmax(recordings_grouped[:, L//2:], dim=1)

    acc_A = (pred_A == latents[:, 0]).float().mean()
    acc_B = (pred_B == latents[:, 1]).float().mean()

    accuracies = torch.tensor([acc_A, acc_B], device=recordings.device)

    return accuracies



def get_cos_sim_torch(x1, x2):
  return torch.dot(x1, x2)/(torch.norm(x1)*torch.norm(x2))
def get_cos_sim_np(x1, x2):
  return np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))


def get_cos_sim_matrix_torch(A, B, eps=1e-12):
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



def train_network(
  net,
  input_params,
  sleep=True,
  print_rate=1,
  true_latent_to_mtl_semantic=False,
  scrambled=False,
):
  input, input_episodes, input_latents = make_input(**input_params)
  permutation = None
  if scrambled:
    permutation = torch.randperm(net.sen_size)
  should_print = print_rate not in [None, np.inf]
  with torch.no_grad():
    for day in range(input_params["num_days"]):
      if should_print and day % int(print_rate) == 0:
        print(day)
      latent_day = input_latents[day] if true_latent_to_mtl_semantic else None
      day_input = input[day] if permutation is None else input[day, :, permutation]
      net(day_input, debug=False, true_latent=latent_day)
      if sleep:
        net.sleep()
  return input, input_episodes, input_latents, net
