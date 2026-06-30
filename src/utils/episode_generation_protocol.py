import itertools
import random
from collections import OrderedDict

import numpy as np
import torch


def make_input(
    num_days,
    day_length,
    mean_duration,
    fixed_duration,
    num_swaps,
    latent_space,
    regions=None,
    satellite=False,
):
    from src.utils.general import get_sample_from_num_swaps

    input_tensor = torch.zeros((num_days, day_length, latent_space.total_size))
    input_episodes = torch.zeros((num_days, day_length), dtype=torch.int32)
    input_latents = torch.zeros(
        (num_days, day_length, latent_space.num_latents),
        dtype=torch.int32,
    )

    for day in range(num_days):
        day_timestep = 0
        while day_timestep < day_length:
            pattern_duration = (
                mean_duration
                if fixed_duration
                else int(torch.poisson(mean_duration * torch.ones(1))[0])
            )
            if satellite:
                latent_index, pattern = latent_space.sample()
                input_latents[day, day_timestep : day_timestep + pattern_duration] = latent_index
            else:
                label, pattern = latent_space.sample()
                input_episodes[day, day_timestep : day_timestep + pattern_duration] = (
                    latent_space.label_to_index[label]
                )
                input_latents[day, day_timestep : day_timestep + pattern_duration, :] = (
                    torch.tensor(label, dtype=torch.int32)
                )

            for pattern_timestep in range(pattern_duration):
                input_tensor[
                    day,
                    (day_timestep + pattern_timestep) : (day_timestep + pattern_timestep + 1),
                ] = get_sample_from_num_swaps(pattern, num_swaps, regions)
            day_timestep += pattern_duration

    return input_tensor, input_episodes, input_latents


class LatentSpace:
    def __init__(self, num, total_sizes, act_sizes, dims, prob_list, random_neurons=False):
        self.num_latents = num
        self.dims = dims
        self.total_sizes = total_sizes
        self.total_size = sum(total_sizes)
        self.act_sizes = act_sizes
        self.random_neurons = random_neurons
        self.latent_patterns = [
            [self.get_sub_latent(latent, sub_dim) for sub_dim in range(self.dims[latent])]
            for latent in range(self.num_latents)
        ]
        self.sub_index_to_neuron_index = [
            self.latent_patterns[latent][sub_dim].nonzero().squeeze(1).detach().numpy()
            + sum(self.total_sizes[:latent])
            for latent in range(self.num_latents)
            for sub_dim in range(self.dims[latent])
        ]
        self.sub_index_to_latent_sub = [
            (latent, sub_dim)
            for latent in range(self.num_latents)
            for sub_dim in range(self.dims[latent])
        ]
        self.index_to_label = list(
            itertools.product(*[[i for i in range(dim)] for dim in self.dims])
        )
        self.label_to_index = {
            label: index for index, label in enumerate(self.index_to_label)
        }
        self.label_to_neurons = OrderedDict(
            {label: self.get_neurons_from_label(label) for label in self.index_to_label}
        )
        self.label_to_probs = OrderedDict(
            {label: prob for label, prob in zip(self.index_to_label, prob_list)}
        )
        self.sub_index_to_marginal = [
            self.get_marginal(latent, sub)
            for (latent, sub) in self.sub_index_to_latent_sub
        ]

    def get_sub_latent(self, latent, sub_dim):
        if self.random_neurons:
            sub_latent = torch.zeros((self.total_sizes[latent]))
            sub_latent[
                torch.randperm(self.total_sizes[latent])[: self.act_sizes[latent]]
            ] = 1
        else:
            sub_latent = torch.zeros((self.total_sizes[latent]))
            start = sub_dim * self.act_sizes[latent]
            end = (sub_dim + 1) * self.act_sizes[latent]
            sub_latent[start:end] = 1
        return sub_latent

    def get_neurons_from_label(self, label):
        return torch.cat(
            tuple(self.latent_patterns[latent][index] for latent, index in enumerate(label))
        )

    def get_marginal(self, latent, sub):
        marginal = 0
        for label, prob in self.label_to_probs.items():
            if label[latent] == sub:
                marginal += prob
        return marginal

    def sample(self):
        return random.choices(
            list(self.label_to_neurons.items()),
            weights=list(self.label_to_probs.values()),
        )[0]


def get_prototypes(
    latent_space,
    sub_pattern=None,
    semantic_load=1,
    return_labels=False,
):
    """
    Return full neural sheets for latent prototypes.

    For ``semantic_load=1``, each returned row contains exactly one sub-pattern
    (for example ``A1`` or ``B5``), with all other latent blocks set to zero.
    For ``semantic_load=2``, each returned row contains one full episode
    prototype (for example ``A1_B5``).

    Args:
        latent_space: ``LatentSpace`` instance.
        sub_pattern: optional selector. If ``None``, returns all prototypes for
            the requested semantic load. If provided:
            - for ``semantic_load=1`` it can be a string like ``"A1"`` or
              ``"B5"``, or a tuple ``(latent_idx, sub_idx)``
            - for ``semantic_load=2`` it can be a string like ``"A1_B5"``, or
              a full latent label tuple like ``(0, 4)``
        semantic_load: ``1`` for sub-pattern prototypes, ``2`` for full
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
                "For semantic_load=2, sub_pattern must be a string like 'A1_B5' or a tuple matching a full label."
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

    if semantic_load == 1:
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

    elif semantic_load == 2:
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
        raise ValueError("semantic_load must be 1 or 2.")

    if len(prototypes) == 0:
        raise ValueError(
            f"Requested selector {sub_pattern!r} is not available for semantic_load={semantic_load} "
            f"and latent dims {tuple(latent_space.dims)}."
        )

    prototypes = torch.stack(prototypes, dim=0)
    if return_labels:
        return prototypes, labels
    return prototypes


def get_cond_matrix(latent_space, weights, eta):
    num_subs = len(latent_space.sub_index_to_neuron_index)
    sim_cond_matrix = np.zeros((num_subs, num_subs))
    th_cond_matrix = np.zeros((num_subs, num_subs))

    for conditioned_sub_index, (
        (conditioned_latent, conditioned_sub),
        conditioned_neuron_index,
    ) in enumerate(
        zip(
            latent_space.sub_index_to_latent_sub,
            latent_space.sub_index_to_neuron_index,
        )
    ):
        for condition_sub_index, (
            (condition_latent, condition_sub),
            condition_neuron_index,
        ) in enumerate(
            zip(
                latent_space.sub_index_to_latent_sub,
                latent_space.sub_index_to_neuron_index,
            )
        ):
            sim_cond_matrix[conditioned_sub_index][condition_sub_index] = np.mean(
                weights[conditioned_neuron_index][:, condition_neuron_index]
            )

            if conditioned_latent != condition_latent:
                label = [0, 0]
                label[conditioned_latent] = conditioned_sub
                label[condition_latent] = condition_sub
                try:
                    numerator = latent_space.label_to_probs[tuple(label)]
                    denominator = (
                        eta * latent_space.sub_index_to_marginal[condition_sub_index]
                        + (1 - eta) * latent_space.sub_index_to_marginal[conditioned_sub_index]
                    )
                    th_cond_matrix[conditioned_sub_index][condition_sub_index] = (
                        numerator / denominator
                    )
                except Exception:
                    th_cond_matrix[conditioned_sub_index][condition_sub_index] = 0
            elif conditioned_sub == condition_sub:
                th_cond_matrix[conditioned_sub_index][condition_sub_index] = 1
            else:
                th_cond_matrix[conditioned_sub_index][condition_sub_index] = 0

    return sim_cond_matrix, th_cond_matrix
