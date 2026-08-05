from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from src.model import SSCNetwork
from src.utils.episode_generation_protocol import LatentSpace, get_prototypes
from src.utils.general import (
    get_accuracy,
    get_ordered_indices,
    get_signal_to_noise_ratio,
    seed_everything,
    train_network,
)


def _make_recording_params(regions=None):
    # Record only the requested activity; Figure 6B records replay in bounded chunks.
    return {
        "regions": [] if regions is None else list(regions),
        "rate_activity": 1 if regions else np.inf,
        "connections": [],
        "rate_connectivity": np.inf,
    }


def _make_input_params(base_input_params, latent_specs, num_days=None, num_swaps=None):
    # Build an independent input configuration for one training or evaluation stage.
    input_params = deepcopy(base_input_params)
    if num_days is not None:
        input_params["num_days"] = int(num_days)
    if num_swaps is not None:
        input_params["num_swaps"] = int(num_swaps)
    input_params["latent_space"] = LatentSpace(**deepcopy(latent_specs))
    return input_params


def get_ordered_accuracy(recordings, binary_latents, labels, assembly_size, num_groups=None):
    # Fit concept-selective cell groups on the first half of samples and decode the second.
    recordings = torch.as_tensor(recordings).float()
    binary_latents = torch.as_tensor(binary_latents).reshape(
        -1, torch.as_tensor(binary_latents).shape[-1]
    ).float()
    labels = torch.as_tensor(labels)

    fit_num_samples = recordings.shape[0] // 2
    test_num_samples = recordings.shape[0] - fit_num_samples
    selectivity, ordered_indices = get_ordered_indices(
        recordings[:fit_num_samples],
        binary_latents[:fit_num_samples],
        assembly_size=assembly_size,
    )
    if num_groups is None:
        num_groups = binary_latents.shape[-1]
    ordered_test = recordings[fit_num_samples:, ordered_indices[: num_groups * assembly_size]]
    labels_test = labels[fit_num_samples:]

    if labels_test.ndim == 1:
        grouped = ordered_test.reshape(-1, num_groups, assembly_size).mean(dim=2)
        accuracy = (grouped.argmax(dim=1) == labels_test.long().reshape(-1)).float().mean()
    else:
        accuracy = get_accuracy(ordered_test, labels_test, assembly_size=assembly_size)

    return {
        "accuracy": accuracy,
        "fit_num_samples": int(fit_num_samples),
        "test_num_samples": int(test_num_samples),
        "selectivity": selectivity,
        "ordered_indices": ordered_indices,
    }


def _get_replay_pattern_snr(replayed_patterns, prototypes, inferred_indices, network):
    # Express replay corruption as the equivalent sensory-input swap count and SNR.
    replayed_patterns = torch.as_tensor(replayed_patterns).float()
    prototypes = torch.as_tensor(prototypes).float()
    inferred_indices = torch.as_tensor(inferred_indices).long()
    closest_prototypes = prototypes[inferred_indices]
    hamming = (replayed_patterns != closest_prototypes).sum(dim=1)
    equivalent_num_swaps = torch.div(hamming, 2, rounding_mode="floor")
    mean_num_swaps = float(equivalent_num_swaps.float().mean().item())
    mean_num_swaps_floor = int(np.floor(mean_num_swaps))
    return {
        "mean_closest_prototype_num_swaps": mean_num_swaps,
        "mean_closest_prototype_snr": float(
            get_signal_to_noise_ratio(
                mean_num_swaps_floor,
                network,
                region="mtl_sensory",
            )
        ),
    }


def _update_replay_reservoir(
    reservoir,
    event_id_reservoir,
    num_seen,
    patterns,
    event_ids,
    capacity,
):
    # Maintain a uniform reservoir of replay events while retaining their unique IDs.
    if reservoir is None:
        reservoir = patterns.new_empty((capacity, patterns.shape[1]))
        event_id_reservoir = torch.empty(capacity, dtype=torch.long, device=patterns.device)
    for pattern, event_id in zip(patterns, event_ids):
        num_seen += 1
        if num_seen <= capacity:
            reservoir[num_seen - 1].copy_(pattern)
            event_id_reservoir[num_seen - 1] = event_id
            continue
        replacement_index = int(torch.randint(num_seen, (), device=patterns.device).item())
        if replacement_index < capacity:
            reservoir[replacement_index].copy_(pattern)
            event_id_reservoir[replacement_index] = event_id
    return reservoir, event_id_reservoir, num_seen


def _update_receptive_field_reservoirs(
    reservoirs,
    event_id_reservoirs,
    counts,
    replayed_patterns,
    label_patterns,
    event_ids,
    prototypes,
    labels,
    max_patterns_per_label,
):
    # Assign replay events to their nearest clean episode prototype and retain up to
    # ``max_patterns_per_label`` uniformly sampled candidate receptive fields per label.
    similarities = F.normalize(label_patterns.float(), dim=1) @ F.normalize(
        prototypes.float(), dim=1
    ).T
    winning_indices = similarities.argmax(dim=1)
    for prototype_index, label in enumerate(labels):
        for pattern, event_id in zip(
            replayed_patterns[winning_indices == prototype_index],
            event_ids[winning_indices == prototype_index],
        ):
            counts[label] += 1
            if counts[label] <= max_patterns_per_label:
                reservoirs[label][counts[label] - 1].copy_(pattern)
                event_id_reservoirs[label][counts[label] - 1] = event_id
                continue
            replacement_index = int(
                torch.randint(counts[label], (), device=pattern.device).item()
            )
            if replacement_index < max_patterns_per_label:
                reservoirs[label][replacement_index].copy_(pattern)
                event_id_reservoirs[label][replacement_index] = event_id


def get_pattern_labels_from_prototypes(patterns, prototypes, labels):
    # Label each replayed pattern by its nearest clean sensory episode prototype.
    patterns = torch.as_tensor(patterns)
    prototypes = torch.as_tensor(prototypes)
    flat_patterns = patterns.reshape(-1, patterns.shape[-1])
    similarities = F.normalize(flat_patterns.float(), dim=1) @ F.normalize(
        prototypes.float(), dim=1
    ).T
    inferred_indices = similarities.argmax(dim=1).tolist()
    return np.asarray([labels[index] for index in inferred_indices], dtype=object).reshape(
        patterns.shape[:-1]
    )


def get_rf_patterns_generalization(receptive_fields, patterns, pattern_labels):
    # For every evaluation day, draw one receptive field per available episode and
    # classify patterns by maximum cosine similarity to those sampled fields.
    patterns = torch.as_tensor(patterns)
    pattern_labels = np.asarray(pattern_labels, dtype=object)
    available_fields = {
        label: torch.as_tensor(fields)
        for label, fields in receptive_fields.items()
        if torch.as_tensor(fields).shape[0] > 0
    }
    labels = list(available_fields)
    label_to_index = {label: index for index, label in enumerate(labels)}
    targets = torch.tensor(
        [label_to_index.get(label, -1) for label in pattern_labels.flat],
        device=patterns.device,
    ).reshape(patterns.shape[:2])

    day_accuracies = []
    for day_patterns, day_targets in zip(patterns, targets):
        sampled_fields = torch.stack(
            [
                fields[torch.randint(fields.shape[0], (), device=fields.device)]
                for fields in available_fields.values()
            ]
        ).to(device=patterns.device, dtype=torch.float32)
        similarities = F.normalize(day_patterns.float(), dim=1) @ F.normalize(
            sampled_fields, dim=1
        ).T
        day_accuracies.append((similarities.argmax(dim=1) == day_targets).float().mean())
    return float(torch.stack(day_accuracies).mean().item())


def _get_ctx_simple_accuracy_from_frozen_eval_net(eval_net, eval_input_latents, latent_specs):
    # Decode the simple latent factors from the simple-concept CTX subregion.
    X_ctx = torch.stack(eval_net.activity_recordings["ctx"], dim=0)[eval_net.awake_indices]
    X_latent_A = F.one_hot(
        eval_input_latents[:, :, 0].long(),
        num_classes=latent_specs["dims"][0],
    )
    X_latent_B = F.one_hot(
        eval_input_latents[:, :, 1].long(),
        num_classes=latent_specs["dims"][1],
    )
    X_latent_AB = torch.cat((X_latent_A, X_latent_B), dim=2)
    results = get_ordered_accuracy(
        X_ctx[:, eval_net.ctx_subregions[0]],
        X_latent_AB,
        eval_input_latents.reshape(-1, eval_input_latents.shape[-1]),
        assembly_size=10,
    )
    return {
        "ctx_accuracy_A": float(results["accuracy"][0].item()),
        "ctx_accuracy_B": float(results["accuracy"][1].item()),
        "ctx_accuracy_mean": float(results["accuracy"].mean().item()),
        "fit_num_samples": results["fit_num_samples"],
        "test_num_samples": results["test_num_samples"],
    }


def _get_ctx_episode_accuracy_from_frozen_eval_net(eval_net, eval_input_episodes, latent_specs):
    # Decode complete episodes from the complex-concept CTX subregion.
    X_ctx = torch.stack(eval_net.activity_recordings["ctx"], dim=0)[eval_net.awake_indices]
    episode_results = get_ordered_accuracy(
        X_ctx[:, eval_net.ctx_subregions[1]],
        F.one_hot(
            eval_input_episodes.long(),
            num_classes=int(np.prod(latent_specs["dims"])),
        ),
        eval_input_episodes.reshape(-1),
        assembly_size=10,
        num_groups=int(np.prod(latent_specs["dims"])),
    )
    return {
        "ctx_episode_accuracy": float(episode_results["accuracy"].item()),
        "fit_num_samples": episode_results["fit_num_samples"],
        "test_num_samples": episode_results["test_num_samples"],
    }


def generalization_simple_complex(
    network_parameters,
    input_params,
    latent_specs,
    network_mode,
    num_swaps,
    pretrain_num_swaps=None,
    seed=None,
    phase_A=200,
    phase_B=400,
    simple_train_days=800,
    simple_eval_days=100,
    complex_train_days=400,
    complex_eval_days=100,
    complex_train_chunk_days=10,
    use_full_mtl=False,
    return_network=False,
    record=False,
):
    # Validate the requested semantic condition and make the run reproducible.
    if network_mode not in {"semantics_present", "semantics_random", "semantics_absent", "semantics_clean"}:
        raise ValueError("Unknown network mode.")
    if record:
        raise ValueError("Full recording is incompatible with this memory-bounded simulation.")
    if seed is not None:
        seed_everything(seed)

    posttrain_num_swaps = int(num_swaps)
    pretrain_num_swaps = posttrain_num_swaps if pretrain_num_swaps is None else int(pretrain_num_swaps)
    if int(complex_train_chunk_days) < 1:
        raise ValueError("complex_train_chunk_days must be at least 1.")

    # Initialise a fresh network and apply the manipulation defining each condition.
    net_params = deepcopy(network_parameters)
    net_params["duration_phase_A"] = int(phase_A)
    net_params["duration_phase_B"] = int(phase_B)
    network = SSCNetwork(net_params, _make_recording_params())
    if network_mode == "semantics_random":
        network.lesioned = {"mtl_semantic"}
    elif network_mode == "semantics_absent":
        network.sensory_replay_only = True
    use_true_latent_to_mtl_semantic = network_mode == "semantics_clean"

    # Pre-train simple concepts before introducing load-2 episodic replay.
    if int(simple_train_days):
        _, _, _, network = train_network(
            network,
            _make_input_params(
                input_params,
                latent_specs,
                num_days=simple_train_days,
                num_swaps=pretrain_num_swaps,
            ),
            sleep=True,
            print_rate=np.inf,
            true_latent_to_mtl_semantic=use_true_latent_to_mtl_semantic,
        )

    # Evaluate simple-concept CTX selectivity under frozen weights (returned for
    # completeness, although Figure 6B reports the complex-concept measures below).
    simple_eval_net = deepcopy(network)
    simple_eval_net.init_recordings(_make_recording_params(regions=["ctx"]))
    simple_eval_net.frozen = True
    _, _, simple_eval_latents, simple_eval_net = train_network(
        simple_eval_net,
        _make_input_params(
            input_params,
            latent_specs,
            num_days=simple_eval_days,
            num_swaps=posttrain_num_swaps,
        ),
        sleep=False,
        print_rate=np.inf,
        true_latent_to_mtl_semantic=use_true_latent_to_mtl_semantic,
    )
    simple_accuracy_results = _get_ctx_simple_accuracy_from_frozen_eval_net(
        simple_eval_net, simple_eval_latents, latent_specs
    )
    del simple_eval_net, simple_eval_latents

    # Continue training the same network with semantic load 2 available during Sleep A.
    complex_train_net = network
    complex_train_net.frozen = False
    complex_train_net.max_semantic_load_replay = 2
    complex_prototypes, complex_labels = get_prototypes(
        _make_input_params(input_params, latent_specs)["latent_space"],
        semantic_load=2,
        return_labels=True,
    )
    complex_labels = [label.replace("_", "") for label in complex_labels]
    max_patterns_per_label = 10
    # Store a small, uniform sample of load-2 replays per inferred episode as
    # candidate receptive fields. Event IDs later enforce a held-out replay test set.
    receptive_field_reservoirs = {
        label: complex_prototypes.new_empty((max_patterns_per_label, complex_prototypes.shape[1]))
        for label in complex_labels
    }
    receptive_field_event_id_reservoirs = {
        label: torch.empty(max_patterns_per_label, dtype=torch.long, device=complex_prototypes.device)
        for label in complex_labels
    }
    receptive_field_counts = {label: 0 for label in complex_labels}
    replayed_replayed_receptive_field_reservoirs = (
        {
            label: complex_prototypes.new_empty(
                (max_patterns_per_label, complex_train_net.mtl_size)
            )
            for label in complex_labels
        }
        if use_full_mtl
        else receptive_field_reservoirs
    )
    replayed_replayed_receptive_field_event_id_reservoirs = (
        {
            label: torch.empty(
                max_patterns_per_label,
                dtype=torch.long,
                device=complex_prototypes.device,
            )
            for label in complex_labels
        }
        if use_full_mtl
        else receptive_field_event_id_reservoirs
    )
    replayed_replayed_receptive_field_counts = (
        {label: 0 for label in complex_labels}
        if use_full_mtl
        else receptive_field_counts
    )
    replay_capacity = int(complex_eval_days) * int(input_params["day_length"])
    replay_reservoir, replay_event_id_reservoir, replay_reservoir_count = None, None, 0

    # Train the full complex phase in chunks only to bound recordings in memory; the
    # network itself is never reset between chunks.
    for day_start in range(0, int(complex_train_days), int(complex_train_chunk_days)):
        chunk_days = min(int(complex_train_chunk_days), int(complex_train_days) - day_start)
        replay_recording_regions = ["mtl_sensory", "mtl"] if use_full_mtl else ["mtl_sensory"]
        complex_train_net.init_recordings(_make_recording_params(regions=replay_recording_regions))
        _, _, _, complex_train_net = train_network(
            complex_train_net,
            _make_input_params(
                input_params,
                latent_specs,
                num_days=chunk_days,
                num_swaps=posttrain_num_swaps,
            ),
            sleep=True,
            print_rate=np.inf,
            true_latent_to_mtl_semantic=use_true_latent_to_mtl_semantic,
        )
        # Keep only the final half of Sleep A, where the schedule sets semantic load to 2.
        sleep_duration_a = int(complex_train_net.sleep_duration_A)
        sleep_a_sensory = torch.stack(complex_train_net.activity_recordings["mtl_sensory"])[
            complex_train_net.sleep_indices_A
        ]
        if sleep_a_sensory.shape[0] != chunk_days * sleep_duration_a:
            raise ValueError("Unexpected number of Sleep-A recordings.")
        replayed_sensory = sleep_a_sensory.reshape(chunk_days, sleep_duration_a, -1)[
            :, sleep_duration_a // 2 :, :
        ].reshape(-1, sleep_a_sensory.shape[-1])
        if use_full_mtl:
            sleep_a_full = torch.stack(complex_train_net.activity_recordings["mtl"])[
                complex_train_net.sleep_indices_A
            ]
            if sleep_a_full.shape[0] != chunk_days * sleep_duration_a:
                raise ValueError("Unexpected number of full-MTL Sleep-A recordings.")
            replayed_complex = sleep_a_full.reshape(chunk_days, sleep_duration_a, -1)[
                :, sleep_duration_a // 2 :, :
            ].reshape(-1, sleep_a_full.shape[-1])
        else:
            replayed_complex = replayed_sensory
        # Give each generated replay event a persistent identity for the later split.
        replay_event_ids = torch.arange(
            replay_reservoir_count,
            replay_reservoir_count + replayed_sensory.shape[0],
            dtype=torch.long,
            device=replayed_sensory.device,
        )
        # Label sensory replay by the closest clean episode and update the R/A fields.
        _update_receptive_field_reservoirs(
            receptive_field_reservoirs,
            receptive_field_event_id_reservoirs,
            receptive_field_counts,
            replayed_sensory,
            replayed_sensory,
            replay_event_ids,
            complex_prototypes,
            complex_labels,
            max_patterns_per_label,
        )
        # Optionally keep a separate full-MTL receptive-field reservoir for R/R.
        if use_full_mtl:
            _update_receptive_field_reservoirs(
                replayed_replayed_receptive_field_reservoirs,
                replayed_replayed_receptive_field_event_id_reservoirs,
                replayed_replayed_receptive_field_counts,
                replayed_complex,
                replayed_sensory,
                replay_event_ids,
                complex_prototypes,
                complex_labels,
                max_patterns_per_label,
            )
        # Keep the global replay pool from which held-out R/R test patterns are drawn.
        replay_reservoir, replay_event_id_reservoir, replay_reservoir_count = _update_replay_reservoir(
            replay_reservoir,
            replay_event_id_reservoir,
            replay_reservoir_count,
            replayed_complex,
            replay_event_ids,
            replay_capacity,
        )
        # Discard the chunk recordings after their replay events have been retained.
        del sleep_a_sensory, replayed_sensory, replayed_complex
        if use_full_mtl:
            del sleep_a_full
        complex_train_net.activity_recordings.clear()
        complex_train_net.awake_indices.clear()
        complex_train_net.sleep_indices_A.clear()
        complex_train_net.sleep_indices_B.clear()

    # Trim reservoirs to the events actually observed for each episode.
    receptive_fields = {
        label: reservoir[: min(receptive_field_counts[label], max_patterns_per_label)].clone()
        for label, reservoir in receptive_field_reservoirs.items()
    }
    replayed_replayed_receptive_fields = {
        label: reservoir[
            : min(replayed_replayed_receptive_field_counts[label], max_patterns_per_label)
        ].clone()
        for label, reservoir in replayed_replayed_receptive_field_reservoirs.items()
    }
    replayed_replayed_receptive_field_event_ids = {
        label: reservoir[
            : min(replayed_replayed_receptive_field_counts[label], max_patterns_per_label)
        ].clone()
        for label, reservoir in replayed_replayed_receptive_field_event_id_reservoirs.items()
    }
    if replay_reservoir_count == 0:
        raise ValueError("Complex training produced no load-2 replay patterns.")

    # Freeze the final network and measure complex-episode decoding in CTX (Fig. 6Bd).
    complex_eval_net = deepcopy(complex_train_net)
    complex_eval_net.init_recordings(_make_recording_params(regions=["ctx"]))
    complex_eval_net.frozen = True
    complex_eval_input, complex_eval_episodes, complex_eval_latents, complex_eval_net = train_network(
        complex_eval_net,
        _make_input_params(
            input_params,
            latent_specs,
            num_days=complex_eval_days,
            num_swaps=posttrain_num_swaps,
        ),
        sleep=False,
        print_rate=np.inf,
        true_latent_to_mtl_semantic=use_true_latent_to_mtl_semantic,
    )
    complex_accuracy_results = _get_ctx_episode_accuracy_from_frozen_eval_net(
        complex_eval_net, complex_eval_episodes, latent_specs
    )
    del complex_eval_net, complex_eval_episodes

    # Use the frozen wake inputs to assess Replayed/Awake generalization (Fig. 6Bf).
    complex_eval_labels = np.asarray(
        [
            "".join(
                f"{chr(ord('A') + index)}{int(value.item()) + 1}"
                for index, value in enumerate(latent)
            )
            for latent in complex_eval_latents.reshape(-1, complex_eval_latents.shape[-1])
        ],
        dtype=object,
    ).reshape(complex_eval_input.shape[:2])
    replayed_awake_generalization = get_rf_patterns_generalization(
        receptive_fields, complex_eval_input, complex_eval_labels
    )

    # Hold out every candidate R/R receptive-field event from the replay test pool.
    # This prevents a replay from serving as both a classifier template and a test item.
    retained_replay_count = min(replay_reservoir_count, replay_capacity)
    replayed_replayed_rf_ids = torch.cat(
        [event_ids for event_ids in replayed_replayed_receptive_field_event_ids.values() if event_ids.numel()]
    )
    replay_test_mask = ~torch.isin(
        replay_event_id_reservoir[:retained_replay_count],
        replayed_replayed_rf_ids.unique(),
    )
    replay_test_pool = replay_reservoir[:retained_replay_count][replay_test_mask]
    if replay_test_pool.shape[0] == 0:
        raise ValueError(
            "No held-out load-2 replay patterns remain after removing classifier receptive fields."
        )
    # Sample a replayed test array from the held-out pool, label it by its nearest
    # prototype, and compute both replay SNR (Fig. 6Be) and R/R generalization (Fig. 6Bg).
    replay_indices = torch.randint(
        replay_test_pool.shape[0],
        (complex_eval_input.shape[0] * complex_eval_input.shape[1],),
        device=replay_reservoir.device,
    )
    replayed_test_input = replay_test_pool[replay_indices].reshape(
        *complex_eval_input.shape[:-1], replay_reservoir.shape[-1]
    )
    replayed_test_sensory = replayed_test_input[..., : complex_train_net.mtl_sensory_size]
    replayed_test_labels = get_pattern_labels_from_prototypes(
        replayed_test_sensory, complex_prototypes, complex_labels
    )
    replayed_test_prototype_indices = (
        F.normalize(replayed_test_sensory.reshape(-1, replayed_test_sensory.shape[-1]).float(), dim=1)
        @ F.normalize(complex_prototypes.float(), dim=1).T
    ).argmax(dim=1)
    replay_snr = _get_replay_pattern_snr(
        replayed_test_sensory.reshape(-1, replayed_test_sensory.shape[-1]),
        complex_prototypes,
        replayed_test_prototype_indices,
        complex_train_net,
    )
    replayed_replayed_generalization = get_rf_patterns_generalization(
        replayed_replayed_receptive_fields, replayed_test_input, replayed_test_labels
    )

    # Return all scalar measures together with the configuration needed for aggregation.
    return {
        "network_mode": network_mode,
        "seed": None if seed is None else int(seed),
        "num_swaps": int(num_swaps),
        "pretrain_num_swaps": int(pretrain_num_swaps),
        "posttrain_num_swaps": int(posttrain_num_swaps),
        "phase_A": int(phase_A),
        "phase_B": int(phase_B),
        "simple_train_days": int(simple_train_days),
        "simple_eval_days": int(simple_eval_days),
        "complex_train_days": int(complex_train_days),
        "complex_eval_days": int(complex_eval_days),
        "use_full_mtl": bool(use_full_mtl),
        **simple_accuracy_results,
        **complex_accuracy_results,
        "replayed_awake_generalization": replayed_awake_generalization,
        "replayed_replayed_generalization": replayed_replayed_generalization,
        "replayed_replayed_equivalent_num_swaps": replay_snr[
            "mean_closest_prototype_num_swaps"
        ],
        "replayed_replayed_snr": replay_snr["mean_closest_prototype_snr"],
        "network": deepcopy(complex_train_net) if return_network else None,
    }


def generalization_simple_complex_job(
    network_mode,
    network_parameters,
    input_params,
    latent_specs,
    num_swaps,
    seed,
    pretrain_num_swaps=None,
    phase_A=200,
    phase_B=400,
    simple_train_days=800,
    simple_eval_days=100,
    complex_train_days=400,
    complex_eval_days=100,
    complex_train_chunk_days=10,
    use_full_mtl=False,
    return_network=False,
    record=False,
):
    # Multiprocessing-friendly positional wrapper used by the Figure 6B notebook.
    return generalization_simple_complex(
        network_parameters=network_parameters,
        input_params=input_params,
        latent_specs=latent_specs,
        network_mode=network_mode,
        num_swaps=num_swaps,
        pretrain_num_swaps=pretrain_num_swaps,
        seed=seed,
        phase_A=phase_A,
        phase_B=phase_B,
        simple_train_days=simple_train_days,
        simple_eval_days=simple_eval_days,
        complex_train_days=complex_train_days,
        complex_eval_days=complex_eval_days,
        complex_train_chunk_days=complex_train_chunk_days,
        use_full_mtl=use_full_mtl,
        return_network=return_network,
        record=record,
    )
