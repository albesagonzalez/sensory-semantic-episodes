import multiprocessing
import os
import random
import time

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from src.model import SSCNetwork
from src.utils.general import (
    LatentSpace,
    get_accuracy,
    get_group_accuracy,
    get_max_overlap,
    get_ordered_accuracy,
    get_ordered_indices,
    get_prototypes,
    get_signal_to_noise_ratio,
    make_input,
    test_network,
)


_GENERALIZATION_SIMPLE_COMPLEX_SHARED = None


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_recording_params(regions=None):
    return {
        "regions": [] if regions is None else list(regions),
        "rate_activity": 1 if regions else np.inf,
        "connections": [],
        "rate_connectivity": np.inf,
    }


def _make_record_everything_params(net):
    return {
        "regions": list(net.regions),
        "rate_activity": 1,
        "connections": [],
        "rate_connectivity": np.inf,
    }


def _make_input_params(base_input_params, latent_specs, num_days=None, num_swaps=None):
    input_params = deepcopy(base_input_params)
    if num_days is not None:
        input_params["num_days"] = int(num_days)
    if num_swaps is not None:
        input_params["num_swaps"] = int(num_swaps)
    input_params["latent_space"] = LatentSpace(**deepcopy(latent_specs))
    return input_params


def _run_network_with_progress(
    net,
    input_params,
    sleep=True,
    print_rate=np.inf,
    verbose=False,
    job_label="job",
    true_latent_to_mtl_semantic=False,
):
    if not verbose:
        return test_network(
            net,
            input_params,
            sleep=sleep,
            print_rate=print_rate,
            true_latent_to_mtl_semantic=true_latent_to_mtl_semantic,
        )

    input_tensor, input_episodes, input_latents = make_input(**input_params)
    num_days = int(input_params["num_days"])
    day_stride = num_days if print_rate in [None, np.inf] else max(int(print_rate), 1)

    print(
        f"[pid={os.getpid()}] {job_label} generated input"
        f" shape={tuple(input_tensor.shape)} sleep={sleep}",
        flush=True,
    )

    with torch.no_grad():
        for day in range(num_days):
            if day % day_stride == 0 or day == num_days - 1:
                print(
                    f"[pid={os.getpid()}] {job_label} day={day + 1}/{num_days}"
                    " stage=awake:start",
                    flush=True,
                )
            latent_day = input_latents[day] if true_latent_to_mtl_semantic else None
            net(input_tensor[day], debug=False, true_latent=latent_day)
            if day % day_stride == 0 or day == num_days - 1:
                print(
                    f"[pid={os.getpid()}] {job_label} day={day + 1}/{num_days}"
                    " stage=awake:end",
                    flush=True,
                )
            if sleep:
                if day % day_stride == 0 or day == num_days - 1:
                    print(
                        f"[pid={os.getpid()}] {job_label} day={day + 1}/{num_days}"
                        " stage=sleep:start",
                        flush=True,
                    )
                net.sleep()
                if day % day_stride == 0 or day == num_days - 1:
                    print(
                        f"[pid={os.getpid()}] {job_label} day={day + 1}/{num_days}"
                        " stage=sleep:end",
                        flush=True,
                    )

    return input_tensor, input_episodes, input_latents, net


def _sample_replay_recordings(replayed_mtl_sensory, num_stored_recordings):
    num_stored_recordings = int(num_stored_recordings)
    num_available = int(replayed_mtl_sensory.shape[0])
    if num_stored_recordings <= 0 or num_available == 0:
        return torch.empty(
            (0, replayed_mtl_sensory.shape[-1]),
            dtype=replayed_mtl_sensory.dtype,
        ), torch.empty((0,), dtype=torch.long)

    num_to_store = min(num_stored_recordings, num_available)
    stored_indices = torch.randperm(num_available)[:num_to_store]
    stored_recordings = replayed_mtl_sensory[stored_indices].detach().cpu().clone()
    return stored_recordings, stored_indices.detach().cpu().clone()


def _get_replay_pattern_snr(
    replayed_patterns,
    prototypes,
    inferred_indices,
    network,
    region="mtl_sensory",
):
    replayed_patterns = torch.as_tensor(replayed_patterns).detach().cpu().float()
    prototypes = torch.as_tensor(prototypes).detach().cpu().float()
    inferred_indices = torch.as_tensor(inferred_indices).detach().cpu().long()

    if replayed_patterns.numel() == 0:
        return {
            "closest_prototype_hamming": torch.empty(0, dtype=torch.float32),
            "closest_prototype_num_swaps": torch.empty(0, dtype=torch.long),
            "closest_prototype_snr": torch.empty(0, dtype=torch.float32),
            "fraction_exact_prototype_replays": float("nan"),
            "mean_closest_prototype_snr": float("nan"),
        }

    closest_prototypes = prototypes[inferred_indices]
    hamming = (replayed_patterns != closest_prototypes).sum(dim=1).to(torch.float32)
    estimated_num_swaps = torch.div(hamming.to(torch.long), 2, rounding_mode="floor")
    snr_values = torch.tensor(
        [
            float(get_signal_to_noise_ratio(int(swaps.item()), network, region=region))
            for swaps in estimated_num_swaps
        ],
        dtype=torch.float32,
    )
    exact_mask = estimated_num_swaps == 0
    mean_num_swaps = float(estimated_num_swaps.to(torch.float32).mean().item())
    mean_num_swaps_floor = int(torch.floor(torch.tensor(mean_num_swaps)).item())
    mean_snr = float(get_signal_to_noise_ratio(mean_num_swaps_floor, network, region=region))

    return {
        "closest_prototype_hamming": hamming,
        "closest_prototype_num_swaps": estimated_num_swaps,
        "closest_prototype_snr": snr_values,
        "fraction_exact_prototype_replays": float(exact_mask.to(torch.float32).mean().item()),
        "mean_closest_prototype_num_swaps": mean_num_swaps,
        "mean_closest_prototype_num_swaps_floor": mean_num_swaps_floor,
        "mean_closest_prototype_snr": mean_snr,
    }


def _get_sleep_a_replays_for_semantic_charge(
    eval_net,
    semantic_charge,
    simulation_num_days,
    simulation_start_day,
    region_name="mtl_sensory",
):
    replayed_mtl_sensory = torch.stack(
        eval_net.activity_recordings[region_name], dim=0
    )[eval_net.sleep_indices_A].detach()

    if semantic_charge == 1:
        return replayed_mtl_sensory

    if semantic_charge != 2:
        raise ValueError("semantic_charge must be 1 or 2.")

    if int(getattr(eval_net, "max_semantic_charge_replay", 1)) < 2:
        return replayed_mtl_sensory[:0]

    sleep_duration_a = int(eval_net.sleep_duration_A)
    if replayed_mtl_sensory.shape[0] != int(simulation_num_days) * sleep_duration_a:
        raise ValueError(
            "Unexpected number of Sleep-A recordings. "
            f"Expected {int(simulation_num_days) * sleep_duration_a}, got {replayed_mtl_sensory.shape[0]}."
        )

    replayed_mtl_sensory = replayed_mtl_sensory.reshape(
        int(simulation_num_days),
        sleep_duration_a,
        -1,
    )

    charge_2_start = int(sleep_duration_a // 2)
    selected_replays = []
    for day_offset in range(int(simulation_num_days)):
        sleep_day = int(simulation_start_day) + day_offset + 1
        if sleep_day <= int(eval_net.duration_phase_B):
            continue
        selected_replays.append(replayed_mtl_sensory[day_offset, charge_2_start:, :])

    if len(selected_replays) == 0:
        return replayed_mtl_sensory.new_empty((0, replayed_mtl_sensory.shape[-1]))

    return torch.cat(selected_replays, dim=0)


def _ensure_recorded_regions(recording_parameters, required_regions):
    recording_parameters = deepcopy(recording_parameters)
    regions = list(recording_parameters.get("regions", []))
    for region_name in required_regions:
        if region_name not in regions:
            regions.append(region_name)
    recording_parameters["regions"] = regions
    return recording_parameters


def train_phase_split_networks(
    network_parameters,
    input_params,
    latent_specs,
    seed=42,
    pretrain_num_swaps=4,
    duration_phase_A=200,
    duration_phase_B=400,
    total_days=600,
    checkpoint_day=200,
    print_rate=50,
):
    if checkpoint_day >= total_days:
        raise ValueError("checkpoint_day must be smaller than total_days.")

    seed_everything(seed)

    net_params = deepcopy(network_parameters)
    net_params["duration_phase_A"] = int(duration_phase_A)
    net_params["duration_phase_B"] = int(duration_phase_B)

    training_recording_parameters = _make_recording_params()
    training_input_params = deepcopy(input_params)
    training_input_params["num_swaps"] = int(pretrain_num_swaps)

    network = SSCNetwork(net_params, training_recording_parameters)

    checkpoint_input_params = _make_input_params(
        training_input_params,
        latent_specs,
        num_days=checkpoint_day,
        num_swaps=pretrain_num_swaps,
    )
    _, _, _, network = _run_network_with_progress(
        network,
        checkpoint_input_params,
        sleep=True,
        print_rate=print_rate,
        verbose=False,
        job_label="pretrain-checkpoint",
    )

    remaining_days = int(total_days) - int(checkpoint_day)
    continuation_input_params = _make_input_params(
        training_input_params,
        latent_specs,
        num_days=remaining_days,
        num_swaps=pretrain_num_swaps,
    )
    branch_networks = {
        "semantics_absent": deepcopy(network),
        "semantics_random": deepcopy(network),
        "semantics_present": deepcopy(network),
    }
    branch_networks["semantics_absent"].sensory_replay_only = True
    branch_networks["semantics_random"].lesioned = {"mtl_semantic"}

    trained_networks = {}
    for branch_name, branch_network in branch_networks.items():
        branch_network.init_recordings(_make_recording_params())
        _, _, _, branch_network = _run_network_with_progress(
            branch_network,
            deepcopy(continuation_input_params),
            sleep=True,
            print_rate=print_rate,
            verbose=False,
            job_label=f"pretrain-continuation:{branch_name}",
        )
        trained_networks[branch_name] = deepcopy(branch_network)

    return trained_networks


def measure_sleep_a_prototype_overlap(
    network,
    recording_parameters,
    input_params,
    latent_specs,
    num_swaps,
    num_days=50,
    semantic_charge=1,
    num_stored_recordings=0,
    seed=None,
    print_rate=np.inf,
    verbose=False,
    network_name=None,
):
    if seed is not None:
        seed_everything(seed)

    start_time = time.time()
    if verbose:
        print(
            f"[pid={os.getpid()}] START overlap"
            f" network={network_name} seed={seed} num_swaps={num_swaps} num_days={num_days}",
            flush=True,
        )

    eval_net = network
    eval_net.init_recordings(deepcopy(recording_parameters))
    eval_net.activity_recordings_rate = 1
    eval_net.connectivity_recordings_rate = np.inf

    eval_input_params = _make_input_params(
        input_params,
        latent_specs,
        num_days=num_days,
        num_swaps=num_swaps,
    )

    prototypes = get_prototypes(eval_input_params["latent_space"], semantic_charge=semantic_charge)
    if verbose:
        print(
            f"[pid={os.getpid()}] overlap"
            f" network={network_name} seed={seed} num_swaps={num_swaps}"
            " checkpoint=prototypes_ready",
            flush=True,
        )

    if network_name == "semantics_clean":
        replayed_mtl_sensory = prototypes.detach().clone()
        if verbose:
            print(
                f"[pid={os.getpid()}] overlap"
                f" network={network_name} seed={seed} num_swaps={num_swaps}"
                " checkpoint=clean_replays_ready",
                flush=True,
            )
    else:
        simulation_start_day = int(eval_net.day)
        _, _, _, eval_net = _run_network_with_progress(
            eval_net,
            eval_input_params,
            sleep=True,
            print_rate=print_rate,
            verbose=verbose,
            job_label=f"overlap:{network_name}:seed={seed}:swaps={num_swaps}",
        )
        if verbose:
            print(
                f"[pid={os.getpid()}] overlap"
                f" network={network_name} seed={seed} num_swaps={num_swaps}"
                " checkpoint=simulation_done",
                flush=True,
            )

        replayed_mtl_sensory = _get_sleep_a_replays_for_semantic_charge(
            eval_net,
            semantic_charge=semantic_charge,
            simulation_num_days=eval_input_params["num_days"],
            simulation_start_day=simulation_start_day,
        )
    if verbose:
        print(
            f"[pid={os.getpid()}] overlap"
            f" network={network_name} seed={seed} num_swaps={num_swaps}"
            f" checkpoint=replays_stacked count={replayed_mtl_sensory.shape[0]}",
            flush=True,
        )

    max_overlaps = get_max_overlap(replayed_mtl_sensory, prototypes)
    stored_recordings, stored_recording_indices = _sample_replay_recordings(
        replayed_mtl_sensory,
        num_stored_recordings=num_stored_recordings,
    )

    if verbose:
        elapsed = time.time() - start_time
        print(
            f"[pid={os.getpid()}] END overlap"
            f" network={network_name} seed={seed} num_swaps={num_swaps}"
            f" mean={max_overlaps.mean().item():.4f} elapsed={elapsed:.1f}s",
            flush=True,
        )

    return {
        "num_swaps": int(num_swaps),
        "num_days": int(num_days),
        "num_replays": int(max_overlaps.numel()),
        "num_stored_recordings": int(stored_recordings.shape[0]),
        "stored_recordings": stored_recordings,
        "stored_recording_indices": stored_recording_indices,
        "max_overlaps": max_overlaps.detach().cpu(),
        "mean_max_overlap": float(max_overlaps.mean().item()),
        "std_max_overlap": float(max_overlaps.std(unbiased=False).item()),
    }


def measure_sleep_a_prototype_overlap_job(
    network_name,
    network,
    recording_parameters,
    input_params,
    latent_specs,
    num_swaps,
    num_days,
    semantic_charge,
    num_stored_recordings,
    seed,
    print_rate,
    verbose=False,
):
    result = measure_sleep_a_prototype_overlap(
        network=deepcopy(network),
        recording_parameters=recording_parameters,
        input_params=input_params,
        latent_specs=latent_specs,
        num_swaps=num_swaps,
        num_days=num_days,
        semantic_charge=semantic_charge,
        num_stored_recordings=num_stored_recordings,
        seed=seed,
        print_rate=print_rate,
        verbose=verbose,
        network_name=network_name,
    )
    result["network_name"] = network_name
    result["seed"] = int(seed)
    return result


def build_codex_figure_5_experiment_params(
    trained_networks,
    recording_parameters,
    input_params,
    latent_specs,
    seeds,
    noise_levels,
    days_per_level=50,
    semantic_charge=1,
    num_stored_recordings=0,
    print_rate=np.inf,
    verbose=False,
):
    experiment_params = []
    for seed in [int(seed) for seed in seeds]:
        for network_name, network in trained_networks.items():
            for num_swaps in [int(level) for level in noise_levels]:
                experiment_params.append(
                    (
                        network_name,
                        network,
                        deepcopy(recording_parameters),
                        deepcopy(input_params),
                        deepcopy(latent_specs),
                        int(num_swaps),
                        int(days_per_level),
                        int(semantic_charge),
                        int(num_stored_recordings),
                        int(seed),
                        print_rate,
                        bool(verbose),
                    )
                )
    return experiment_params


def aggregate_codex_figure_5_results(job_results, noise_levels=None, seeds=None):
    if len(job_results) == 0:
        raise ValueError("job_results cannot be empty.")

    if noise_levels is None:
        noise_levels = sorted({int(job["num_swaps"]) for job in job_results})
    else:
        noise_levels = [int(level) for level in noise_levels]

    if seeds is None:
        seeds = sorted({int(job["seed"]) for job in job_results})
    else:
        seeds = [int(seed) for seed in seeds]

    aggregate = {
        "seeds": seeds,
        "noise_levels": np.array(noise_levels, dtype=int),
        "days_per_level": int(job_results[0]["num_days"]),
        "semantic_charge": int(job_results[0].get("semantic_charge", 1)),
        "num_stored_recordings": int(job_results[0].get("num_stored_recordings", 0)),
        "per_seed": {},
        "summary": {},
        "raw_results": job_results,
    }

    network_names = sorted({job["network_name"] for job in job_results})
    for network_name in network_names:
        aggregate["per_seed"][network_name] = {}
        seed_curves = []
        replay_snr_seed_curves = []
        for seed in seeds:
            per_seed_jobs = [
                job
                for job in job_results
                if job["network_name"] == network_name and int(job["seed"]) == int(seed)
            ]
            per_seed_jobs.sort(key=lambda job: int(job["num_swaps"]))
            if len(per_seed_jobs) != len(noise_levels):
                raise ValueError(
                    f"Missing jobs for network={network_name!r}, seed={seed}. "
                    f"Expected {len(noise_levels)} noise levels, got {len(per_seed_jobs)}."
                )

            aggregate["per_seed"][network_name][int(seed)] = per_seed_jobs
            seed_curves.append(
                np.array([job["mean_max_overlap"] for job in per_seed_jobs], dtype=float)
            )

        seed_curves = np.stack(seed_curves, axis=0)
        aggregate["summary"][network_name] = {
            "seed_curves": seed_curves,
            "mean_curve": seed_curves.mean(axis=0),
            "std_curve": seed_curves.std(axis=0),
        }

    return aggregate


def aggregate_codex_figure_5_generalization_results(job_results, noise_levels=None, seeds=None):
    if len(job_results) == 0:
        raise ValueError("job_results cannot be empty.")

    if noise_levels is None:
        noise_levels = sorted({int(job["num_swaps"]) for job in job_results})
    else:
        noise_levels = [int(level) for level in noise_levels]

    if seeds is None:
        seeds = sorted({int(job["seed"]) for job in job_results})
    else:
        seeds = [int(seed) for seed in seeds]

    aggregate = {
        "seeds": seeds,
        "noise_levels": np.array(noise_levels, dtype=int),
        "days_per_level": int(job_results[0]["num_days"]),
        "semantic_charge": int(job_results[0].get("semantic_charge", 1)),
        "num_stored_recordings": int(job_results[0].get("num_stored_recordings", 0)),
        "per_seed": {},
        "summary": {},
        "raw_results": job_results,
    }

    network_names = sorted({job["network_name"] for job in job_results})
    for network_name in network_names:
        aggregate["per_seed"][network_name] = {}
        seed_curves = []
        replay_snr_seed_curves = []
        for seed in seeds:
            per_seed_jobs = [
                job
                for job in job_results
                if job["network_name"] == network_name and int(job["seed"]) == int(seed)
            ]
            per_seed_jobs.sort(key=lambda job: int(job["num_swaps"]))
            if len(per_seed_jobs) != len(noise_levels):
                raise ValueError(
                    f"Missing jobs for network={network_name!r}, seed={seed}. "
                    f"Expected {len(noise_levels)} noise levels, got {len(per_seed_jobs)}."
                )

            aggregate["per_seed"][network_name][int(seed)] = per_seed_jobs
            seed_curves.append(
                np.array([job["mean_margin"] for job in per_seed_jobs], dtype=float)
            )
            replay_snr_seed_curves.append(
                np.array(
                    [job.get("mean_closest_prototype_snr", np.nan) for job in per_seed_jobs],
                    dtype=float,
                )
            )

        seed_curves = np.stack(seed_curves, axis=0)
        replay_snr_seed_curves = np.stack(replay_snr_seed_curves, axis=0)
        aggregate["summary"][network_name] = {
            "seed_curves": seed_curves,
            "mean_curve": seed_curves.mean(axis=0),
            "std_curve": seed_curves.std(axis=0),
            "replay_snr_seed_curves": replay_snr_seed_curves,
            "replay_snr_mean_curve": np.nanmean(replay_snr_seed_curves, axis=0),
            "replay_snr_std_curve": np.nanstd(replay_snr_seed_curves, axis=0),
        }

    return aggregate


def measure_sleep_a_awake_generalization(
    network,
    recording_parameters,
    input_params,
    latent_specs,
    num_swaps,
    num_days=50,
    semantic_charge=1,
    num_stored_recordings=0,
    seed=None,
    print_rate=np.inf,
    verbose=False,
    network_name=None,
):
    if seed is not None:
        seed_everything(seed)

    start_time = time.time()
    if verbose:
        print(
            f"[pid={os.getpid()}] START generalization"
            f" network={network_name} seed={seed} num_swaps={num_swaps} num_days={num_days}",
            flush=True,
        )

    eval_input_params = _make_input_params(
        input_params,
        latent_specs,
        num_days=num_days,
        num_swaps=num_swaps,
    )

    prototypes = get_prototypes(eval_input_params["latent_space"], semantic_charge=semantic_charge)
    if verbose:
        print(
            f"[pid={os.getpid()}] generalization"
            f" network={network_name} seed={seed} num_swaps={num_swaps}"
            " checkpoint=prototypes_ready",
            flush=True,
        )

    if network_name == "semantics_clean":
        replayed_mtl_sensory = prototypes.detach().clone()
        if verbose:
            print(
                f"[pid={os.getpid()}] generalization"
                f" network={network_name} seed={seed} num_swaps={num_swaps}"
                " checkpoint=clean_replays_ready",
                flush=True,
            )
    else:
        eval_net = network
        eval_net.init_recordings(deepcopy(recording_parameters))
        eval_net.activity_recordings_rate = 1
        eval_net.connectivity_recordings_rate = np.inf

        simulation_start_day = int(eval_net.day)
        _, _, _, eval_net = _run_network_with_progress(
            eval_net,
            eval_input_params,
            sleep=True,
            print_rate=print_rate,
            verbose=verbose,
            job_label=f"generalization:{network_name}:seed={seed}:swaps={num_swaps}",
        )
        if verbose:
            print(
                f"[pid={os.getpid()}] generalization"
                f" network={network_name} seed={seed} num_swaps={num_swaps}"
                " checkpoint=simulation_done",
                flush=True,
            )

        replayed_mtl_sensory = _get_sleep_a_replays_for_semantic_charge(
            eval_net,
            semantic_charge=semantic_charge,
            simulation_num_days=eval_input_params["num_days"],
            simulation_start_day=simulation_start_day,
        )
    stored_recordings, stored_recording_indices = _sample_replay_recordings(
        replayed_mtl_sensory,
        num_stored_recordings=num_stored_recordings,
    )

    awake_input_params = deepcopy(eval_input_params)
    awake_input_params["num_days"] = 1
    awake_input_params["day_length"] = 500
    awake_input_params["mean_duration"] = 1
    awake_input_params["fixed_duration"] = True
    awake_input, _, awake_input_latents = make_input(**awake_input_params)
    awake_mtl_sensory = awake_input.reshape(-1, awake_input.shape[-1]).detach()
    awake_latents_flat = awake_input_latents.reshape(-1, awake_input_latents.shape[-1])

    if verbose:
        print(
            f"[pid={os.getpid()}] generalization"
            f" network={network_name} seed={seed} num_swaps={num_swaps}"
            f" checkpoint=recordings_ready awake={awake_mtl_sensory.shape[0]} replay={replayed_mtl_sensory.shape[0]}",
            flush=True,
        )

    num_A = int(latent_specs["dims"][0])
    num_B = int(latent_specs["dims"][1])
    latent_space = eval_input_params["latent_space"]

    if network_name == "semantics_clean":
        concept_overlap = torch.ones(replayed_mtl_sensory.shape[0], dtype=torch.float32)
        inferred_concepts = torch.arange(replayed_mtl_sensory.shape[0], dtype=torch.long)
    else:
        concept_overlap, inferred_concepts = get_max_overlap(
            replayed_mtl_sensory,
            prototypes,
            return_indices=True,
        )

    if semantic_charge == 2:
        replay_pattern_snr = _get_replay_pattern_snr(
            replayed_mtl_sensory,
            prototypes,
            inferred_concepts,
            network,
            region="mtl_sensory",
        )
    else:
        replay_pattern_snr = {
            "closest_prototype_hamming": torch.empty(0, dtype=torch.float32),
            "closest_prototype_num_swaps": torch.empty(0, dtype=torch.long),
            "closest_prototype_snr": torch.empty(0, dtype=torch.float32),
            "fraction_exact_prototype_replays": float("nan"),
            "mean_closest_prototype_snr": float("nan"),
        }
    if verbose:
        print(
            f"[pid={os.getpid()}] generalization"
            f" network={network_name} seed={seed} num_swaps={num_swaps}"
            " checkpoint=inference_done",
            flush=True,
        )

    awake_norm = F.normalize(awake_mtl_sensory.float(), dim=1)
    replay_norm = F.normalize(replayed_mtl_sensory.float(), dim=1)
    similarity_matrix = awake_norm @ replay_norm.T
    if verbose:
        print(
            f"[pid={os.getpid()}] generalization"
            f" network={network_name} seed={seed} num_swaps={num_swaps}"
            f" checkpoint=similarity_matrix_done shape={tuple(similarity_matrix.shape)}",
            flush=True,
        )

    sim_present = []
    sim_absent = []
    margins = []
    inferred_concepts_list = []
    awake_target_indices = []
    inferred_concepts_cpu = inferred_concepts.detach().cpu()
    progress_stride = max(int(awake_mtl_sensory.shape[0] // 10), 1)
    if verbose:
        print(
            f"[pid={os.getpid()}] generalization"
            f" network={network_name} seed={seed} num_swaps={num_swaps}"
            f" checkpoint=scoring_start num_awake={awake_mtl_sensory.shape[0]} num_replays={replayed_mtl_sensory.shape[0]}",
            flush=True,
        )

    for awake_idx in range(awake_mtl_sensory.shape[0]):
        if verbose and (
            awake_idx == 0
            or (awake_idx + 1) % progress_stride == 0
            or awake_idx == awake_mtl_sensory.shape[0] - 1
        ):
            print(
                f"[pid={os.getpid()}] generalization"
                f" network={network_name} seed={seed} num_swaps={num_swaps}"
                f" checkpoint=scoring_progress awake={awake_idx + 1}/{awake_mtl_sensory.shape[0]}",
                flush=True,
            )
        a_idx = int(awake_latents_flat[awake_idx, 0].item())
        b_idx = int(awake_latents_flat[awake_idx, 1].item())
        if semantic_charge == 1:
            concept_a = a_idx
            concept_b = num_A + b_idx
            present_mask = (inferred_concepts_cpu == concept_a) | (
                inferred_concepts_cpu == concept_b
            )
            target_entry = (concept_a, concept_b)
            target_index = -1
        elif semantic_charge == 2:
            target_index = int(latent_space.label_to_index[(a_idx, b_idx)])
            present_mask = inferred_concepts_cpu == target_index
            target_entry = (a_idx, b_idx)
        else:
            raise ValueError("semantic_charge must be 1 or 2.")

        if (not bool(present_mask.any().item())) or (not bool((~present_mask).any().item())):
            continue

        sim_vals = similarity_matrix[awake_idx]
        present_mean = sim_vals[present_mask].mean().item()
        absent_mean = sim_vals[~present_mask].mean().item()

        inferred_concepts_list.append(target_entry)
        awake_target_indices.append(target_index)
        sim_present.append(present_mean)
        sim_absent.append(absent_mean)
        margins.append(present_mean - absent_mean)

    margins_tensor = torch.tensor(margins, dtype=torch.float32)
    if verbose:
        print(
            f"[pid={os.getpid()}] generalization"
            f" network={network_name} seed={seed} num_swaps={num_swaps}"
            " checkpoint=scoring_done",
            flush=True,
        )

    if verbose:
        elapsed = time.time() - start_time
        print(
            f"[pid={os.getpid()}] END generalization"
            f" network={network_name} seed={seed} num_swaps={num_swaps}"
            f" margin={margins_tensor.mean().item():.4f} elapsed={elapsed:.1f}s",
            flush=True,
        )

    return {
        "num_swaps": int(num_swaps),
        "num_days": int(num_days),
        "semantic_charge": int(semantic_charge),
        "num_awake": int(awake_mtl_sensory.shape[0]),
        "num_valid_awake": int(len(margins)),
        "num_replays": int(replayed_mtl_sensory.shape[0]),
        "num_stored_recordings": int(stored_recordings.shape[0]),
        "stored_recordings": stored_recordings,
        "stored_recording_indices": stored_recording_indices,
        "inferred_replay_concepts": inferred_concepts_cpu.clone(),
        "awake_concepts": torch.tensor(inferred_concepts_list, dtype=torch.long),
        "awake_target_indices": torch.tensor(awake_target_indices, dtype=torch.long),
        "prototype_overlap": concept_overlap.detach().cpu(),
        "closest_prototype_hamming": replay_pattern_snr["closest_prototype_hamming"],
        "closest_prototype_num_swaps": replay_pattern_snr["closest_prototype_num_swaps"],
        "closest_prototype_snr": replay_pattern_snr["closest_prototype_snr"],
        "fraction_exact_prototype_replays": replay_pattern_snr["fraction_exact_prototype_replays"],
        "mean_closest_prototype_snr": replay_pattern_snr["mean_closest_prototype_snr"],
        "sim_present": torch.tensor(sim_present, dtype=torch.float32),
        "sim_absent": torch.tensor(sim_absent, dtype=torch.float32),
        "margins": margins_tensor,
        "mean_present": float(np.mean(sim_present)),
        "mean_absent": float(np.mean(sim_absent)),
        "mean_margin": float(margins_tensor.mean().item()),
        "std_margin": float(margins_tensor.std(unbiased=False).item()),
    }


def measure_sleep_a_replay_episode_clustering(
    network,
    recording_parameters,
    input_params,
    latent_specs,
    num_swaps,
    num_days=50,
    semantic_charge=1,
    num_stored_recordings=0,
    seed=None,
    print_rate=np.inf,
    verbose=False,
    network_name=None,
):
    if seed is not None:
        seed_everything(seed)

    start_time = time.time()
    if verbose:
        print(
            f"[pid={os.getpid()}] START replay-clustering"
            f" network={network_name} seed={seed} num_swaps={num_swaps} num_days={num_days}",
            flush=True,
        )

    if network_name == "semantics_clean":
        raise ValueError(
            "Replay episode clustering requires full MTL replay states and does not support 'semantics_clean'."
        )

    eval_input_params = _make_input_params(
        input_params,
        latent_specs,
        num_days=num_days,
        num_swaps=num_swaps,
    )
    label_prototypes = get_prototypes(
        eval_input_params["latent_space"],
        semantic_charge=semantic_charge,
    )

    eval_net = network
    eval_recording_parameters = _ensure_recorded_regions(
        recording_parameters,
        required_regions=["mtl_sensory", "mtl"],
    )
    eval_net.init_recordings(eval_recording_parameters)
    eval_net.activity_recordings_rate = 1
    eval_net.connectivity_recordings_rate = np.inf

    simulation_start_day = int(eval_net.day)
    _, _, _, eval_net = _run_network_with_progress(
        eval_net,
        eval_input_params,
        sleep=True,
        print_rate=print_rate,
        verbose=verbose,
        job_label=f"replay-clustering:{network_name}:seed={seed}:swaps={num_swaps}",
    )

    replayed_mtl_sensory = _get_sleep_a_replays_for_semantic_charge(
        eval_net,
        semantic_charge=semantic_charge,
        simulation_num_days=eval_input_params["num_days"],
        simulation_start_day=simulation_start_day,
        region_name="mtl_sensory",
    )
    replayed_mtl_full = _get_sleep_a_replays_for_semantic_charge(
        eval_net,
        semantic_charge=semantic_charge,
        simulation_num_days=eval_input_params["num_days"],
        simulation_start_day=simulation_start_day,
        region_name="mtl",
    )

    if replayed_mtl_sensory.shape[0] != replayed_mtl_full.shape[0]:
        raise ValueError(
            "Mismatch between replayed sensory and full-MTL recordings: "
            f"{replayed_mtl_sensory.shape[0]} vs {replayed_mtl_full.shape[0]}."
        )

    label_overlap, inferred_label_indices = get_max_overlap(
        replayed_mtl_sensory,
        label_prototypes,
        return_indices=True,
    )

    replay_mtl_norm = F.normalize(replayed_mtl_full.float(), dim=1)
    similarity_matrix = replay_mtl_norm @ replay_mtl_norm.T

    sim_same = []
    sim_different = []
    margins = []
    inferred_label_indices_cpu = inferred_label_indices.detach().cpu()

    for replay_idx in range(replayed_mtl_full.shape[0]):
        same_mask = inferred_label_indices_cpu == inferred_label_indices_cpu[replay_idx]
        same_mask[replay_idx] = False
        different_mask = inferred_label_indices_cpu != inferred_label_indices_cpu[replay_idx]

        if (not bool(same_mask.any().item())) or (not bool(different_mask.any().item())):
            continue

        sim_vals = similarity_matrix[replay_idx]
        same_mean = sim_vals[same_mask].mean().item()
        different_mean = sim_vals[different_mask].mean().item()

        sim_same.append(same_mean)
        sim_different.append(different_mean)
        margins.append(same_mean - different_mean)

    stored_recordings_sensory, stored_recording_indices = _sample_replay_recordings(
        replayed_mtl_sensory,
        num_stored_recordings=num_stored_recordings,
    )
    stored_recordings_full = replayed_mtl_full[stored_recording_indices].detach().cpu().clone()
    stored_recording_label_indices = inferred_label_indices_cpu[stored_recording_indices].clone()

    margins_tensor = torch.tensor(margins, dtype=torch.float32)
    if verbose:
        elapsed = time.time() - start_time
        print(
            f"[pid={os.getpid()}] END replay-clustering"
            f" network={network_name} seed={seed} num_swaps={num_swaps}"
            f" margin={margins_tensor.mean().item():.4f} elapsed={elapsed:.1f}s",
            flush=True,
        )

    return {
        "num_swaps": int(num_swaps),
        "num_days": int(num_days),
        "semantic_charge": int(semantic_charge),
        "num_replays": int(replayed_mtl_full.shape[0]),
        "num_valid_replays": int(len(margins)),
        "num_stored_recordings": int(stored_recordings_sensory.shape[0]),
        "stored_recordings": stored_recordings_sensory,
        "stored_recordings_mtl": stored_recordings_full,
        "stored_recording_indices": stored_recording_indices,
        "stored_recording_label_indices": stored_recording_label_indices,
        "replay_label_overlap": label_overlap.detach().cpu(),
        "inferred_replay_labels": inferred_label_indices_cpu.clone(),
        # Backward-compatible aliases retained for existing notebook cells.
        "stored_recording_episode_indices": stored_recording_label_indices,
        "replay_episode_overlap": label_overlap.detach().cpu(),
        "inferred_replay_episodes": inferred_label_indices_cpu.clone(),
        "sim_same": torch.tensor(sim_same, dtype=torch.float32),
        "sim_different": torch.tensor(sim_different, dtype=torch.float32),
        "margins": margins_tensor,
        "mean_same": float(np.mean(sim_same)),
        "mean_different": float(np.mean(sim_different)),
        "mean_margin": float(margins_tensor.mean().item()),
        "std_margin": float(margins_tensor.std(unbiased=False).item()),
    }


def measure_sleep_a_awake_generalization_job(
    network_name,
    network,
    recording_parameters,
    input_params,
    latent_specs,
    num_swaps,
    num_days,
    semantic_charge,
    num_stored_recordings,
    seed,
    print_rate,
    verbose=False,
):
    result = measure_sleep_a_awake_generalization(
        network=deepcopy(network),
        recording_parameters=recording_parameters,
        input_params=input_params,
        latent_specs=latent_specs,
        num_swaps=num_swaps,
        num_days=num_days,
        semantic_charge=semantic_charge,
        num_stored_recordings=num_stored_recordings,
        seed=seed,
        print_rate=print_rate,
        verbose=verbose,
        network_name=network_name,
    )
    result["network_name"] = network_name
    result["seed"] = int(seed)
    return result


def measure_ctx_latent_accuracy(
    network,
    recording_parameters,
    input_params,
    latent_specs,
    num_swaps,
    num_days=50,
    semantic_charge=1,
    num_stored_recordings=0,
    seed=None,
    print_rate=np.inf,
    verbose=False,
    network_name=None,
):
    del recording_parameters, num_stored_recordings

    if semantic_charge != 1:
        raise ValueError("measure_ctx_latent_accuracy expects semantic_charge=1.")

    if seed is not None:
        seed_everything(seed)

    eval_recording_parameters = {
        "regions": ["ctx"],
        "rate_activity": 1,
        "connections": [],
        "rate_connectivity": np.inf,
    }
    eval_net = deepcopy(network)
    eval_net.init_recordings(eval_recording_parameters)
    eval_net.frozen = True
    eval_net.activity_recordings_rate = 1
    eval_net.connectivity_recordings_rate = np.inf

    eval_input_params = _make_input_params(
        input_params,
        latent_specs,
        num_days=num_days,
        num_swaps=num_swaps,
    )
    eval_input, _, eval_input_latents = make_input(**eval_input_params)

    with torch.no_grad():
        for day in range(eval_input_params["num_days"]):
            if verbose and (
                day == 0
                or (day + 1) % max(int(print_rate), 1) == 0
                or day == eval_input_params["num_days"] - 1
            ):
                print(
                    f"[pid={os.getpid()}] ctx-accuracy"
                    f" network={network_name} seed={seed} num_swaps={num_swaps}"
                    f" day={day + 1}/{eval_input_params['num_days']}",
                    flush=True,
                )
            eval_net(eval_input[day], debug=False)

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
    eval_latents_flat = eval_input_latents.reshape(-1, eval_input_latents.shape[-1])
    X_latent_AB_flat = X_latent_AB.reshape(-1, X_latent_AB.shape[-1])
    num_eval_samples = int(X_ctx.shape[0])
    fit_num_samples = num_eval_samples // 2
    test_num_samples = num_eval_samples - fit_num_samples
    if fit_num_samples == 0 or test_num_samples == 0:
        raise ValueError(
            "measure_ctx_latent_accuracy requires at least two recorded awake samples "
            f"to split decoder-fit and decoder-test data, got {num_eval_samples}."
        )

    X_ctx_fit = X_ctx[:fit_num_samples, :100]
    X_ctx_test = X_ctx[fit_num_samples:, :100]
    X_latent_AB_fit = X_latent_AB_flat[:fit_num_samples].unsqueeze(0)
    eval_latents_flat_test = eval_latents_flat[fit_num_samples:]

    selectivity_ctx, ordered_indices_ctx = get_ordered_indices(
        X_ctx_fit,
        X_latent_AB_fit,
        assembly_size=10,
    )
    ctx_accuracy = get_accuracy(
        X_ctx_test[:, ordered_indices_ctx[:100]],
        eval_latents_flat_test,
        assembly_size=10,
    )

    return {
        "num_swaps": int(num_swaps),
        "num_days": int(num_days),
        "semantic_charge": int(semantic_charge),
        "ctx_accuracy_A": float(ctx_accuracy[0].item()),
        "ctx_accuracy_B": float(ctx_accuracy[1].item()),
        "ctx_accuracy_mean": float(ctx_accuracy.mean().item()),
        "mean_margin": float(ctx_accuracy.mean().item()),
        "fit_num_samples": int(fit_num_samples),
        "test_num_samples": int(test_num_samples),
        "ordered_indices_ctx": ordered_indices_ctx.detach().cpu(),
        "selectivity_ctx": selectivity_ctx.detach().cpu(),
    }


def measure_ctx_latent_accuracy_job(
    network_name,
    network,
    recording_parameters,
    input_params,
    latent_specs,
    num_swaps,
    num_days,
    semantic_charge,
    num_stored_recordings,
    seed,
    print_rate,
    verbose=False,
):
    result = measure_ctx_latent_accuracy(
        network=deepcopy(network),
        recording_parameters=recording_parameters,
        input_params=input_params,
        latent_specs=latent_specs,
        num_swaps=num_swaps,
        num_days=num_days,
        semantic_charge=semantic_charge,
        num_stored_recordings=num_stored_recordings,
        seed=seed,
        print_rate=print_rate,
        verbose=verbose,
        network_name=network_name,
    )
    result["network_name"] = network_name
    result["seed"] = int(seed)
    return result


def _get_ctx_simple_accuracy_from_frozen_eval_net(eval_net, eval_input_latents, latent_specs):
    X_ctx = torch.stack(eval_net.activity_recordings["ctx"], dim=0)[eval_net.awake_indices]
    ctx_simple_indices = eval_net.ctx_subregions[0]
    X_mtl_sensory = None
    if "mtl_sensory" in eval_net.activity_recordings:
        X_mtl_sensory = torch.stack(
            eval_net.activity_recordings["mtl_sensory"], dim=0
        )[eval_net.awake_indices]
    X_mtl_semantic = None
    if "mtl_semantic" in eval_net.activity_recordings:
        X_mtl_semantic = torch.stack(
            eval_net.activity_recordings["mtl_semantic"], dim=0
        )[eval_net.awake_indices]
    X_latent_A = F.one_hot(
        eval_input_latents[:, :, 0].long(),
        num_classes=latent_specs["dims"][0],
    )
    X_latent_B = F.one_hot(
        eval_input_latents[:, :, 1].long(),
        num_classes=latent_specs["dims"][1],
    )
    X_latent_AB = torch.cat((X_latent_A, X_latent_B), dim=2)
    simple_ctx_results = get_ordered_accuracy(
        X_ctx[:, ctx_simple_indices],
        X_latent_AB,
        eval_input_latents.reshape(-1, eval_input_latents.shape[-1]),
        assembly_size=10,
    )
    result = {
        "ctx_accuracy_A": float(simple_ctx_results["accuracy"][0].item()),
        "ctx_accuracy_B": float(simple_ctx_results["accuracy"][1].item()),
        "ctx_accuracy_mean": float(simple_ctx_results["accuracy"].mean().item()),
        "fit_num_samples": int(simple_ctx_results["fit_num_samples"]),
        "test_num_samples": int(simple_ctx_results["test_num_samples"]),
        "ordered_indices_ctx": simple_ctx_results["ordered_indices"].detach().cpu(),
        "selectivity_ctx": simple_ctx_results["selectivity"].detach().cpu(),
    }
    X_latent_AB_flat = X_latent_AB.reshape(-1, X_latent_AB.shape[-1])
    if X_mtl_sensory is not None:
        selectivity_mtl_sensory, ordered_indices_mtl_sensory = get_ordered_indices(
            X_mtl_sensory[: simple_ctx_results["fit_num_samples"]],
            X_latent_AB_flat[: simple_ctx_results["fit_num_samples"]].unsqueeze(0),
            assembly_size=10,
        )
        result["ordered_indices_mtl_sensory"] = ordered_indices_mtl_sensory.detach().cpu()
        result["selectivity_mtl_sensory"] = selectivity_mtl_sensory.detach().cpu()
    if X_mtl_semantic is not None:
        selectivity_mtl_semantic, ordered_indices_mtl_semantic = get_ordered_indices(
            X_mtl_semantic[: simple_ctx_results["fit_num_samples"]],
            X_latent_AB_flat[: simple_ctx_results["fit_num_samples"]].unsqueeze(0),
            assembly_size=10,
        )
        result["ordered_indices_mtl_semantic"] = ordered_indices_mtl_semantic.detach().cpu()
        result["selectivity_mtl_semantic"] = selectivity_mtl_semantic.detach().cpu()
    return result


def _get_episode_accuracy(recordings, episode_indices, assembly_size):
    return get_group_accuracy(recordings, episode_indices, assembly_size=assembly_size)


def _get_ctx_episode_accuracy_from_frozen_eval_net(
    eval_net,
    eval_input_episodes,
    latent_specs,
    verbose=False,
    debug_label="ctx-episode-accuracy",
):
    if verbose:
        print(f"{debug_label}: stack_ctx_start", flush=True)
    X_ctx = torch.stack(eval_net.activity_recordings["ctx"], dim=0)[eval_net.awake_indices]
    if verbose:
        print(f"{debug_label}: stack_ctx_done shape={tuple(X_ctx.shape)}", flush=True)
    ctx_episode_indices = eval_net.ctx_subregions[1]
    if verbose:
        print(f"{debug_label}: one_hot_start", flush=True)
    X_episodes = F.one_hot(
        eval_input_episodes.long(),
        num_classes=int(np.prod(latent_specs["dims"])),
    )
    if verbose:
        print(f"{debug_label}: one_hot_done shape={tuple(X_episodes.shape)}", flush=True)
        print(f"{debug_label}: ordered_accuracy_start", flush=True)
    episode_results = get_ordered_accuracy(
        X_ctx[:, ctx_episode_indices],
        X_episodes,
        eval_input_episodes.reshape(-1),
        assembly_size=10,
        num_groups=int(np.prod(latent_specs["dims"])),
        debug_label=None if not verbose else f"{debug_label}: ordered_accuracy",
    )
    if verbose:
        print(f"{debug_label}: ordered_accuracy_done", flush=True)
    ordered_indices_ctx_episodes = ctx_episode_indices[episode_results["ordered_indices"]]
    return {
        "ctx_episode_accuracy": float(episode_results["accuracy"].item()),
        "fit_num_samples": int(episode_results["fit_num_samples"]),
        "test_num_samples": int(episode_results["test_num_samples"]),
        "ordered_indices_ctx_episodes": ordered_indices_ctx_episodes.detach().cpu(),
        "selectivity_ctx_episodes": episode_results["selectivity"].detach().cpu(),
    }


def attach_ctx_replay_readout_orderings(
    network,
    input_params,
    latent_specs,
    num_days=100,
    num_swaps=None,
    include_simple=True,
    include_complex=False,
):
    if (not include_simple) and (not include_complex):
        return network

    eval_recording_parameters = {
        "regions": ["ctx"],
        "rate_activity": 1,
        "connections": [],
        "rate_connectivity": np.inf,
    }
    eval_net = deepcopy(network)
    eval_net.init_recordings(eval_recording_parameters)
    eval_net.frozen = True
    eval_net.activity_recordings_rate = 1
    eval_net.connectivity_recordings_rate = np.inf

    eval_input_params = _make_input_params(
        input_params,
        latent_specs,
        num_days=num_days,
        num_swaps=input_params["num_swaps"] if num_swaps is None else num_swaps,
    )
    eval_input, eval_input_episodes, eval_input_latents = make_input(**eval_input_params)

    with torch.no_grad():
        for day in range(eval_input_params["num_days"]):
            eval_net(eval_input[day], debug=False)

    if include_simple:
        simple_results = _get_ctx_simple_accuracy_from_frozen_eval_net(
            eval_net,
            eval_input_latents,
            latent_specs,
        )
        for attr_name in [
            "ordered_indices_ctx",
            "selectivity_ctx",
            "ordered_indices_mtl_sensory",
            "selectivity_mtl_sensory",
            "ordered_indices_mtl_semantic",
            "selectivity_mtl_semantic",
        ]:
            if attr_name in simple_results:
                setattr(network, attr_name, simple_results[attr_name].detach().cpu())

    if include_complex:
        complex_results = _get_ctx_episode_accuracy_from_frozen_eval_net(
            eval_net,
            eval_input_episodes,
            latent_specs,
        )
        for attr_name in [
            "ordered_indices_ctx_episodes",
            "selectivity_ctx_episodes",
        ]:
            if attr_name in complex_results:
                setattr(network, attr_name, complex_results[attr_name].detach().cpu())

    return network


def attach_ctx_replay_readout_orderings_to_dict(
    trained_networks,
    input_params,
    latent_specs,
    num_days=100,
    num_swaps=None,
    include_simple=True,
    include_complex=False,
):
    for network_name, network in trained_networks.items():
        trained_networks[network_name] = attach_ctx_replay_readout_orderings(
            network,
            input_params=input_params,
            latent_specs=latent_specs,
            num_days=num_days,
            num_swaps=num_swaps,
            include_simple=include_simple,
            include_complex=include_complex,
        )
    return trained_networks


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
    print_rate=np.inf,
    verbose=False,
    return_network=False,
    record=False,
):
    if network_mode not in {"semantics_present", "semantics_random", "semantics_absent", "semantics_clean"}:
        raise ValueError(
            "network_mode must be one of "
            "{'semantics_present', 'semantics_random', 'semantics_absent', 'semantics_clean'}."
        )

    if seed is not None:
        seed_everything(seed)

    if verbose:
        print(
            f"[pid={os.getpid()}] START generalization_simple_complex "
            f"mode={network_mode} seed={seed} num_swaps={num_swaps}",
            flush=True,
        )

    posttrain_num_swaps = int(num_swaps)
    if pretrain_num_swaps is None:
        pretrain_num_swaps = posttrain_num_swaps
    pretrain_num_swaps = int(pretrain_num_swaps)

    net_params = deepcopy(network_parameters)
    net_params["duration_phase_A"] = int(phase_A)
    net_params["duration_phase_B"] = int(phase_B)
    network = SSCNetwork(net_params, _make_recording_params())
    if return_network and record:
        network.record_everything_params = _make_record_everything_params(network)
        network.return_network_records_everything = True
        network.init_recordings(network.record_everything_params)

    if network_mode == "semantics_random":
        network.lesioned = {"mtl_semantic"}
    elif network_mode == "semantics_absent":
        network.sensory_replay_only = True
    use_true_latent_to_mtl_semantic = network_mode == "semantics_clean"

    pretrain_days = min(int(simple_train_days), int(phase_B))
    posttrain_simple_days = max(0, int(simple_train_days) - pretrain_days)

    if pretrain_days > 0:
        train_input_params_pre = _make_input_params(
            input_params,
            latent_specs,
            num_days=pretrain_days,
            num_swaps=pretrain_num_swaps,
        )
        if verbose:
            print(
                f"[pid={os.getpid()}] generalization_simple_complex "
                f"mode={network_mode} seed={seed} num_swaps={num_swaps} "
                "checkpoint=simple_train_prephaseB_start",
                flush=True,
            )
        _, _, _, network = _run_network_with_progress(
            network,
            train_input_params_pre,
            sleep=True,
            print_rate=print_rate,
            verbose=verbose,
            job_label=(
                f"generalization-simple-complex:{network_mode}:seed={seed}:"
                f"swaps={num_swaps}:simple-train-prephaseB"
            ),
            true_latent_to_mtl_semantic=use_true_latent_to_mtl_semantic,
        )
        if verbose:
            print(
                f"[pid={os.getpid()}] generalization_simple_complex "
                f"mode={network_mode} seed={seed} num_swaps={num_swaps} "
                "checkpoint=simple_train_prephaseB_done",
                flush=True,
            )

    if posttrain_simple_days > 0:
        train_input_params_post = _make_input_params(
            input_params,
            latent_specs,
            num_days=posttrain_simple_days,
            num_swaps=posttrain_num_swaps,
        )
        if verbose:
            print(
                f"[pid={os.getpid()}] generalization_simple_complex "
                f"mode={network_mode} seed={seed} num_swaps={num_swaps} "
                "checkpoint=simple_train_postphaseB_start",
                flush=True,
            )
        _, _, _, network = _run_network_with_progress(
            network,
            train_input_params_post,
            sleep=True,
            print_rate=print_rate,
            verbose=verbose,
            job_label=(
                f"generalization-simple-complex:{network_mode}:seed={seed}:"
                f"swaps={num_swaps}:simple-train-postphaseB"
            ),
            true_latent_to_mtl_semantic=use_true_latent_to_mtl_semantic,
        )
        if verbose:
            print(
                f"[pid={os.getpid()}] generalization_simple_complex "
                f"mode={network_mode} seed={seed} num_swaps={num_swaps} "
                "checkpoint=simple_train_postphaseB_done",
                flush=True,
            )

    simple_eval_recording_parameters = {
        "regions": ["ctx", "mtl_sensory", "mtl_semantic"],
        "rate_activity": 1,
        "connections": [],
        "rate_connectivity": np.inf,
    }
    simple_eval_net = deepcopy(network)
    simple_eval_net.init_recordings(simple_eval_recording_parameters)
    simple_eval_net.frozen = True
    simple_eval_net.activity_recordings_rate = 1
    simple_eval_net.connectivity_recordings_rate = np.inf

    simple_eval_input_params = _make_input_params(
        input_params,
        latent_specs,
        num_days=simple_eval_days,
        num_swaps=posttrain_num_swaps,
    )
    if verbose:
        print(
            f"[pid={os.getpid()}] generalization_simple_complex "
            f"mode={network_mode} seed={seed} num_swaps={num_swaps} "
            "checkpoint=simple_eval_start",
            flush=True,
        )
    simple_eval_input, _, simple_eval_input_latents = make_input(**simple_eval_input_params)
    with torch.no_grad():
        for day in range(simple_eval_input_params["num_days"]):
            latent_day = (
                simple_eval_input_latents[day] if use_true_latent_to_mtl_semantic else None
            )
            simple_eval_net(simple_eval_input[day], debug=False, true_latent=latent_day)
    if verbose:
        print(
            f"[pid={os.getpid()}] generalization_simple_complex "
            f"mode={network_mode} seed={seed} num_swaps={num_swaps} "
            "checkpoint=simple_eval_done",
            flush=True,
        )

    simple_accuracy_results = _get_ctx_simple_accuracy_from_frozen_eval_net(
        simple_eval_net,
        simple_eval_input_latents,
        latent_specs,
    )

    complex_train_net = network if return_network else deepcopy(network)
    complex_train_net.frozen = False
    complex_train_net.max_semantic_charge_replay = 2
    if not return_network:
        complex_train_net.init_recordings(_make_recording_params())

    complex_train_input_params = _make_input_params(
        input_params,
        latent_specs,
        num_days=complex_train_days,
        num_swaps=posttrain_num_swaps,
    )
    if verbose:
        print(
            f"[pid={os.getpid()}] generalization_simple_complex "
            f"mode={network_mode} seed={seed} num_swaps={num_swaps} "
            "checkpoint=complex_train_start",
            flush=True,
        )
    _, _, _, complex_train_net = _run_network_with_progress(
        complex_train_net,
        complex_train_input_params,
        sleep=True,
        print_rate=print_rate,
        verbose=verbose,
        job_label=f"generalization-simple-complex:{network_mode}:seed={seed}:swaps={num_swaps}:complex-train",
        true_latent_to_mtl_semantic=use_true_latent_to_mtl_semantic,
    )
    if verbose:
        print(
            f"[pid={os.getpid()}] generalization_simple_complex "
            f"mode={network_mode} seed={seed} num_swaps={num_swaps} "
            "checkpoint=complex_train_done",
            flush=True,
        )

    complex_eval_recording_parameters = {
        "regions": ["ctx"],
        "rate_activity": 1,
        "connections": [],
        "rate_connectivity": np.inf,
    }
    complex_eval_net = deepcopy(complex_train_net)
    complex_eval_net.init_recordings(complex_eval_recording_parameters)
    complex_eval_net.frozen = True
    complex_eval_net.activity_recordings_rate = 1
    complex_eval_net.connectivity_recordings_rate = np.inf

    complex_eval_input_params = _make_input_params(
        input_params,
        latent_specs,
        num_days=complex_eval_days,
        num_swaps=posttrain_num_swaps,
    )
    if verbose:
        print(
            f"[pid={os.getpid()}] generalization_simple_complex "
            f"mode={network_mode} seed={seed} num_swaps={num_swaps} "
            "checkpoint=complex_eval_start",
            flush=True,
        )
    complex_eval_input, complex_eval_input_episodes, complex_eval_input_latents = make_input(**complex_eval_input_params)
    with torch.no_grad():
        for day in range(complex_eval_input_params["num_days"]):
            latent_day = (
                complex_eval_input_latents[day] if use_true_latent_to_mtl_semantic else None
            )
            complex_eval_net(complex_eval_input[day], debug=False, true_latent=latent_day)
    if verbose:
        print(
            f"[pid={os.getpid()}] generalization_simple_complex "
            f"mode={network_mode} seed={seed} num_swaps={num_swaps} "
            "checkpoint=complex_eval_done",
            flush=True,
        )

    if verbose:
        print(
            f"[pid={os.getpid()}] generalization_simple_complex "
            f"mode={network_mode} seed={seed} num_swaps={num_swaps} "
            "checkpoint=episode_accuracy_start",
            flush=True,
        )
    complex_accuracy_results = _get_ctx_episode_accuracy_from_frozen_eval_net(
        complex_eval_net,
        complex_eval_input_episodes,
        latent_specs,
        verbose=verbose,
        debug_label=(
            f"[pid={os.getpid()}] generalization_simple_complex "
            f"mode={network_mode} seed={seed} num_swaps={num_swaps}"
        ),
    )
    if verbose:
        print(
            f"[pid={os.getpid()}] generalization_simple_complex "
            f"mode={network_mode} seed={seed} num_swaps={num_swaps} "
            "checkpoint=episode_accuracy_done",
            flush=True,
        )

    if verbose:
        print(
            f"[pid={os.getpid()}] generalization_simple_complex "
            f"mode={network_mode} seed={seed} num_swaps={num_swaps} "
            f"checkpoint=return_network_{'start' if return_network else 'skip'}",
            flush=True,
        )
    returned_network = deepcopy(complex_train_net) if return_network else None
    if verbose and return_network:
        print(
            f"[pid={os.getpid()}] generalization_simple_complex "
            f"mode={network_mode} seed={seed} num_swaps={num_swaps} "
            "checkpoint=return_network_done",
            flush=True,
        )
    if returned_network is not None:
        returned_network.record_everything_params = _make_record_everything_params(returned_network)
        returned_network.return_network_records_everything = bool(record)
        for attr_name in [
            "ordered_indices_ctx",
            "selectivity_ctx",
            "ordered_indices_mtl_sensory",
            "selectivity_mtl_sensory",
            "ordered_indices_mtl_semantic",
            "selectivity_mtl_semantic",
        ]:
            if attr_name in simple_accuracy_results:
                setattr(returned_network, attr_name, simple_accuracy_results[attr_name].detach().cpu())

    result = {
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
        "record": bool(record),
        **simple_accuracy_results,
        **complex_accuracy_results,
        "network": returned_network,
    }
    for key in [
        "ordered_indices_ctx",
        "selectivity_ctx",
        "ordered_indices_mtl_sensory",
        "selectivity_mtl_sensory",
        "ordered_indices_mtl_semantic",
        "selectivity_mtl_semantic",
        "ordered_indices_ctx_episodes",
        "selectivity_ctx_episodes",
    ]:
        result.pop(key, None)
    if verbose:
        print(
            f"[pid={os.getpid()}] DONE generalization_simple_complex "
            f"mode={network_mode} seed={seed} num_swaps={num_swaps}",
            flush=True,
        )
    return result


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
    print_rate=np.inf,
    verbose=False,
    return_network=False,
    record=False,
):
    return generalization_simple_complex(
        network_parameters=deepcopy(network_parameters),
        input_params=deepcopy(input_params),
        latent_specs=deepcopy(latent_specs),
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
        print_rate=print_rate,
        verbose=verbose,
        return_network=return_network,
        record=record,
    )


def set_generalization_simple_complex_shared_config(
    network_parameters,
    input_params,
    latent_specs,
    phase_A=200,
    phase_B=400,
    simple_train_days=800,
    simple_eval_days=100,
    complex_train_days=400,
    complex_eval_days=100,
    print_rate=np.inf,
    verbose=False,
):
    global _GENERALIZATION_SIMPLE_COMPLEX_SHARED
    _GENERALIZATION_SIMPLE_COMPLEX_SHARED = {
        "network_parameters": deepcopy(network_parameters),
        "input_params": deepcopy(input_params),
        "latent_specs": deepcopy(latent_specs),
        "phase_A": int(phase_A),
        "phase_B": int(phase_B),
        "simple_train_days": int(simple_train_days),
        "simple_eval_days": int(simple_eval_days),
        "complex_train_days": int(complex_train_days),
        "complex_eval_days": int(complex_eval_days),
        "print_rate": print_rate,
        "verbose": bool(verbose),
    }


def generalization_simple_complex_shared_job(
    network_mode,
    num_swaps,
    seed=None,
    return_network=False,
    pretrain_num_swaps=None,
    record=False,
):
    if _GENERALIZATION_SIMPLE_COMPLEX_SHARED is None:
        raise RuntimeError(
            "Shared generalization config is not set. "
            "Call set_generalization_simple_complex_shared_config(...) before starting the pool."
        )

    cfg = _GENERALIZATION_SIMPLE_COMPLEX_SHARED
    return generalization_simple_complex(
        network_parameters=deepcopy(cfg["network_parameters"]),
        input_params=deepcopy(cfg["input_params"]),
        latent_specs=deepcopy(cfg["latent_specs"]),
        network_mode=network_mode,
        num_swaps=num_swaps,
        pretrain_num_swaps=pretrain_num_swaps,
        seed=seed,
        phase_A=cfg["phase_A"],
        phase_B=cfg["phase_B"],
        simple_train_days=cfg["simple_train_days"],
        simple_eval_days=cfg["simple_eval_days"],
        complex_train_days=cfg["complex_train_days"],
        complex_eval_days=cfg["complex_eval_days"],
        print_rate=cfg["print_rate"],
        verbose=cfg["verbose"],
        return_network=return_network,
        record=record,
    )


def measure_charge_2_ctx_episode_selectivity(
    network,
    recording_parameters,
    input_params,
    latent_specs,
    num_swaps,
    num_days=200,
    semantic_charge=2,
    num_stored_recordings=0,
    seed=None,
    print_rate=np.inf,
    verbose=False,
    network_name=None,
    eval_num_days=100,
):
    del recording_parameters, num_stored_recordings

    if semantic_charge != 2:
        raise ValueError("measure_charge_2_ctx_episode_selectivity expects semantic_charge=2.")

    if seed is not None:
        seed_everything(seed)

    train_net = deepcopy(network)
    train_net.max_semantic_charge_replay = 2
    train_net.frozen = False
    train_net.init_recordings(_make_recording_params())

    train_input_params = _make_input_params(
        input_params,
        latent_specs,
        num_days=num_days,
        num_swaps=num_swaps,
    )
    _, _, _, train_net = _run_network_with_progress(
        train_net,
        train_input_params,
        sleep=True,
        print_rate=print_rate,
        verbose=verbose,
        job_label=f"ctx-episode-selectivity:{network_name}:seed={seed}:swaps={num_swaps}:train",
    )

    eval_recording_parameters = {
        "regions": ["mtl_sensory", "mtl_semantic", "mtl", "ctx"],
        "rate_activity": 1,
        "connections": [],
        "rate_connectivity": np.inf,
    }
    eval_net = deepcopy(train_net)
    eval_net.init_recordings(eval_recording_parameters)
    eval_net.frozen = True
    eval_net.activity_recordings_rate = 1
    eval_net.connectivity_recordings_rate = np.inf

    eval_input_params = _make_input_params(
        input_params,
        latent_specs,
        num_days=eval_num_days,
        num_swaps=num_swaps,
    )
    eval_input, eval_input_episodes, eval_input_latents = make_input(**eval_input_params)

    with torch.no_grad():
        for day in range(eval_input_params["num_days"]):
            eval_net(eval_input[day], debug=False)

    X_ctx = torch.stack(eval_net.activity_recordings["ctx"], dim=0)[eval_net.awake_indices]
    X_mtl_semantic = torch.stack(
        eval_net.activity_recordings["mtl_semantic"], dim=0
    )[eval_net.awake_indices]
    X_mtl_sensory = torch.stack(
        eval_net.activity_recordings["mtl_sensory"], dim=0
    )[eval_net.awake_indices]

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

    selectivity_ctx, ordered_indices_ctx = get_ordered_indices(
        X_ctx[:, :100],
        X_latent_AB,
        assembly_size=10,
    )
    selectivity_mtl_semantic, ordered_indices_mtl_semantic = get_ordered_indices(
        X_mtl_semantic,
        X_latent_AB,
        assembly_size=5,
    )
    selectivity_mtl_sensory, ordered_indices_mtl_sensory = get_ordered_indices(
        X_mtl_sensory,
        X_latent_AB,
        assembly_size=10,
    )
    selectivity_ctx_episodes, ordered_indices_ctx_episodes = get_ordered_indices(
        X_ctx[:, 100:],
        X_episodes,
        assembly_size=10,
    )
    ordered_indices_ctx_episodes = ordered_indices_ctx_episodes + 100

    ctx_episode_selectivity = (
        selectivity_ctx_episodes[ordered_indices_ctx_episodes - 100].max(dim=1)[0][:250].mean()
    )

    return {
        "num_swaps": int(num_swaps),
        "num_days": int(num_days),
        "semantic_charge": int(semantic_charge),
        "ctx_episode_selectivity": float(ctx_episode_selectivity.item()),
        # Alias for reusing the existing scalar-results aggregator/plot path.
        "mean_margin": float(ctx_episode_selectivity.item()),
        "ordered_indices_ctx": ordered_indices_ctx.detach().cpu(),
        "ordered_indices_mtl_semantic": ordered_indices_mtl_semantic.detach().cpu(),
        "ordered_indices_mtl_sensory": ordered_indices_mtl_sensory.detach().cpu(),
        "ordered_indices_ctx_episodes": ordered_indices_ctx_episodes.detach().cpu(),
        "selectivity_ctx_episodes": selectivity_ctx_episodes.detach().cpu(),
    }


def measure_charge_2_ctx_episode_selectivity_job(
    network_name,
    network,
    recording_parameters,
    input_params,
    latent_specs,
    num_swaps,
    num_days,
    semantic_charge,
    num_stored_recordings,
    seed,
    print_rate,
    verbose=False,
):
    result = measure_charge_2_ctx_episode_selectivity(
        network=deepcopy(network),
        recording_parameters=recording_parameters,
        input_params=input_params,
        latent_specs=latent_specs,
        num_swaps=num_swaps,
        num_days=num_days,
        semantic_charge=semantic_charge,
        num_stored_recordings=num_stored_recordings,
        seed=seed,
        print_rate=print_rate,
        verbose=verbose,
        network_name=network_name,
    )
    result["network_name"] = network_name
    result["seed"] = int(seed)
    return result


def measure_sleep_a_replay_episode_clustering_job(
    network_name,
    network,
    recording_parameters,
    input_params,
    latent_specs,
    num_swaps,
    num_days,
    semantic_charge,
    num_stored_recordings,
    seed,
    print_rate,
    verbose=False,
):
    result = measure_sleep_a_replay_episode_clustering(
        network=deepcopy(network),
        recording_parameters=recording_parameters,
        input_params=input_params,
        latent_specs=latent_specs,
        num_swaps=num_swaps,
        num_days=num_days,
        semantic_charge=semantic_charge,
        num_stored_recordings=num_stored_recordings,
        seed=seed,
        print_rate=print_rate,
        verbose=verbose,
        network_name=network_name,
    )
    result["network_name"] = network_name
    result["seed"] = int(seed)
    return result


def evaluate_noise_sweep(
    semantics_absent_network,
    semantics_present_network,
    recording_parameters,
    input_params,
    latent_specs,
    noise_levels=None,
    days_per_level=50,
    semantic_charge=1,
    num_stored_recordings=0,
    print_rate=np.inf,
    num_workers=None,
    multiprocessing_context="fork",
):
    if noise_levels is None:
        noise_levels = list(range(int(input_params["num_swaps"]), 13))
    else:
        noise_levels = [int(level) for level in noise_levels]

    networks = {
        "semantics_absent": semantics_absent_network,
        "semantics_present": semantics_present_network,
    }

    results = {
        "noise_levels": np.array(noise_levels, dtype=int),
        "days_per_level": int(days_per_level),
        "semantic_charge": int(semantic_charge),
        "num_stored_recordings": int(num_stored_recordings),
        "by_network": {name: [] for name in networks},
    }

    experiment_params = []
    for network_name, network in networks.items():
        for num_swaps in noise_levels:
            experiment_params.append(
                (
                    network_name,
                    network,
                    deepcopy(recording_parameters),
                    deepcopy(input_params),
                    deepcopy(latent_specs),
                    int(num_swaps),
                    int(days_per_level),
                    int(semantic_charge),
                    int(num_stored_recordings),
                    0,
                    print_rate,
                )
            )

    if num_workers is None or int(num_workers) <= 1:
        job_results = [
            measure_sleep_a_prototype_overlap_job(*params)
            for params in experiment_params
        ]
    else:
        try:
            mp_context = multiprocessing.get_context(multiprocessing_context)
        except ValueError:
            mp_context = multiprocessing.get_context()

        with mp_context.Pool(processes=int(num_workers)) as pool:
            job_results = pool.starmap(
                measure_sleep_a_prototype_overlap_job,
                experiment_params,
            )

    for network_name in results["by_network"]:
        network_results = [job for job in job_results if job["network_name"] == network_name]
        network_results.sort(key=lambda job: job["num_swaps"])
        for job in network_results:
            job = dict(job)
            job.pop("network_name", None)
            results["by_network"][network_name].append(job)

    return results


def run_codex_figure_5_seed(
    network_parameters,
    recording_parameters,
    input_params,
    latent_specs,
    seed,
    pretrain_num_swaps=4,
    duration_phase_A=200,
    duration_phase_B=400,
    total_days=600,
    checkpoint_day=200,
    noise_levels=None,
    days_per_level=50,
    semantic_charge=1,
    num_stored_recordings=0,
    print_rate=50,
    num_workers=None,
    multiprocessing_context="fork",
):
    trained_networks = train_phase_split_networks(
        network_parameters=network_parameters,
        input_params=input_params,
        latent_specs=latent_specs,
        seed=seed,
        pretrain_num_swaps=pretrain_num_swaps,
        duration_phase_A=duration_phase_A,
        duration_phase_B=duration_phase_B,
        total_days=total_days,
        checkpoint_day=checkpoint_day,
        print_rate=print_rate,
    )

    sweep = evaluate_noise_sweep(
        semantics_absent_network=trained_networks["semantics_absent"],
        semantics_present_network=trained_networks["semantics_present"],
        recording_parameters=recording_parameters,
        input_params=input_params,
        latent_specs=latent_specs,
        noise_levels=noise_levels,
        days_per_level=days_per_level,
        semantic_charge=semantic_charge,
        num_stored_recordings=num_stored_recordings,
        print_rate=np.inf,
        num_workers=num_workers,
        multiprocessing_context=multiprocessing_context,
    )
    sweep["seed"] = int(seed)
    return sweep


def run_codex_figure_5_experiment(
    network_parameters,
    recording_parameters,
    input_params,
    latent_specs,
    seeds,
    pretrain_num_swaps=4,
    duration_phase_A=200,
    duration_phase_B=400,
    total_days=600,
    checkpoint_day=200,
    noise_levels=None,
    days_per_level=50,
    semantic_charge=1,
    num_stored_recordings=0,
    print_rate=50,
    num_workers=None,
    multiprocessing_context="fork",
):
    seeds = [int(seed) for seed in seeds]
    if noise_levels is None:
        noise_levels = list(range(int(pretrain_num_swaps), 13))
    else:
        noise_levels = [int(level) for level in noise_levels]

    per_seed_results = []
    for seed in seeds:
        per_seed_results.append(
            run_codex_figure_5_seed(
                network_parameters=network_parameters,
                recording_parameters=recording_parameters,
                input_params=input_params,
                latent_specs=latent_specs,
                seed=seed,
                pretrain_num_swaps=pretrain_num_swaps,
                duration_phase_A=duration_phase_A,
                duration_phase_B=duration_phase_B,
                total_days=total_days,
                checkpoint_day=checkpoint_day,
                noise_levels=noise_levels,
                days_per_level=days_per_level,
                semantic_charge=semantic_charge,
                num_stored_recordings=num_stored_recordings,
                print_rate=print_rate,
                num_workers=num_workers,
                multiprocessing_context=multiprocessing_context,
            )
        )

    aggregate = {
        "seeds": seeds,
        "noise_levels": np.array(noise_levels, dtype=int),
        "days_per_level": int(days_per_level),
        "semantic_charge": int(semantic_charge),
        "num_stored_recordings": int(num_stored_recordings),
        "per_seed": per_seed_results,
        "summary": {},
    }

    for network_name in ["semantics_absent", "semantics_present"]:
        seed_curves = np.stack(
            [
                np.array(
                    [entry["mean_max_overlap"] for entry in seed_result["by_network"][network_name]],
                    dtype=float,
                )
                for seed_result in per_seed_results
            ],
            axis=0,
        )

        aggregate["summary"][network_name] = {
            "seed_curves": seed_curves,
            "mean_curve": seed_curves.mean(axis=0),
            "std_curve": seed_curves.std(axis=0),
        }

    return aggregate


def plot_codex_figure_5(
    results,
    ax=None,
    colors=None,
    labels=None,
    title=None,
    ylabel="Mean Max Overlap",
    xlabel=None,
    x_values=None,
    xticklabels=None,
    xscale=None,
    scatter_alpha=0.35,
    line_width=2.5,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.5))

    if colors is None:
        colors = {
            "semantics_absent": "#F15A2A",
            "semantics_clean": "#00A651",
            "semantics_random": "#7A7A7A",
            "semantics_present": "#00AEEF",
        }

    if x_values is None:
        if "snr_levels" in results:
            x = np.asarray(results["snr_levels"], dtype=float)
            if xlabel is None:
                xlabel = "SNR"
            if xscale is None:
                xscale = "log"
            if xticklabels is None:
                xticklabels = [f"{value:.2f}" for value in x]
        else:
            x = np.asarray(results["noise_levels"])
    else:
        x = np.asarray(x_values)

    if xlabel is None:
        xlabel = "num_swaps"

    if labels is None:
        labels = {
            "semantics_absent": "semantics absent",
            "semantics_clean": "semantics clean",
            "semantics_random": "semantics random",
            "semantics_present": "semantics present",
        }

    for network_name in results["summary"]:
        seed_curves = results["summary"][network_name]["seed_curves"]
        mean_curve = results["summary"][network_name]["mean_curve"]
        std_curve = results["summary"][network_name]["std_curve"]
        color = colors.get(network_name, "#000000")

        for seed_curve in seed_curves:
            ax.scatter(x, seed_curve, color=color, alpha=scatter_alpha, s=24)

        ax.plot(
            x,
            mean_curve,
            color=color,
            linewidth=line_width,
            label=labels.get(network_name, network_name),
        )
        ax.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.15,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if xscale is not None:
        ax.set_xscale(xscale)
    ax.set_ylim(0, 1.02)
    if title is not None:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=16)

    return ax
