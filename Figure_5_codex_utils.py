import multiprocessing
import os
import random
import time

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from model import SSCNetwork
from src.utils.general import (
    LatentSpace,
    get_max_overlap,
    get_ordered_indices,
    get_prototypes,
    make_input,
    test_network,
)


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
):
    if not verbose:
        return test_network(net, input_params, sleep=sleep, print_rate=print_rate)

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
            net(input_tensor[day], debug=False)
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

    network_semantics_absent = deepcopy(network)
    network_semantics_absent.sensory_replay_only = True

    remaining_days = int(total_days) - int(checkpoint_day)
    continuation_input_params = _make_input_params(
        training_input_params,
        latent_specs,
        num_days=remaining_days,
        num_swaps=pretrain_num_swaps,
    )
    _, _, _, network = _run_network_with_progress(
        network,
        continuation_input_params,
        sleep=True,
        print_rate=print_rate,
        verbose=False,
        job_label="pretrain-continuation",
    )

    network_semantics_present = deepcopy(network)

    return {
        "semantics_absent": network_semantics_absent,
        "semantics_present": network_semantics_present,
    }


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

        seed_curves = np.stack(seed_curves, axis=0)
        aggregate["summary"][network_name] = {
            "seed_curves": seed_curves,
            "mean_curve": seed_curves.mean(axis=0),
            "std_curve": seed_curves.std(axis=0),
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
    episode_prototypes = get_prototypes(
        eval_input_params["latent_space"],
        semantic_charge=2,
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

    episode_overlap, inferred_episode_indices = get_max_overlap(
        replayed_mtl_sensory,
        episode_prototypes,
        return_indices=True,
    )

    replay_mtl_norm = F.normalize(replayed_mtl_full.float(), dim=1)
    similarity_matrix = replay_mtl_norm @ replay_mtl_norm.T

    sim_same = []
    sim_different = []
    margins = []
    inferred_episode_indices_cpu = inferred_episode_indices.detach().cpu()

    for replay_idx in range(replayed_mtl_full.shape[0]):
        same_mask = inferred_episode_indices_cpu == inferred_episode_indices_cpu[replay_idx]
        same_mask[replay_idx] = False
        different_mask = inferred_episode_indices_cpu != inferred_episode_indices_cpu[replay_idx]

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
    stored_recording_episode_indices = inferred_episode_indices_cpu[stored_recording_indices].clone()

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
        "stored_recording_episode_indices": stored_recording_episode_indices,
        "replay_episode_overlap": episode_overlap.detach().cpu(),
        "inferred_replay_episodes": inferred_episode_indices_cpu.clone(),
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
    scatter_alpha=0.35,
    line_width=2.5,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.5))

    if colors is None:
        colors = {
            "semantics_absent": "#F15A2A",
            "semantics_clean": "#00A651",
            "semantics_random": "#7A7A7A",
            "semantics_present": "#00AEEF",
        }

    x = results["noise_levels"]
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

    ax.set_xlabel("num_swaps")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_ylim(0, 1.02)
    if title is not None:
        ax.set_title(title)
    ax.legend(frameon=False)

    return ax
