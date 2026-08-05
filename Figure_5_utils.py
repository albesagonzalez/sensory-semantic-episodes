"""Utilities for Figure_experiments_circuits notebook experiments."""

import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from src.utils.episode_generation_protocol import (
    LatentSpace,
    make_input,
)
from src.utils.general import (
    get_cos_sim_torch,
    get_ordered_indices,
    get_sample_from_num_swaps,
    get_selectivity,
    seed_everything,
)


def build_two_example_prob_list(
    dims,
    target_latent,
    target_idx,
    fixed_opposite_idx,
    contrast_idx,
):
    probs = []
    dim_a, dim_b = dims

    for a_idx in range(dim_a):
        for b_idx in range(dim_b):
            prob = 0.0
            if target_latent == 0:
                if b_idx == fixed_opposite_idx and a_idx == target_idx:
                    prob = 0.5
                elif b_idx == fixed_opposite_idx and a_idx == contrast_idx:
                    prob = 0.5
            else:
                if a_idx == fixed_opposite_idx and b_idx == target_idx:
                    prob = 0.5
                elif a_idx == fixed_opposite_idx and b_idx == contrast_idx:
                    prob = 0.5
            probs.append(prob)

    return probs


def build_excluding_train_context_prob_list(
    dims,
    target_latent,
    target_idx,
    excluded_opposite_idx,
):
    # Test distribution:
    # - Exclude all pairs containing the training opposite context.
    # - Set target marginal to 0.5.
    # - Distribute remaining 0.5 equally across all non-target values.
    dim_a, dim_b = dims

    if target_latent == 0 and dim_b <= 1:
        raise ValueError("Need at least 2 B concepts to exclude one training context.")
    if target_latent == 1 and dim_a <= 1:
        raise ValueError("Need at least 2 A concepts to exclude one training context.")

    probs = []
    for a_idx in range(dim_a):
        for b_idx in range(dim_b):
            prob = 0.0
            if target_latent == 0:
                # A-target task: exclude B_train.
                if b_idx != excluded_opposite_idx:
                    if a_idx == target_idx:
                        prob = 0.5 / (dim_b - 1)
                    else:
                        prob = 0.5 / ((dim_a - 1) * (dim_b - 1))
            else:
                # B-target task: exclude A_train.
                if a_idx != excluded_opposite_idx:
                    if b_idx == target_idx:
                        prob = 0.5 / (dim_a - 1)
                    else:
                        prob = 0.5 / ((dim_b - 1) * (dim_a - 1))
            probs.append(prob)

    return probs


def get_binary_labels(target_latent, target_idx, fixed_opposite_idx, contrast_idx):
    if target_latent == 0:
        positive_label = (target_idx, fixed_opposite_idx)
        negative_label = (contrast_idx, fixed_opposite_idx)
    else:
        positive_label = (fixed_opposite_idx, target_idx)
        negative_label = (fixed_opposite_idx, contrast_idx)
    return positive_label, negative_label


def make_deterministic_two_block_input(
    latent_space,
    positive_label,
    negative_label,
    mean_duration,
    num_swaps,
):
    if mean_duration < 1:
        raise ValueError("mean_duration must be >= 1 for deterministic two-block input")

    day_length = 2 * int(mean_duration)
    input_tensor = torch.zeros((1, day_length, latent_space.total_size))
    input_latents = torch.zeros((1, day_length, latent_space.num_latents), dtype=torch.int32)

    block_labels = [positive_label, negative_label]
    for block_idx, label in enumerate(block_labels):
        pattern = latent_space.label_to_neurons[tuple(label)]
        start = block_idx * int(mean_duration)
        end = (block_idx + 1) * int(mean_duration)

        input_latents[0, start:end, :] = torch.tensor(label, dtype=torch.int32)
        for t in range(start, end):
            input_tensor[0, t] = get_sample_from_num_swaps(pattern, num_swaps)

    return input_tensor, input_latents


def make_excluding_context_test_input_params(
    test_sampling_params,
    base_latent_specs,
    target_latent,
    target_idx,
    excluded_opposite_idx,
):
    input_params = deepcopy(test_sampling_params)

    latent_specs = deepcopy(base_latent_specs)
    latent_specs["prob_list"] = build_excluding_train_context_prob_list(
        dims=latent_specs["dims"],
        target_latent=target_latent,
        target_idx=target_idx,
        excluded_opposite_idx=excluded_opposite_idx,
    )
    input_params["latent_space"] = LatentSpace(**latent_specs)
    return input_params


def rollout_activity_from_input_tensor(net, recording_parameters, input_tensor, input_latents, target_latent, target_idx):
    net.init_recordings(recording_parameters)
    net.frozen = True

    with torch.no_grad():
        for day in range(input_tensor.shape[0]):
            net(input_tensor[day], debug=False)

    n_samples = input_tensor.shape[0] * input_tensor.shape[1]

    X_mtl_sensory = (
        torch.stack(net.activity_recordings["mtl_sensory"], dim=0)[net.awake_indices][-n_samples:]
        .float()
        .numpy()
    )
    X_mtl_semantic = (
        torch.stack(net.activity_recordings["mtl_semantic"], dim=0)[net.awake_indices][-n_samples:]
        .float()
        .numpy()
    )

    y = (input_latents[:, :, target_latent] == target_idx).reshape(-1).long().numpy()
    return X_mtl_sensory, X_mtl_semantic, y


def rollout_train_two_block(
    net,
    recording_parameters,
    base_latent_specs,
    train_within_day_stats,
    target_latent,
    target_idx,
    fixed_opposite_idx,
    contrast_idx,
):
    latent_specs = deepcopy(base_latent_specs)
    latent_space = LatentSpace(**latent_specs)

    positive_label, negative_label = get_binary_labels(
        target_latent=target_latent,
        target_idx=target_idx,
        fixed_opposite_idx=fixed_opposite_idx,
        contrast_idx=contrast_idx,
    )

    input_tensor, input_latents = make_deterministic_two_block_input(
        latent_space=latent_space,
        positive_label=positive_label,
        negative_label=negative_label,
        mean_duration=int(train_within_day_stats["mean_duration"]),
        num_swaps=int(train_within_day_stats["num_swaps"]),
    )

    return rollout_activity_from_input_tensor(
        net=net,
        recording_parameters=recording_parameters,
        input_tensor=input_tensor,
        input_latents=input_latents,
        target_latent=target_latent,
        target_idx=target_idx,
    )


def rollout_test_random_until_binary(
    net,
    recording_parameters,
    input_params,
    target_latent,
    target_idx,
    seed,
    max_attempts=25,
):
    last_counts = None

    for attempt in range(max_attempts):
        seed_everything(seed + attempt)
        input_tensor, _, input_latents = make_input(**input_params)

        X_sens, X_sem, y = rollout_activity_from_input_tensor(
            net=net,
            recording_parameters=recording_parameters,
            input_tensor=input_tensor,
            input_latents=input_latents,
            target_latent=target_latent,
            target_idx=target_idx,
        )

        unique, counts = np.unique(y, return_counts=True)
        if unique.size == 2:
            return X_sens, X_sem, y, attempt

        last_counts = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}

    raise ValueError(
        f"Could not sample both test classes after {max_attempts} attempts. "
        f"Last class counts: {last_counts}"
    )


def fit_linear_decoder_accuracy(X_train, y_train, X_test, y_test, seed):
    decoder = LogisticRegression(
        random_state=seed,
        solver="liblinear",
        max_iter=500,
    )
    decoder.fit(X_train, y_train)
    return float(decoder.score(X_test, y_test))


def run_cross_context_concept_cell_experiment(
    net,
    recording_parameters,
    base_latent_specs,
    train_within_day_stats,
    test_sampling_params,
    num_repeats=10,
    seed=0,
    max_test_attempts=25,
):
    dims = base_latent_specs["dims"]
    rng = np.random.default_rng(seed)

    concept_specs = [("A", 0, idx) for idx in range(dims[0])] + [("B", 1, idx) for idx in range(dims[1])]

    results = []
    base_day = int(net.day)

    for concept_name, target_latent, target_idx in concept_specs:
        opposite_dim = dims[1 - target_latent]
        target_dim = dims[target_latent]
        print(f"Running concept {concept_name}{target_idx + 1}...")

        for repeat in range(num_repeats):
            train_context = int(rng.integers(opposite_dim))
            contrast_candidates = [c for c in range(target_dim) if c != target_idx]
            contrast_idx = int(rng.choice(contrast_candidates))

            net.day = base_day
            X_sens_train, X_sem_train, y_train = rollout_train_two_block(
                net=net,
                recording_parameters=recording_parameters,
                base_latent_specs=base_latent_specs,
                train_within_day_stats=train_within_day_stats,
                target_latent=target_latent,
                target_idx=target_idx,
                fixed_opposite_idx=train_context,
                contrast_idx=contrast_idx,
            )

            test_input_params = make_excluding_context_test_input_params(
                test_sampling_params=test_sampling_params,
                base_latent_specs=base_latent_specs,
                target_latent=target_latent,
                target_idx=target_idx,
                excluded_opposite_idx=train_context,
            )

            rollout_seed = seed + 100000 * target_latent + 10000 * target_idx + 100 * repeat

            net.day = base_day
            X_sens_test, X_sem_test, y_test, test_retry = rollout_test_random_until_binary(
                net=net,
                recording_parameters=recording_parameters,
                input_params=test_input_params,
                target_latent=target_latent,
                target_idx=target_idx,
                seed=rollout_seed,
                max_attempts=max_test_attempts,
            )

            acc_sensory = fit_linear_decoder_accuracy(
                X_train=X_sens_train,
                y_train=y_train,
                X_test=X_sens_test,
                y_test=y_test,
                seed=rollout_seed,
            )

            acc_semantic = fit_linear_decoder_accuracy(
                X_train=X_sem_train,
                y_train=y_train,
                X_test=X_sem_test,
                y_test=y_test,
                seed=rollout_seed,
            )

            train_pos = int(np.sum(y_train == 1))
            train_neg = int(np.sum(y_train == 0))
            test_pos = int(np.sum(y_test == 1))
            test_neg = int(np.sum(y_test == 0))

            results.append(
                {
                    "concept": f"{concept_name}{target_idx + 1}",
                    "target_latent": target_latent,
                    "target_index": target_idx,
                    "contrast_index": contrast_idx,
                    "repeat": repeat,
                    "train_context": train_context,
                    "excluded_test_context": train_context,
                    "train_yes": train_pos,
                    "train_no": train_neg,
                    "test_yes": test_pos,
                    "test_no": test_neg,
                    "test_retry": test_retry,
                    "region": "MTL-sensory",
                    "accuracy": acc_sensory,
                }
            )
            results.append(
                {
                    "concept": f"{concept_name}{target_idx + 1}",
                    "target_latent": target_latent,
                    "target_index": target_idx,
                    "contrast_index": contrast_idx,
                    "repeat": repeat,
                    "train_context": train_context,
                    "excluded_test_context": train_context,
                    "train_yes": train_pos,
                    "train_no": train_neg,
                    "test_yes": test_pos,
                    "test_no": test_neg,
                    "test_retry": test_retry,
                    "region": "MTL-semantic",
                    "accuracy": acc_semantic,
                }
            )

    return pd.DataFrame(results)


def _build_balanced_object_to_scene(num_objects, num_scenes, seed, shuffle_assignments=False):
    rng = np.random.default_rng(seed)
    object_to_scene = np.repeat(np.arange(num_scenes), int(np.ceil(num_objects / num_scenes)))[:num_objects]
    if shuffle_assignments:
        rng.shuffle(object_to_scene)
    return object_to_scene.astype(int)


def _build_unique_object_latent_specs(
    num_objects,
    num_scenes,
    total_sizes,
    act_sizes,
):
    if len(total_sizes) != 2 or len(act_sizes) != 2:
        raise ValueError("total_sizes and act_sizes must each have length 2.")
    return {
        "num": 2,
        "total_sizes": [int(total_sizes[0]), int(total_sizes[1])],
        "act_sizes": [int(act_sizes[0]), int(act_sizes[1])],
        "dims": [int(num_objects), int(num_scenes)],
        "prob_list": [1.0 / (num_objects * num_scenes) for _ in range(num_objects * num_scenes)],
        "random_neurons": False,
    }


def _make_deterministic_object_scene_day(
    latent_space,
    object_to_scene,
    mean_duration,
    num_swaps,
    presentations_per_object=1,
    shuffle_objects=False,
    seed=0,
):
    num_objects = len(object_to_scene)
    event_objects = np.repeat(np.arange(num_objects), int(presentations_per_object))
    if shuffle_objects:
        rng = np.random.default_rng(seed)
        rng.shuffle(event_objects)

    day_length = int(mean_duration) * len(event_objects)
    input_day = torch.zeros((day_length, latent_space.total_size))
    latents_day = torch.zeros((day_length, 2), dtype=torch.int32)

    time_idx = 0
    for obj_idx in event_objects:
        scene_idx = int(object_to_scene[int(obj_idx)])
        label = (int(obj_idx), scene_idx)
        pattern = latent_space.label_to_neurons[label]
        for _ in range(int(mean_duration)):
            input_day[time_idx] = get_sample_from_num_swaps(pattern, int(num_swaps))
            latents_day[time_idx] = torch.tensor(label, dtype=torch.int32)
            time_idx += 1

    return input_day, latents_day


def _collect_partial_object_cue_patterns(
    net,
    latent_space,
    num_objects,
    cue_num_swaps=0,
):
    net_eval = deepcopy(net)
    net_eval.frozen = True

    a_size = int(latent_space.total_sizes[0])
    patterns_mtl = []
    patterns_ctx = []

    with torch.no_grad():
        for obj_idx in range(int(num_objects)):
            cue = torch.zeros(latent_space.total_size)
            cue[:a_size] = latent_space.latent_patterns[0][obj_idx]

            if int(cue_num_swaps) > 0:
                cue_a = get_sample_from_num_swaps(cue[:a_size], int(cue_num_swaps))
                cue[:a_size] = cue_a
            cue[a_size:] = 0

            # 1) Pattern-complete in MTL from a partial object cue.
            mtl_init = torch.zeros(net_eval.mtl_size)
            mtl_init[: net_eval.mtl_sensory_size] = cue
            mtl_conditioned = torch.zeros(net_eval.mtl_size)
            mtl_conditioned[: net_eval.mtl_sensory_size] = cue
            mtl_completed = net_eval.pattern_complete(
                "mtl",
                h_0=mtl_init,
                h_conditioned=mtl_conditioned,
            )
            mtl_sensory_final = mtl_completed[: net_eval.mtl_sensory_size].clone()
            mtl_semantic_final = mtl_completed[net_eval.mtl_sensory_size :].clone()

            # 2) Feed completed MTL through the network inference path to get final CTX/MTL.
            # First pass: sensory-driven CTX activation.
            ctx_hat = (
                F.linear(
                    mtl_sensory_final,
                    net_eval.ctx_mtl[:, : net_eval.mtl_sensory_size],
                )
                + net_eval.ctx_b * net_eval.ctx_IM
            )
            ctx, _ = net_eval.activation(ctx_hat, "ctx")

            # CTX -> MTL-semantic feedback (active after phase A).
            if net_eval.day >= net_eval.duration_phase_A:
                mtl_semantic_hat = (
                    F.linear(ctx, net_eval.mtl_semantic_ctx)
                    + net_eval.mtl_semantic_b * net_eval.mtl_semantic_IM
                )
                mtl_semantic_final, _ = net_eval.activation(mtl_semantic_hat, "mtl_semantic")

            # Compose final MTL explicitly from sensory + semantic components.
            mtl_final = torch.zeros_like(mtl_completed)
            mtl_final[: net_eval.mtl_sensory_size] = mtl_sensory_final
            mtl_final[net_eval.mtl_sensory_size :] = mtl_semantic_final

            # Final CTX update with full MTL (active after phase B).
            if net_eval.day >= net_eval.duration_phase_B:
                ctx_hat = F.linear(mtl_final, net_eval.ctx_mtl) + net_eval.ctx_b * net_eval.ctx_IM
                ctx, _ = net_eval.activation(ctx_hat, "ctx")

            net_eval.mtl = mtl_final.clone()
            net_eval.mtl_sensory = mtl_sensory_final.clone()
            net_eval.mtl_semantic = mtl_semantic_final.clone()
            net_eval.ctx = ctx.clone()

            patterns_mtl.append(net_eval.mtl.detach().clone())
            patterns_ctx.append(net_eval.ctx.detach().clone())

    return {
        "MTL": torch.stack(patterns_mtl, dim=0).float().numpy(),
        "CTX": torch.stack(patterns_ctx, dim=0).float().numpy(),
    }


def _compute_overlap_dataframe(patterns, object_to_scene, region, phase):
    patterns = np.asarray(patterns, dtype=float)
    object_to_scene = np.asarray(object_to_scene, dtype=int)
    n_items = patterns.shape[0]

    corr = np.corrcoef(patterns)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    rows = []
    for i in range(n_items):
        same_mask = object_to_scene == object_to_scene[i]
        same_mask[i] = False
        diff_mask = object_to_scene != object_to_scene[i]

        overlap = float(np.mean(corr[i, same_mask])) if np.any(same_mask) else np.nan
        non_overlap = float(np.mean(corr[i, diff_mask])) if np.any(diff_mask) else np.nan

        rows.append(
            {
                "item_index": int(i),
                "scene_index": int(object_to_scene[i]),
                "region": str(region),
                "phase": str(phase),
                "overlap_similarity": overlap,
                "non_overlap_similarity": non_overlap,
                "overlap_index": overlap - non_overlap,
            }
        )

    return pd.DataFrame(rows)


def run_increase_overlap_experiment(
    network_parameters,
    recording_parameters,
    num_objects=128,
    num_scenes=4,
    num_days_train=1,
    presentations_per_object=1,
    train_mean_duration=1,
    train_num_swaps=2,
    train_shuffle_objects=False,
    act_sizes=(5, 5),
    shuffle_object_scene_assignments=False,
    consolidation_sleep_cycles=100,
    cue_num_swaps=0,
    seed=0,
    duration_phase_A=0,
    duration_phase_B=None,
    sleep_duration_A=None,
    sleep_duration_B=None,
    stop_after="recall",
):
    from src.model import SSCNetwork

    seed_everything(seed)

    if num_days_train < 1:
        raise ValueError("num_days_train must be >= 1")
    if consolidation_sleep_cycles < 0:
        raise ValueError("consolidation_sleep_cycles must be >= 0")
    if stop_after not in {"encoding", "sleep", "recall"}:
        raise ValueError("stop_after must be one of {'encoding', 'sleep', 'recall'}")
    net_params = deepcopy(network_parameters)
    # Enable semantic replay from the beginning while preserving episodic replay.
    net_params["duration_phase_A"] = int(duration_phase_A)
    if duration_phase_B is not None:
        net_params["duration_phase_B"] = int(duration_phase_B)
    if sleep_duration_A is not None:
        net_params["sleep_duration_A"] = int(sleep_duration_A)
    if sleep_duration_B is not None:
        net_params["sleep_duration_B"] = int(sleep_duration_B)

    act_sizes_local = tuple(int(x) for x in act_sizes)
    # Allocate disjoint sensory and MTL-sensory blocks for every object and scene.
    total_sizes_local = (
        int(num_objects) * int(act_sizes_local[0]),
        int(num_scenes) * int(act_sizes_local[1]),
    )

    sen_total_size = int(total_sizes_local[0] + total_sizes_local[1])
    pattern_active_count = int(act_sizes_local[0] + act_sizes_local[1])
    sen_sparse = float(pattern_active_count) / float(sen_total_size)
    sen_sparse_sleep = max(sen_sparse * 0.5, 1.0 / float(sen_total_size))

    # Keep identity sensory->MTL-sensory when resizing, and align sizes/sparsities.
    net_params["mtl_sensory_sen_projection"] = False
    net_params["sen_size_subregions"] = torch.tensor([sen_total_size])
    net_params["mtl_sensory_size_subregions"] = torch.tensor([sen_total_size])
    net_params["sen_sparsity"] = torch.tensor([sen_sparse])
    net_params["sen_sparsity_sleep"] = torch.tensor([sen_sparse_sleep])
    net_params["mtl_sensory_sparsity"] = torch.tensor([sen_sparse])
    net_params["mtl_sensory_sparsity_sleep"] = torch.tensor([sen_sparse_sleep])

    # Keep MTL first subregion matched to mtl_sensory size.
    mtl_sem_size = int(net_params["mtl_size_subregions"][1])
    net_params["mtl_size_subregions"] = torch.tensor([sen_total_size, mtl_sem_size])
    mtl_sparse_sleep_0 = max(sen_sparse * 0.5, 1.0 / float(sen_total_size))
    net_params["mtl_sparsity"] = torch.tensor([sen_sparse, float(net_params["mtl_sparsity"][1])])
    net_params["mtl_sparsity_sleep"] = torch.tensor([mtl_sparse_sleep_0, float(net_params["mtl_sparsity_sleep"][1])])

    rec_params = deepcopy(recording_parameters)
    net = SSCNetwork(net_params, rec_params)

    latent_specs = _build_unique_object_latent_specs(
        num_objects=int(num_objects),
        num_scenes=int(num_scenes),
        total_sizes=total_sizes_local,
        act_sizes=act_sizes_local,
    )
    latent_space = LatentSpace(**latent_specs)

    object_to_scene = _build_balanced_object_to_scene(
        num_objects=int(num_objects),
        num_scenes=int(num_scenes),
        seed=seed,
        shuffle_assignments=bool(shuffle_object_scene_assignments),
    )

    for day_idx in range(int(num_days_train)):
        input_day, _ = _make_deterministic_object_scene_day(
            latent_space=latent_space,
            object_to_scene=object_to_scene,
            mean_duration=int(train_mean_duration),
            num_swaps=int(train_num_swaps),
            presentations_per_object=int(presentations_per_object),
            shuffle_objects=bool(train_shuffle_objects),
            seed=seed + day_idx,
        )
        net(input_day, debug=False)

    if stop_after == "encoding":
        return {"network": net}

    pre_patterns = None
    if stop_after == "recall":
        pre_patterns = _collect_partial_object_cue_patterns(
            net=net,
            latent_space=latent_space,
            num_objects=int(num_objects),
            cue_num_swaps=int(cue_num_swaps),
        )

    for _ in range(int(consolidation_sleep_cycles)):
        net.sleep()

    if stop_after == "sleep":
        return {"network": net}

    post_patterns = _collect_partial_object_cue_patterns(
        net=net,
        latent_space=latent_space,
        num_objects=int(num_objects),
        cue_num_swaps=int(cue_num_swaps),
    )

    overlap_df = pd.concat(
        [
            _compute_overlap_dataframe(pre_patterns["MTL"], object_to_scene, region="MTL", phase="pre"),
            _compute_overlap_dataframe(pre_patterns["CTX"], object_to_scene, region="CTX", phase="pre"),
            _compute_overlap_dataframe(post_patterns["MTL"], object_to_scene, region="MTL", phase="post"),
            _compute_overlap_dataframe(post_patterns["CTX"], object_to_scene, region="CTX", phase="post"),
        ],
        ignore_index=True,
    )

    summary_df = (
        overlap_df.groupby(["region", "phase"])["overlap_index"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_df["sem"] = summary_df["std"] / np.sqrt(summary_df["count"])

    return {
        "network": net,
        "latent_space": latent_space,
        "object_to_scene": object_to_scene,
        "overlap_df": overlap_df,
        "summary_df": summary_df,
        "pre_patterns": pre_patterns,
        "post_patterns": post_patterns,
    }


def _permute_channel(input_tensor, channel_start, channel_end, seed, time_start=None, time_end=None):
    torch.manual_seed(int(seed))
    t0 = 0 if time_start is None else int(time_start)
    t1 = input_tensor.shape[1] if time_end is None else int(time_end)
    source = input_tensor[:, t0:t1, channel_start:channel_end].clone()
    perms = torch.argsort(torch.rand_like(source), dim=2)
    input_tensor[:, t0:t1, channel_start:channel_end] = torch.gather(source, 2, perms)
    return input_tensor


def _cell_indices_from_mean_activity(network, region, activity_tensor):
    mean_vec = activity_tensor.mean(axis=0)
    active_vec, _ = network.activation(mean_vec, region)
    active_cells = torch.nonzero(active_vec == 1, as_tuple=True)[0]
    inactive_cells = torch.nonzero(active_vec != 1, as_tuple=True)[0]
    return active_cells, inactive_cells, active_vec


def _safe_weight_slice(weights, row_idx, col_idx):
    if len(row_idx) == 0 or len(col_idx) == 0:
        return np.array([], dtype=float)
    return weights[row_idx][:, col_idx].flatten().detach().cpu().numpy()


def _sampled_weight_slice(weights, row_idx, col_idx, n_samples=5):
    if len(row_idx) == 0 or len(col_idx) == 0:
        return np.array([], dtype=float)
    row_sel = row_idx[torch.randperm(len(row_idx))[: min(int(n_samples), len(row_idx))]]
    col_sel = col_idx[torch.randperm(len(col_idx))[: min(int(n_samples), len(col_idx))]]
    return weights[row_sel][:, col_sel].flatten().detach().cpu().numpy()


def _copy_network_without_recordings(net):
    """Copy network parameters/state without duplicating potentially long traces."""
    recording_attrs = (
        "activity_recordings",
        "activity_recordings_time",
        "connectivity_recordings",
        "connectivity_recordings_time",
    )
    saved_recordings = {attr: getattr(net, attr) for attr in recording_attrs}
    try:
        net.activity_recordings = {}
        net.activity_recordings_time = []
        net.connectivity_recordings = {}
        net.connectivity_recordings_time = []
        return deepcopy(net)
    finally:
        for attr, value in saved_recordings.items():
            setattr(net, attr, value)


def _get_ctx_ordering_from_frozen_uniform_probe(
    net,
    seed,
    num_days=100,
    day_length=100,
    mean_duration=5,
    num_swaps=4,
):
    """Derive a single CTX concept ordering for all stored network states.

    The probe is evaluated after recall on a frozen copy of the network.  The
    returned indices are global CTX indices, so they can be used directly to
    slice connectivity matrices from any earlier network snapshot.
    """
    eval_net = _copy_network_without_recordings(net)
    eval_net.init_recordings(
        {
            "regions": ["ctx", "mtl_semantic"],
            "rate_activity": 1,
            "connections": [],
            "rate_connectivity": np.inf,
        }
    )
    eval_net.frozen = True

    latent_specs = {
        "num": 2,
        "total_sizes": [50, 50],
        "act_sizes": [10, 10],
        "dims": [5, 5],
        "prob_list": [1 / 25] * 25,
    }
    # Keep this evaluation deterministic without changing the random stream of
    # the subsequent extinction simulation.
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    try:
        seed_everything(seed + 3)
        probe_params = {
            "num_days": int(num_days),
            "day_length": int(day_length),
            "mean_duration": int(mean_duration),
            "fixed_duration": True,
            "num_swaps": int(num_swaps),
            "latent_space": LatentSpace(**latent_specs),
        }
        probe_input, probe_episodes, probe_latents = make_input(**probe_params)
        with torch.no_grad():
            for day in range(probe_params["num_days"]):
                eval_net(probe_input[day], debug=False)
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(torch_rng_state)

    X_ctx = torch.stack(eval_net.activity_recordings["ctx"], dim=0)[eval_net.awake_indices]
    X_mtl_semantic = torch.stack(eval_net.activity_recordings["mtl_semantic"], dim=0)[
        eval_net.awake_indices
    ]
    X_latent_A = F.one_hot(probe_latents[:, :, 0].long(), num_classes=latent_specs["dims"][0])
    X_latent_B = F.one_hot(probe_latents[:, :, 1].long(), num_classes=latent_specs["dims"][1])
    X_latent_AB = torch.cat((X_latent_A, X_latent_B), axis=2)
    X_episodes = F.one_hot(probe_episodes.long(), num_classes=np.prod(latent_specs["dims"]))

    ctx_simple = eval_net.ctx_subregions[0]
    ctx_complex = eval_net.ctx_subregions[1]
    simple_selectivity, simple_ordered_indices = get_ordered_indices(
        X_ctx[:, ctx_simple], X_latent_AB, assembly_size=10, seed=seed + 3
    )
    complex_selectivity, complex_ordered_indices_local = get_ordered_indices(
        X_ctx[:, ctx_complex], X_episodes, assembly_size=10, seed=seed + 3
    )
    mtl_semantic_selectivity, mtl_semantic_ordered_indices = get_ordered_indices(
        X_mtl_semantic, X_latent_AB, assembly_size=10, seed=seed + 3
    )
    simple_labels = [("A", idx) for idx in range(5)] + [("B", idx) for idx in range(5)]

    simple_ordered_indices_global = simple_ordered_indices + ctx_simple[0]
    complex_ordered_indices_global = complex_ordered_indices_local + ctx_complex[0]

    return {
        "simple_selectivity": simple_selectivity,
        "simple_ordered_indices": simple_ordered_indices_global,
        "simple_labels": simple_labels,
        "complex_selectivity": complex_selectivity,
        "complex_ordered_indices": complex_ordered_indices_global,
        "complex_labels": probe_params["latent_space"].index_to_label,
        "ordered_indices": torch.cat(
            (simple_ordered_indices_global, complex_ordered_indices_global)
        ),
        "mtl_semantic_selectivity": mtl_semantic_selectivity,
        "mtl_semantic_ordered_indices": mtl_semantic_ordered_indices,
        "mtl_semantic_labels": simple_labels,
        "input_latents": probe_latents,
        "input_episodes": probe_episodes,
        "parameters": {
            "num_days": int(num_days),
            "day_length": int(day_length),
            "mean_duration": int(mean_duration),
            "num_swaps": int(num_swaps),
            "prob_list": latent_specs["prob_list"],
            "simple_assembly_size": 10,
            "complex_assembly_size": 10,
        },
    }


def _get_frozen_uniform_concept_selectivity(
    net,
    seed,
    num_days=100,
    day_length=100,
    mean_duration=5,
    num_swaps=4,
):
    """Measure frozen post-training responses to every simple and joint concept.

    The probe samples all 25 A_iB_j episodes equally.  Each recorded neuron's
    selectivity is its correlation with the 10 simple-concept indicators
    (A_i, B_j) and the 25 joint-episode indicators (A_iB_j).  These are raw
    correlations, rather than a mutually exclusive concept assignment.
    """
    eval_net = _copy_network_without_recordings(net)
    eval_net.init_recordings(
        {
            "regions": ["mtl_sensory", "mtl_semantic", "mtl", "ctx"],
            "rate_activity": 1,
            "connections": [],
            "rate_connectivity": np.inf,
        }
    )
    eval_net.frozen = True

    latent_specs = {
        "num": 2,
        "total_sizes": [50, 50],
        "act_sizes": [10, 10],
        "dims": [5, 5],
        "prob_list": [1 / 25] * 25,
    }
    # As in the ordering probe, preserve the random stream of the calling
    # simulation while making the diagnostic probe reproducible.
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    try:
        seed_everything(seed + 4)
        probe_params = {
            "num_days": int(num_days),
            "day_length": int(day_length),
            "mean_duration": int(mean_duration),
            "fixed_duration": True,
            "num_swaps": int(num_swaps),
            "latent_space": LatentSpace(**latent_specs),
        }
        probe_input, probe_episodes, probe_latents = make_input(**probe_params)
        with torch.no_grad():
            for day in range(probe_params["num_days"]):
                eval_net(probe_input[day], debug=False)
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(torch_rng_state)

    awake_idx = eval_net.awake_indices
    activities = {
        region: torch.stack(eval_net.activity_recordings[region], dim=0)[awake_idx]
        for region in ("mtl_sensory", "mtl_semantic", "mtl", "ctx")
    }
    latent_a = F.one_hot(probe_latents[:, :, 0].long(), num_classes=latent_specs["dims"][0])
    latent_b = F.one_hot(probe_latents[:, :, 1].long(), num_classes=latent_specs["dims"][1])
    episodes = F.one_hot(probe_episodes.long(), num_classes=np.prod(latent_specs["dims"]))
    simple_concepts = torch.cat((latent_a, latent_b), dim=2)
    concepts = torch.cat((latent_a, latent_b, episodes), dim=2).reshape(-1, 35)

    ctx_simple = eval_net.ctx_subregions[0]
    ctx_complex = eval_net.ctx_subregions[1]
    selectivity = {
        "mtl_sensory": get_selectivity(activities["mtl_sensory"], concepts).detach().cpu(),
        "mtl_semantic": get_selectivity(activities["mtl_semantic"], concepts).detach().cpu(),
        "mtl": get_selectivity(activities["mtl"], concepts).detach().cpu(),
        "ctx": get_selectivity(activities["ctx"], concepts).detach().cpu(),
        "ctx_simple": get_selectivity(activities["ctx"][:, ctx_simple], concepts).detach().cpu(),
        "ctx_complex": get_selectivity(activities["ctx"][:, ctx_complex], concepts).detach().cpu(),
    }

    # This order is deliberately re-derived after extinction.  The fixed
    # post-conditioning ordering is useful for comparing weight matrices over
    # time, but it need not contain the neurons that acquire selectivity while
    # the animal is exposed to the extinction distribution.
    _, mtl_semantic_ordered_indices = get_ordered_indices(
        activities["mtl_semantic"], simple_concepts, assembly_size=10, seed=seed + 4
    )
    _, ctx_simple_ordered_indices = get_ordered_indices(
        activities["ctx"][:, ctx_simple], simple_concepts, assembly_size=10, seed=seed + 4
    )
    _, ctx_complex_ordered_indices_local = get_ordered_indices(
        activities["ctx"][:, ctx_complex], episodes, assembly_size=10, seed=seed + 4
    )
    ctx_complex_ordered_indices = ctx_complex_ordered_indices_local + ctx_complex[0]
    concept_labels = (
        [f"A{i + 1}" for i in range(5)]
        + [f"B{i + 1}" for i in range(5)]
        + [f"A{i + 1}B{j + 1}" for i in range(5) for j in range(5)]
    )

    return {
        "selectivity": selectivity,
        "concept_labels": concept_labels,
        "simple_concept_indices": torch.arange(10),
        "episode_concept_indices": torch.arange(10, 35),
        "plot_ordering": {
            "mtl_sensory": torch.arange(eval_net.mtl_sensory_size),
            "mtl_semantic": mtl_semantic_ordered_indices,
            "ctx": torch.cat((ctx_simple_ordered_indices, ctx_complex_ordered_indices)),
        },
        "parameters": {
            "num_days": int(num_days),
            "day_length": int(day_length),
            "mean_duration": int(mean_duration),
            "num_swaps": int(num_swaps),
            "prob_list": latent_specs["prob_list"],
        },
    }


def run_synaptic_engrams_experiment(
    network_parameters,
    seed=0,
    run_extinction=True,
    conditioning_day_length=100,
    conditioning_mean_duration=5,
    conditioning_num_swaps=4,
    extinction_days=1000,
    extinction_day_length=20,
    extinction_mean_duration=5,
    extinction_num_swaps=4,
    pre_extinction_probe_days=100,
    post_extinction_probe_days=100,
    num_stability_seeds=50,
    record_rate_activity=1,
    fear_probe_fraction=0.25,
):
    from src.model import SSCNetwork

    seed_everything(seed)
    if int(record_rate_activity) != 1:
        raise ValueError(
            "run_synaptic_engrams_experiment reproduces the source notebook only with "
            "record_rate_activity=1 (awake-time indexing assumes full-rate recordings)."
        )

    net_params = deepcopy(network_parameters)
    net_params["duration_phase_A"] = 1
    net_params["duration_phase_B"] = 1
    net_params["max_semantic_load_replay"] = 2

    recording_parameters = {
        "regions": ["mtl_sensory", "mtl_semantic", "mtl", "ctx", "ctx_hat"],
        "rate_activity": int(record_rate_activity),
        "connections": ["ctx_mtl"],
        "rate_connectivity": 1,
    }

    latent_specs = {
        "num": 2,
        "total_sizes": [50, 50],
        "act_sizes": [10, 10],
        "dims": [5, 5],
    }

    input_params = {
        "num_days": 1,
        "day_length": int(conditioning_day_length),
        "mean_duration": int(conditioning_mean_duration),
        "fixed_duration": True,
        "num_swaps": int(conditioning_num_swaps),
    }

    network = SSCNetwork(net_params, recording_parameters)
    network_0 = deepcopy(network)


    # 1. Pre-conditioning: B1 (the fear/US value) is present throughout,
    # while A_i varies. This phase recruits the CTX fear-cell subset.
    latent_specs["prob_list"] = [0.2 if j == 0 else 0 for i in range(5) for j in range(5)]
    input_params["latent_space"] = LatentSpace(**latent_specs)
    preconditioning_input, _, _ = make_input(**input_params)
    preconditioning_input = _permute_channel(preconditioning_input, 0, 50, seed=seed + 1)



    preconditioning_awake_start = len(network.awake_indices)
    with torch.no_grad():
        network(preconditioning_input[0], debug=False)
    with torch.no_grad():
        network.sleep()
    preconditioning_awake_end = len(network.awake_indices)

    network_preconditioning = deepcopy(network)
    network_naive = deepcopy(network)
    # This pre-conditioning-defined subset is retained only to exclude cells
    # that were already recruited by persistent B1 exposure from the auxiliary
    # CS-engram analysis below.  It is not the population used to measure the
    # extinction trace.
    preconditioning_fear_cells = torch.nonzero(network.ctx_IM[:100] == 0, as_tuple=True)[0]

    # 2. Conditioning: A1 is first paired with a shuffled B, then A1B1
    latent_specs["prob_list"] = [1 if i == 0 and j == 0 else 0 for i in range(5) for j in range(5)]
    input_params["latent_space"] = LatentSpace(**latent_specs)
    conditioning_input, _, _ = make_input(**input_params)
    # Permute the US channel only in the first 50 timesteps.
    conditioning_input = _permute_channel(
        conditioning_input, 50, 100, seed=seed + 2, time_start=0, time_end=50
    )

    conditioning_awake_start = len(network.awake_indices)
    with torch.no_grad():
        network(conditioning_input[0], debug=False)
    conditioning_awake_end = len(network.awake_indices)
    with torch.no_grad():
        network.sleep()

    network_conditioning = deepcopy(network)

    # 3 Engram-cell extraction from activity across the complete conditioning
    # episode, mirroring the experiment's activity-dependent CFC labeling.
    ctx_awake = torch.stack(network.activity_recordings["ctx"], dim=0)[network.awake_indices]
    mtl_sem_awake = torch.stack(network.activity_recordings["mtl_semantic"], dim=0)[network.awake_indices]

    ctx_preconditioning = ctx_awake[preconditioning_awake_start:preconditioning_awake_end]
    ctx_conditioning = ctx_awake[conditioning_awake_start:conditioning_awake_end]
    mtl_semantic_encoding = mtl_sem_awake[conditioning_awake_start:conditioning_awake_end]

    ctx_engram_cells, ctx_nonengram_cells, ctx_conditioning_vec = _cell_indices_from_mean_activity(
        network, "ctx", ctx_conditioning
    )
    # Retain the conditioning-defined engram identities, but exclude CTX cells
    # that were already recruited by persistent fear during pre-conditioning.
    # This is the CS-associated, non-fear subset used for the additional
    # pre-conditioning baseline analysis below.
    ctx_engram_nonfear_cells = ctx_engram_cells[
        ~torch.isin(ctx_engram_cells, preconditioning_fear_cells)
    ]
    mtl_sem_engram_cells, mtl_sem_nonengram_cells, _ = _cell_indices_from_mean_activity(
        network, "mtl_semantic", mtl_semantic_encoding
    )

    # 4 Recall pass (same cue)
    recall_probe_awake_start = len(network.awake_indices)
    with torch.no_grad():
        network(conditioning_input[0], debug=False)
    recall_probe_awake_end = len(network.awake_indices)
    ctx_awake = torch.stack(network.activity_recordings["ctx"], dim=0)[network.awake_indices]
    ctx_recall = ctx_awake[recall_probe_awake_start:recall_probe_awake_end]

    _, _, ctx_recall_vec = _cell_indices_from_mean_activity(network, "ctx", ctx_recall)
    ctx_only_recall_cells = torch.nonzero(
        (ctx_recall_vec == 1) & (ctx_conditioning_vec != 1), as_tuple=True
    )[0]

    ctx_ordering = _get_ctx_ordering_from_frozen_uniform_probe(
        network,
        seed=seed,
        num_days=100,
        day_length=conditioning_day_length,
        mean_duration=conditioning_mean_duration,
        num_swaps=conditioning_num_swaps,
    )
    mtl_semantic_ordering = {
        "selectivity": ctx_ordering.pop("mtl_semantic_selectivity"),
        "ordered_indices": ctx_ordering.pop("mtl_semantic_ordered_indices"),
        "labels": ctx_ordering.pop("mtl_semantic_labels"),
    }
    # Define the extinction readout from the frozen, pre-extinction concept
    # probe: the cortical assembly selective for B1 (the fear/US feature).
    # ``simple_ordered_indices`` contains consecutive K-cell blocks ordered by
    # ``simple_labels`` (A1--A5, then B1--B5).
    b1_block = ctx_ordering["simple_labels"].index(("B", 0))
    simple_assembly_size = ctx_ordering["parameters"]["simple_assembly_size"]
    fear_cells = ctx_ordering["simple_ordered_indices"][
        b1_block * simple_assembly_size : (b1_block + 1) * simple_assembly_size
    ].detach().clone()

    # A frozen, uniform probe of the exact state that begins extinction.
    pre_extinction_selectivity = _get_frozen_uniform_concept_selectivity(
        network,
        seed=seed,
        num_days=pre_extinction_probe_days,
        day_length=conditioning_day_length,
        mean_duration=conditioning_mean_duration,
        num_swaps=conditioning_num_swaps,
    )

    # Synaptic distributions
    synaptic_distributions = {
        "ctx_EE": _safe_weight_slice(network.ctx_ctx, ctx_engram_cells, ctx_engram_cells),
        "ctx_ENE": _safe_weight_slice(network.ctx_ctx, ctx_nonengram_cells, ctx_engram_cells),
        "mtl_sem_to_ctx_E": _safe_weight_slice(
            network.mtl_semantic_ctx, mtl_sem_engram_cells, ctx_engram_cells
        ),
        "mtl_sem_to_ctx_NE": _safe_weight_slice(
            network.mtl_semantic_ctx, mtl_sem_nonengram_cells, ctx_engram_cells
        ),
        "ctx_recall_to_ctx_E": _safe_weight_slice(
            network.ctx_ctx, ctx_only_recall_cells, ctx_engram_cells
        ),
    }

    # Keep the existing g2 comparison above intact.  This additional analysis
    # asks whether the same future conditioning-engram, after excluding the
    # pre-existing fear subset, preferentially targets future BLA engram cells
    # before conditioning versus after conditioning.  Rows are MTL-semantic
    # (BLA-like, postsynaptic) cells and columns are CTX (presynaptic) cells.
    ctx_to_mtl_semantic_nonfear_by_stage = {
        "preconditioning": {
            "E_to_BLA_E": _safe_weight_slice(
                network_preconditioning.mtl_semantic_ctx,
                mtl_sem_engram_cells,
                ctx_engram_nonfear_cells,
            ),
            "E_to_BLA_NE": _safe_weight_slice(
                network_preconditioning.mtl_semantic_ctx,
                mtl_sem_nonengram_cells,
                ctx_engram_nonfear_cells,
            ),
        },
        "postconditioning": {
            "E_to_BLA_E": _safe_weight_slice(
                network_conditioning.mtl_semantic_ctx,
                mtl_sem_engram_cells,
                ctx_engram_nonfear_cells,
            ),
            "E_to_BLA_NE": _safe_weight_slice(
                network_conditioning.mtl_semantic_ctx,
                mtl_sem_nonengram_cells,
                ctx_engram_nonfear_cells,
            ),
        },
    }

    # A conditioning-only run supports analyses of the newly formed engram
    # without incurring the cost of the long extinction protocol.  Explicit
    # None/empty outputs prevent post-extinction results from being mistaken
    # for pre-extinction data.
    if not run_extinction:
        X_mtl_all = torch.stack(network.activity_recordings["mtl"], dim=0)[network.awake_indices]
        X_ctx_all = torch.stack(network.activity_recordings["ctx"], dim=0)[network.awake_indices]
        plot_data = {
            "mtl_preconditioning": X_mtl_all[preconditioning_awake_start:preconditioning_awake_end].detach().cpu().numpy(),
            "mtl_conditioning": X_mtl_all[conditioning_awake_start:conditioning_awake_end].detach().cpu().numpy(),
            "ctx_preconditioning": X_ctx_all[preconditioning_awake_start:preconditioning_awake_end].detach().cpu().numpy(),
            "ctx_conditioning": X_ctx_all[conditioning_awake_start:conditioning_awake_end].detach().cpu().numpy(),
            "ctx_recall": X_ctx_all[recall_probe_awake_start:recall_probe_awake_end].detach().cpu().numpy(),
            "mtl_late_extinction": None,
        }
        summary_df = pd.DataFrame([
            {
                "metric": "ctx_EE_mean",
                "value": float(np.mean(synaptic_distributions["ctx_EE"]))
                if synaptic_distributions["ctx_EE"].size > 0 else np.nan,
            },
            {
                "metric": "ctx_ENE_mean",
                "value": float(np.mean(synaptic_distributions["ctx_ENE"]))
                if synaptic_distributions["ctx_ENE"].size > 0 else np.nan,
            },
            {
                "metric": "mtl_sem_to_ctx_E_mean",
                "value": float(np.mean(synaptic_distributions["mtl_sem_to_ctx_E"]))
                if synaptic_distributions["mtl_sem_to_ctx_E"].size > 0 else np.nan,
            },
            {
                "metric": "mtl_sem_to_ctx_NE_mean",
                "value": float(np.mean(synaptic_distributions["mtl_sem_to_ctx_NE"]))
                if synaptic_distributions["mtl_sem_to_ctx_NE"].size > 0 else np.nan,
            },
        ])
        return {
            "network": network,
            "network_0": network_0,
            "network_preconditioning": network_preconditioning,
            "network_conditioning": network_conditioning,
            "network_naive": network_naive,
            "recording_parameters": recording_parameters,
            "run_extinction": False,
            "fear_cells": fear_cells,
            "preconditioning_fear_cells": preconditioning_fear_cells,
            "ctx_engram_cells": ctx_engram_cells,
            "ctx_engram_nonfear_cells": ctx_engram_nonfear_cells,
            "ctx_nonengram_cells": ctx_nonengram_cells,
            "ctx_only_recall_cells": ctx_only_recall_cells,
            "mtl_semantic_engram_cells": mtl_sem_engram_cells,
            "mtl_semantic_nonengram_cells": mtl_sem_nonengram_cells,
            "synaptic_distributions": synaptic_distributions,
            "ctx_to_mtl_semantic_nonfear_by_stage": ctx_to_mtl_semantic_nonfear_by_stage,
            "synaptic_distributions_post_extinction": None,
            "synaptic_distributions_post_extinction_sampled": None,
            "fear_input": torch.empty(0),
            "fear_probe_index": None,
            "stability_savings": torch.empty(0),
            "stability_naive": torch.empty(0),
            "ctx_ordering": ctx_ordering,
            "mtl_semantic_ordering": mtl_semantic_ordering,
            "pre_extinction_selectivity": pre_extinction_selectivity,
            "post_extinction_selectivity": None,
            "plot_data": plot_data,
            "summary_df": summary_df,
        }

    # Extinction dynamics (US absent)
    extinction_params = {
        "num_days": int(extinction_days),
        "day_length": int(extinction_day_length),
        "mean_duration": int(extinction_mean_duration),
        "fixed_duration": True,
        "num_swaps": int(extinction_num_swaps),
    }
    latent_specs["prob_list"] = [1 / 20 if j != 0 else 0 for i in range(5) for j in range(5)]
    extinction_params["latent_space"] = LatentSpace(**latent_specs)
    extinction_input, _, _ = make_input(**extinction_params)

    if not (0.0 <= float(fear_probe_fraction) <= 1.0):
        raise ValueError("fear_probe_fraction must be in [0, 1].")

    fear_input = torch.zeros(extinction_params["num_days"])
    recall_len = max(int(ctx_recall.shape[0]), 1)
    # Match source notebook logic: use an early recall-state cue (index ~25 for len=100),
    # which corresponds to [-75] in the original hard-coded indexing.
    fear_probe_index = min(max(int(float(fear_probe_fraction) * recall_len), 0), recall_len - 1)
    ctx_0 = ctx_recall[fear_probe_index].detach().clone()

    extinction_awake_boundaries = [len(network.awake_indices)]
    with torch.no_grad():
        for day in range(extinction_params["num_days"]):
            network(extinction_input[day], debug=False)
            network.sleep()
            # Total recurrent input that the recalled conditioning pattern
            # delivers to the pre-extinction B1-selective cortical assembly.
            fear_input[day] = (network.ctx_ctx[fear_cells] @ ctx_0).sum()
            extinction_awake_boundaries.append(len(network.awake_indices))

    post_extinction_selectivity = _get_frozen_uniform_concept_selectivity(
        network,
        seed=seed,
        num_days=post_extinction_probe_days,
        day_length=conditioning_day_length,
        mean_duration=conditioning_mean_duration,
        num_swaps=conditioning_num_swaps,
    )

    # Post-extinction CTX synaptic distributions (same E-E vs E-NE comparison as source notebook late plot)
    synaptic_distributions_post = {
        "ctx_EE": _safe_weight_slice(network.ctx_ctx, ctx_engram_cells, ctx_engram_cells),
        "ctx_ENE": _safe_weight_slice(network.ctx_ctx, ctx_nonengram_cells, ctx_engram_cells),
    }
    synaptic_distributions_post_sampled = {
        "ctx_EE": _sampled_weight_slice(network.ctx_ctx, ctx_engram_cells, ctx_engram_cells, n_samples=5),
        "ctx_ENE": _sampled_weight_slice(network.ctx_ctx, ctx_nonengram_cells, ctx_engram_cells, n_samples=5),
    }

    # Store plot-ready activity snapshots before any recording reset.
    X_mtl_all = torch.stack(network.activity_recordings["mtl"], dim=0)[network.awake_indices].detach().clone()
    X_ctx_all = torch.stack(network.activity_recordings["ctx"], dim=0)[network.awake_indices].detach().clone()
    late_extinction_start_day = max(int(extinction_days) - 10, 0)
    late_extinction_start = int(extinction_awake_boundaries[late_extinction_start_day])
    late_extinction_end = int(extinction_awake_boundaries[-1])
    plot_data = {
        "mtl_preconditioning": X_mtl_all[preconditioning_awake_start:preconditioning_awake_end].detach().cpu().numpy(),
        "mtl_conditioning": X_mtl_all[conditioning_awake_start:conditioning_awake_end].detach().cpu().numpy(),
        "ctx_preconditioning": X_ctx_all[preconditioning_awake_start:preconditioning_awake_end].detach().cpu().numpy(),
        "ctx_conditioning": X_ctx_all[conditioning_awake_start:conditioning_awake_end].detach().cpu().numpy(),
        # Match source notebook's recall activity snapshot (extra recall pass, no sleep).
        "ctx_recall": X_ctx_all[recall_probe_awake_start:recall_probe_awake_end].detach().cpu().numpy(),
        "mtl_late_extinction": X_mtl_all[late_extinction_start:late_extinction_end].detach().cpu().numpy(),
    }

    # Episode recall stability: post-extinction network vs conditioning-naive checkpoint.
    stability_savings = torch.zeros(int(num_stability_seeds))
    stability_naive = torch.zeros(int(num_stability_seeds))
    recall_probe_params = {
        "num_days": 1,
        "day_length": int(conditioning_day_length),
        "mean_duration": int(conditioning_mean_duration),
        "fixed_duration": True,
        "num_swaps": int(conditioning_num_swaps),
    }
    latent_specs["prob_list"] = [1 if i == 0 and j == 0 else 0 for i in range(5) for j in range(5)]
    recall_probe_params["latent_space"] = LatentSpace(**latent_specs)
    # Match original notebook speed-up while preserving main-network recordings for plotting.
    stability_base = deepcopy(network)
    stability_base.init_recordings(recording_parameters)

    for s in range(int(num_stability_seeds)):
        seed_everything(s)
        network_test = deepcopy(stability_base)
        probe_input, _, _ = make_input(**recall_probe_params)
        with torch.no_grad():
            network_test(probe_input[0], debug=False)
            ctx_completed_savings = network_test.pattern_complete("ctx", network_test.ctx)
            ctx_completed_naive = network_naive.pattern_complete("ctx", network_test.ctx)
            stability_savings[s] = get_cos_sim_torch(network_test.ctx, ctx_completed_savings)
            stability_naive[s] = get_cos_sim_torch(network_test.ctx, ctx_completed_naive)

    summary_rows = [
        {
            "metric": "ctx_EE_mean",
            "value": float(np.mean(synaptic_distributions["ctx_EE"]))
            if synaptic_distributions["ctx_EE"].size > 0
            else np.nan,
        },
        {
            "metric": "ctx_ENE_mean",
            "value": float(np.mean(synaptic_distributions["ctx_ENE"]))
            if synaptic_distributions["ctx_ENE"].size > 0
            else np.nan,
        },
        {
            "metric": "mtl_sem_to_ctx_E_mean",
            "value": float(np.mean(synaptic_distributions["mtl_sem_to_ctx_E"]))
            if synaptic_distributions["mtl_sem_to_ctx_E"].size > 0
            else np.nan,
        },
        {
            "metric": "mtl_sem_to_ctx_NE_mean",
            "value": float(np.mean(synaptic_distributions["mtl_sem_to_ctx_NE"]))
            if synaptic_distributions["mtl_sem_to_ctx_NE"].size > 0
            else np.nan,
        },
        {
            "metric": "ctx_EE_post_extinction_mean",
            "value": float(np.mean(synaptic_distributions_post["ctx_EE"]))
            if synaptic_distributions_post["ctx_EE"].size > 0
            else np.nan,
        },
        {
            "metric": "ctx_ENE_post_extinction_mean",
            "value": float(np.mean(synaptic_distributions_post["ctx_ENE"]))
            if synaptic_distributions_post["ctx_ENE"].size > 0
            else np.nan,
        },
        {"metric": "fear_input_last", "value": float(fear_input[-1])},
        {"metric": "stability_savings_mean", "value": float(stability_savings.mean())},
        {"metric": "stability_naive_mean", "value": float(stability_naive.mean())},
    ]
    summary_df = pd.DataFrame(summary_rows)

    return {
        "network": network,
        "network_0": network_0,
        "network_preconditioning": network_preconditioning,
        "network_conditioning": network_conditioning,
        "network_naive": network_naive,
        "recording_parameters": recording_parameters,
        "run_extinction": True,
        "fear_cells": fear_cells,
        "preconditioning_fear_cells": preconditioning_fear_cells,
        "ctx_engram_cells": ctx_engram_cells,
        "ctx_engram_nonfear_cells": ctx_engram_nonfear_cells,
        "ctx_nonengram_cells": ctx_nonengram_cells,
        "ctx_only_recall_cells": ctx_only_recall_cells,
        "mtl_semantic_engram_cells": mtl_sem_engram_cells,
        "mtl_semantic_nonengram_cells": mtl_sem_nonengram_cells,
        "synaptic_distributions": synaptic_distributions,
        "ctx_to_mtl_semantic_nonfear_by_stage": ctx_to_mtl_semantic_nonfear_by_stage,
        "synaptic_distributions_post_extinction": synaptic_distributions_post,
        "synaptic_distributions_post_extinction_sampled": synaptic_distributions_post_sampled,
        "fear_input": fear_input,
        "fear_probe_index": int(fear_probe_index),
        "stability_savings": stability_savings,
        "stability_naive": stability_naive,
        "ctx_ordering": ctx_ordering,
        "mtl_semantic_ordering": mtl_semantic_ordering,
        "pre_extinction_selectivity": pre_extinction_selectivity,
        "post_extinction_selectivity": post_extinction_selectivity,
        "plot_data": plot_data,
        "summary_df": summary_df,
    }
