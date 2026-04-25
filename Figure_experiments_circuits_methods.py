"""Utilities for Figure_experiments_circuits notebook experiments."""

import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from src.utils.general import (
    LatentSpace,
    get_cos_sim_torch,
    get_sample_from_num_swaps,
    make_input,
)

def seed_everything(seed=0):
    """Set all relevant RNGs for reproducible sampling and decoder fitting.

    Example:
        seed=0 -> deterministic concept/context sampling across repeats.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_two_example_prob_list(
    dims,
    target_latent,
    target_idx,
    fixed_opposite_idx,
    contrast_idx,
):
    """Build a probability list with exactly two equiprobable latent labels.

    Intuition:
        This is the train-time distribution for a binary decoder task:
        one positive pair and one negative pair, each with probability 0.5.

    Example (target_latent=0):
        target A concept = A3, fixed B context = B2, contrast A = A1
        -> non-zero labels are (A3,B2) and (A1,B2), each with p=0.5.
    """
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
    """Build a test distribution that excludes the training opposite context.

    Intuition:
        We test cross-context generalization by removing all pairs that contain
        the training context on the opposite latent.

    Example:
        If training for A_i used B_train=B2, then test sampling excludes all
        (A_k, B2) pairs, sets marginal P(A_i)=0.5 across remaining B's, and
        spreads the remaining 0.5 over non-target A values.
    """
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
    """Return the positive/negative latent labels for the binary task.

    Example:
        target_latent=1 (B-task), target_idx=2, fixed_opposite_idx=0, contrast_idx=4
        -> positive=(A1,B3), negative=(A1,B5) in 1-based naming.
    """
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
    """Create one deterministic mini-day with two contiguous blocks.

    Output shapes:
        input_tensor: (1, 2*mean_duration, total_input_size)
        input_latents: (1, 2*mean_duration, 2)

    Intuition:
        Block 1 is the positive class, block 2 is the negative class.
        This guarantees exactly one example of each class in training.
    """
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
    """Create test input params with a latent space excluding training context.

    Intuition:
        Keeps the notebook's global test settings but swaps in a context-excluding
        probability table for the current concept task.
    """
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
    """Run the network on provided inputs and return region activity + labels.

    Inputs:
        input_tensor shape is typically (num_days, day_length, input_size).
        input_latents shape is (num_days, day_length, 2).

    Returns:
        X_mtl_sensory: (num_days*day_length, mtl_sensory_size)
        X_mtl_semantic: (num_days*day_length, mtl_semantic_size)
        y: binary vector for target concept presence.
    """
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
    """Build deterministic two-block training data and return neural features/labels.

    Intuition:
        Encapsulates the full train-set creation for one concept-repeat.
    """
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
    """Sample random test input until both binary classes are present.

    Why:
        Random sampling can occasionally produce only one class for short tests.
        This retry loop avoids invalid decoder evaluation.
    """
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
    """Fit a logistic regression decoder and return test accuracy."""
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
    """Main concept-cell cross-context decoding experiment.

    Procedure (per concept and repeat):
        1) Train decoder features from deterministic two-block input.
        2) Build excluded-context test distribution.
        3) Sample test rollout with retry until binary.
        4) Fit/evaluate linear decoders separately on MTL-sensory/semantic.

    Output:
        Long-form DataFrame with metadata + accuracy for both regions.
    """
    # dims[0] = number of A concepts, dims[1] = number of B concepts.
    # Example: dims=[5,5] -> A1..A5 and B1..B5.
    dims = base_latent_specs["dims"]
    # Local RNG so repeat-level sampling (contexts/contrasts) is reproducible.
    rng = np.random.default_rng(seed)

    # Build a full sweep over all target concepts:
    # ("A",0,idx) means decode presence of A_idx, ("B",1,idx) for B_idx.
    concept_specs = [("A", 0, idx) for idx in range(dims[0])] + [("B", 1, idx) for idx in range(dims[1])]

    # Accumulate one row per (concept, repeat, region).
    results = []
    # Keep simulation day fixed across repeats so each rollout starts from same phase.
    base_day = int(net.day)

    for concept_name, target_latent, target_idx in concept_specs:
        # Number of concepts on opposite latent (e.g., B-count when decoding A).
        opposite_dim = dims[1 - target_latent]
        # Number of concepts on target latent (e.g., A-count when decoding A).
        target_dim = dims[target_latent]
        print(f"Running concept {concept_name}{target_idx + 1}...")

        for repeat in range(num_repeats):
            # Randomly choose training context from opposite latent.
            # Example: if decoding A_i, train_context might be B3.
            train_context = int(rng.integers(opposite_dim))
            # Pick one non-target concept as negative class (e.g., A2 vs A5).
            contrast_candidates = [c for c in range(target_dim) if c != target_idx]
            contrast_idx = int(rng.choice(contrast_candidates))

            # Reset day so all repeats use same network phase/state schedule.
            net.day = base_day
            # Deterministic train rollout: exactly one positive and one negative block.
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

            # Build test generator that excludes the training opposite context.
            # Example: trained on B3 -> test excludes all (*,B3) pairs.
            test_input_params = make_excluding_context_test_input_params(
                test_sampling_params=test_sampling_params,
                base_latent_specs=base_latent_specs,
                target_latent=target_latent,
                target_idx=target_idx,
                excluded_opposite_idx=train_context,
            )

            # Derive a unique but reproducible seed per concept/repeat.
            # This avoids accidental identical test draws across tasks.
            rollout_seed = seed + 100000 * target_latent + 10000 * target_idx + 100 * repeat

            net.day = base_day
            # Random test rollout with retries until both classes appear.
            X_sens_test, X_sem_test, y_test, test_retry = rollout_test_random_until_binary(
                net=net,
                recording_parameters=recording_parameters,
                input_params=test_input_params,
                target_latent=target_latent,
                target_idx=target_idx,
                seed=rollout_seed,
                max_attempts=max_test_attempts,
            )

            # Fit/evaluate one linear decoder per region on identical labels.
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

            # Track class balance for debugging and later QA plots/tables.
            train_pos = int(np.sum(y_train == 1))
            train_neg = int(np.sum(y_train == 0))
            test_pos = int(np.sum(y_test == 1))
            test_neg = int(np.sum(y_test == 0))

            # Row for MTL-sensory decoder result.
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
            # Row for MTL-semantic decoder result (same metadata, different region/accuracy).
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
    """Assign objects to scenes in balanced blocks.

    Example:
        num_objects=128, num_scenes=4 -> [0...0,1...1,2...2,3...3] with 32 each.
    """
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
    random_neurons=True,
):
    """Create latent-space spec where object and scene dimensions are explicit.

    Example:
        num_objects=128, num_scenes=4 gives dims=[128,4].
    """
    if len(total_sizes) != 2 or len(act_sizes) != 2:
        raise ValueError("total_sizes and act_sizes must each have length 2.")
    return {
        "num": 2,
        "total_sizes": [int(total_sizes[0]), int(total_sizes[1])],
        "act_sizes": [int(act_sizes[0]), int(act_sizes[1])],
        "dims": [int(num_objects), int(num_scenes)],
        "prob_list": [1.0 / (num_objects * num_scenes) for _ in range(num_objects * num_scenes)],
        "random_neurons": bool(random_neurons),
    }


def _build_object_scene_prob_list(num_objects, num_scenes, object_to_scene):
    """Build probability list constrained by the object->scene map.

    Intuition:
        For each object, only its assigned scene has non-zero probability.
    """
    probs = []
    for obj_idx in range(int(num_objects)):
        for scene_idx in range(int(num_scenes)):
            probs.append(1.0 / float(num_objects) if int(object_to_scene[obj_idx]) == scene_idx else 0.0)
    return probs


def _make_deterministic_object_scene_day(
    latent_space,
    object_to_scene,
    mean_duration,
    num_swaps,
    presentations_per_object=1,
    shuffle_objects=False,
    seed=0,
    object_patterns=None,
    scene_patterns=None,
):
    """Generate a deterministic wake day with ordered object-scene events.

    Output:
        input_day: (day_length, input_size)
        latents_day: (day_length, 2) with (object_idx, scene_idx)
    """
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
        if object_patterns is not None and scene_patterns is not None:
            pattern = torch.cat((object_patterns[int(obj_idx)], scene_patterns[int(scene_idx)]))
        else:
            pattern = latent_space.label_to_neurons[label]
        for _ in range(int(mean_duration)):
            input_day[time_idx] = get_sample_from_num_swaps(pattern, int(num_swaps))
            latents_day[time_idx] = torch.tensor(label, dtype=torch.int32)
            time_idx += 1

    return input_day, latents_day


def _build_structured_component_patterns(num_codes, total_size, act_size, allow_wrap=False):
    """Create block-structured binary codes for objects/scenes.

    Example:
        total_size=40, act_size=10 -> 4 disjoint blocks.
        num_codes > 4 raises unless allow_wrap=True.
    """
    num_codes = int(num_codes)
    total_size = int(total_size)
    act_size = int(act_size)
    if act_size <= 0:
        raise ValueError("act_size must be > 0")
    num_blocks = max(total_size // act_size, 1)
    if (not allow_wrap) and num_codes > num_blocks:
        raise ValueError(
            "Not enough disjoint blocks for structured patterns: "
            f"num_codes={num_codes}, available_blocks={num_blocks}. "
            "Increase total_size, decrease act_size, or use latent pattern source."
        )

    patterns = []
    for idx in range(num_codes):
        block_idx = idx % num_blocks
        start = block_idx * act_size
        end = min(start + act_size, total_size)
        pattern = torch.zeros(total_size)
        pattern[start:end] = 1
        patterns.append(pattern)
    return patterns


def _collect_partial_object_cue_patterns(
    net,
    latent_space,
    num_objects,
    cue_num_swaps=0,
    object_patterns=None,
    condition_mtl_on_cue=True,
):
    """Probe object cues and collect final MTL/CTX patterns.

    Intuition:
        For each object A_i we present only the object part (B channels zero),
        complete in MTL, then propagate through network inference to read out
        final MTL/CTX states.
    """
    net_eval = deepcopy(net)
    net_eval.frozen = True

    a_size = int(latent_space.total_sizes[0])
    patterns_mtl = []
    patterns_ctx = []

    with torch.no_grad():
        for obj_idx in range(int(num_objects)):
            cue = torch.zeros(latent_space.total_size)
            if object_patterns is not None:
                cue[:a_size] = object_patterns[int(obj_idx)]
            else:
                cue[:a_size] = latent_space.latent_patterns[0][obj_idx]

            if int(cue_num_swaps) > 0:
                cue_a = get_sample_from_num_swaps(cue[:a_size], int(cue_num_swaps))
                cue[:a_size] = cue_a
            cue[a_size:] = 0

            # 1) Pattern-complete in MTL from a partial object cue.
            mtl_init = torch.zeros(net_eval.mtl_size)
            mtl_init[: net_eval.mtl_sensory_size] = cue
            mtl_conditioned = None
            if bool(condition_mtl_on_cue):
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
    """Compute overlap metrics from pairwise pattern correlations.

    For each item i:
        overlap_similarity = mean corr with same-scene items
        non_overlap_similarity = mean corr with different-scene items
        overlap_index = overlap_similarity - non_overlap_similarity
    """
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


def _decode_scene_from_mtl_sensory_blocks(
    mtl_patterns,
    object_to_scene,
    total_sizes,
    act_sizes,
    num_scenes,
    phase,
):
    """
    Decode scene identity from B-block activity in the MTL sensory component.
    """
    mtl_patterns = np.asarray(mtl_patterns, dtype=float)
    object_to_scene = np.asarray(object_to_scene, dtype=int)

    num_items = int(mtl_patterns.shape[0])
    a_size = int(total_sizes[0])
    b_size = int(total_sizes[1])
    b_block = int(act_sizes[1])
    num_scenes = int(num_scenes)

    if b_size < num_scenes * b_block:
        raise ValueError(
            "Scene decoding requires b_size >= num_scenes * act_sizes[1]. "
            f"Got b_size={b_size}, num_scenes={num_scenes}, act_sizes[1]={b_block}."
        )

    mtl_sensory = mtl_patterns[:, : a_size + b_size]
    b_slice = mtl_sensory[:, a_size : a_size + b_size]

    scene_scores = np.zeros((num_items, num_scenes), dtype=float)
    for scene_idx in range(num_scenes):
        start = int(scene_idx * b_block)
        end = int(start + b_block)
        scene_scores[:, scene_idx] = b_slice[:, start:end].sum(axis=1)

    predicted_scene = np.argmax(scene_scores, axis=1).astype(int)
    true_scene = object_to_scene[:num_items].astype(int)
    correct = (predicted_scene == true_scene).astype(int)

    rows = []
    for item_idx in range(num_items):
        rows.append(
            {
                "item_index": int(item_idx),
                "phase": str(phase),
                "true_scene": int(true_scene[item_idx]),
                "predicted_scene": int(predicted_scene[item_idx]),
                "correct": int(correct[item_idx]),
                "winning_score": float(scene_scores[item_idx, predicted_scene[item_idx]]),
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
    train_fixed_duration=True,
    train_day_length=None,
    train_input_mode="deterministic",
    deterministic_pattern_source="latent",
    train_shuffle_objects=False,
    total_sizes=(80, 20),
    act_sizes=(5, 5),
    latent_random_neurons=True,
    auto_allocate_mtl_sensory=True,
    shuffle_object_scene_assignments=False,
    consolidation_sleep_cycles=100,
    cue_num_swaps=0,
    condition_mtl_on_cue=True,
    seed=0,
    duration_phase_A=0,
    duration_phase_B=None,
    sleep_duration_A=None,
    sleep_duration_B=None,
    stop_after="recall",
):
    """Reproduce consolidation-induced overlap using object-only cue probes.

    High-level flow:
        1) Configure network and latent space.
        2) Train object-scene associations (deterministic or make_input).
        3) Collect pre-consolidation cue patterns.
        4) Run sleep consolidation cycles.
        5) Collect post-consolidation cue patterns.
        6) Compute overlap metrics and scene-decoding accuracy.
    """
    from src.model_engrams import SSCNetwork

    seed_everything(seed)

    if num_days_train < 1:
        raise ValueError("num_days_train must be >= 1")
    if consolidation_sleep_cycles < 0:
        raise ValueError("consolidation_sleep_cycles must be >= 0")
    if stop_after not in {"encoding", "sleep", "recall"}:
        raise ValueError("stop_after must be one of {'encoding', 'sleep', 'recall'}")
    if train_input_mode not in {"make_input", "deterministic"}:
        raise ValueError("train_input_mode must be one of {'make_input', 'deterministic'}")
    if deterministic_pattern_source not in {"latent", "structured"}:
        raise ValueError("deterministic_pattern_source must be one of {'latent', 'structured'}")

    net_params = deepcopy(network_parameters)
    # Enable semantic replay from the beginning while preserving episodic replay.
    net_params["duration_phase_A"] = int(duration_phase_A)
    if duration_phase_B is not None:
        net_params["duration_phase_B"] = int(duration_phase_B)
    if sleep_duration_A is not None:
        net_params["sleep_duration_A"] = int(sleep_duration_A)
    if sleep_duration_B is not None:
        net_params["sleep_duration_B"] = int(sleep_duration_B)

    total_sizes_local = tuple(int(x) for x in total_sizes)
    act_sizes_local = tuple(int(x) for x in act_sizes)

    if bool(auto_allocate_mtl_sensory):
        # Allocate enough sensory/MTL-sensory units to represent every object/scene concept
        # with disjoint blocks of size act_sizes.
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
        random_neurons=latent_random_neurons,
    )
    if (not bool(latent_random_neurons)) and int(num_objects) * int(act_sizes_local[0]) > int(total_sizes_local[0]):
        raise ValueError(
            "With latent_random_neurons=False, num_objects * act_sizes[0] must be <= total_sizes[0]. "
            "For 128 unique objects this is usually impossible unless total_sizes[0] is very large."
        )
    latent_space = LatentSpace(**latent_specs)

    object_to_scene = _build_balanced_object_to_scene(
        num_objects=int(num_objects),
        num_scenes=int(num_scenes),
        seed=seed,
        shuffle_assignments=bool(shuffle_object_scene_assignments),
    )

    object_patterns = None
    scene_patterns = None
    if train_input_mode == "deterministic" and deterministic_pattern_source == "structured":
        object_patterns = _build_structured_component_patterns(
            num_codes=int(num_objects),
            total_size=int(total_sizes_local[0]),
            act_size=int(act_sizes_local[0]),
            allow_wrap=False,
        )
        scene_patterns = _build_structured_component_patterns(
            num_codes=int(num_scenes),
            total_size=int(total_sizes_local[1]),
            act_size=int(act_sizes_local[1]),
            allow_wrap=False,
        )

    if train_input_mode == "make_input":
        latent_specs_train = deepcopy(latent_specs)
        latent_specs_train["prob_list"] = _build_object_scene_prob_list(
            num_objects=int(num_objects),
            num_scenes=int(num_scenes),
            object_to_scene=object_to_scene,
        )
        latent_space_train = LatentSpace(**latent_specs_train)

        inferred_day_length = int(num_objects) * int(presentations_per_object) * int(train_mean_duration)
        input_params_train = {
            "num_days": int(num_days_train),
            "day_length": int(inferred_day_length if train_day_length is None else train_day_length),
            "mean_duration": int(train_mean_duration),
            "fixed_duration": bool(train_fixed_duration),
            "num_swaps": int(train_num_swaps),
            "latent_space": latent_space_train,
        }
        input_tensor_train, _, _ = make_input(**input_params_train)
        for day_idx in range(int(num_days_train)):
            net(input_tensor_train[day_idx], debug=False)
    else:
        for day_idx in range(int(num_days_train)):
            input_day, _ = _make_deterministic_object_scene_day(
                latent_space=latent_space,
                object_to_scene=object_to_scene,
                mean_duration=int(train_mean_duration),
                num_swaps=int(train_num_swaps),
                presentations_per_object=int(presentations_per_object),
                shuffle_objects=bool(train_shuffle_objects),
                seed=seed + day_idx,
                object_patterns=object_patterns,
                scene_patterns=scene_patterns,
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
            object_patterns=object_patterns,
            condition_mtl_on_cue=bool(condition_mtl_on_cue),
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
        object_patterns=object_patterns,
        condition_mtl_on_cue=bool(condition_mtl_on_cue),
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

    scene_decoding_df = pd.concat(
        [
            _decode_scene_from_mtl_sensory_blocks(
                mtl_patterns=pre_patterns["MTL"],
                object_to_scene=object_to_scene,
                total_sizes=total_sizes_local,
                act_sizes=act_sizes_local,
                num_scenes=int(num_scenes),
                phase="pre",
            ),
            _decode_scene_from_mtl_sensory_blocks(
                mtl_patterns=post_patterns["MTL"],
                object_to_scene=object_to_scene,
                total_sizes=total_sizes_local,
                act_sizes=act_sizes_local,
                num_scenes=int(num_scenes),
                phase="post",
            ),
        ],
        ignore_index=True,
    )
    scene_decoding_summary_df = (
        scene_decoding_df.groupby("phase")["correct"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy"})
    )
    scene_decoding_summary_df["sem"] = (
        scene_decoding_summary_df["std"] / np.sqrt(scene_decoding_summary_df["count"])
    )

    return {
        "network": net,
        "latent_space": latent_space,
        "object_to_scene": object_to_scene,
        "overlap_df": overlap_df,
        "summary_df": summary_df,
        "scene_decoding_df": scene_decoding_df,
        "scene_decoding_summary_df": scene_decoding_summary_df,
        "pre_patterns": pre_patterns,
        "post_patterns": post_patterns,
    }


def _permute_channel(input_tensor, channel_start, channel_end, seed, time_start=None, time_end=None):
    """Randomly permute feature order within a selected channel slice."""
    torch.manual_seed(int(seed))
    t0 = 0 if time_start is None else int(time_start)
    t1 = input_tensor.shape[1] if time_end is None else int(time_end)
    source = input_tensor[:, t0:t1, channel_start:channel_end].clone()
    perms = torch.argsort(torch.rand_like(source), dim=2)
    input_tensor[:, t0:t1, channel_start:channel_end] = torch.gather(source, 2, perms)
    return input_tensor


def _cell_indices_from_mean_activity(network, region, activity_tensor):
    """Return active/inactive cell indices after mean-activity thresholding."""
    mean_vec = activity_tensor.mean(axis=0)
    active_vec, _ = network.activation(mean_vec, region)
    active_cells = torch.nonzero(active_vec == 1, as_tuple=True)[0]
    inactive_cells = torch.nonzero(active_vec != 1, as_tuple=True)[0]
    return active_cells, inactive_cells, active_vec


def _safe_weight_slice(weights, row_idx, col_idx):
    """Flatten a weight submatrix safely; return empty array on empty indices."""
    if len(row_idx) == 0 or len(col_idx) == 0:
        return np.array([], dtype=float)
    return weights[row_idx][:, col_idx].flatten().detach().cpu().numpy()


def _sampled_weight_slice(weights, row_idx, col_idx, n_samples=5):
    """Sample up to n row/column indices and flatten the corresponding weights."""
    if len(row_idx) == 0 or len(col_idx) == 0:
        return np.array([], dtype=float)
    row_sel = row_idx[torch.randperm(len(row_idx))[: min(int(n_samples), len(row_idx))]]
    col_sel = col_idx[torch.randperm(len(col_idx))[: min(int(n_samples), len(col_idx))]]
    return weights[row_sel][:, col_sel].flatten().detach().cpu().numpy()


def run_synaptic_engrams_experiment(
    network_parameters,
    seed=0,
    conditioning_day_length=100,
    conditioning_mean_duration=5,
    conditioning_num_swaps=4,
    extinction_days=1000,
    extinction_day_length=20,
    extinction_mean_duration=5,
    extinction_num_swaps=4,
    num_stability_seeds=50,
    record_rate_activity=1,
    fear_probe_fraction=0.25,
):
    """Synaptic engram reproduction pipeline (conditioning -> recall -> extinction).

    Returns:
        Network objects, engram cell indices, synaptic distributions, fear-input
        trajectory, stability metrics, and plot-ready activity snapshots.
    """
    from src.model_engrams import SSCNetwork

    seed_everything(seed)
    if int(record_rate_activity) != 1:
        raise ValueError(
            "run_synaptic_engrams_experiment reproduces the source notebook only with "
            "record_rate_activity=1 (awake-time indexing assumes full-rate recordings)."
        )

    net_params = deepcopy(network_parameters)
    net_params["duration_phase_A"] = 1
    net_params["duration_phase_B"] = 1
    net_params["max_semantic_charge_replay"] = 2

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

    # Phase 1: Conditioning day (US = B1; varied contexts A_i)
    latent_specs["prob_list"] = [0.2 if j == 0 else 0 for i in range(5) for j in range(5)]
    input_params["latent_space"] = LatentSpace(**latent_specs)
    conditioning_input, _, _ = make_input(**input_params)
    conditioning_input = _permute_channel(conditioning_input, 0, 50, seed=seed + 1)

    conditioning_awake_start = len(network.awake_indices)
    with torch.no_grad():
        network(conditioning_input[0], debug=False)
    conditioning_awake_end = len(network.awake_indices)
    with torch.no_grad():
        network.sleep()

    network_naive = deepcopy(network)
    fear_cells = torch.nonzero(network.ctx_IM[:100] == 0, as_tuple=True)[0]

    # Phase 2: Recall day (A1-B1 cue; shuffled US channel as in source notebook)
    latent_specs["prob_list"] = [1 if i == 0 and j == 0 else 0 for i in range(5) for j in range(5)]
    input_params["latent_space"] = LatentSpace(**latent_specs)
    recall_input, _, _ = make_input(**input_params)
    # Match original notebook snippet: permute US channel only in first 50 timesteps.
    recall_input = _permute_channel(recall_input, 50, 100, seed=seed + 2, time_start=0, time_end=50)

    recall_awake_start = len(network.awake_indices)
    with torch.no_grad():
        network(recall_input[0], debug=False)
    recall_awake_end = len(network.awake_indices)
    with torch.no_grad():
        network.sleep()

    # Engram cell extraction (from conditioning + first recall day, as in source notebook)
    ctx_awake = torch.stack(network.activity_recordings["ctx"], dim=0)[network.awake_indices]
    mtl_sem_awake = torch.stack(network.activity_recordings["mtl_semantic"], dim=0)[network.awake_indices]

    ctx_encoding = ctx_awake[conditioning_awake_start:conditioning_awake_end]
    mtl_semantic_encoding = mtl_sem_awake[recall_awake_start:recall_awake_end]

    ctx_engram_cells, ctx_nonengram_cells, ctx_encoding_vec = _cell_indices_from_mean_activity(
        network, "ctx", ctx_encoding
    )
    mtl_sem_engram_cells, mtl_sem_nonengram_cells, _ = _cell_indices_from_mean_activity(
        network, "mtl_semantic", mtl_semantic_encoding
    )

    # Source notebook runs an extra recall pass (same cue) before recall-only analysis and extinction.
    recall_probe_awake_start = len(network.awake_indices)
    with torch.no_grad():
        network(recall_input[0], debug=False)
    recall_probe_awake_end = len(network.awake_indices)
    ctx_awake = torch.stack(network.activity_recordings["ctx"], dim=0)[network.awake_indices]
    ctx_recall = ctx_awake[recall_probe_awake_start:recall_probe_awake_end]

    _, _, ctx_recall_vec = _cell_indices_from_mean_activity(network, "ctx", ctx_recall)
    ctx_only_recall_cells = torch.nonzero(
        (ctx_recall_vec == 1) & (ctx_encoding_vec != 1), as_tuple=True
    )[0]

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
            fear_input[day] = (network.ctx_ctx @ ctx_0)[fear_cells].sum()
            extinction_awake_boundaries.append(len(network.awake_indices))

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
        "mtl_conditioning": X_mtl_all[conditioning_awake_start:conditioning_awake_end].detach().cpu().numpy(),
        "mtl_recall": X_mtl_all[recall_awake_start:recall_awake_end].detach().cpu().numpy(),
        "ctx_conditioning": X_ctx_all[conditioning_awake_start:conditioning_awake_end].detach().cpu().numpy(),
        # Match source notebook's recall activity snapshot (extra recall pass, no sleep).
        "ctx_recall": X_ctx_all[recall_probe_awake_start:recall_probe_awake_end].detach().cpu().numpy(),
        "mtl_late_extinction": X_mtl_all[late_extinction_start:late_extinction_end].detach().cpu().numpy(),
    }

    # Episode recall stability (savings vs naive)
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
        "network_naive": network_naive,
        "recording_parameters": recording_parameters,
        "fear_cells": fear_cells,
        "ctx_engram_cells": ctx_engram_cells,
        "ctx_nonengram_cells": ctx_nonengram_cells,
        "ctx_only_recall_cells": ctx_only_recall_cells,
        "mtl_semantic_engram_cells": mtl_sem_engram_cells,
        "mtl_semantic_nonengram_cells": mtl_sem_nonengram_cells,
        "synaptic_distributions": synaptic_distributions,
        "synaptic_distributions_post_extinction": synaptic_distributions_post,
        "synaptic_distributions_post_extinction_sampled": synaptic_distributions_post_sampled,
        "fear_input": fear_input,
        "fear_probe_index": int(fear_probe_index),
        "stability_savings": stability_savings,
        "stability_naive": stability_naive,
        "plot_data": plot_data,
        "summary_df": summary_df,
    }
