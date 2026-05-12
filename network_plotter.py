from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Affine2D
import numpy as np
import torch

from src.plotting_parameters import plotting_parameters


blue_yellow = LinearSegmentedColormap.from_list(
    "blue_yellow",
    ["#0073B7", "#FFD23F"],
)


@dataclass
class DayRun:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class RecordingSchedule:
    awake_runs: List[DayRun]
    sleep_a_runs: List[DayRun]
    sleep_b_runs: List[DayRun]

    @property
    def num_awake_days(self) -> int:
        return len(self.awake_runs)

    @property
    def num_sleep_a_days(self) -> int:
        return len(self.sleep_a_runs)

    @property
    def num_sleep_b_days(self) -> int:
        return len(self.sleep_b_runs)

    @property
    def awake_timesteps_per_day(self) -> List[int]:
        return [run.length for run in self.awake_runs]

    @property
    def sleep_a_timesteps_per_day(self) -> List[int]:
        return [run.length for run in self.sleep_a_runs]

    @property
    def sleep_b_timesteps_per_day(self) -> List[int]:
        return [run.length for run in self.sleep_b_runs]


def _get_index_signature(indices: Sequence[int]) -> Tuple[int, int | None]:
    if len(indices) == 0:
        return (0, None)
    return (len(indices), int(indices[-1]))


def _split_consecutive_runs(indices: Sequence[int]) -> List[DayRun]:
    if len(indices) == 0:
        return []

    runs: List[DayRun] = []
    run_start = int(indices[0])
    prev = int(indices[0])

    for value in indices[1:]:
        value = int(value)
        if value != prev + 1:
            runs.append(DayRun(start=run_start, end=prev))
            run_start = value
        prev = value

    runs.append(DayRun(start=run_start, end=prev))
    return runs


def get_recording_schedule(net) -> RecordingSchedule:
    signature = (
        _get_index_signature(getattr(net, "awake_indices", [])),
        _get_index_signature(getattr(net, "sleep_indices_A", [])),
        _get_index_signature(getattr(net, "sleep_indices_B", [])),
    )

    cached_signature = getattr(net, "_network_plotter_schedule_signature", None)
    cached_schedule = getattr(net, "_network_plotter_schedule", None)
    if cached_schedule is not None and cached_signature == signature:
        return cached_schedule

    schedule = RecordingSchedule(
        awake_runs=_split_consecutive_runs(getattr(net, "awake_indices", [])),
        sleep_a_runs=_split_consecutive_runs(getattr(net, "sleep_indices_A", [])),
        sleep_b_runs=_split_consecutive_runs(getattr(net, "sleep_indices_B", [])),
    )
    net._network_plotter_schedule_signature = signature
    net._network_plotter_schedule = schedule
    return schedule


def summarize_recording_schedule(net) -> Dict[str, object]:
    schedule = get_recording_schedule(net)
    return {
        "num_awake_days": schedule.num_awake_days,
        "num_sleep_a_days": schedule.num_sleep_a_days,
        "num_sleep_b_days": schedule.num_sleep_b_days,
        "awake_timesteps_per_day": schedule.awake_timesteps_per_day,
        "sleep_a_timesteps_per_day": schedule.sleep_a_timesteps_per_day,
        "sleep_b_timesteps_per_day": schedule.sleep_b_timesteps_per_day,
    }


def _resolve_phase_runs(net, wake_sleep: int | str) -> Tuple[str, List[DayRun]]:
    schedule = get_recording_schedule(net)

    if wake_sleep in {0, "awake"}:
        return "awake", schedule.awake_runs
    if wake_sleep in {1, "sleep", "sleep_a"}:
        return "sleep_a", schedule.sleep_a_runs
    if wake_sleep in {2, "sleep_b"}:
        return "sleep_b", schedule.sleep_b_runs

    raise ValueError(
        "wake_sleep must be one of {0, 1, 2, 'awake', 'sleep', 'sleep_a', 'sleep_b'}."
    )


def _resolve_recording_index(net, day: int, wake_sleep: int | str, timestep: int) -> Tuple[str, int]:
    phase_name, phase_runs = _resolve_phase_runs(net, wake_sleep=wake_sleep)
    if day < 0 or day >= len(phase_runs):
        raise IndexError(
            f"Requested day={day} for phase '{phase_name}', but only "
            f"{len(phase_runs)} days are available."
        )

    run = phase_runs[day]
    if timestep < 0 or timestep >= run.length:
        raise IndexError(
            f"Requested timestep={timestep} for day={day}, phase='{phase_name}', "
            f"but valid timesteps are [0, {run.length - 1}]."
        )

    return phase_name, run.start + timestep


def _get_region_activity(net, region: str, recording_index: int) -> torch.Tensor:
    recordings = torch.stack(net.activity_recordings[region], dim=0)
    return recordings[recording_index].detach().cpu().float()


def _get_connection_snapshot(
    net,
    connection: str,
    *,
    from_recordings: bool = False,
    day: int | None = None,
    wake_sleep: int | str | None = None,
    timestep: int | None = None,
) -> torch.Tensor:
    if not from_recordings:
        return getattr(net, connection).detach().cpu().float()

    if day is None or wake_sleep is None or timestep is None:
        raise ValueError(
            "When from_recordings=True, day, wake_sleep, and timestep must all be provided."
        )

    if connection not in getattr(net, "connectivity_recordings", {}):
        raise KeyError(f"Connection '{connection}' is not available in connectivity_recordings.")

    _, recording_index = _resolve_recording_index(
        net,
        day=day,
        wake_sleep=wake_sleep,
        timestep=timestep,
    )
    target_step = max(recording_index - 1, 0)

    recordings = torch.stack(net.connectivity_recordings[connection], dim=0).detach().cpu().float()
    if recordings.shape[0] == 1:
        return recordings[0]

    times = np.asarray(getattr(net, "connectivity_recordings_time", []), dtype=int)
    if times.size == 0:
        return recordings[min(recording_index, recordings.shape[0] - 1)]

    unique_times = np.unique(times)
    recording_slot = int(np.searchsorted(unique_times, target_step, side="right"))
    recording_slot = min(recording_slot, recordings.shape[0] - 1)
    return recordings[recording_slot]


def _maybe_reorder(activity: torch.Tensor, ordering: torch.Tensor | None) -> torch.Tensor:
    if ordering is None:
        return activity
    return activity[ordering.detach().cpu()]


def _get_order_attr_name(region: str, subregion_index: int | None = None) -> str | None:
    if region == "ctx":
        return "ordered_indices_ctx"
    if region == "mtl_sensory":
        return "ordered_indices_mtl_sensory"
    if region == "mtl_semantic":
        return "ordered_indices_mtl_semantic"
    if region == "mtl":
        if subregion_index == 0:
            return "ordered_indices_mtl_sensory"
        if subregion_index == 1:
            return "ordered_indices_mtl_semantic"
    return None


def _get_order_for_selected_indices(
    net,
    region: str,
    selected_indices: torch.Tensor,
    *,
    subregion_index: int | None = None,
    explicit_order_attr: str | None = None,
):
    order_attr = explicit_order_attr if explicit_order_attr is not None else _get_order_attr_name(region, subregion_index)
    if order_attr is None or not hasattr(net, order_attr):
        return None

    ordering = getattr(net, order_attr).detach().cpu().long()
    selected_indices = selected_indices.detach().cpu().long()

    # Ordering may already be local to the selected subregion.
    if ordering.numel() > 0 and int(ordering.max()) < selected_indices.numel() and int(ordering.min()) >= 0:
        return ordering

    # Otherwise treat ordering as global and convert to local positions.
    selected_set = {int(v): idx for idx, v in enumerate(selected_indices.tolist())}
    local_positions = [selected_set[int(v)] for v in ordering.tolist() if int(v) in selected_set]
    if not local_positions:
        return None
    return torch.tensor(local_positions, dtype=torch.long)


def _get_connection_regions(connection: str) -> Tuple[str, str]:
    post_region, pre_region = connection.split("_", 1)
    return post_region, pre_region


def _get_region_indices(
    net,
    region: str,
    *,
    subregion_index: int | None = None,
) -> torch.Tensor:
    if subregion_index is None:
        return torch.arange(int(getattr(net, f"{region}_size")), dtype=torch.long)
    return getattr(net, f"{region}_subregions")[subregion_index].detach().cpu().long()


def _reshape_strip(activity: torch.Tensor, n_rows: int = 10) -> np.ndarray:
    activity = activity.detach().cpu().float()
    if activity.numel() % n_rows != 0:
        raise ValueError(
            f"Cannot reshape activity with {activity.numel()} neurons into "
            f"{n_rows} rows."
        )
    return activity.reshape(n_rows, activity.numel() // n_rows).numpy()


def _reshape_concept_columns(activity: torch.Tensor, neurons_per_concept: int = 10) -> np.ndarray:
    activity = activity.detach().cpu().float()
    if activity.numel() % neurons_per_concept != 0:
        raise ValueError(
            f"Cannot reshape activity with {activity.numel()} neurons into concept "
            f"columns of size {neurons_per_concept}."
        )
    return activity.reshape(activity.numel() // neurons_per_concept, neurons_per_concept).T.numpy()


def _oblique_panel_transform(
    ax,
    *,
    shear_deg_x: float = -18.0,
    shear_deg_y: float = 0.0,
):
    return Affine2D().skew_deg(shear_deg_x, shear_deg_y) + ax.transData


def _draw_panel(
    ax,
    activity: np.ndarray,
    *,
    title: str,
    cmap,
    perspective: bool,
    perspective_kwargs: Dict[str, float] | None = None,
    draw_grid: bool = True,
    grid_color: str = "black",
    grid_linewidth: float = 0.35,
    grid_alpha: float = 0.8,
):
    n_rows, n_cols = activity.shape

    if perspective:
        transform = _oblique_panel_transform(ax, **(perspective_kwargs or {}))
        ax.imshow(
            activity,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
            interpolation="nearest",
            origin="upper",
            extent=(0, n_cols, n_rows, 0),
            transform=transform,
        )
        shear = np.tan(np.deg2rad((perspective_kwargs or {}).get("shear_deg_x", -18.0)))
        x_shift = abs(shear) * n_rows
        ax.set_xlim(-x_shift - 0.5, n_cols + 0.5)
        ax.set_ylim(n_rows + 0.5, -0.5)

        if draw_grid:
            for x in range(n_cols + 1):
                ax.plot(
                    [x, x],
                    [0, n_rows],
                    color=grid_color,
                    linewidth=grid_linewidth,
                    alpha=grid_alpha,
                    transform=transform,
                    solid_capstyle="butt",
                )
            for y in range(n_rows + 1):
                ax.plot(
                    [0, n_cols],
                    [y, y],
                    color=grid_color,
                    linewidth=grid_linewidth,
                    alpha=grid_alpha,
                    transform=transform,
                    solid_capstyle="butt",
                )
    else:
        ax.imshow(
            activity,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
            interpolation="nearest",
            origin="upper",
        )
        if draw_grid:
            for x in range(n_cols + 1):
                ax.plot(
                    [x - 0.5, x - 0.5],
                    [-0.5, n_rows - 0.5],
                    color=grid_color,
                    linewidth=grid_linewidth,
                    alpha=grid_alpha,
                    solid_capstyle="butt",
                )
            for y in range(n_rows + 1):
                ax.plot(
                    [-0.5, n_cols - 0.5],
                    [y - 0.5, y - 0.5],
                    color=grid_color,
                    linewidth=grid_linewidth,
                    alpha=grid_alpha,
                    solid_capstyle="butt",
                )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=14, pad=6)


def _draw_perspective_vertical_separator(
    ax,
    n_rows: int,
    n_cols: int,
    boundary_col: float,
    *,
    perspective_kwargs: Dict[str, float] | None = None,
    color: str = "white",
    linewidth: float = 2.0,
    alpha: float = 0.9,
):
    transform = _oblique_panel_transform(ax, **(perspective_kwargs or {}))
    ax.plot(
        [boundary_col, boundary_col],
        [0, n_rows],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        transform=transform,
    )


def get_network_snapshot(net, day: int, wake_sleep: int | str, timestep: int) -> Dict[str, object]:
    phase_name, recording_index = _resolve_recording_index(
        net,
        day=day,
        wake_sleep=wake_sleep,
        timestep=timestep,
    )

    ctx_order = getattr(net, "ordered_indices_ctx", None)
    mtl_semantic_order = getattr(net, "ordered_indices_mtl_semantic", None)

    ctx = _get_region_activity(net, "ctx", recording_index)
    mtl_sensory = _get_region_activity(net, "mtl_sensory", recording_index)
    mtl_semantic = _maybe_reorder(
        _get_region_activity(net, "mtl_semantic", recording_index),
        mtl_semantic_order,
    )

    ctx_subregion_0 = ctx[net.ctx_subregions[0]]
    ctx_subregion_1 = ctx[net.ctx_subregions[1]]
    if ctx_order is not None:
        ctx_subregion_0 = ctx_subregion_0[ctx_order.detach().cpu()[: net.ctx_size_subregions[0]]]

    return {
        "phase": phase_name,
        "recording_index": recording_index,
        "ctx_subregion_0": ctx_subregion_0,
        "ctx_subregion_1": ctx_subregion_1,
        "mtl_sensory": mtl_sensory,
        "mtl_semantic": mtl_semantic,
    }


def plot_region_snapshot(
    net,
    region: str,
    day: int,
    wake_sleep: int | str,
    timestep: int,
    *,
    cmap=blue_yellow,
    figsize: Tuple[float, float] = (2, 2),
    reshape_rows: int = 10,
    order: torch.Tensor | np.ndarray | None = None,
    auto_order: bool = False,
    ylabel: str | None = "Neuron",
    xlabel: str | None = "Neuron",
    show_xticks: bool = False,
    show_yticks: bool = True,
    xticks=None,
    yticks=None,
    yticklabels=None,
    ytick_fontsize: int = 14,
    label_fontsize: int = 20,
    vmin=None,
    vmax=None,
    apply_tight_layout: bool = False,
    draw_grid: bool = True,
    grid_color: str = "black",
    grid_linewidth: float = 0.35,
    grid_alpha: float = 0.8,
):
    region_defaults = plotting_parameters.get(region, {})

    if figsize == (2, 2) and "figsize" in region_defaults:
        figsize = tuple(region_defaults["figsize"])
    if reshape_rows == 10 and "reshape_rows" in region_defaults:
        reshape_rows = int(region_defaults["reshape_rows"])
    if ylabel == "Neuron" and "ylabel" in region_defaults:
        ylabel = region_defaults["ylabel"]
    if xlabel == "Neuron" and "xlabel" in region_defaults:
        xlabel = region_defaults["xlabel"]
    if show_xticks is False and "show_xticks" in region_defaults:
        show_xticks = bool(region_defaults["show_xticks"])
    if show_yticks is True and "show_yticks" in region_defaults:
        show_yticks = bool(region_defaults["show_yticks"])
    if xticks is None and "xticks" in region_defaults:
        xticks = region_defaults["xticks"]
    if yticks is None and "yticks" in region_defaults:
        yticks = region_defaults["yticks"]
    if yticklabels is None and "yticklabels" in region_defaults:
        yticklabels = region_defaults["yticklabels"]
    if ytick_fontsize == 14 and "ytick_fontsize" in region_defaults:
        ytick_fontsize = int(region_defaults["ytick_fontsize"])
    if label_fontsize == 20 and "label_fontsize" in region_defaults:
        label_fontsize = int(region_defaults["label_fontsize"])
    if auto_order is False and "auto_order" in region_defaults:
        auto_order = bool(region_defaults["auto_order"])
    if vmin is None and "vmin" in region_defaults:
        vmin = region_defaults["vmin"]
    if vmax is None and "vmax" in region_defaults:
        vmax = region_defaults["vmax"]

    _, recording_index = _resolve_recording_index(
        net,
        day=day,
        wake_sleep=wake_sleep,
        timestep=timestep,
    )

    activity = _get_region_activity(net, region, recording_index)

    if auto_order:
        if region == "ctx" and hasattr(net, "ordered_indices_ctx"):
            activity = activity[net.ordered_indices_ctx.detach().cpu()]
        elif region == "mtl_semantic" and hasattr(net, "ordered_indices_mtl_semantic"):
            activity = activity[net.ordered_indices_mtl_semantic.detach().cpu()]

    if order is not None:
        if torch.is_tensor(order):
            order = order.detach().cpu()
        activity = activity[order]

    panel = _reshape_strip(activity, n_rows=reshape_rows)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax)

    if draw_grid:
        n_rows, n_cols = panel.shape
        for x in range(n_cols + 1):
            ax.plot(
                [x - 0.5, x - 0.5],
                [-0.5, n_rows - 0.5],
                color=grid_color,
                linewidth=grid_linewidth,
                alpha=grid_alpha,
                solid_capstyle="butt",
            )
        for y in range(n_rows + 1):
            ax.plot(
                [-0.5, n_cols - 0.5],
                [y - 0.5, y - 0.5],
                color=grid_color,
                linewidth=grid_linewidth,
                alpha=grid_alpha,
                solid_capstyle="butt",
            )

    if show_xticks:
        if xticks is None:
            ax.tick_params(axis="x", labelsize=18)
        else:
            if len(xticks) == 2 and xticks[1] is None:
                ax.set_xticks(xticks[0])
                ax.tick_params(axis="x", labelsize=18)
            else:
                ax.set_xticks(xticks[0], xticks[1])
    else:
        ax.set_xticks([])

    if show_yticks:
        if yticks is not None:
            if yticklabels is None:
                ax.set_yticks(yticks)
                ax.tick_params(axis="y", labelsize=ytick_fontsize, length=0)
            else:
                ax.set_yticks(yticks, yticklabels)
                ax.tick_params(axis="y", labelsize=ytick_fontsize, length=0)
        else:
            ax.tick_params(axis="y", labelsize=ytick_fontsize, length=0)
    else:
        ax.set_yticks([])

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)

    if apply_tight_layout:
        fig.tight_layout()
    return fig, ax


def plot_weight_snapshot(
    net,
    connection: str,
    day: int | None = None,
    wake_sleep: int | str | None = None,
    timestep: int | None = None,
    *,
    cmap,
    from_recordings: bool = False,
    figsize: Tuple[float, float] | None = None,
    row_order_attr: str | None = None,
    row_slice=None,
    col_order_attr: str | None = None,
    col_slice=None,
    subregion_post: int | None = None,
    subregion_pre: int | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    show_xticks: bool | None = None,
    show_yticks: bool | None = None,
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    xtick_rotation: float | None = None,
    tick_fontsize: int | None = None,
    label_fontsize: int | None = None,
    draw_grid: bool | None = None,
    grid_color: str = "black",
    grid_linewidth: float | None = None,
    grid_alpha: float = 0.8,
    vmin=None,
    vmax=None,
    apply_tight_layout: bool = False,
    subplots_adjust: Dict[str, float] | None = None,
):
    defaults = plotting_parameters.get(connection, {})

    figsize = tuple(defaults.get("figsize", (3, 3))) if figsize is None else figsize
    row_order_attr = defaults.get("row_order_attr") if row_order_attr is None else row_order_attr
    row_slice = defaults.get("row_slice") if row_slice is None else row_slice
    col_order_attr = defaults.get("col_order_attr") if col_order_attr is None else col_order_attr
    col_slice = defaults.get("col_slice") if col_slice is None else col_slice
    subregion_post = defaults.get("subregion_post") if subregion_post is None else subregion_post
    subregion_pre = defaults.get("subregion_pre") if subregion_pre is None else subregion_pre
    ylabel = defaults.get("ylabel") if ylabel is None else ylabel
    xlabel = defaults.get("xlabel") if xlabel is None else xlabel
    show_xticks = defaults.get("show_xticks", True) if show_xticks is None else show_xticks
    show_yticks = defaults.get("show_yticks", True) if show_yticks is None else show_yticks
    xticks = defaults.get("xticks") if xticks is None else xticks
    xticklabels = defaults.get("xticklabels") if xticklabels is None else xticklabels
    yticks = defaults.get("yticks") if yticks is None else yticks
    yticklabels = defaults.get("yticklabels") if yticklabels is None else yticklabels
    xtick_rotation = defaults.get("xtick_rotation", 0) if xtick_rotation is None else xtick_rotation
    tick_fontsize = defaults.get("tick_fontsize", 18) if tick_fontsize is None else tick_fontsize
    label_fontsize = defaults.get("label_fontsize", 20) if label_fontsize is None else label_fontsize
    draw_grid = defaults.get("draw_grid", True) if draw_grid is None else draw_grid
    grid_linewidth = defaults.get("grid_linewidth", 0.1) if grid_linewidth is None else grid_linewidth
    vmin = defaults.get("vmin") if vmin is None else vmin
    vmax = defaults.get("vmax") if vmax is None else vmax
    subplots_adjust = defaults.get("subplots_adjust") if subplots_adjust is None else subplots_adjust

    weights = _get_connection_snapshot(
        net,
        connection,
        from_recordings=from_recordings,
        day=day,
        wake_sleep=wake_sleep,
        timestep=timestep,
    )

    post_region, pre_region = _get_connection_regions(connection)

    post_indices = _get_region_indices(net, post_region, subregion_index=subregion_post)
    pre_indices = _get_region_indices(net, pre_region, subregion_index=subregion_pre)
    weights = weights[post_indices][:, pre_indices]

    row_order = _get_order_for_selected_indices(
        net,
        post_region,
        post_indices,
        subregion_index=subregion_post,
        explicit_order_attr=row_order_attr,
    )
    if row_order is not None:
        weights = weights[row_order]

    col_order = _get_order_for_selected_indices(
        net,
        pre_region,
        pre_indices,
        subregion_index=subregion_pre,
        explicit_order_attr=col_order_attr,
    )
    if col_order is not None:
        weights = weights[:, col_order]
    if row_slice is not None:
        weights = weights[row_slice]
    if col_slice is not None:
        weights = weights[:, col_slice]

    panel = weights.detach().cpu().numpy()

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax)

    if draw_grid:
        n_rows, n_cols = panel.shape
        for x in range(n_cols + 1):
            ax.plot(
                [x - 0.5, x - 0.5],
                [-0.5, n_rows - 0.5],
                color=grid_color,
                linewidth=grid_linewidth,
                alpha=grid_alpha,
                solid_capstyle="butt",
            )
        for y in range(n_rows + 1):
            ax.plot(
                [-0.5, n_cols - 0.5],
                [y - 0.5, y - 0.5],
                color=grid_color,
                linewidth=grid_linewidth,
                alpha=grid_alpha,
                solid_capstyle="butt",
            )

    if show_xticks:
        if xticks is not None:
            ax.set_xticks(xticks)
            if xticklabels is not None:
                ax.set_xticklabels(xticklabels, rotation=xtick_rotation)
        ax.tick_params(
            axis="x",
            labelsize=tick_fontsize,
            length=0,
            bottom=False,
            top=False,
            labelbottom=True,
            pad=2,
        )
    else:
        ax.set_xticks([])

    if show_yticks:
        if yticks is not None:
            ax.set_yticks(yticks)
            if yticklabels is not None:
                ax.set_yticklabels(yticklabels)
        ax.tick_params(
            axis="y",
            labelsize=tick_fontsize,
            length=0,
            left=False,
            right=False,
            labelleft=True,
            pad=2,
        )
    else:
        ax.set_yticks([])

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)

    if apply_tight_layout:
        fig.tight_layout()
    elif subplots_adjust is not None:
        fig.subplots_adjust(**subplots_adjust)
    return fig, ax


def plot_network_snapshot(
    net,
    day: int,
    wake_sleep: int | str,
    timestep: int,
    *,
    cmap=blue_yellow,
    figsize: Tuple[float, float] = (11, 5.8),
    show_titles: bool = True,
    perspective: bool = True,
    draw_grid: bool = True,
):
    snapshot = get_network_snapshot(net, day=day, wake_sleep=wake_sleep, timestep=timestep)
    neurons_per_concept = 10

    ctx_panel = np.concatenate(
        [
            _reshape_concept_columns(snapshot["ctx_subregion_0"], neurons_per_concept=neurons_per_concept),
            _reshape_concept_columns(snapshot["ctx_subregion_1"], neurons_per_concept=neurons_per_concept),
        ],
        axis=1,
    )
    mtl_sensory_panel = _reshape_concept_columns(
        snapshot["mtl_sensory"],
        neurons_per_concept=neurons_per_concept,
    )
    mtl_semantic_panel = _reshape_concept_columns(
        snapshot["mtl_semantic"],
        neurons_per_concept=neurons_per_concept,
    )

    perspective_settings = {
        "ctx": {"shear_deg_x": -18.0, "shear_deg_y": 0.0},
        "mtl_sensory": {"shear_deg_x": -18.0, "shear_deg_y": 0.0},
        "mtl_semantic": {"shear_deg_x": -18.0, "shear_deg_y": 0.0},
    }

    def _effective_panel_width(n_rows: int, n_cols: int, shear_deg_x: float) -> float:
        return n_cols + abs(np.tan(np.deg2rad(shear_deg_x))) * n_rows

    ctx_width_units = _effective_panel_width(
        ctx_panel.shape[0],
        ctx_panel.shape[1],
        perspective_settings["ctx"]["shear_deg_x"],
    )
    mtl_width_units = _effective_panel_width(
        mtl_sensory_panel.shape[0],
        mtl_sensory_panel.shape[1],
        perspective_settings["mtl_sensory"]["shear_deg_x"],
    )
    center_gap_units = 2.5
    side_gap_units = max((ctx_width_units - 2 * mtl_width_units - center_gap_units) / 2, 0.5)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        1,
        # Top panel is 10x40 and spans the full width.
        # Bottom row is handled by a nested grid so the two 10x10 panels can
        # keep the same neuron size while allowing a tunable central gap.
        height_ratios=[1.0, 1.0],
        hspace=0.06,
    )

    ax_ctx = fig.add_subplot(grid[0, 0])

    bottom_grid = grid[1, 0].subgridspec(
        1,
        5,
        width_ratios=[
            side_gap_units,
            mtl_width_units,
            center_gap_units,
            mtl_width_units,
            side_gap_units,
        ],
        wspace=0.0,
    )
    ax_mtl_sensory = fig.add_subplot(bottom_grid[0, 1])
    ax_mtl_semantic = fig.add_subplot(bottom_grid[0, 3])

    panels = [
        (ax_ctx, ctx_panel, "CTX"),
        (ax_mtl_sensory, mtl_sensory_panel, "MTL-sensory"),
        (ax_mtl_semantic, mtl_semantic_panel, "MTL-semantic"),
    ]

    for ax, activity, title in panels:
        _draw_panel(
            ax,
            activity,
            title=title if show_titles else "",
            cmap=cmap,
            perspective=perspective,
            perspective_kwargs=perspective_settings[title.lower().replace("-", "_")],
            draw_grid=draw_grid,
        )

    if perspective:
        _draw_perspective_vertical_separator(
            ax_ctx,
            ctx_panel.shape[0],
            ctx_panel.shape[1],
            boundary_col=10.0,
            perspective_kwargs=perspective_settings["ctx"],
        )
    else:
        ax_ctx.axvline(ctx_panel.shape[1] / 4 - 0.5, color="white", linewidth=2.0, alpha=0.9)

    phase_display = {
        "awake": "Awake",
        "sleep_a": "Sleep-A",
        "sleep_b": "Sleep-B",
    }.get(snapshot["phase"], str(snapshot["phase"]))
    fig.suptitle(
        f"Network snapshot | day={day} | phase={phase_display} | timestep={timestep}",
        fontsize=16,
        y=1.01,
    )

    return fig, {
        "ctx": ax_ctx,
        "mtl_sensory": ax_mtl_sensory,
        "mtl_semantic": ax_mtl_semantic,
    }
