import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


DEFAULT_ENTITY = "" # TODO
DEFAULT_PROJECT = "" # TODO
DEFAULT_ENV_NAME = "" # TODO
DEFAULT_TOTAL_TIMESTEPS = int(1e9)
DEFAULT_OUTPUT = "" # TODO
DEFAULT_NUM_BINS = 200

ALGORITHM_ORDER = ["PPO-RNN", "PPO", "ICM", "E3B", "RND"]
ALGORITHM_COLORS = {
    "PPO-RNN": "tab:blue",
    "PPO": "orange",
    "ICM": "#11a579",
    "E3B": "#dd6a00",
    "RND": "#c97acb",
}

FIGURE5_ACHIEVEMENTS = [
    "make_wood_pickaxe",
    "enter_dungeon",
    "defeat_zombie",
    "enter_gnomish_mines",
    "eat_plant",
    "make_diamond_sword",
]

FIGURE5_YLIMS = {
    "make_wood_pickaxe": 105.0,
    "enter_dungeon": 90.0,
    "defeat_zombie": 86.0,
    "enter_gnomish_mines": 0.37,
    "eat_plant": 8.8,
    "make_diamond_sword": 2.4,
}


@dataclass
class MetricSeries:
    timesteps: np.ndarray
    values: np.ndarray


@dataclass
class RunHistory:
    algorithm: str
    run_id: str
    run_name: str
    state: str
    seed: int | None
    total_timesteps: int
    num_envs: int
    num_steps: int
    num_repeats: int
    metrics: dict[str, MetricSeries]


@dataclass
class AggregatedMetric:
    timesteps_millions: np.ndarray
    mean: np.ndarray
    sem: np.ndarray
    num_runs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Craftax Figure 5 selected-achievement plot from Weights & Biases runs."
    )
    parser.add_argument("--entity", type=str, default=DEFAULT_ENTITY)
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--env-name", type=str, default=DEFAULT_ENV_NAME)
    parser.add_argument(
        "--total-timesteps",
        type=lambda x: int(float(x)),
        default=DEFAULT_TOTAL_TIMESTEPS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Output path whose stem will be used for all requested formats.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="File formats to save, for example: png pdf.",
    )
    parser.add_argument(
        "--achievements",
        nargs="+",
        default=FIGURE5_ACHIEVEMENTS,
        help="Achievement names without the Achievements/ prefix.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=ALGORITHM_ORDER,
        choices=ALGORITHM_ORDER,
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=["finished", "failed"],
        help="Run states to include. Failed runs can still provide usable histories.",
    )
    parser.add_argument(
        "--exclude-name-fragment",
        nargs="*",
        default=["wrong"],
        help="Case-insensitive run-name fragments to exclude.",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display the figure interactively after saving.",
    )
    parser.add_argument(
        "--downsample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bin raw per-update traces before plotting for readability.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=DEFAULT_NUM_BINS,
        help="Number of timestep bins to use when downsampling.",
    )
    return parser.parse_args()


def infer_total_timesteps(run_name: str, config: dict) -> int | None:
    total_timesteps = config.get("TOTAL_TIMESTEPS")
    if total_timesteps is not None:
        return int(total_timesteps)

    match = re.search(r"-(\d+)M\b", run_name)
    if match:
        return int(match.group(1)) * int(1e6)
    return None


def infer_algorithm(run_name: str, config: dict) -> str:
    name = run_name.upper()
    if config.get("USE_RND") or "PPO_RND" in name or name.endswith("RND"):
        return "RND"
    if "PPO_RNN" in name:
        return "PPO-RNN"
    if config.get("TRAIN_ICM") and config.get("USE_E3B"):
        return "E3B"
    if "E3B" in name:
        return "E3B"
    if config.get("TRAIN_ICM"):
        return "ICM"
    if "ICM" in name:
        return "ICM"
    return "PPO"


def should_skip_run(run_name: str, excluded_fragments: list[str]) -> bool:
    lowered = run_name.lower()
    return any(fragment.lower() in lowered for fragment in excluded_fragments)


def fetch_metric_histories(
    run: wandb.apis.public.Run, achievement_metrics: list[str]
) -> dict[str, MetricSeries]:
    raw_by_metric: dict[str, dict[int, float]] = {metric: {} for metric in achievement_metrics}
    keys = ["_step", *achievement_metrics]

    for row in run.scan_history(keys=keys):
        step = row.get("_step")
        if step is None:
            continue
        step = int(step)

        for metric in achievement_metrics:
            value = row.get(metric)
            if value is None or not np.isfinite(value):
                continue
            raw_by_metric[metric][step] = float(value)

    metric_histories: dict[str, MetricSeries] = {}
    for metric, mapping in raw_by_metric.items():
        if not mapping:
            continue
        steps = np.asarray(sorted(mapping.keys()), dtype=np.int64)
        values = np.asarray([mapping[int(step)] for step in steps], dtype=np.float64)
        metric_histories[metric] = MetricSeries(timesteps=steps, values=values)

    return metric_histories


def load_matching_runs(args: argparse.Namespace) -> list[RunHistory]:
    api = wandb.Api()
    base_path = f"{args.entity}/{args.project}"
    selected_runs: list[RunHistory] = []
    available_summaries: list[str] = []
    achievement_metrics = [f"Achievements/{name}" for name in args.achievements]

    for stub in api.runs(base_path):
        if should_skip_run(stub.name or "", args.exclude_name_fragment):
            continue

        run = api.run(f"{base_path}/{stub.id}")
        config = run.config or {}
        total_timesteps = infer_total_timesteps(run.name or "", config)
        env_name = config.get("ENV_NAME")
        algorithm = infer_algorithm(run.name or "", config)

        available_summaries.append(
            f"{run.id}: {run.name} [{run.state}] total={total_timesteps} env={env_name} alg={algorithm}"
        )

        if run.state not in args.states:
            continue
        if total_timesteps != args.total_timesteps:
            continue
        if env_name is not None and env_name != args.env_name:
            continue
        if env_name is None and args.env_name not in (run.name or ""):
            continue
        if algorithm not in args.algorithms:
            continue

        num_envs = int(config["NUM_ENVS"])
        num_steps = int(config["NUM_STEPS"])
        num_repeats = int(config.get("NUM_REPEATS", 1))
        seed = config.get("SEED")

        metric_histories = fetch_metric_histories(run, achievement_metrics)
        if not metric_histories:
            continue

        steps_per_update = num_envs * num_steps * num_repeats
        converted_metrics: dict[str, MetricSeries] = {}
        for metric_name, metric_series in metric_histories.items():
            # update_step 0 corresponds to 0 environment steps in the paper's plots.
            timesteps = metric_series.timesteps * steps_per_update
            converted_metrics[metric_name] = MetricSeries(
                timesteps=timesteps,
                values=metric_series.values,
            )

        selected_runs.append(
            RunHistory(
                algorithm=algorithm,
                run_id=run.id,
                run_name=run.name or run.id,
                state=run.state,
                seed=seed,
                total_timesteps=total_timesteps,
                num_envs=num_envs,
                num_steps=num_steps,
                num_repeats=num_repeats,
                metrics=converted_metrics,
            )
        )

    if not selected_runs:
        available = "\n".join(sorted(available_summaries)) or "No runs found in W&B project."
        raise ValueError(
            "No matching runs found for the requested Figure 5 settings.\n"
            f"Requested env={args.env_name}, total_timesteps={args.total_timesteps}, states={args.states}.\n"
            f"Available runs:\n{available}"
        )

    return selected_runs


def aggregate_metric_for_algorithm(
    runs: list[RunHistory], metric_name: str, args: argparse.Namespace
) -> AggregatedMetric | None:
    metric_runs = [run for run in runs if metric_name in run.metrics]
    if not metric_runs:
        return None

    if args.downsample:
        bin_edges = np.linspace(0.0, float(args.total_timesteps), args.num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        values = np.full((len(metric_runs), args.num_bins), np.nan, dtype=np.float64)

        for run_index, run in enumerate(metric_runs):
            series = run.metrics[metric_name]
            bin_ids = np.digitize(series.timesteps.astype(np.float64), bin_edges[1:-1], right=False)
            for bin_index in range(args.num_bins):
                mask = bin_ids == bin_index
                if np.any(mask):
                    values[run_index, bin_index] = np.mean(series.values[mask])

        mean = np.nanmean(values, axis=0)
        sem = np.zeros_like(mean)
        for idx in range(values.shape[1]):
            valid = values[:, idx]
            valid = valid[~np.isnan(valid)]
            if valid.size > 1:
                sem[idx] = np.std(valid, ddof=1) / math.sqrt(valid.size)

        return AggregatedMetric(
            timesteps_millions=bin_centers / 1e6,
            mean=mean,
            sem=sem,
            num_runs=len(metric_runs),
        )

    all_timesteps = sorted(
        {
            int(timestep)
            for run in metric_runs
            for timestep in run.metrics[metric_name].timesteps.tolist()
        }
    )
    timestep_array = np.asarray(all_timesteps, dtype=np.int64)
    values = np.full((len(metric_runs), len(timestep_array)), np.nan, dtype=np.float64)
    position_lookup = {value: index for index, value in enumerate(all_timesteps)}

    for run_index, run in enumerate(metric_runs):
        series = run.metrics[metric_name]
        for timestep, value in zip(series.timesteps, series.values, strict=True):
            values[run_index, position_lookup[int(timestep)]] = value

    mean = np.nanmean(values, axis=0)
    sem = np.zeros_like(mean)
    for idx in range(values.shape[1]):
        valid = values[:, idx]
        valid = valid[~np.isnan(valid)]
        if valid.size > 1:
            sem[idx] = np.std(valid, ddof=1) / math.sqrt(valid.size)

    return AggregatedMetric(
        timesteps_millions=timestep_array.astype(np.float64) / 1e6,
        mean=mean,
        sem=sem,
        num_runs=len(metric_runs),
    )


def aggregate_runs(
    runs: list[RunHistory], achievements: list[str], args: argparse.Namespace
) -> dict[str, dict[str, AggregatedMetric]]:
    by_algorithm: dict[str, list[RunHistory]] = defaultdict(list)
    for run in runs:
        by_algorithm[run.algorithm].append(run)

    aggregated: dict[str, dict[str, AggregatedMetric]] = defaultdict(dict)
    for algorithm, algorithm_runs in by_algorithm.items():
        for achievement in achievements:
            metric_name = f"Achievements/{achievement}"
            metric = aggregate_metric_for_algorithm(algorithm_runs, metric_name, args)
            if metric is not None:
                aggregated[algorithm][achievement] = metric

    return aggregated


def resolve_ylim(achievement: str, aggregated: dict[str, dict[str, AggregatedMetric]]) -> tuple[float, float]:
    configured_upper = FIGURE5_YLIMS.get(achievement)
    if configured_upper is not None:
        return 0.0, configured_upper

    observed_upper = 0.0
    for algorithm in ALGORITHM_ORDER:
        metric = aggregated.get(algorithm, {}).get(achievement)
        if metric is None:
            continue
        observed_upper = max(observed_upper, float(np.nanmax(metric.mean + metric.sem)))

    return 0.0, max(1.0, observed_upper * 1.1)


def plot_figure5(
    aggregated: dict[str, dict[str, AggregatedMetric]], args: argparse.Namespace
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9), constrained_layout=True)
    axes = axes.reshape(2, 3)

    legend_handles = []
    legend_labels = []

    for plot_index, achievement in enumerate(args.achievements):
        row = plot_index // 3
        col = plot_index % 3
        ax = axes[row, col]

        for algorithm in ALGORITHM_ORDER:
            metric = aggregated.get(algorithm, {}).get(achievement)
            if metric is None:
                continue

            color = ALGORITHM_COLORS[algorithm]
            line = ax.plot(
                metric.timesteps_millions,
                metric.mean,
                color=color,
                linewidth=2.5,
                label=algorithm,
            )[0]
            ax.fill_between(
                metric.timesteps_millions,
                metric.mean - metric.sem,
                metric.mean + metric.sem,
                color=color,
                alpha=0.2,
            )

            if algorithm not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(algorithm)

        ax.set_title(achievement, fontsize=18, pad=10)
        ax.set_xlim(0, args.total_timesteps / 1e6)
        ax.set_ylim(*resolve_ylim(achievement, aggregated))
        ax.set_xticks([0, 500, 1000])
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=14)

        if col == 0:
            ax.set_ylabel("Success Rate (%)", fontsize=18)
        if row == 1:
            ax.set_xlabel("Timestep (M)", fontsize=18)

        if achievement == "make_diamond_sword":
            ax.legend(
                legend_handles,
                legend_labels,
                loc="upper left",
                framealpha=0.95,
                fontsize=14,
            )

    return fig, axes


def print_run_summary(runs: list[RunHistory], achievements: list[str]) -> None:
    print("Selected runs:")
    for run in sorted(runs, key=lambda item: (ALGORITHM_ORDER.index(item.algorithm), item.run_name)):
        available = [
            achievement
            for achievement in achievements
            if f"Achievements/{achievement}" in run.metrics
        ]
        print(
            f"  {run.algorithm:7s} | {run.run_id} | seed={run.seed} | "
            f"state={run.state} | steps/update={run.num_envs * run.num_steps * run.num_repeats} | "
            f"metrics={len(available)} | {run.run_name}"
        )


def save_figure(fig: plt.Figure, output: Path, formats: list[str]) -> list[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    stem = output.with_suffix("")

    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=200)
        saved_paths.append(path)

    return saved_paths


def main() -> None:
    args = parse_args()
    runs = load_matching_runs(args)
    aggregated = aggregate_runs(runs, args.achievements, args)
    print_run_summary(runs, args.achievements)

    fig, _ = plot_figure5(aggregated, args)
    saved_paths = save_figure(fig, args.output, args.formats)
    print("\nSaved figure files:")
    for path in saved_paths:
        print(f"  {path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
