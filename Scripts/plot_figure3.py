import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


PAPER_MAX_REWARD = 226.0
DEFAULT_ENTITY = "" # TODO
DEFAULT_PROJECT = "" # TODO
DEFAULT_ENV_NAME = "" # TODO
DEFAULT_TOTAL_TIMESTEPS = int(1e9)
DEFAULT_OUTPUT = "" # TODO

ALGORITHM_ORDER = ["PPO-RNN", "PPO", "ICM", "E3B", "RND"]
ALGORITHM_COLORS = {
    "PPO-RNN": "tab:blue",
    "PPO": "tab:orange",
    "ICM": "tab:green",
    "E3B": "tab:red",
    "RND": "tab:purple",
}


@dataclass
class RunSeries:
    algorithm: str
    run_id: str
    run_name: str
    state: str
    seed: int | None
    total_timesteps: int
    num_envs: int
    num_steps: int
    num_repeats: int
    timesteps: np.ndarray
    values: np.ndarray


@dataclass
class AggregatedSeries:
    algorithm: str
    timesteps: np.ndarray
    mean: np.ndarray
    sem: np.ndarray
    num_runs: int
    run_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Craftax Figure 3 reward plot from Weights & Biases runs."
    )
    parser.add_argument("--entity", type=str, default=DEFAULT_ENTITY)
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--env-name", type=str, default=DEFAULT_ENV_NAME)
    parser.add_argument(
        "--total-timesteps",
        type=lambda x: int(float(x)),
        default=DEFAULT_TOTAL_TIMESTEPS,
    )
    parser.add_argument("--metric", type=str, default="episode_return")
    parser.add_argument("--max-reward", type=float, default=PAPER_MAX_REWARD)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Output path for the saved figure.",
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
        help="Run states to include. Failed runs are useful if they still logged history.",
    )
    parser.add_argument(
        "--exclude-name-fragment",
        nargs="*",
        default=["wrong"],
        help="Case-insensitive name fragments to exclude.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display the figure interactively after saving.",
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


def fetch_history(run: wandb.apis.public.Run, metric: str) -> tuple[np.ndarray, np.ndarray]:
    steps: list[int] = []
    values: list[float] = []
    last_seen: dict[int, float] = {}

    for row in run.scan_history(keys=["_step", metric]):
        step = row.get("_step")
        value = row.get(metric)
        if step is None or value is None:
            continue
        if not np.isfinite(value):
            continue
        last_seen[int(step)] = float(value)

    if not last_seen:
        return np.array([]), np.array([])

    for step in sorted(last_seen):
        steps.append(step)
        values.append(last_seen[step])

    return np.asarray(steps, dtype=np.int64), np.asarray(values, dtype=np.float64)


def load_matching_runs(args: argparse.Namespace) -> list[RunSeries]:
    api = wandb.Api()
    base_path = f"{args.entity}/{args.project}"
    selected_runs: list[RunSeries] = []
    available_summaries: list[str] = []

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

        update_steps, values = fetch_history(run, args.metric)
        if update_steps.size == 0:
            continue

        # The local training loop logs once per update with update_step starting at 0.
        # Mapping step 0 to timestep 0 matches the paper's visual convention.
        timesteps = update_steps * num_envs * num_steps * num_repeats
        values = values * 100.0 / args.max_reward

        selected_runs.append(
            RunSeries(
                algorithm=algorithm,
                run_id=run.id,
                run_name=run.name or run.id,
                state=run.state,
                seed=seed,
                total_timesteps=total_timesteps,
                num_envs=num_envs,
                num_steps=num_steps,
                num_repeats=num_repeats,
                timesteps=timesteps,
                values=values,
            )
        )

    if not selected_runs:
        available = "\n".join(sorted(available_summaries)) or "No runs found in W&B project."
        raise ValueError(
            "No matching runs found for the requested Figure 3 settings.\n"
            f"Requested env={args.env_name}, total_timesteps={args.total_timesteps}, states={args.states}.\n"
            f"Available runs:\n{available}"
        )

    return selected_runs


def aggregate_runs(runs: list[RunSeries]) -> dict[str, AggregatedSeries]:
    grouped: dict[str, list[RunSeries]] = defaultdict(list)
    for run in runs:
        grouped[run.algorithm].append(run)

    aggregated: dict[str, AggregatedSeries] = {}
    for algorithm, algorithm_runs in grouped.items():
        all_timesteps = sorted(
            {
                int(timestep)
                for run in algorithm_runs
                for timestep in run.timesteps.tolist()
            }
        )
        timestep_array = np.asarray(all_timesteps, dtype=np.int64)
        values = np.full((len(algorithm_runs), len(timestep_array)), np.nan, dtype=np.float64)

        position_lookup = {value: index for index, value in enumerate(all_timesteps)}
        for run_index, run in enumerate(algorithm_runs):
            for timestep, value in zip(run.timesteps, run.values, strict=True):
                values[run_index, position_lookup[int(timestep)]] = value

        mean = np.nanmean(values, axis=0)
        sem = np.zeros_like(mean)
        for idx in range(values.shape[1]):
            valid = values[:, idx]
            valid = valid[~np.isnan(valid)]
            if valid.size > 1:
                sem[idx] = np.std(valid, ddof=1) / math.sqrt(valid.size)

        aggregated[algorithm] = AggregatedSeries(
            algorithm=algorithm,
            timesteps=timestep_array,
            mean=mean,
            sem=sem,
            num_runs=len(algorithm_runs),
            run_ids=[run.run_id for run in algorithm_runs],
        )

    return aggregated


def plot_series(
    aggregated: dict[str, AggregatedSeries], args: argparse.Namespace
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)

    for algorithm in ALGORITHM_ORDER:
        series = aggregated.get(algorithm)
        if series is None:
            continue

        color = ALGORITHM_COLORS[algorithm]
        ax.plot(
            series.timesteps,
            series.mean,
            color=color,
            linewidth=2.5,
            label=algorithm,
        )
        ax.fill_between(
            series.timesteps,
            series.mean - series.sem,
            series.mean + series.sem,
            color=color,
            alpha=0.18,
        )

    ax.set_xlim(left=0, right=args.total_timesteps)
    ax.set_xlabel("Timestep", fontsize=16)
    ax.set_ylabel("Reward (% of max)", fontsize=16)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.95)

    if args.title:
        ax.set_title(args.title, fontsize=16)
    else:
        ax.set_title(
            f"Craftax Figure 3 Reproduction ({args.env_name}, {args.total_timesteps // int(1e6)}M)",
            fontsize=16,
        )

    return fig, ax


def print_run_summary(runs: list[RunSeries], aggregated: dict[str, AggregatedSeries]) -> None:
    print("Selected runs:")
    for run in sorted(runs, key=lambda item: (ALGORITHM_ORDER.index(item.algorithm), item.run_name)):
        print(
            f"  {run.algorithm:7s} | {run.run_id} | seed={run.seed} | "
            f"state={run.state} | steps/update={run.num_envs * run.num_steps * run.num_repeats} | "
            f"{run.run_name}"
        )

    print("\nAggregated series:")
    for algorithm in ALGORITHM_ORDER:
        series = aggregated.get(algorithm)
        if series is None:
            continue
        print(
            f"  {algorithm:7s} | runs={series.num_runs} | "
            f"last_timestep={int(series.timesteps[-1])} | run_ids={','.join(series.run_ids)}"
        )


def main() -> None:
    args = parse_args()
    runs = load_matching_runs(args)
    aggregated = aggregate_runs(runs)
    print_run_summary(runs, aggregated)

    fig, _ = plot_series(aggregated, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"\nSaved figure to {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
