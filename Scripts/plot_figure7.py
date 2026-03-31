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

ALGORITHM_ORDER = ["DR", "PLR", "PLR⊥", "ACCEL (Swap)", "ACCEL (RSwap)", "ACCEL (Noise)"]
ALGORITHM_COLORS = {
    "DR": "#0173b2",
    "PLR": "#de8f05",
    "PLR⊥": "#029e73",
    "ACCEL (Swap)": "#d55e00",
    "ACCEL (RSwap)": "#cc78bc",
    "ACCEL (Noise)": "#ca9161",
}

EXPLICIT_NAME_TO_ALGORITHM = {
    "dr": "DR",
    "plr": "PLR",
    "rplr": "PLR⊥",
    "accelswap": "ACCEL (Swap)",
    "accelrswap": "ACCEL (RSwap)",
    "accelnoise": "ACCEL (Noise)",
}

RUN_NAME_TO_ALGORITHM = {
    "dr": "DR",
    "plr": "PLR",
    "plr2": "PLR",
    "rplr": "PLR⊥",
    "rplr2": "PLR⊥",
    "accelswap": "ACCEL (Swap)",
    "accelswap2": "ACCEL (Swap)",
    "accelswaprestricted": "ACCEL (RSwap)",
    "accelrswap": "ACCEL (RSwap)",
    "accelrswap2": "ACCEL (RSwap)",
    "accelnoise": "ACCEL (Noise)",
    "accelnoise2": "ACCEL (Noise)",
}


@dataclass
class RunSeries:
    algorithm: str
    run_id: str
    run_name: str
    state: str
    seed: int | None
    total_timesteps: int
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
        description="Reproduce the Craftax Figure 7 UED reward plot from Weights & Biases runs."
    )
    parser.add_argument("--entity", type=str, default=DEFAULT_ENTITY)
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--env-name", type=str, default=DEFAULT_ENV_NAME)
    parser.add_argument(
        "--total-timesteps",
        type=lambda x: int(float(x)),
        default=DEFAULT_TOTAL_TIMESTEPS,
    )
    parser.add_argument("--metric", type=str, default="return/mean")
    parser.add_argument("--max-reward", type=float, default=PAPER_MAX_REWARD)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path whose stem will be used for all requested formats.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="File formats to save, for example: png pdf.",
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
        default=["finished"],
        help="Run states to include.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=2.5,
        help="Lower y-limit for the plot.",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display the figure interactively after saving.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def infer_total_timesteps(run_name: str, config: dict) -> int | None:
    total_timesteps = config.get("num_env_steps")
    if total_timesteps is not None:
        return int(total_timesteps)

    match = re.search(r"(\d+)B\b", run_name)
    if match:
        return int(match.group(1)) * int(1e9)
    return None


def infer_algorithm(run_name: str, config: dict) -> str | None:
    normalized_name = normalize_name(run_name)
    if normalized_name in EXPLICIT_NAME_TO_ALGORITHM:
        return EXPLICIT_NAME_TO_ALGORITHM[normalized_name]

    normalized_run_name = normalize_name(str(config.get("run_name", "")))
    if normalized_run_name in RUN_NAME_TO_ALGORITHM:
        return RUN_NAME_TO_ALGORITHM[normalized_run_name]

    if config.get("use_accel"):
        mutation = config.get("accel_mutation")
        if mutation == "swap":
            return "ACCEL (Swap)"
        if mutation == "swap_restricted":
            return "ACCEL (RSwap)"
        if mutation == "noise":
            return "ACCEL (Noise)"

    replay_prob = config.get("replay_prob")
    if replay_prob in (0, 0.0):
        return "DR"
    if normalized_run_name.startswith("rplr"):
        return "PLR⊥"
    if normalized_run_name.startswith("plr"):
        return "PLR"
    return None


def is_explicit_named_run(run_name: str) -> bool:
    return normalize_name(run_name) in EXPLICIT_NAME_TO_ALGORITHM


def fetch_history(run: wandb.apis.public.Run, metric: str, max_reward: float) -> tuple[np.ndarray, np.ndarray]:
    config = run.config or {}
    batch_size = (
        int(config.get("num_train_envs", 1))
        * int(config.get("num_steps", 1))
        * int(config.get("outer_rollout_steps", 1))
    )
    last_seen: dict[int, float] = {}

    for row in run.scan_history(keys=["num_env_steps", "num_updates", metric]):
        value = row.get(metric)
        if value is None or not np.isfinite(value):
            continue

        timestep = row.get("num_env_steps")
        if timestep is None:
            update = row.get("num_updates")
            if update is None:
                continue
            timestep = int(update) * batch_size

        last_seen[int(timestep)] = float(value) * 100.0 / max_reward

    if not last_seen:
        return np.array([]), np.array([])

    timesteps = np.asarray(sorted(last_seen.keys()), dtype=np.int64)
    values = np.asarray([last_seen[int(timestep)] for timestep in timesteps], dtype=np.float64)
    return timesteps, values


def load_matching_runs(args: argparse.Namespace) -> tuple[list[RunSeries], list[str]]:
    api = wandb.Api()
    base_path = f"{args.entity}/{args.project}"
    available_summaries: list[str] = []
    duplicate_notes: list[str] = []
    chosen_runs: dict[tuple[str, int | str], tuple[int, wandb.apis.public.Run]] = {}

    for stub in api.runs(base_path):
        run = api.run(f"{base_path}/{stub.id}")
        config = run.config or {}
        total_timesteps = infer_total_timesteps(run.name or "", config)
        env_name = config.get("env_name") or config.get("ENV_NAME")
        algorithm = infer_algorithm(run.name or "", config)

        available_summaries.append(
            f"{run.id}: {run.name} [{run.state}] total={total_timesteps} env={env_name} alg={algorithm}"
        )

        if run.state not in args.states:
            continue
        if total_timesteps is not None and total_timesteps != args.total_timesteps:
            continue
        if env_name is not None and env_name != args.env_name:
            continue
        if algorithm is None or algorithm not in args.algorithms:
            continue

        seed = config.get("seed")
        if seed is None:
            seed = config.get("SEED")
        dedupe_key = (algorithm, seed if seed is not None else run.id)
        priority = 1 if is_explicit_named_run(run.name or "") else 0

        current = chosen_runs.get(dedupe_key)
        if current is None or priority > current[0]:
            if current is not None:
                duplicate_notes.append(
                    f"  {algorithm:13s} | replaced {current[1].name} ({current[1].id}) "
                    f"with {run.name} ({run.id}) for seed={dedupe_key[1]}"
                )
            chosen_runs[dedupe_key] = (priority, run)
        else:
            duplicate_notes.append(
                f"  {algorithm:13s} | skipped duplicate {run.name} ({run.id}) for seed={dedupe_key[1]}"
            )

    selected_runs: list[RunSeries] = []
    for _, (_, run) in sorted(chosen_runs.items(), key=lambda item: (ALGORITHM_ORDER.index(item[0][0]), str(item[0][1]))):
        config = run.config or {}
        total_timesteps = infer_total_timesteps(run.name or "", config)
        seed = config.get("seed")
        if seed is None:
            seed = config.get("SEED")

        timesteps, values = fetch_history(run, args.metric, args.max_reward)
        if timesteps.size == 0:
            continue

        selected_runs.append(
            RunSeries(
                algorithm=infer_algorithm(run.name or "", config) or "UNKNOWN",
                run_id=run.id,
                run_name=run.name or run.id,
                state=run.state,
                seed=seed,
                total_timesteps=int(total_timesteps or args.total_timesteps),
                timesteps=timesteps,
                values=values,
            )
        )

    if not selected_runs:
        available = "\n".join(sorted(available_summaries)) or "No runs found in W&B project."
        raise ValueError(
            "No matching runs found for the requested Figure 7 settings.\n"
            f"Requested env={args.env_name}, total_timesteps={args.total_timesteps}, states={args.states}.\n"
            f"Available runs:\n{available}"
        )

    return selected_runs, duplicate_notes


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


def resolve_ylim(aggregated: dict[str, AggregatedSeries], ymin: float) -> tuple[float, float]:
    ymax = ymin + 1.0
    for algorithm in ALGORITHM_ORDER:
        series = aggregated.get(algorithm)
        if series is None:
            continue
        ymax = max(ymax, float(np.nanmax(series.mean + series.sem)))
    return ymin, ymax * 1.05


def plot_series(
    aggregated: dict[str, AggregatedSeries], args: argparse.Namespace
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10.5, 7.0), constrained_layout=True)

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
    ax.set_ylim(*resolve_ylim(aggregated, args.ymin))
    ax.set_xlabel("# Env Steps", fontsize=16)
    ax.set_ylabel("Reward (% of max)", fontsize=16)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(9, 9))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=2, framealpha=0.95, fontsize=13)

    if args.title:
        ax.set_title(args.title, fontsize=16)

    return fig, ax


def print_run_summary(runs: list[RunSeries], aggregated: dict[str, AggregatedSeries], duplicate_notes: list[str]) -> None:
    print("Selected runs:")
    for run in sorted(runs, key=lambda item: (ALGORITHM_ORDER.index(item.algorithm), item.run_name)):
        print(
            f"  {run.algorithm:13s} | {run.run_id} | seed={run.seed} | "
            f"state={run.state} | points={len(run.timesteps)} | {run.run_name}"
        )

    print("\nAggregated series:")
    for algorithm in ALGORITHM_ORDER:
        series = aggregated.get(algorithm)
        if series is None:
            continue
        print(
            f"  {algorithm:13s} | runs={series.num_runs} | "
            f"last_timestep={int(series.timesteps[-1])} | run_ids={','.join(series.run_ids)}"
        )

    if duplicate_notes:
        print("\nDuplicate handling:")
        for note in duplicate_notes:
            print(note)


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
    runs, duplicate_notes = load_matching_runs(args)
    aggregated = aggregate_runs(runs)
    print_run_summary(runs, aggregated, duplicate_notes)

    fig, _ = plot_series(aggregated, args)
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
