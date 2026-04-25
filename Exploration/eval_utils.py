# eval_utils.py
from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name


def _get_achievement_enum(env_name: str):
    if "Classic" in env_name:
        from craftax.craftax_classic.constants import Achievement
    else:
        from craftax.craftax.constants import Achievement
    return Achievement


def mean_metric_dicts(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {k: float(np.mean([m[k] for m in metric_dicts])) for k in keys}


def evaluate_feedforward_policy(
    train_state,
    env_name: str,
    eval_seeds: Iterable[int],
    network_factory: Callable[[int, tuple], object],
    greedy: bool = False,
    prefix: str = "eval_vanilla",
) -> Dict[str, float]:
    # No auto-reset: we want to stop on terminal state and read terminal achievements.
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params

    action_dim = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape
    network = network_factory(action_dim, obs_shape)

    Achievement = _get_achievement_enum(env_name)

    returns: List[float] = []
    achievement_hits = np.zeros(len(Achievement), dtype=np.float64)

    for seed in eval_seeds:
        rng = jax.random.PRNGKey(int(seed))
        rng, reset_rng = jax.random.split(rng)

        obs, env_state = env.reset(reset_rng, env_params)
        done = False
        ep_return = 0.0

        while not done:
            rng, act_rng, step_rng = jax.random.split(rng, 3)

            out = network.apply(train_state.params, jnp.expand_dims(obs, axis=0))
            pi = out[0]

            action = pi.mode() if greedy else pi.sample(seed=act_rng)
            action = jnp.asarray(action).squeeze()

            obs, env_state, reward, done_arr, _ = env.step(
                step_rng, env_state, action, env_params
            )

            ep_return += float(reward)
            done = bool(np.asarray(done_arr))

        returns.append(ep_return)
        achievement_hits += np.asarray(env_state.achievements, dtype=np.float64)

    num_eps = float(len(list(eval_seeds)) if not isinstance(eval_seeds, list) else len(eval_seeds))
    # Safer if eval_seeds is a generator:
    if num_eps == 0:
        raise ValueError("eval_seeds must be non-empty")

    metrics: Dict[str, float] = {
        f"{prefix}/episode_return": float(np.mean(returns)),
        f"{prefix}/achievement_count": float(achievement_hits.sum() / num_eps),
    }

    for achievement in Achievement:
        metrics[f"{prefix}/Achievements/{achievement.name.lower()}"] = float(
            100.0 * achievement_hits[achievement.value] / num_eps
        )

    return metrics


def evaluate_rnn_policy(
    train_state,
    env_name: str,
    eval_seeds: Iterable[int],
    network_factory: Callable[[int, tuple], object],
    init_carry_fn: Callable[[int, int], object],
    hidden_size: int,
    greedy: bool = False,
    prefix: str = "eval_vanilla",
) -> Dict[str, float]:
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params

    action_dim = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape
    network = network_factory(action_dim, obs_shape)

    Achievement = _get_achievement_enum(env_name)

    returns: List[float] = []
    achievement_hits = np.zeros(len(Achievement), dtype=np.float64)

    for seed in eval_seeds:
        rng = jax.random.PRNGKey(int(seed))
        rng, reset_rng = jax.random.split(rng)

        obs, env_state = env.reset(reset_rng, env_params)

        hstate = init_carry_fn(1, hidden_size)
        last_done = jnp.zeros((1, 1), dtype=bool)

        done = False
        ep_return = 0.0

        while not done:
            rng, act_rng, step_rng = jax.random.split(rng, 3)

            ac_in = (obs[None, None, ...], last_done)
            hstate, pi, _ = network.apply(train_state.params, hstate, ac_in)

            action = pi.mode() if greedy else pi.sample(seed=act_rng)
            action = jnp.asarray(action).squeeze()

            obs, env_state, reward, done_arr, _ = env.step(
                step_rng, env_state, action, env_params
            )

            done = bool(np.asarray(done_arr))
            last_done = jnp.array([[done]], dtype=bool)
            ep_return += float(reward)

        returns.append(ep_return)
        achievement_hits += np.asarray(env_state.achievements, dtype=np.float64)

    num_eps = float(len(list(eval_seeds)) if not isinstance(eval_seeds, list) else len(eval_seeds))
    if num_eps == 0:
        raise ValueError("eval_seeds must be non-empty")

    metrics: Dict[str, float] = {
        f"{prefix}/episode_return": float(np.mean(returns)),
        f"{prefix}/achievement_count": float(achievement_hits.sum() / num_eps),
    }

    for achievement in Achievement:
        metrics[f"{prefix}/Achievements/{achievement.name.lower()}"] = float(
            100.0 * achievement_hits[achievement.value] / num_eps
        )

    return metrics