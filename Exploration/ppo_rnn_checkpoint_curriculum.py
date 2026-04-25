import argparse
import os
import sys

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time

from flax.training import orbax_utils
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

import wandb
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
from flax.training.train_state import TrainState
import distrax
import functools

from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from logz.batch_logging import create_log_dict, batch_log

from craftax.craftax_env import make_craftax_env_from_name

try:
    from eval_utils import evaluate_rnn_policy, mean_metric_dicts
except ImportError:
    # The upstream Craftax_Baselines repository does not currently include
    # eval_utils.py. Keep a simple clean-eval fallback so this file can still run
    # when copied into a fresh checkout. If your local eval_utils.py exists, it
    # will be used instead.
    def mean_metric_dicts(metric_dicts):
        if len(metric_dicts) == 0:
            return {}
        keys = metric_dicts[0].keys()
        return {
            k: float(np.mean([metrics[k] for metrics in metric_dicts]))
            for k in keys
        }

    def evaluate_rnn_policy(
        train_state,
        env_name,
        eval_seeds,
        network_factory,
        init_carry_fn,
        hidden_size,
        greedy=False,
    ):
        del network_factory  # train_state.apply_fn already points to the network.
        eval_env = make_craftax_env_from_name(env_name, True)
        eval_params = eval_env.default_params

        metric_dicts = []
        for seed in eval_seeds:
            rng = jax.random.PRNGKey(seed)
            rng, reset_rng = jax.random.split(rng)
            obs, state = eval_env.reset(reset_rng, eval_params)
            hstate = init_carry_fn(1, hidden_size)
            last_done = jnp.zeros((1,), dtype=bool)
            episode_return = 0.0
            episode_length = 0

            for _ in range(int(eval_params.max_timesteps)):
                ac_in = (obs[None, None, ...], last_done[None, :])
                hstate, pi, _ = train_state.apply_fn(
                    train_state.params, hstate, ac_in
                )
                rng, action_rng, step_rng = jax.random.split(rng, 3)
                action = pi.mode() if greedy else pi.sample(seed=action_rng)
                action = jnp.asarray(action).squeeze()
                obs, state, reward, done, info = eval_env.step(
                    step_rng, state, action, eval_params
                )
                episode_return += float(reward)
                episode_length += 1
                last_done = jnp.asarray([done], dtype=bool)
                if bool(done):
                    break

            metric_dicts.append(
                {
                    "eval/episode_return": episode_return,
                    "eval/episode_length": episode_length,
                }
            )

        return mean_metric_dicts(metric_dicts)

# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def _tree_map(fn, *trees):
    """Compatibility wrapper for JAX pytree mapping."""
    return jax.tree.map(fn, *trees)


def _batch_where(mask, new, old):
    """Select per environment from arrays whose first axis is NUM_ENVS."""
    mask = mask.astype(bool)
    while mask.ndim < new.ndim:
        mask = mask[..., None]
    return jnp.where(mask, new, old)


def _tree_batch_where(mask, new_tree, old_tree):
    return _tree_map(lambda new, old: _batch_where(mask, new, old), new_tree, old_tree)


def _tree_take_first_axis(tree, idx):
    """Take one or many entries from the leading axis of each pytree leaf."""
    return _tree_map(lambda x: x[idx], tree)


def _zero_log_counters(log_state):
    """Start a synthetic checkpoint episode with clean LogWrapper counters.

    The raw Craftax env_state is intentionally preserved, including achievements and
    inventory. Only the episode accounting fields maintained by LogWrapper are reset.
    """
    return log_state.replace(
        episode_returns=jnp.zeros_like(log_state.episode_returns),
        episode_lengths=jnp.zeros_like(log_state.episode_lengths),
        returned_episode_returns=jnp.zeros_like(log_state.returned_episode_returns),
        returned_episode_lengths=jnp.zeros_like(log_state.returned_episode_lengths),
        timestep=jnp.zeros_like(log_state.timestep),
    )


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Create environment
    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params

    # Wrap with some extra logging
    env = LogWrapper(env)

    # Wrap with a batcher, maybe using optimistic resets
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env_params).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )

        # INIT CHECKPOINT CURRICULUM BANK
        # The bank stores full LogWrapper states and their matching observations. It
        # is initialized from the first environment state but marked invalid until
        # real curriculum states are inserted.
        checkpoint_bank_size = config["CHECKPOINT_BANK_SIZE"]
        checkpoint_state = _tree_map(
            lambda x: jnp.repeat(x[:1], checkpoint_bank_size, axis=0), env_state
        )
        checkpoint_obs = jnp.repeat(obsv[:1], checkpoint_bank_size, axis=0)
        checkpoint_valid = jnp.zeros((checkpoint_bank_size,), dtype=bool)
        checkpoint_ptr = jnp.array(0, dtype=jnp.int32)

        checkpoint_enabled = config["CHECKPOINT_CURRICULUM"]

        def _maybe_insert_checkpoint(
            bank_state, bank_obs, bank_valid, bank_ptr, current_state, current_obs, update_step, rng
        ):
            """Insert one safe/progress state into a ring buffer, if available."""
            if not checkpoint_enabled:
                return bank_state, bank_obs, bank_valid, bank_ptr

            raw_state = current_state.env_state
            rng_save, rng_pick = jax.random.split(rng)

            safe = (
                (raw_state.player_health >= config["CHECKPOINT_MIN_HEALTH"])
                & (raw_state.player_food >= config["CHECKPOINT_MIN_FOOD"])
                & (raw_state.player_drink >= config["CHECKPOINT_MIN_DRINK"])
                & (
                    raw_state.timestep
                    < (env_params.max_timesteps - config["CHECKPOINT_TIMEOUT_MARGIN"])
                )
            )
            progress_ok = raw_state.player_level >= config["CHECKPOINT_MIN_LEVEL"]
            warm_enough = update_step >= config["CHECKPOINT_WARMUP_UPDATES"]
            save_roll = (
                jax.random.uniform(rng_save, (config["NUM_ENVS"],))
                < config["CHECKPOINT_SAVE_PROB"]
            )
            eligible = safe & progress_ok & warm_enough & save_roll

            has_candidate = eligible.any()
            logits = jnp.where(eligible, 0.0, -1e9)
            candidate_idx = jax.random.categorical(rng_pick, logits)
            candidate_state = _tree_take_first_axis(current_state, candidate_idx)
            candidate_obs = current_obs[candidate_idx]

            updated_bank_state = _tree_map(
                lambda bank_leaf, candidate_leaf: bank_leaf.at[bank_ptr].set(candidate_leaf),
                bank_state,
                candidate_state,
            )
            updated_bank_obs = bank_obs.at[bank_ptr].set(candidate_obs)
            updated_bank_valid = bank_valid.at[bank_ptr].set(True)

            bank_state = _tree_map(
                lambda old, new: jnp.where(has_candidate, new, old),
                bank_state,
                updated_bank_state,
            )
            bank_obs = jnp.where(has_candidate, updated_bank_obs, bank_obs)
            bank_valid = jnp.where(has_candidate, updated_bank_valid, bank_valid)
            bank_ptr = jnp.where(
                has_candidate,
                (bank_ptr + 1) % checkpoint_bank_size,
                bank_ptr,
            )
            return bank_state, bank_obs, bank_valid, bank_ptr

        def _maybe_resume_from_checkpoint(
            next_state, next_obs, done, bank_state, bank_obs, bank_valid, update_step, rng
        ):
            """For terminal envs, optionally replace the fresh reset with a checkpoint."""
            if not checkpoint_enabled:
                return next_state, next_obs

            rng_sample, rng_use = jax.random.split(rng)
            has_valid = bank_valid.any()
            logits = jnp.where(bank_valid, 0.0, -1e9)
            sampled_idx = jax.random.categorical(
                rng_sample, logits, shape=(config["NUM_ENVS"],)
            )
            resume_state = _tree_take_first_axis(bank_state, sampled_idx)
            resume_obs = bank_obs[sampled_idx]
            resume_state = _zero_log_counters(resume_state)

            warm_enough = update_step >= config["CHECKPOINT_WARMUP_UPDATES"]
            use_resume = (
                done
                & has_valid
                & warm_enough
                & (
                    jax.random.uniform(rng_use, (config["NUM_ENVS"],))
                    < config["CHECKPOINT_RESET_PROB"]
                )
            )
            next_state = _tree_batch_where(use_resume, resume_state, next_state)
            next_obs = _batch_where(use_resume, resume_obs, next_obs)
            return next_state, next_obs

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    rng,
                    update_step,
                    checkpoint_state,
                    checkpoint_obs,
                    checkpoint_valid,
                    checkpoint_ptr,
                ) = runner_state

                if checkpoint_enabled:
                    rng, _rng = jax.random.split(rng)
                    (
                        checkpoint_state,
                        checkpoint_obs,
                        checkpoint_valid,
                        checkpoint_ptr,
                    ) = _maybe_insert_checkpoint(
                        checkpoint_state,
                        checkpoint_obs,
                        checkpoint_valid,
                        checkpoint_ptr,
                        env_state,
                        last_obs,
                        update_step,
                        _rng,
                    )

                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                if checkpoint_enabled:
                    rng, _rng = jax.random.split(rng)
                    env_state, obsv = _maybe_resume_from_checkpoint(
                        env_state,
                        obsv,
                        done,
                        checkpoint_state,
                        checkpoint_obs,
                        checkpoint_valid,
                        update_step,
                        _rng,
                    )

                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done,
                    hstate,
                    rng,
                    update_step,
                    checkpoint_state,
                    checkpoint_obs,
                    checkpoint_valid,
                    checkpoint_ptr,
                )
                return runner_state, transition

            # hstate is field 4. Do not use negative indexing because the checkpoint
            # bank is appended to the runner_state when curriculum is enabled.
            initial_hstate = runner_state[4]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step,
                checkpoint_state,
                checkpoint_obs,
                checkpoint_valid,
                checkpoint_ptr,
            ) = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    )
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            rng = update_state[-1]
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(callback, metric, update_step)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step + 1,
                checkpoint_state,
                checkpoint_obs,
                checkpoint_valid,
                checkpoint_ptr,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
            0,
            checkpoint_state,
            checkpoint_obs,
            checkpoint_valid,
            checkpoint_ptr,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        # Strip the checkpoint bank from the returned state so evaluation/checkpoint
        # saving remain compatible with the original PPO-RNN script. In particular,
        # out["runner_state"][0] is still the TrainState.
        eval_runner_state = runner_state[:7]
        return {"runner_state": eval_runner_state, "metric": metric}

    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    # Defaults for backward compatibility with older argument sets.
    config.setdefault("CHECKPOINT_CURRICULUM", False)
    config.setdefault("CHECKPOINT_BANK_SIZE", 64)
    config.setdefault("CHECKPOINT_RESET_PROB", 0.25)
    config.setdefault("CHECKPOINT_SAVE_PROB", 0.05)
    config.setdefault("CHECKPOINT_MIN_LEVEL", 1)
    config.setdefault("CHECKPOINT_MIN_HEALTH", 3.0)
    config.setdefault("CHECKPOINT_MIN_FOOD", 1)
    config.setdefault("CHECKPOINT_MIN_DRINK", 1)
    config.setdefault("CHECKPOINT_TIMEOUT_MARGIN", 500)
    config.setdefault("CHECKPOINT_WARMUP_UPDATES", 0)

    if config["CHECKPOINT_BANK_SIZE"] < 1:
        raise ValueError("--checkpoint_bank_size must be >= 1")
    if not 0.0 <= config["CHECKPOINT_RESET_PROB"] <= 1.0:
        raise ValueError("--checkpoint_reset_prob must be in [0, 1]")
    if not 0.0 <= config["CHECKPOINT_SAVE_PROB"] <= 1.0:
        raise ValueError("--checkpoint_save_prob must be in [0, 1]")

    # Keep disabled runs close to the original memory footprint.
    if not config["CHECKPOINT_CURRICULUM"]:
        config["CHECKPOINT_BANK_SIZE"] = 1

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-PPO_RNN"
            + ("-CKPT" if config["CHECKPOINT_CURRICULUM"] else "")
            + "-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))

    ## Vanilla Evaluation ##
    eval_seeds = list(range(config["EVAL_EPISODES"]))

    network_factory = lambda action_dim, obs_shape: ActorCriticRNN(
        action_dim, config=config
    )

    eval_metrics_all = []
    for repeat_idx in range(config["NUM_REPEATS"]):
        rep_train_state = jax.tree.map(
            lambda x: x[repeat_idx], out["runner_state"][0]
        )
        rep_metrics = evaluate_rnn_policy(
            train_state=rep_train_state,
            env_name=config["ENV_NAME"],
            eval_seeds=eval_seeds,
            network_factory=network_factory,
            init_carry_fn=ScannedRNN.initialize_carry,
            hidden_size=config["LAYER_SIZE"],
            greedy=config["EVAL_GREEDY"],
        )
        eval_metrics_all.append(rep_metrics)

    eval_metrics = mean_metric_dicts(eval_metrics_all)

    print("\nVanilla evaluation:")
    for k, v in sorted(eval_metrics.items()):
        print(f"{k}: {v}")

    if config["USE_WANDB"]:
        wandb.log(eval_metrics)
    ## Vanilla Evaluation ##

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                config["TOTAL_TIMESTEPS"],
                train_state,
                save_kwargs={"save_args": save_args},
            )

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1_000_000_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=np.random.randint(2**31))
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_policy", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument(
        "--eval_greedy",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--checkpoint_curriculum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable checkpoint-state curriculum resets during training only.",
    )
    parser.add_argument(
        "--checkpoint_bank_size",
        type=int,
        default=64,
        help="Number of saved full environment states in the checkpoint ring buffer.",
    )
    parser.add_argument(
        "--checkpoint_reset_prob",
        type=float,
        default=0.25,
        help="Probability of replacing a terminal reset with a saved checkpoint state.",
    )
    parser.add_argument(
        "--checkpoint_save_prob",
        type=float,
        default=0.05,
        help="Per-env probability of admitting an eligible state as a checkpoint candidate.",
    )
    parser.add_argument(
        "--checkpoint_min_level",
        type=int,
        default=1,
        help="Minimum player_level required for a state to be saved. Use 0 for overworld smoke tests, 1 for dungeon-focused curriculum.",
    )
    parser.add_argument(
        "--checkpoint_min_health",
        type=float,
        default=3.0,
        help="Minimum player health required for checkpoint saving.",
    )
    parser.add_argument(
        "--checkpoint_min_food",
        type=int,
        default=1,
        help="Minimum food required for checkpoint saving.",
    )
    parser.add_argument(
        "--checkpoint_min_drink",
        type=int,
        default=1,
        help="Minimum drink required for checkpoint saving.",
    )
    parser.add_argument(
        "--checkpoint_timeout_margin",
        type=int,
        default=500,
        help="Do not save states this close to the max episode timestep.",
    )
    parser.add_argument(
        "--checkpoint_warmup_updates",
        type=int,
        default=0,
        help="Number of PPO updates before saving or resuming from checkpoints.",
    )

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)
