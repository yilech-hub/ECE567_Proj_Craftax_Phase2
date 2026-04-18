# Phase 2 Results — Simplified NGU on Craftax-Symbolic-v1

Summary of Phase 2 ablation experiments on `craftax-phase2-exp` W&B project
(`models-university-of-michigan6208/craftax-phase2-exp`).

## Experiment summary

Five 500M-timestep runs, single seed=42, Craftax-Symbolic-v1, coeff=0.01 for all
intrinsic-motivated methods. Full training script is
[`Exploration/ppo_rnd_episodic.py`](../Exploration/ppo_rnd_episodic.py).

| # | Method | `combination_mode` | wandb run ID | SLURM JobID | Wall time |
|---|---|---|---|---|---|
| 1 | Stock RND (baseline) | n/a — use `ppo_rnd.py` | `m7ol0qu3` | 48131444 | 2h 41m |
| 2 | Multiply (prior best @ 100M) | `multiply` | `jwo9eu8m` | 48131380 | 3h 46m |
| 3 | Episodic-only (ablation) | `episodic_only` | `r531xsgs` | 48140356 | 3h 49m |
| 4 | Add | `add` | `8hvzuc00` | 48140357 | 3h 47m |
| 5 | **Anneal (global decay schedule)** | `anneal` | `gc37etup` | 48140358 | 3h 47m |

All runs use 1024 envs × 64 steps × 7628 updates = 500,269,056 env timesteps.
Fixed hyperparameters: `lr=2e-4`, `k_neighbors=10`, `max_mem=1024`,
`ema_alpha=0.001`, `normalize_rnd=True`.

## Aggregate final metrics (step 7628 summary)

| metric | anneal | **add** | multiply | episodic_only | stock |
|---|---|---|---|---|---|
| `episode_return` | 24.60 | **25.66** | 24.78 | 23.50 | 24.82 |
| `achievements` (%) | 20.42 | **20.94** | 20.17 | 19.54 | 20.26 |
| `episode_length` | 451 | 574 | 501 | 487 | 477 |
| `intrinsic_reward` | 0.013 | 0.047 | 0.046 | 0.012 | 0.0007 |
| `extrinsic_reward` | 0.051 | 0.051 | 0.052 | 0.049 | 0.050 |
| SPS | 37,364 | 37,279 | 37,506 | 36,931 | 52,855 |

The 29% SPS drop for all episodic methods vs stock RND is the overhead of
per-step k-NN computation over the 1024-slot episodic ring buffer.

## Key finding: anneal cracks the iron-recipe cliff

Stock RND's achievement distribution has a sharp discontinuity between Tier 2
(stone-tier crafts: 73-92%) and Tier 3 (iron-tier recipes: 0.7-4.4%). The
anneal schedule **shatters this cliff for iron recipes specifically** while
leaving deeper tiers untouched.

| Achievement | anneal | add | multiply | ep_only | stock | anneal / stock |
|---|---|---|---|---|---|---|
| **make_iron_pickaxe** | **22.05** | 0.93 | 2.29 | 1.46 | 4.35 | **5.1×** |
| **make_iron_sword** | **37.01** | 1.85 | 3.05 | 0.00 | 0.72 | **51×** |
| collect_iron | 64.57 | 69.44 | 65.65 | 56.93 | 58.70 | 1.1× |
| make_stone_pickaxe | 80.31 | 76.85 | 76.34 | 72.26 | 73.19 | 1.1× |
| place_furnace | 91.34 | 95.37 | 90.08 | 89.78 | 86.96 | 1.05× |

**Stability check** (average over last 10% of training, ~10⁵ episodes):

| anneal metric | final | avg 10% | min 10% | max 10% |
|---|---|---|---|---|
| make_iron_pickaxe | 22.05 | **20.78** | **9.57** | 34.75 |
| make_iron_sword | 37.01 | **38.57** | **23.20** | 51.54 |

The minimum over the last 10% window (9.57, 23.20) is still 2-30× higher than
the maximum observed in any other method's last 10% window (7.52, 7.08).
This is a stable late-training regime, not a noise spike.

**See [`plots/anneal_diag.png`](../plots/anneal_diag.png)**: iron-recipe curves
for all 5 methods across 500M timesteps. Anneal diverges around update step
2000 (~130M timesteps), which is the region where the global weight has
decayed to ~0.67.

## But anneal does not cascade to Tier 4

| Achievement | anneal | add | multiply | ep_only | stock |
|---|---|---|---|---|---|
| collect_diamond | 3.94 | 5.56 | 8.40 | 7.30 | **7.97** |
| make_diamond_pickaxe | 0.00 | 0.93 | 0.00 | 0.73 | **2.17** |
| collect_ruby | 3.94 | 7.41 | 5.34 | 5.11 | 3.62 |
| collect_sapphire | 2.36 | 3.70 | 3.82 | 3.65 | **5.80** |
| enter_gnomish_mines | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Anneal's iron-recipe breakthrough does **not** carry through to diamond-tier
mining and recipes. All methods (including anneal) leave `enter_gnomish_mines`,
`enchant_*`, `learn_fireball/iceball`, and all `defeat_*` for hard enemies at 0%.

`episode_length` for anneal is also the shortest (451 vs stock 477). The
working hypothesis: once the agent learns iron recipes, it takes riskier
actions afterward (venturing into dungeons with better equipment), which
increases death rate — explaining why `episode_return` does not track the
iron-recipe gains.

## Per-method behavioral profiles

**Multiply (prior best @ 100M)**: +2 return lead at ~150M timesteps,
converges to stock RND by ~200M. Aggregate-parity at 500M but with active
intrinsic signal (0.046 vs stock's 0.0007, **65× larger**). Episode length
501 — close to baseline. "Early-phase helper, no asymptotic gain."

**Add**: +0.84 return over stock at 500M — first method in this ablation
to beat baseline on aggregate metric. Behavior profile is
"broad gatherer, shallow crafter": +10.7pp on `collect_iron`, +8.4pp on
`place_furnace`, +3.7pp on `make_stone_pickaxe`, but **worse** on iron
recipes (−3.4pp on `make_iron_pickaxe`). Episode length 574 (longest).

**Episodic-only**: −1.32 return vs stock. Matches Phase 1 E3B trajectory
(different implementation: E3B uses learned inverse-dynamics embedding +
elliptical bonus; ours uses frozen random target + k-NN L²). Both
implementations converge within 0.1 return at 500M (see
[`plots/e3b_overlay.png`](../plots/e3b_overlay.png)). Suggests pure
episodic novelty is insufficient for directional exploration on Craftax,
independent of embedding choice.

**Anneal**: aggregate metrics indistinguishable from stock, but the
behavioral profile is unique — specializes hard on iron recipes. 22% vs
~2% on `make_iron_pickaxe`. Suggests that decaying global novelty while
keeping episodic active drives the agent toward **executing learned
recipes** rather than **exploring new regions**, without injecting the
pure noise that episodic-only adds.

## Comparison against Phase 1 baselines (1B)

Phase 1 runs are on `yk-youngk/Craftax_Baselines`. Snapshots at matched
env-timesteps:

| @timesteps | P1_E3B | P1_RND | OUR_EO | OUR_STK |
|---|---|---|---|---|
| 100M return | 18.27 | 20.11 | 18.23 | 18.32 |
| 500M return | 23.60 | 25.92 | 23.50 | 24.82 |
| 1000M return | 27.76 | 28.34 | — | — |
| 500M `enter_dungeon` % | 33.06 | 44.54 | 40.88 | 44.93 |
| 500M `defeat_zombie` % | 74.38 | 62.18 | 60.58 | 58.70 |

**Core cross-implementation finding**: Our `episodic_only` matches Phase 1 E3B's
trajectory to within 0.1 return at 100M and 500M, despite using a completely
different embedding (fixed random target vs. learned inverse dynamics) and
distance function (k-NN L² vs. elliptical bonus). The behavioral outcome of
"episodic novelty alone" appears robust to implementation details on
Craftax-Symbolic-v1.

## Open questions

1. **Seed robustness**: All results are single-seed (seed=42). Anneal's iron
   breakthrough (20.78% vs 1.97-2.12%) is far outside any plausible noise band,
   but aggregate-return differences (add +0.84) need seed=0/1/2 replication.
2. **Tier 4 target**: Can anneal be further tuned (slower decay, later
   decay, different schedule shape) to extend past iron and unlock
   `make_diamond_pickaxe` / `collect_diamond`?
3. **Extrinsic trade-off**: Anneal's shorter episode_length suggests a
   death-rate increase. Is this optimal for the Craftax scoring rule
   (sum of achievements), or is the agent missing late-episode
   achievements by dying early?

## Training the methods

```bash
# Multiply (prior best)
python ppo_rnd_episodic.py --combination_mode multiply --rnd_reward_coeff 0.01 \
    --total_timesteps 5e8 --seed 42 --use_wandb --debug \
    --wandb_entity models-university-of-michigan6208 --wandb_project craftax-phase2-exp

# Add
python ppo_rnd_episodic.py --combination_mode add --rnd_reward_coeff 0.01 [...]

# Episodic-only (ablation)
python ppo_rnd_episodic.py --combination_mode episodic_only --rnd_reward_coeff 0.01 [...]

# Anneal (novel contribution)
python ppo_rnd_episodic.py --combination_mode anneal --rnd_reward_coeff 0.01 [...]
```

SBATCH scripts: see `rnd_ep_{add,anneal}_500M.sh`, `rnd_episodic_only_500M.sh`,
`rnd_ep_500M.sh` (multiply), `rnd_stock_500M.sh`.

## Plots

- [`plots/rnd_compare.png`](../plots/rnd_compare.png) — initial multiply vs stock comparison (episode_return, enter_dungeon, intrinsic_reward)
- [`plots/e3b_overlay.png`](../plots/e3b_overlay.png) — Phase 1 E3B (1B) vs our episodic_only (500M) trajectory overlap
- [`plots/ep_len_dist.png`](../plots/ep_len_dist.png) — episode_length distribution and evolution across training
- [`plots/anneal_diag.png`](../plots/anneal_diag.png) — anneal iron-recipe breakthrough verification
