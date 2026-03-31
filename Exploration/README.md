# Craftax Baselines

This folder contains the code for running the baselines from the [Craftax paper](https://arxiv.org/abs/2402.16801).

This folder is modified based on [Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines).

## Installation
We run the experiments using python 3.10, and cuda 12.8.
```commandline
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pre-commit install
```

# Run Experiments

### PPO
```commandline
python ppo.py
```

### PPO-RNN
```commandline
python ppo_rnn.py
```

### ICM
```commandline
python ppo.py --train_icm
```

### E3B
```commandline
python ppo.py --train_icm --use_e3b --icm_reward_coeff 0 --e3b_reward_coeff 0.001
```

### RND
```commandline
python ppo_rnd.py
```

# Visualisation
You can save trained policies with the `--save_policy` flag.  These can then be viewed with the `view_ppo_agent` script (pass in the path up to the `files` directory).