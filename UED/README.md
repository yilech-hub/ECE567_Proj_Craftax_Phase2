# Craftax Baselines: UED

This folder contains the code for running the UED methods from the [Craftax paper](https://arxiv.org/abs/2402.16801), discussed in Section 4.5.

This folder is modified based on [NCC-UED](https://github.com/nmonette/NCC-UED).


## Installation
We run the experiments using python 3.10, and cuda 12.8.
```commandline
pip install -r requirements.txt
```

## Run Experiments

### DR
```commandline
python train/craftax_plr.py --exploratory_grad_updates --replay_prob 0.0
```

### PLR
```commandline
python train/craftax_plr.py --run_name plr2 --exploratory_grad_updates --replay_prob 0.5 --lr 0.0003
```

### RPLR
```commandline
python train/craftax_plr.py --exploratory_grad_updates --replay_prob 0.8 --lr 0.0003
```

### ACCEL Swap
```commandline
python train/craftax_plr.py --exploratory_grad_updates --replay_prob 0.5 --use_accel --num_edits 10 --accel_mutation swap --lr 0.0003
```

### ACCEL RSwap
```commandline
python train/craftax_plr.py --exploratory_grad_updates --replay_prob 0.5 --use_accel --num_edits 200 --accel_mutation swap_restricted --lr 0.0003
```

### ACCEL Noise
```commandline
python train/craftax_plr.py --exploratory_grad_updates --replay_prob 0.5 --use_accel --num_edits 100 --accel_mutation noise --lr 0.0003
```
