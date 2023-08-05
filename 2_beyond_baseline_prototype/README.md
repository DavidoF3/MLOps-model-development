# Moving beyond baseline prototype

This sub-directory is about optimising the performance (more specifically the IoU) of a baseline segmentation model developed in [here](../1_build_end2end_prototype/).

Topics covered:
* Optimisation of hyperparameters.
* Analysis and reporting/sharing of optimisation results. 
* Retraining model with optimised hyperparameters. 

W&B tools used: 
* Tables
* Artifacts
* Experiments
* Sweeps
* Reports

## Notes

### Basic model training

Check inputs to interactive python program.
```
python train.py --help
```

To run train.py with modified inputs (other than default).
```
python train.py --batch_size 16
```

### Model training with sweeps

To start a sweep, run the following command. Where, sweep.yaml contains the sweep configuration.
```
wandb sweep sweep.yaml
```

Once a sweep has been started, individual sweep runs are conducted by agents. To check available options for W&B sweep agents:
```
wandb agent --help
```

If performing a sweep `random` search, we should define a max number of runs. Otherwise the sweep will run forever. For example:
```
wandb agent <this/will/be/specific/to/sweep> --count 50
```

Multiple sweep agents can be run in parallel if the machine being used has multiple GPUs. To check the available GPU's.
```
watch nvidia-smi
```

We can run a sweep agent per GPU. Each run is started in a different terminal.
```
CUDA_VISIBLE_DEVICES=<GPU_number> wandb agent <this/will/be/specific/to/sweep> --count 50
```

### Visualisation of sweep data

Example ideas:
* In the runs panel, delete all columns and only display/ pin columns of interest.
* Can filter runs by their status eg. finished (ignore incompleted or crashed ones).
* Can group runs by attribute eg. encoder architecture.
* In the workspace panel, can add additional axes (of different performance parameters - eg. IoU of a specific class) to the sweep config tracking plot.

