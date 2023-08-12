# Moving beyond baseline prototype

In this sub-directory, the baseline model developed [here](../1_build_end2end_prototype/README.md) is further optimised to achieve higher IOU's on a specific class.

Topics covered:
* Optimisation of hyperparameters using W&B Sweeps (see [sweep config](sweep.yaml)).
* Analysis and reporting/sharing of optimisation results using W&B Reports (see [report](https://wandb.ai/doc93/mlops-course-001/reports/Sweep-analysis--Vmlldzo1MDY0MTA0)). 
* Retraining model with optimised hyperparameters (see [train script](train.py)).. 

W&B tools used: 
* Tables
* Artifacts
* Experiments
* Sweeps
* Reports

# Commands for running optimisation

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

# Visualisation of sweep data

Example ideas:
* In the runs panel, delete all columns and only display/ pin columns of interest.
* Can filter runs by their status eg. finished (ignore incompleted or crashed ones).
* Can group runs by attribute eg. encoder architecture.
* If dealing with a multi class problem (eg. semantic segmentation), in the workspace panel- go to the sweep config tracking plot and add additional axes of different performance parameters - eg. IoU of a specific class.

