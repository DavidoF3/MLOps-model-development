# Effective MLOps: Model Development using W&B

This repository details work done to:
* Building a segmentation model prototype end-to-end ([link](/1_build_end2end_prototype/)).
* Optimising the performance of the baseline model ([link](/2_beyond_baseline_prototype/)).
* 

W&B tools used: 
* Tables
* Artifacts
* Experiments
* Reports

Dataset used:

* BDD1K dataset to be used for semantic segmentation.

# Repository setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
wandb login
```