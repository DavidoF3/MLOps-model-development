# Effective MLOps: Model Development using W&B

This repository details work done on:
* Building a segmentation model prototype end-to-end ([link](/1_build_end2end_prototype/README.md)).
* Optimising the performance of the baseline model ([link](/2_beyond_baseline_prototype/README.md)).
* Best practices for model evaluation ([link](/3_model_eval/README.md)).

W&B tools used: 
* Tables
* Artifacts
* Experiments
* Reports
* Model registry

Dataset used:

* BDD1K dataset to be used for semantic segmentation.

Further W&B resources:

* https://docs.wandb.ai/tutorials

# Repository setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
wandb login
```