# MLOps-model-development
This repository details a project where an prototype end-to-end semantic segmentation model prototype is developed.

Topics covered:
* Exploratory data analysis (EDA) with Tables and Reports in W&B.
* Versioning datasets and models with Artifacts and Model Registry in W&B.
* Tracking and analyzing experiments

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