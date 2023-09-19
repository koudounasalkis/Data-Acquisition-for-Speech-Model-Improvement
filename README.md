# Data-Acquisition-for-Speech-Model-Improvement
This repo contains the code for "Prioritizing Data Acquisition For End-to-End Speech Model Improvement"

In this repository, you will find the code to replicate our experiments by training and testing our models.

## Getting Started

### Data
We do not include the datasets used in the paper as they are publicly available and downloadable from the respective authors. To make it work, you should put data files under data.

### Python Environment
Our code was tested on Python 3.11.2. To make it work, you will need:
- a working environment with the libraries listed in requirements.txt;
- a functioning torch installation in the same environment.

## Running the Experiments
Use the `ft_main.py` to finetune the required models, `inference.py` to evaluate them, and `divexplorer_analysis.ipynb` to explore subgroup divergence.

We will extend the documentation in this README upon acceptance.
