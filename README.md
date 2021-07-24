# climate-super-resolution

This repo contains training and evaluation scripts of Neural Network for Climate Modelling Super Resolution

## Local Env Installation

1. Run:
```shell
conda create -n sr python=3.8
conda activate sr
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda env update -f environment.yml
```
2. Done

## Data pre-processing
1. Download World Clim and CRU-TS datasets
2. Ensure folder structure as in arguments in `preprocessing.py`
3. Run pre-processing script `preprocessing.py`

## How to train
Run:
```shell
python train.py <args>
```
See `train.py`, `datamodules.py` and `pl_sr_module.py` for full list of arguments.

## How to evaluate on the hidden test set
Testing should happen automatically when the test dataset was defined. See `test.py` for more details.

## Inference
To run inference on **CRU-TS**, please see `inference.py`. The results can be visualized using provided notebooks.

Less interactive version of the notebook can be found in the `inspect_results.py`.
