# Toy-ABR-Gym
A controlled version of ABR for repeatable, controlled experiments

## Installation
To run and install this code, we assume that you are in this directory.

This code uses [Pytorch](https://pytorch.org/get-started/locally/) as a backend and assumes a Python version >= 3.10. If you would like to install a specific Pytorch a specific way (for example though conda), please do so now.

We recommend that you create a new conda environment regardless, to ensure version conflicts do not happen. If you had already done so for `puffer_abr`, can you simply activate the `abr` environment and continue.

Do so with 
```bash
conda create -n abr python=3.10
conda activate abr
```


```bash
pip install -e .
```

Unfortunately, the code does not support the new Rllib formatting yet. So, after installing the environment, you must install the earlier version and ignore the version conflicts:
```bash
pip install 'ray[rllib]==1.13'
```


## Training
To train, you can change to the `toy_abr_training` directory run `train.py`. Running this file requires that we start a ray cluster. You can do so with: `ray start --head`.

To train with random sampling on `majority_fast` for example, you may run `python train.py --sampling_func random --idx 0 --dataset majority_fast`. We evaluate and aggregate four indices in our experiments. 

This file will produce logs, which you can then process using `plot.py` using `plot_args.json`.