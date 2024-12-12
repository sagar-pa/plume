# Puffer-ABR

This is an Gynasium simulation environment that uses the logs produced by [Puffer](https://puffer.stanford.edu), a free and open-source live TV streaming website and a research study at Stanford University. This is the environment that implements Gelato.

## Installation
To run and install this code, we assume that you are in this directory.

This code uses [Pytorch](https://pytorch.org/get-started/locally/) as a backend and assumes a Python version >= 3.10. If you would like to install a specific Pytorch a specific way (for example though conda), please do so now.

We recommend that you create a new conda environment regardless, to ensure version conflicts do not happen.

Do so with 
```bash
conda create -n abr python=3.10
conda activate abr
```


```bash
pip install -e .
```

Unfortunately, the code does not support the new Rllib formatting yet. If you would like use the rllib code for APE-X training, after installing the environment, you must install the earlier version and ignore the version conflicts:
```bash
pip install 'ray[rllib]==1.13'
```

## Setting up traces
From this point on, we assume that you are in the `puffer_abr_training/` directory. You may change to it using `cd puffer_abr_training/`

In our experiments, we used the traces for the months of April and May 2021. To use the traces used in the paper, you can go to this [Google Drive folder](https://drive.google.com/drive/folders/1zLbD94Yd8BkLEnLHsSBLOFPEpoMD5DiN?usp=sharing), download the traces, and unzip it to a destination traces folder. 
```bash
unzip 4.zip -d ./traces
unzip 5.zip -d ./traces
```

Alternatively, you may download a new set of traces and extracting the features using the `setup_env.py` file. You can run `python setup_env.py --days [NUM OF DAYS OF TRACE TO DOWNLOAD] --start_date [YYYY/MM/DD] --trace_dir ./traces`. It will start at the start_date, and go **backwards** till number of days are reached. This logic follows the results plots shown: starting from 2020/05/01 and requesting 14 days will produce the data for the 14 day plot found on the [results page](https://puffer.stanford.edu/results/) for that day. However, note that this script is *slow and expensive* and should be avoided unless necessary.

If you use any other directory, please appropriately change the value in `global_constants.py`


## Training
To train, you can run `train.py`. Running this file requires that we start a ray cluster. You can do so with: `ray start --head`.

To train gelato with plume_static for example, you may run `python train.py --kind gelato --sampling_func random --idx 0`. We evaluate and aggregate four indices in our experiments. 

Running this file will give us a pickle file in the summary directory: an object of `StreamData` Class (plot.py). You can may use this plot the logs produces and read (To aggregate the data of multiple indices together, you can aggregate the lists of `TraceData` within each pickle file).

## Deploying to Puffer
Unfortunately, there is no automatic way to deploy a trained controller here to Puffer. We provide this public repository as a guideline: https://github.com/sagar-pa/abr_rl_test.
This process can be done with a little bit of effort by following these steps:
1. Re-implementing the controller in Pure Pytorch. 
    - You can load a saved policy using for example `A2C.load()`, and obtain the `state_dict()`
    - The `state_dict` is a dictionary with the name of the neural parameter as a key and the corresponding weight
    - Your task now is to recreate the model (without a value function) and match the names in the `state_dict` to the new class
    - Ensure that the model loads the `state_dict` correctly and the output for a tensor matches the controller
2. Re-implementing the controller features in Python
    - This can be done by implementing the `BaseEnv` class (please see `maguro_env.py` as an example)

