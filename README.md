# pytorch_per
This repo implements the datastructures, algorithms, and architectures from Schaul et al., 2015 - 'Prioritized Experience Replay'.

## Getting Started

Install the following system dependencies:
#### Ubuntu     
```bash
sudo apt-get update
sudo apt-get install -y cmake openmpi-bin openmpi-doc libopenmpi-dev
```

#### Mac OS X
Installation of the system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

#### Everyone
Once the system dependencies have been installed, it's time to install the python dependencies. 
Install the conda package manager from https://docs.conda.io/en/latest/miniconda.html

Then run
```bash
conda create --name pytorch_rl2 python=3.8.1
conda activate pytorch_rl2
git clone https://github.com/lucaslingle/pytorch_rl2
cd pytorch_rl2
pip install -e .
```

## Usage

### Training
To train the default settings, you can simply type:
```bash
mpiexec -np 8 python -m run
```

This will launch 8 parallel processes, each running the ```run.py``` script. These processes each generate and store experience tuples in separate replay memories, and then synchronously train on the collected experience in a data-parallel manner, with gradient information and model parameters synchronized across processes using mpi4py.

To see additional configuration options, you can simply type ```python train.py --help```. Among other options, we support various architectures including GRU, LSTM, SNAIL, and Transformer models.

### Checkpoints
By default, checkpoints are saved to ```./models_dir/default_hparams/```. 
When training different models, you should set unique run names using the 
```--run_name``` command line argument.

## Wall-Clock Time
We currently support distributed data parallel training using mpi4py directly on CPUs. On a laptop with a ```2.3 GHz 8-Core Intel Core i9``` CPU, we obtained a throughput of about 150,000 frames per hour, meaning that training for 50M timesteps should take about 333 hours, or 13.9 days.

We are currently evaluating the throughput on an NVIDIA V100 GPU. 