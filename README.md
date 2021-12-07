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
conda create --name pytorch_per python=3.8.1
conda activate pytorch_per
git clone https://github.com/lucaslingle/pytorch_per
cd pytorch_per
pip install -e .
```

## Usage

### Training

#### Non-Distributed Training
To train the default settings, you can simply type:
```bash
python -m script
```

#### Single-Machine Distributed Training
To run in distributed mode on N processes, you can type
```bash
mpiexec -np N python -m script
```
This will launch N parallel processes, each running the ```script.py``` script. These processes each generate and store experience tuples in separate replay memories, and then synchronously train on the collected experience in a data-parallel manner. 

To see additional configuration options, you can simply type ```python -m script --help```. 

### Checkpoints
By default, checkpoints are saved to ```./models_dir/default_hparams/```. 
When training different models, you should set unique run names using the 
```--run_name``` command line argument.

### Videos
Videos can be saved to the checkpoint directory for a given model by using the command line argument ```--mode=video```.

## Wall-Clock Time
We currently support distributed data parallel training using mpi4py directly on CPUs. On a laptop with a ```2.3 GHz 8-Core Intel Core i9``` CPU, we obtained a throughput of about 150,000 frames per hour, meaning that training for 50M timesteps should take about 333 hours, or 13.9 days.

We are currently evaluating the throughput on an NVIDIA V100 GPU. 
