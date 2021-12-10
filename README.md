# pytorch_per
This repo implements the datastructures, algorithms, and architectures from Schaul et al., 2015 - ['Prioritized Experience Replay'](https://arxiv.org/pdf/1511.05952.pdf).

## Getting Started

Install the following system dependencies:
#### Ubuntu     
```bash
sudo apt-get update
sudo apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev zlib1g zlib1g-dev swig
sudo apt-get install -y cmake python3-dev
```

#### Mac OS X
Installation of the system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake
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

To train with the default settings, you can simply type:
```bash
python -m script
```
To see additional configuration options, append the ```--help``` command line argument.

### Checkpoints
By default, checkpoints are saved to ```./models_dir/default_hparams/```. 
When training different models, you should set unique run names using the 
```--run_name``` command line argument.

### Videos
Videos can be saved to the checkpoint directory for a given model by using the command line argument ```--mode=video```.

## Wall-Clock Time
We currently support distributed data parallel training on CPUs using mpi4py directly. On a laptop with a ```2.3 GHz 8-Core Intel Core i9``` CPU, we obtained a throughput of about 150,000 timesteps per hour, meaning that training for 50M timesteps (200M frames) should take about 333 hours, or 13.9 days.

Presently, the throughput on an NVIDIA V100 GPU is not significantly better. This will be the subject of additional investigation using torch's profiling tools. Feel free to check back later. 
