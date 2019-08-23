# Deep learning for dense reconstruction of neurons from electron microscopic images

*Note:* This repository is a subproject of https://github.com/tbullmann/heuhaufen.

### Prerequisites
- Windows, Linux or OSX (following getting started guide for Linux users)
- Python3
- CPU or NVIDIA GPU + CUDA CuDNN

### Requirements
- Argparse 1.4.0 
- Tifffile 2019.1.30
- Pillow 5.4.1
- Scipy 1.2.0
- Imageio 2.4.1
- Skimage 0.14.2
- Neuroglancer 1.0.11 (https://github.com/google/neuroglancer)

- Tornado 4.5.3 (*Note:* Make sure you installed the specified version for Tornado)
- Tensorflow 1.12.0 

##Getting started

-Create a new environment and install all requirements
(*Note:* for more in depth instructions see https://github.com/tbullmann/heuhaufen/blob/master/CONDA.md)

-Clone this repository

```bash
git clone https://github.com/mweber95/neuron3d.git
```

- Clone other repositories used for computation and visualization

```bash
git clone https://github.com/tbullmann/imagetranslation-tensorflow.git
```

- Symlink repositories

```bash
cd neuron3d
ln -s ../imagetranslation-tensorflow/ imagetranslation (command for Linux)
```

- Create directory

```bash
mkdir datasets
```

- Place the following .tif files in the *datasets* folder
- Download the SNEMI3D dataset from http://brainiac2.mit.edu/SNEMI3D/home
- Download evaluation tif files from https://doi.org/10.5281/zenodo.3371325
