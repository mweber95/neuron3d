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
ln -s ../imagetranslation-tensorflow/ imagetranslation (Linux command)
```

- Create directory

```bash
mkdir datasets
```

- Place the following .tif files in the *datasets* folder
- Download the SNEMI3D dataset from http://brainiac2.mit.edu/SNEMI3D/home
- Download evaluation tif files from https://doi.org/10.5281/zenodo.3371325

# Background information of the workflow

![workflow](documentation/full_workflow.png)

The major goal of this thesis was to combine machine learning and deterministic algorithms
for a dense reconstruction of three dimensional objects such as neurons from a
series of EM images. In particular, my workflow combined established 2D CNNs suited
for image prediction with the classic connected components algorithm. Cytoplasm images
represent cross-sections of putative three-dimensional objects in the image stack,
whereas overlap images describe the individual connectivity of these cross-sections.
Therefore an intermediate representation consisting of cross-sections and corresponding
features for the final object prediction was used. There are several paths to go from
EM images to mentioned features, which are marked as NN-1 to NN-5. 

## Python scripts and associated functionalities

### *process3D.py*
- creating membrane images from SNEMI3D dataset (train-labels.tif)
- creating cytoplasm images from SNEMI3D dataset (train-labels.tif)
- creating overlap images from SNEMI3D dataset (train-labels.tif)
- postprocessing of overlap images
- converting png to tif
- converting tif to png
### *reconstruct3D.py*
- relabeling of final predicted cytoplasm and overlap images and preparation for evaluation and 3D visualization 
### *evaluate3D.py*
- Calculating adjusted RAND index, precision, recall and split and merge errors
### *neuroglancer3D.py*
- 3D visualization of relabeled predicted neurons via neuroglancer

## Automatic workflow

A full workflow consisting of training, prediction and evaluation of NN-3, NN-4 and NN-5 is already attached to this repostiory.
- Executing bash script
```bash
bash experiment_8.sh
```
- This script takes around 4-5 hours with a Nvidia Tesla V100 and includes the training to predict membranes from EM images, cytoplasm from predicted membrane images and overlap from predicted membrane images.
