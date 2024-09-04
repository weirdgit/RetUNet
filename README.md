# Ret-UNet: Enhancing Medical Image Segmentation with Retentive Self-Attention


---

## Installation

#### 1. System requirements

We run nnFormer on a system running Ubuntu 20.04, with Python 3.9, PyTorch 2.1.2, and CUDA 12.1. For a full list of software packages and version numbers, see the  file `pyproject.toml`.

The software was tested with the NVIDIA RTX 3090 GPU, though we anticipate that other GPUs will also work, provided that the unit offers sufficient memory.

#### 2. Installation guide

We recommend installation of the required packages using the conda package manager, available through the Anaconda Python distribution. Anaconda is available free of charge for non-commercial use through [Anaconda Inc](https://www.anaconda.com/products/individual). After installing Anaconda and cloning this repository, For use as integrative framework：

```
git https://github.com/weirdgit/RetUNet.git
cd RetUNet
pip install -e .
```

---

## Training

#### 1. Dataset download

Datasets can be acquired via following links:

**Dataset I**
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

**Dataset II**
[CAMUS dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html)

#### 2. Setting up the datasets

After you have downloaded the datasets, you can follow the settings in [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) for path configurations and preprocessing procedures. Finally, your folders should be organized as follows:

```
./RetUNet/
./RetUNet_raw/
    ├── Dataset027_ACDC/
      ├── imagesTr/
      ├── imagesTs/
      ├── labelsTr/
      ├── dataset.json
    ├── Dataset555_CAMUS/
      ├── imagesTr/
      ├── imagesTs/
      ├── labelsTr/
      ├── dataset.json
./RetUNet_preprocessed/
./RetUNet_results/
```

Please make sure you have the following environment variables set:

```
RetUNet_raw = xxx/xxx/RetUNet_raw
RetUNet_preprocessed = xxx/xxx/RetUNet_preprocessed
RetUNet_results = xxx/xxx/RetUNet_results
```

After that, you can preprocess the above data using following commands:

```
RetUNet_plan_and_preprocess -d 27 -c 2d
RetUNet_plan_and_preprocess -d 555 -c 2d
```

#### 3. Training and Testing

Commands for training and testing:

```
RetUNet_train 27 2d 0
RetUNet_train 555 2d 0

RetUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c 2d--save_probabilities
```
