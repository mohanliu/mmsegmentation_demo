# Demo of `MMSegmentation` Toolbox 
> Reproduce inference and training processes using provided backbone and methods for semantic segmentation. Source Code: https://github.com/open-mmlab/mmsegmentation

## Installation

### Create a conda virtual environment and activate it.

```bash
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

### Install PyTorch and torchvision

```bash
conda install pytorch=1.7.0 torchvision cudatoolkit=11.0 -c pytorch
```

### Install MMCV 
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

[link](https://mmcv.readthedocs.io/en/latest/#installation)

### Install MMSegmentation.

```bash
pip install git+https://github.com/open-mmlab/mmsegmentation.git
```

## Useful links

- Pretrained backbone model: https://github.com/open-mmlab/mmcv/tree/master/mmcv/model_zoo
- MMSegmentation Tutorial Docs: https://github.com/open-mmlab/mmsegmentation/tree/master/docs/tutorials
- Prepare new dataset: https://github.com/open-mmlab/mmsegmentation/blob/master/demo/MMSegmentation_Tutorial.ipynb

