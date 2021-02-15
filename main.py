import torch, torchvision
import mmseg
import mmcv

import matplotlib.pyplot as plt
import os
import glob
import argparse
import numpy as np
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor, set_random_seed, inference_segmentor

# define class and plaette for better visualization
classes = (
    'sky', 'tree', 'road', 'grass', 
    'water', 'bldg', 'mntn', 'fg obj'
)

palette = [
    [128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
    [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]
]

@DATASETS.register_module()
class StandfordBackgroundDataset(CustomDataset):
  CLASSES = classes
  PALETTE = palette
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png', 
                     split=split, **kwargs)
    assert os.path.exists(self.img_dir) and self.split is not None
    
def main(args):
    cfg = mmcv.Config.fromfile(f'./experiments/config_standfordbackground_{args.version}.py')
    
    if args.config:
        print(cfg.pretty_text)

    set_random_seed(cfg.seed, deterministic=False)

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_segmentor(
        cfg.model
    )
    
    if args.model:
        print(model)

    # Launch training
    if args.train:
        print("Start training...")
        
        # Create work_dir
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
        
        # Training process
        train_segmentor(
            model, 
            datasets, 
            cfg, 
            distributed=False, 
            validate=True, 
            meta={
                "CLASSES": classes,
                "PALETTE": palette,
            }
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--version", "-V", type=str, help="Model Version", required=True)
    parser.add_argument("--config", "-C", action="store_true", help="Show config")
    parser.add_argument("--model", "-M", action="store_true", help="Show model")
    parser.add_argument("--train", "-T", action="store_true", help="Launch training")

    args = parser.parse_args()
    
    main(args)