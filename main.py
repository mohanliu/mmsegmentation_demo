import torch, torchvision
import mmseg
import mmcv

import matplotlib.pyplot as plt
import os
import glob
import argparse
import numpy as np
from PIL import Image

from semanticsegmentation.class_names import *
from semanticsegmentation.dataset import *
from semanticsegmentation.evaluation import *

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor, set_random_seed, inference_segmentor
    
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
    
    # evaluation
    if args.evaluation:
        eval_ = evaluate_dataset(
            checkpoint="{}/iter_{}.pth".format(cfg.work_dir, args.evaluation),
            device='cuda:{}'.format(cfg.gpu_ids[0]),
            config=cfg
        )
        print(f"Overall accuracy: {eval_[0]}")
        print(f"Accuracies: {eval_[1]}")
        print(f"IoUs: {eval_[2]}")
        print(f"mIoU: {eval_[3]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--version", "-V", type=str, help="Model Version", required=True)
    parser.add_argument("--config", "-C", action="store_true", help="Show config")
    parser.add_argument("--model", "-M", action="store_true", help="Show model")
    parser.add_argument("--train", "-T", action="store_true", help="Launch training")
    parser.add_argument("--evaluation", "-E", type=int, help="Evaluation iteration")

    args = parser.parse_args()
    
    main(args)