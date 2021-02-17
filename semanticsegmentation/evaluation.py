from tqdm import tqdm
import mmseg
import mmcv

import os
import glob
import json
import numpy as np

from mmseg.apis import init_segmentor, inference_segmentor

def evaluate_dataset(
    checkpoint,
    device,
    config,
    data_dir="/rd-temp/mohan/iccv09Data",
    split_file="/rd-temp/mohan/iccv09Data/splits/val.txt",
    **kwargs
):  
    # initiate model
    model = init_segmentor(config, checkpoint, device=device)
    
    # assign config to model (Required)
    model.config = config
    
    result_list = []
    gt_lst = []
    raw_image_ids = []
    
    with open(split_file, "r") as f:
        for line in f.readlines():       
            raw_image_ids.append(line.strip())

    with tqdm(total=len(raw_image_ids)) as pbar:
        
        for image_id in raw_image_ids:
            # inference
            img = mmcv.imread(f"{data_dir}/images/{image_id}.jpg")
            result = inference_segmentor(model, img)

            # prepare image data and ground truth
            gt_image_path = f"{data_dir}/labels/{image_id}.regions.txt"
            seg_map = np.loadtxt(gt_image_path).astype(np.uint8)

            result_list.extend(result)
            gt_lst.append(seg_map)

            pbar.update(1)

    all_acc, acc, iou = mmseg.core.evaluation.mean_iou(
        results=result_list, 
        gt_seg_maps=gt_lst, 
        num_classes=config.num_classes, 
        ignore_index=kwargs.get("ignore_index", 255),
        label_map=kwargs.get("label_map", {})
    )
    
    return (all_acc, acc, iou, iou[~np.isnan(iou)].mean())