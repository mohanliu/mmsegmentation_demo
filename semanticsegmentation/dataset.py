import os

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from .class_names import *

@DATASETS.register_module()
class StandfordBackgroundDataset(CustomDataset):
  CLASSES = classes
  PALETTE = palette
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png', 
                     split=split, **kwargs)
    assert os.path.exists(self.img_dir) and self.split is not None