"""
Dataset to load the Pascal VOC & MS COCO datasets
"""

import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    """Some Information about MyDataset"""
    def __init__(
        self, 
        csv_file, 
        img_dir, 
        label_dir, 
        anchors, 
        image_size=416,
        S=[13,26,52]
        C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir,  self.annotations.iloc[index, 1]) # Because 1 is the second column 
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image , bboxes=bboxes)
            image = augmentations[image]
            bboxes = augmentations[bboxes]

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        for box in bboxes: 
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)




        return 

    def __len__(self):
        return len(self.annotations)












