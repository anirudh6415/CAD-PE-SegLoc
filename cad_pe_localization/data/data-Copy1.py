import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchmetrics.functional import accuracy
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchmetrics.functional import accuracy
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import os
from tqdm import tqdm
import tifffile
import random
import sys
import cv2
import torch.nn as nn

def nor_image(org_img):
    normalized_img = ((org_img - np.min(org_img)) / (np.max(org_img) - np.min(org_img)))
    return normalized_img.astype(np.float32)

class cadpedataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = annotation.copy()
        self.ids = [i['image_id'] for i in annotation['annotations']][:3000
        #print(self.ids)
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = img_id 
        coco_annotation = [i for i in self.coco['annotations'] if i['image_id']==img_id] 
        #print(coco_annotation)
        # path for input image
        path = [i['file_name'] for i in self.coco['images'] if i['id']==img_id][0] #self.image_names[index]
        #print(f"File Name: {path}")
        # Image.open(os.path.join(self.root, self.image_names[index])) #coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = np.load(os.path.join(self.root, path))
        #print(img.dtype)

        # img_np = img.astype(np.uint8)
        
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        resized_img_width, resized_img_height = 128,128
        original_img_width, original_img_height= 512,512
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = coco_annotation[i]['bbox'][2]
            ymax = coco_annotation[i]['bbox'][3]
            
           
            
            
            boxes.append([xmin, ymin, xmax, ymax])
        if boxes:
            boxes = torch.as_tensor(boxes,dtype=torch.float32)
        else:
            pass
            # boxes = torch.as_tensor([[0,0,0,0]],dtype=torch.float32)
        
        #print(len(boxes))
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        img = nor_image(img)
        if self.transforms is not None:
            img = self.transforms(img)
        #print(my_annotation)
        return img, my_annotation
