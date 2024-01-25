import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2

def nor_image(org_img):
    normalized_img = ((org_img - np.min(org_img)) / (np.max(org_img) - np.min(org_img)))
    return normalized_img.astype(np.float32)

class cadpedataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None, target_size=(128, 128)):
        self.root = root
        self.transforms = transforms
        self.coco = annotation.copy()
        self.ids = [i['image_id'] for i in annotation['annotations']]
        self.target_size = target_size
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = img_id 
        coco_annotation = [i for i in self.coco['annotations'] if i['image_id']==img_id]
        
        # path for input image
        path = [i['file_name'] for i in self.coco['images'] if i['id']==img_id][0]
        
        # open the input image
        img = np.load(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = coco_annotation[i]['bbox'][2]
            ymax = coco_annotation[i]['bbox'][3]
            
            # Resize bounding boxes to target size
            xmin = xmin * self.target_size[1] / img.shape[1]
            xmax = xmax * self.target_size[1] / img.shape[1]
            ymin = ymin * self.target_size[0] / img.shape[0]
            ymax = ymax * self.target_size[0] / img.shape[0]
            
            boxes.append([xmin, ymin, xmax, ymax])
        
        # if boxes:
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # else:
        #     boxes = torch.as_tensor([[0,0,0,0]], dtype=torch.float32)
        
        # Labels (In my case, I only have one class: target class or background)
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