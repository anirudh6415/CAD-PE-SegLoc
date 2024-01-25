import os 
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def binary(org_img):
    #print(np.any(org_img))
    if np.any(org_img):
        normalized_img = (org_img - np.min(org_img)) / (np.max(org_img) - np.min(org_img))
        normalized_img[normalized_img < 0.5] = 0
        normalized_img[normalized_img >= 0.5] = 1
        return normalized_img.astype(np.float32)
    else:
        return org_img.astype(np.float32)
    
def nor_image(org_img):
    normalized_img = ((org_img - np.min(org_img)) / (np.max(org_img) - np.min(org_img)))
    return normalized_img.astype(np.float32)



class segmentation_dataset(Dataset):
    def __init__(self,image_filenames,mask_filenames,transforms=None):
        self.image_dir = "/data/jliang12/nuislam/CAD_PE_Challenge_Data/np_images/"
        self.mask_dir = "/data/jliang12/nuislam/CAD_PE_Challenge_Data/np_masks/"
        self.transform = transforms
        # self.mask_transform = transform.Compose([ transform.ToPILImage(),
        #     transform.Resize((128,128), InterpolationMode.BICUBIC),
        #     transform.Grayscale(num_output_channels = 1),
        #     transform.ToTensor()])
    
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        
    def __len__(self):
        return len(self.image_filenames)
    
    
    def __getitem__(self, idx):
        img = np.load(os.path.join(self.image_dir, self.image_filenames[idx]))
        label = np.load(os.path.join(self.mask_dir, self.mask_filenames[idx]))
        
        img = nor_image(img)
        label = binary(label)
        if self.transform is not None:
            img, label = self.transform(img), self.transform(label)
            
        return img, label
        