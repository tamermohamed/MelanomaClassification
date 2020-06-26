import os
import  csv
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd 

class ImagesDataset(Dataset):
    

    def __init__(self, csv_file, root_dir, transform=None):
      
     
        self.root_dir = root_dir
        self.transform = transform
        self.lables = pd.read_csv(csv_file)
        

    def __len__(self):
        return len(self.lables)

    def __getitem__(self, idx):
        
        img_name = self.lables.iloc[idx,0] 
        
        lable = int(self.lables.iloc[idx,7])

        image_path =  os.path.join(self.root_dir, img_name)
                               
        image = Image.open(open(f'{image_path}.jpg','rb'))

        if self.transform:
            image = self.transform(image)
           

        
        return  {'image': image, 'lable': lable}