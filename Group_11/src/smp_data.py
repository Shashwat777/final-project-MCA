import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
c=1
class SMP_data(Dataset):

     def __init__(self, csv_file, root_dir, transform):
        self.data = pd.read_csv(csv_file, header=None) 
        self.root_dir = root_dir 
        self.transform = transform
        

        # Univariate and Ablation Study
        self.use_img = True#False
        self.user = True# False#
        self.text = True# False# 
        self.ts = True# False# 
        self.num =  True#False
        self.img_score = True# False#

     def __len__(self):
        return len(self.data)

     def __getitem__(self, idx): 
    
        if self.use_img:
            img_name = os.path.join(self.root_dir, str(self.data.iloc[idx, 0]))
            

       
           
           
            
            image = Image.open(img_name).convert("RGB")
            image = self.transform(image)
                
          
            
            
        else:
            image = torch.zeros(3,224,224)

        if self.text:
            category = str(self.data.iloc[idx, 1])
            tags = str(self.data.iloc[idx, 2])
            title = str(self.data.iloc[idx, 3])
        else:
            category = '0'
            tags = '0'
            title = '0'
        
        if self.ts:
            ts = torch.from_numpy(self.data.iloc[idx, 4:7].values.astype(np.float32))
        else:
            ts = torch.zeros(3)

        if self.user:
            user = torch.from_numpy(self.data.iloc[idx, 7:10].values.astype(np.float32))
        else:
            user = torch.zeros(3)

        if self.num:
            num = torch.from_numpy(self.data.iloc[idx, 10:12].values.astype(np.float32))
        else:
            num = torch.zeros(2)

        if self.img_score:
            img_score = torch.from_numpy(self.data.iloc[idx, 12:14].values.astype(np.float32))
        else:
            img_score = torch.zeros(2)

        meta = torch.cat([ts,user,num,img_score])

        id = str(self.data.iloc[idx, 0])[:-4]
        label = torch.from_numpy(np.array(self.data.iloc[idx,14])).float().view(1)
        text = category+' | '+ tags+' | '+title 
        sample = {'img':image, 'meta':meta,'user':user,'ts':ts, 'label':label, 'id':id, 'category':category, 'tags':tags, 'title':title, 'text': text}

        return sample

