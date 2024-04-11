#-*- coding:utf-8 -*-

from typing import List, Dict
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from PIL import Image
import random
import torch
import json 
import os


IMAGE_SIZE = 224

image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
preprocess = T.Compose([
    T.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(image_mean, image_std)
])

class ImageTextDataset(Dataset):
    def __init__(self, image_folder_path:str, label_file_path:str, transform:any, tokenizer:any, image_transform:any = None):
        # Initialize image paths and corresponding texts
        self.image_folder_path = image_folder_path
        # Tokenize text using CLIP's tokenizer
        self.label_file_path = label_file_path
        self.labels = self.load_labels()
        self.transform = transform
        self.image_transform = image_transform
        self.tokenizer = tokenizer
    
    def load_labels(self):
        """
            Label file structure:
            [
                {
                    'file': file_name.jpg (not full path),
                    'label': "LABEL"
                },
                ...
            ]
        """
        with open(self.label_file_path) as f:
            labels = json.loads(f.read())
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        data = self.labels[idx]
        image_file = os.path.join(self.image_folder_path, data['file'])
        title = data['label']
        if self.image_transform is not None:
            image = self.image_transform(image)
        image = self.transform(Image.open(image_file))
        # title = self.tokenizer(title)
        return image, title
    
if __name__ == '__main__':
    dataset = ImageTextDataset("../raw_datas/images", "..//raw_datas/labels.json")
    print(">", len(dataset))
    im, tx = dataset[0]
    print(im.shape, torch.max(im), torch.min(im))
    print(tx)