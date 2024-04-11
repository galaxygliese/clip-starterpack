#-*- coding:utf-8 -*-

from dataset import ImageTextDataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from glob import glob
from tqdm import tqdm
import japanese_clip as ja_clip
import torch.nn as nn
import numpy as np
import argparse
import torch
import clip
import os

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-b', '--batchsize', type=int, default=256)
parser.add_argument('--save-per-epochs', type=int, default=100)
opt = parser.parse_args()

def load_pretrained_model(device):
    model, preprocess = ja_clip.load("rinna/japanese-clip-vit-b-16", device=device)
    tokenizer = ja_clip.load_tokenizer()
    return model, preprocess, tokenizer

def load_pretrained_model_openai(device):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = clip.tokenize
    return model, preprocess, tokenizer

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def main():
    device = 'cuda'
    IMAGE_FOLDER = "<PATH>/raw_datas/images"
    LABEL_FILE = "<PATH>/raw_datas/labels.json"
    
    # model, preprocess, tokenizer = load_pretrained_model(device)
    model, preprocess, tokenizer = load_pretrained_model_openai(device)
    print("Models Loaded!")
    dataset = ImageTextDataset(IMAGE_FOLDER, LABEL_FILE, transform=preprocess, tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True)
    print("Training Datas:", len(dataset))
    
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=5e-5,
        betas=(0.9,0.98),
        eps=1e-6,
        weight_decay=0.2
    )
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    for epoch in range(opt.epochs):
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch in pbar:
            optimizer.zero_grad()
            images, texts = batch 
            # texts = ja_clip.tokenize(
            #     texts=texts,
            #     max_seq_len=77,
            #     device=device,
            #     tokenizer=tokenizer, # this is optional. if you don't pass, load tokenizer each time
            # )['input_ids']
            texts = list(texts)
            texts = tokenizer(texts)
        
            images= images.to(device)
            texts = texts.to(device)

            # Forward pass
            
            logits_per_image, logits_per_text = model(images, texts)
            # logits_per_image = model.get_image_features(images)
            # logits_per_text = model.get_text_features(input_ids=texts)

            # Compute loss
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

            # Backward pass
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else : 
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
        print("Loss:", total_loss.detach())
        if epoch % opt.save_per_epochs == 0 and epoch > 0:
            torch.save(model.state_dict(), os.path.join('checkpoints', f'training-clip-epoch{epoch}.pt'))
    print("Finished!")
    torch.save(model.state_dict(), os.path.join('checkpoints', f'clip-epoch{opt.epochs}.pt'))

if __name__ == '__main__':
    main()
