import numpy as np
import pandas as pd
import os
import csv
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from UNet import UNet

class MRI_dataset(Dataset):
    '''
    x: imgs
    y: masks if none -> predict
    '''
    def __init__(self, path, masks = False):
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        self.masks = masks
        
    def __getitem__(self, idx):
        f_name = self.files[idx]
        x = cv2.imread(f_name, 0)
        if self.masks is False:
            return torch.tensor(x)
        else:
            y = cv2.imread(f_name.replace('imgs', 'mask'), 0)
            y[y <= 10] = 0
            y[y > 10] = 1
            return x, y
            # return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return len(self.files)


def calculate_iou(prediction, target):
    # print(prediction.shape, target.shape)
    intersection = np.logical_and(prediction, target)
    union = np.logical_or(prediction, target)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def iou(pred, target):
    iou_values = []
    for i in range(len(pred)):
        iou = calculate_iou(pred[i], target[i])
        iou_values.append(iou)
    return sum(iou_values)/len(iou_values)
    

config = {
    'name' : 'model0927',
    'train_imgs_path' : 'mri_denoise_checked/train/imgs',
    'valid_imgs_path' : 'mri_denoise_checked/valid/imgs',
    'test_imgs_path' : 'mri_denoise_checked/test/imgs',
    'batch_size' : 128,
    'epoch_num' : 50,
    'device' : "cuda" if torch.cuda.is_available() else "cpu",
    'early_stop' : 50
}

train_set = MRI_dataset(config['train_imgs_path'], masks=True)
train_loader = DataLoader(train_set, batch_size = config['batch_size'], shuffle=True)

valid_set = MRI_dataset(config['valid_imgs_path'], masks=True)
valid_loader = DataLoader(valid_set, batch_size = config['batch_size'], shuffle=True)

device = config['device']
n_epochs = config['epoch_num']
patience = config['early_stop']

model = UNet(1, 1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) 

if __name__ == '__main__':
    stale = 0
    best_acc = 0

    total_train_loss = []
    total_train_accs = []
    total_val_loss = []
    total_val_accs = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = []
        train_accs = []

        for img in tqdm(train_loader):    
            DWI_imgs, masks = img
            DWI_imgs, masks = DWI_imgs.to(torch.float32).to(device), masks.to(torch.float32).to(device)
            out = model(DWI_imgs.unsqueeze(1)).squeeze(1)
            loss = criterion(out, masks)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            predicted_mask = (out.cpu().detach().numpy() > 0.5).astype(np.uint8)
            masks = masks.cpu().detach().numpy().astype(np.uint8)
            acc = iou(predicted_mask, masks)

            train_loss.append(loss.item())
            train_accs.append(acc)
            
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        total_train_loss.append(train_loss)
        total_train_accs.append(train_acc)

        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        model.eval()
        valid_loss = []
        valid_accs = []

        for img in tqdm(valid_loader):
            DWI_imgs, masks = img
            DWI_imgs, masks = DWI_imgs.to(torch.float32).to(device), masks.to(torch.float32).to(device)
            with torch.no_grad():
                out = model(DWI_imgs.unsqueeze(1)).squeeze(1)

            loss = criterion(out, masks)
            predicted_mask = (out.cpu().detach().numpy() > 0.5).astype(np.uint8)
            masks = masks.cpu().detach().numpy().astype(np.uint8)
            acc = iou(predicted_mask, masks)

            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        total_val_loss.append(valid_loss)
        total_val_accs.append(valid_acc)
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    

        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"./weight/{config['name']}_best.ckpt") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break


        fig = plt.figure()
        plt.subplot(221)
        plt.title('train_loss')
        plt.plot(total_train_loss)

        plt.subplot(222)
        plt.title('train_acc')
        plt.plot(total_train_accs)

        plt.subplot(223)
        plt.title('val_loss')
        plt.plot(total_val_loss)

        plt.subplot(224)
        plt.title('val_acc')
        plt.plot(total_val_accs)

        fig.tight_layout()
        fig.savefig(f"{config['name']}_output.jpg")
        plt.close()