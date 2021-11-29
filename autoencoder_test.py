import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU, Sigmoid
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import os
import sys
from pathlib import Path
import argparse
import copy
import pathlib


BASE_VAL_PATH = "data/validation/"

class PatchDataset(Dataset): #a dataset object has to be definied that specifies how PyTorch can access the training data

    def __init__(self, X):
        'Initialization'
        self.X = X
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.X[index]
        X = self.transform(image)
        return X

    transform = T.Compose([
        T.ToPILImage(),
        #T.Resize(image_size),
        T.ToTensor()])

class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            #the stride parameter is describing how much the resoltion decreases in each step
            nn. Conv2d(3, 16, 3, stride=2, padding=1), # N, 16, 32, 32
            nn. ReLU() ,
            nn. Conv2d(16, 32, 3 , stride=2, padding=1), # N, 32, 16, 16
            nn. ReLU(),
            nn. Conv2d(32, 64, 3 , stride=2, padding=1), # N, 64, 8, 8
            nn. ReLU(),
            #in the last step, we make the stride equal to the image resolution, thus reducing it to just one pixel
            nn. Conv2d(64, 128, 8) # N, 128, 1, 1 -> this results a simple 128d vector
        )

        self.decoder = nn.Sequential(
            nn. ConvTranspose2d(128, 64, 8), # N, 64, 8, 8
            nn. ReLU() ,
            nn. ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # N, 64, 8, 8
            nn. ReLU() ,
            nn. ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # N, 32, 16, 16
            nn. ReLU() ,
            nn. ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), # N, 3, 64, 64
            nn. Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def loadData():
    transform = transforms.ToTensor()
    mnist_data = datasets.MNIST(root='./mnistdata',train=True,download=True,transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data,batch_size=64,shuffle=True)
    return data_loader

def trainEncoder(data_loader):
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model. parameters (), lr=1e-3, weight_decay=1e-5)

    outputs = []
    for epoch in range (num_epochs) :
        for (img) in data_loader:
            #print(img.shape)
            #img = img. reshape (-1, 28*28)
            recon = model (img)
            loss = criterion (recon, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print (f' Epoch: {epoch+1}, Loss: {loss. item ():.4f}')
        outputs.append((epoch, img, recon))

    return outputs



def displayResults(outputs):
    for k in range (0, num_epochs, 5):
        plt. figure (figsize=(9, 2))
        #plt. gray ()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate (imgs):
            if i >= 9: break
            plt. subplot (2, 9, i+1)
            plt.imshow(item[0])
        for i, item in enumerate(recon) :
            if i >= 9: break
            plt. subplot(2, 9, 9+i+1) # row length + i + 1
            plt.imshow(item[0])

    plt.show()


def sliceImage(imagepath):
    image = cv2.imread(imagepath)
    patchSize = (64,64)
    imgResolution = image.shape[:-1]
    patchCount = (int(imgResolution[0]/patchSize[0]),int(imgResolution[1]/patchSize[1]))
    print(patchCount)
    patches = []
    for x in range(patchCount[0]):
        for y in range(patchCount[1]):
            xCoord = x*patchSize[0]
            yCoord = y*patchSize[1]
            patch = image[xCoord:xCoord+patchSize[0],yCoord:yCoord+patchSize[1],:]
            #patch = patch.astype(np.float32)
            #patch /= 255. #convert image to float, so that every value is between 0 an 1
            if cv2.countNonZero(patch[::,0]) > 0: #discard all completely black patches
                patches.append(patch)
                # print(patch)
                # cv2.imshow(imagepath,patch)
                # cv2.waitKey(100)
    
    return patches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train autoencoder')
    parser.add_argument('--epochs', default=15, type=int,
                    help='Number of training epochs')
    args = parser.parse_args()
    global num_epochs
    num_epochs = args.epochs
    img_path = "data/train/train-1-0/3-B01.png"
    print(pathlib.Path().resolve())
    slice_img_path = os.path.join(pathlib.Path().resolve(), img_path)
    print(slice_img_path)
    patches = sliceImage(slice_img_path)
    dataloader = torch.utils.data.DataLoader(PatchDataset(patches), len(patches), shuffle=True,num_workers=3, pin_memory=True)
    outputs = trainEncoder(dataloader)
    displayResults(outputs)