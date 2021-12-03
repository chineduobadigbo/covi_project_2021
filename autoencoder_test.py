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
import time
import pathlib

num_epochs = 2000
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
    start = time.time()
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

    print("training time: ",str(time.time()-start))
    torch.save(model.state_dict(), "testmodel.txt")

    return outputs



def displayResults(outputs):
    for k in range (0, num_epochs, 200):
        plt. figure (figsize=(9, 2))
        #plt. gray ()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate (imgs):
            if i >= 9: break
            plt. subplot (2, 9, i+1)
            item = np.moveaxis(item, 0, 2)
            plt.imshow(item)
        for i, item in enumerate(recon) :
            if i >= 9: break
            plt. subplot(2, 9, 9+i+1) # row length + i + 1
            item = np.moveaxis(item, 0, 2)
            plt.imshow(item)

    plt.show()


def sliceImage(imagepath):
    image = cv2.imread(imagepath)
    patchSize = (64,64)
    imgResolution = image.shape[:-1]
    patchCount = (int(imgResolution[0]/patchSize[0]),int(imgResolution[1]/patchSize[1]))
    print(patchCount)
    patches = []
    mapping = [] #stores the positions for each patch on the original image
    for x in range(patchCount[0]):
        for y in range(patchCount[1]):
            xCoord = x*patchSize[0]
            yCoord = y*patchSize[1]
            patch = image[xCoord:xCoord+patchSize[0],yCoord:yCoord+patchSize[1],:]
            #patch = patch.astype(np.float32)
            #patch /= 255. #convert image to float, so that every value is between 0 an 1
            if cv2.countNonZero(patch[::,0]) > 0: #discard all completely black patches
                patches.append(patch)
                mapping.append((xCoord,yCoord))
                # print(patch)
                # cv2.imshow(imagepath,patch)
                # cv2.waitKey(100)
    
    return patches,mapping,image.shape


def loadModel():
    model = Autoencoder()
    model.load_state_dict(torch.load("testmodel.txt"))
    return model

def compareImages(dataloader,model,mapping,resolution):
    print("sers")
    canvas = np.zeros(resolution)
    rebuilt = np.zeros(resolution)
    cv2.imshow("original",canvas)
    cv2.imshow("reconstruction",rebuilt)
    cv2.waitKey(0)
    for (img) in dataloader:
        npimg = img.numpy()
        recon = model (img)
        nprecon = recon.detach().numpy()

        mses = []
        for i in range(len(img)):

            location=mapping[i]

            patch=npimg[i]
            patch = np.moveaxis(patch, 0, 2)
            canvas[location[0]:location[0]+patch.shape[0],location[1]:location[1]+patch.shape[1],:]=patch

            reconpatch = nprecon[i]
            reconpatch = np.moveaxis(reconpatch, 0, 2)
            #rebuilt[location[0]:location[0]+reconpatch.shape[0],location[1]:location[1]+reconpatch.shape[1],:]=reconpatch

            mse = np.square(np.subtract(patch,reconpatch)).mean()
            mses.append(mse)

    mean_mse=np.mean(mses)
    print(mean_mse)

    for i in range(len(img)):

            location=mapping[i]

            reconpatch = nprecon[i]
            reconpatch = np.moveaxis(reconpatch, 0, 2)
            mse = np.square(np.subtract(patch,reconpatch)).mean()
            msediff = np.power(np.subtract(mean_mse,mse),2)*10
            #print(msediff)
            reconpatch[:,:,2] +=msediff
            canvas[location[0]:location[0]+patch.shape[0],location[1]:location[1]+patch.shape[1],0]+=msediff
            rebuilt[location[0]:location[0]+reconpatch.shape[0],location[1]:location[1]+reconpatch.shape[1],:]=reconpatch


            cv2.imshow("original",canvas)
            cv2.imshow("reconstruction",rebuilt)
            cv2.waitKey(2)
    cv2.waitKey(0)
    


if __name__ == "__main__":


    print("***OKAAAAY LETS GO***")
    patches,mapping,resolution = sliceImage("covi_project_2021/data/train/train-1-0/0-B01.png")
    dataloader = torch.utils.data.DataLoader(PatchDataset(patches), len(patches), shuffle=False,num_workers=0, pin_memory=True)
    model = loadModel()
    compareImages(dataloader,model,mapping,resolution)

    #outputs = trainEncoder(dataloader)
    #displayResults(outputs)
    # parser = argparse.ArgumentParser(description='Train autoencoder')
    # parser.add_argument('--epochs', default=15, type=int,
    #                 help='Number of training epochs')
    # args = parser.parse_args()
    # global num_epochs
    # num_epochs = args.epochs
    # img_path = "data/train/train-1-0/3-B01.png"
    # print(pathlib.Path().resolve())
    # slice_img_path = os.path.join(pathlib.Path().resolve(), img_path)
    # print(slice_img_path)
    # patches = sliceImage(slice_img_path)
    # dataloader = torch.utils.data.DataLoader(PatchDataset(patches), len(patches), shuffle=True,num_workers=3, pin_memory=True)
    # outputs = trainEncoder(dataloader)
    # displayResults(outputs)
