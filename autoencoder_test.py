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
import itertools

BASE_TRAIN_PATH = "data/train/"
SAVE_PATH = "data/train_images_mse/"

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
    return model, outputs



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


def loadModel(name='testmodel.txt'):
    model = Autoencoder()
    model.load_state_dict(torch.load(name))
    return model

def compareImages(patches_list,model,mapping_list,resolution_list):
    global_mse = []
    canvas_list = []
    rebuilt_list = []
    nprecon_list = []
    new_patches = []
    print('Paches len {}'.format(len(patches_list)))
    print('mappings len {}'.format(len(mapping_list)))
    print('resolutions len {}'.format(len(resolution_list)))
    for dataloader, mapping, resolution in zip(patches_list, mapping_list, resolution_list):
        canvas = np.zeros(resolution)
        rebuilt = np.zeros(resolution)
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

                mse = np.square(np.subtract(patch,reconpatch)).mean()
                mses.append(mse)
            canvas_list.append(canvas)
            rebuilt_list.append(rebuilt)
            global_mse.append(np.mean(mses))
            nprecon_list.append(nprecon)
            new_patches.append(patch)
    mean_mse=np.mean(global_mse)
    print(global_mse)
    print(mean_mse)
    save_image_id = 0
    for dataloader, mapping, canvas, rebuilt, nprecon, patch in zip(patches_list, mapping_list, canvas_list, rebuilt_list, nprecon_list, new_patches):
        for (img) in dataloader:
            print('Img len: {}'.format(len(img)))
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
            write_dir = SAVE_PATH+'{}'.format(save_image_id)
            print(write_dir)
            # create dir if not exists
            Path(write_dir).mkdir(parents=True, exist_ok=True)
            coonverted_range_canvas = canvas * 255
            coonverted_range_rebuilt = rebuilt * 255
            cv2.imwrite(write_dir+'/'+"original.png", coonverted_range_canvas)
            cv2.imwrite(write_dir+'/'+"rebuilt.png", coonverted_range_rebuilt)
            save_image_id += 1
    
def flatten_list_of_lists(t):
    return [item for sublist in t for item in sublist]

def data_load_wrapper(train_patches):
    return torch.utils.data.DataLoader(PatchDataset(train_patches), len(train_patches), shuffle=False,num_workers=0, pin_memory=True)

def get_dataloader_list(patches_list):
    dataloaders = []
    for patch in patches_list:
        dataloaders.append(data_load_wrapper(patch))
    return dataloaders            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train autoencoder')
    parser.add_argument('--epochs', default=15, type=int,
                    help='Number of training epochs')
    parser.add_argument('--train', dest='train', default=False, action='store_true',
                    help='Dont train the model, load it instead')
    args = parser.parse_args()
    global num_epochs
    num_epochs = args.epochs
    img_path = "data/train/train-1-0/0-B01.png"
    print(pathlib.Path().resolve())
    patches_list = []
    mapping_list = []
    resolution_list = []
    for root, dirs, files in os.walk(BASE_TRAIN_PATH, topdown=False):
        for name in dirs:
            img_path = BASE_TRAIN_PATH + name +'/0-B01.png'
            slice_img_path = os.path.join(pathlib.Path().resolve(), img_path)
            patches,mapping,resolution = sliceImage(slice_img_path)
            patches_list.append(patches)
            mapping_list.append(mapping)
            resolution_list.append(resolution)
    print('Len Patch List: {}'.format(len(patches_list)))
    train_patches = flatten_list_of_lists(patches_list)
    print('Len Train Patches: {}'.format(len(train_patches)))
    imgs_per_patch = len(patches_list[0])
    train_dataloader = data_load_wrapper(train_patches)

    if args.train:
        model, outputs = trainEncoder(train_dataloader)
        torch.save(model.state_dict(), "testmodel.txt")
    else:
        model = loadModel(name='testmodel_2000epochs.txt')
    compareImages(get_dataloader_list(patches_list),model,mapping_list,resolution_list)

