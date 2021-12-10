import cv2
import torch
import numpy as np
import os
import pathlib

from torch._C import device

import utils.autoencoder_boilerplate as ae

COLOR_MAPPING = cv2.COLOR_RGB2HSV
INVERSE_COLOR_MAPPING = cv2.COLOR_HSV2RGB
BASE_TRAIN_PATH = "data/train/"
SAVE_PATH = "data/train_images_mse/"
VAL_SAVE_PATH = "data/val_train_images_mse/"


def flattenListOfLists(t):
    return [item for sublist in t for item in sublist]

def dataLoadWrapper(patches): #dataloader length is number of images/batchsize
    return torch.utils.data.DataLoader(ae.PatchDataset(patches), batch_size=1024, shuffle=False,num_workers=0, pin_memory=True)

def loadImages(file_name='/0-B01.png', size=None): #so far, this only loads the images of one camera in the training set
    patches_list = []
    mapping_list = []
    resolution_list = []
    image_name_list = []
    tileCount_list = []
    for root, dirs, files in os.walk(BASE_TRAIN_PATH, topdown=False):
        for name in dirs:
            sub_folder_image_name = name +file_name
            img_path = BASE_TRAIN_PATH + sub_folder_image_name
            slice_img_path = os.path.join(pathlib.Path().resolve(), img_path)
            patches,mapping,resolution, tileCount = sliceImage(slice_img_path)
            patches_list.append(patches)
            mapping_list.append(mapping)
            resolution_list.append(resolution)
            image_name_list.append(sub_folder_image_name)
            tileCount_list.append(tileCount)
    if type(size) == int:
        return patches_list[0:size], mapping_list[0:size], resolution_list[0:size], image_name_list[0:size]
    else:
        return patches_list, mapping_list, resolution_list, image_name_list, tileCount_list

def sliceImage(imagepath):
    image = cv2.imread(imagepath)
    patchSize = (64,64)
    imgResolution = image.shape[:-1]
    patchCount = (int(imgResolution[0]/patchSize[0]),int(imgResolution[1]/patchSize[1]))
    patches = []
    mapping = [] #stores the positions for each patch on the original image
    tileCount = 0
    for x in range(patchCount[0]):
        for y in range(patchCount[1]):
            xCoord = x*patchSize[0]
            yCoord = y*patchSize[1]
            patch = image[xCoord:xCoord+patchSize[0],yCoord:yCoord+patchSize[1],:]
            patch = cv2.cvtColor(patch, COLOR_MAPPING)

            if cv2.countNonZero(patch[::,0]) > 0: #discard all completely black patches
                tileCount += 1
                patches.append(patch)
                mapping.append((xCoord,yCoord))

    return patches,mapping,image.shape,tileCount

def convertTensorToImage(tensor):
    img = tensor.detach().numpy()
    img = np.moveaxis(img,0,2)
    img *= 255
    img = img.astype(np.uint8)
    rgbimg = cv2.cvtColor(img, INVERSE_COLOR_MAPPING)
    return rgbimg

def pickBestDevice():
    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using device: ", device)
    return device

def puzzleBackTogether(atEachBatch,atEachPatch,atEachImage,dataloader,resolutionsList,mappingsList,tileCountsList,imageNamesList,canvasCount):

    currentTileIndex=0 #index of the current tile within one image
    currentImageIndex=0

    canvasArray = [None]*canvasCount
    for c in range(canvasCount):
        canvasArray[c] = np.zeros(resolutionsList[currentImageIndex])
        canvasArray[c] = canvasArray[c].astype(np.uint8)
    
    for item in dataloader:
        item = item.to(device)
        reconstructedTensor = atEachBatch(item)
        for i in range(len(item)):
            patchTensor = item[i].cpu()
            reconstructedPatchTensor = reconstructedTensor[i].cpu()
            patchArray = atEachPatch(patchTensor,reconstructedPatchTensor)
            location = mappingsList[currentImageIndex][currentTileIndex]

            for c in range(canvasCount):    
                canvasArray[c][location[0]:location[0]+patchArray[c].shape[0],location[1]:location[1]+patchArray[c].shape[1],:] = patchArray[c]
            
            currentTileIndex+=1
            if currentTileIndex >= tileCountsList[currentImageIndex]:
                atEachImage(canvasArray,imageNamesList[currentImageIndex])
                currentImageIndex+=1
                currentTileIndex=0

                for c in range(canvasCount):
                    canvasArray[c] = np.zeros(resolutionsList[currentImageIndex])
                    canvasArray[c] = canvasArray[c].astype(np.uint8)

def loadModel(name='models/testmodel.txt'):

    global device
    model = ae.Autoencoder()
    model.load_state_dict(torch.load(name, map_location=device))
    return model