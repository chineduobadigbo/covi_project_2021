import cv2
import torch
import numpy as np
import os
import json
import pathlib
from sklearn.cluster import MiniBatchKMeans

from torch._C import device

import utils.autoencoder_boilerplate as ae

COLOR_MAPPING = cv2.COLOR_BGR2HSV
INVERSE_COLOR_MAPPING = cv2.COLOR_HSV2BGR


def flattenListOfLists(t):
    return [item for sublist in t for item in sublist]

def dataLoadWrapper(patches, batchSize=1024): #dataloader length is number of images/batchsize
    batchSize = batchSize if batchSize <= len(patches) else len(patches)
    return torch.utils.data.DataLoader(ae.PatchDataset(patches), batch_size=batchSize, shuffle=False,num_workers=0, pin_memory=True)


def loadValidationImages(timepoints = [3]):
    validationDir = "data/validation/"

    returnDict = {}

    for root, dirs, _ in os.walk(validationDir, topdown=False):
        for folder in dirs:
            #print(folder)
            folderDict = {}
            for p in timepoints:
                folderDict[p] = [[],[],[],[],[]]



            for _, _, files in os.walk(os.path.join(pathlib.Path().resolve(), validationDir+folder), topdown=False):
                for file in files:
                    if(file.endswith(".png")):
                        timepoint = int(file[0])

                        if(timepoint in timepoints):
                            imagePath = os.path.join(pathlib.Path().resolve(), validationDir+folder+"/"+file)
                            patches,mapping,resolution,tileCount = sliceImage(imagePath)
                            folderDict[timepoint][0].append(patches)
                            folderDict[timepoint][1].append(mapping)
                            folderDict[timepoint][2].append(resolution)
                            folderDict[timepoint][3].append(folder+"/"+file)
                            folderDict[timepoint][4].append(tileCount)

            returnDict[folder] = folderDict

    return returnDict


def loadImages(fileName='/0-B01.png', baseDir='data/train/', size=None): #so far, this only loads the images of one camera in the training set
    patches_list = []
    mapping_list = []
    resolution_list = []
    image_name_list = []
    tileCount_list = []
    for root, dirs, files in os.walk(baseDir, topdown=False):
        for name in dirs:
            sub_folder_image_name = name +fileName
            img_path = baseDir + sub_folder_image_name
            slice_img_path = os.path.join(pathlib.Path().resolve(), img_path)
            patches,mapping,resolution, tileCount = sliceImage(slice_img_path)
            patches_list.append(patches)
            mapping_list.append(mapping)
            resolution_list.append(resolution)
            image_name_list.append(sub_folder_image_name)
            tileCount_list.append(tileCount)
    if type(size) == int:
        return patches_list[0:size], mapping_list[0:size], resolution_list[0:size], image_name_list[0:size], tileCount_list[0:size]
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
            #patch = cv2.cvtColor(patch, COLOR_MAPPING)

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
    #rgbimg = cv2.cvtColor(img, INVERSE_COLOR_MAPPING)
    rgbimg = img
    return rgbimg

def pickBestDevice():
    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    return device

def puzzleBackTogether(atEachBatch,atEachPatch,atEachImage,dataloader,resolutionsList,mappingsList,tileCountsList,imageNamesList,canvasCount, model):

    currentTileIndex=0 #index of the current tile within one image
    currentImageIndex=0

    canvasArray = [None]*canvasCount
    for c in range(canvasCount):
        canvasArray[c] = np.zeros(resolutionsList[currentImageIndex])
        canvasArray[c] = canvasArray[c].astype(np.uint8)
    
    for item in dataloader:
        item = item.to(device)
        reconstructedTensor = atEachBatch(model, item)
        print(f'{item.shape = }')
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
                for c in range(canvasCount):
                    canvasArray[c] = np.zeros(resolutionsList[currentImageIndex])
                    canvasArray[c] = canvasArray[c].astype(np.uint8)

                currentImageIndex+=1
                currentTileIndex=0

def loadModel(name='models/testmodel.txt'):

    global device
    model = ae.Autoencoder()
    model.load_state_dict(torch.load(name, map_location=device))
    return model

def saveTrainMetrics(fileName, metricDict):
    with open(fileName, 'w') as fp:
        json.dump(metricDict, fp)

def createModelOptimizer():
    model = ae.Autoencoder()
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    return model, criterion, optimizer

# applies blur to a single patch
def blurMapWrapper(patch):
    blurredPatch = cv2.GaussianBlur(patch, (5,5), cv2.BORDER_DEFAULT)
    return blurredPatch

# effifciently applies blur to list of patches
def blurPatches(patchList):
    vFuncBlur = np.vectorize(blurMapWrapper, signature='(m,n,c)->(m,n,c)')
    blurredPatchList = vFuncBlur(np.array(patchList))
    return blurredPatchList

def quantizeMapWrapper(patch):
    bgrPatch = patch
    #bgrPatch = cv2.cvtColor(patch, INVERSE_COLOR_MAPPING)
    (h, w) = bgrPatch.shape[:2]
    labPatch = cv2.cvtColor(bgrPatch, cv2.COLOR_RGB2LAB)
    labPatch = labPatch.reshape((labPatch.shape[0] * labPatch.shape[1], 3))
    clt = MiniBatchKMeans(16)
    labels = clt.fit_predict(labPatch)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h, w, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)
    #quant = cv2.cvtColor(quant, COLOR_MAPPING)
    return quant

def quantizePatches(patchList):
    vFuncQuant = np.vectorize(quantizeMapWrapper, signature='(m,n,c)->(m,n,c)')
    quantizedPatchList = vFuncQuant(np.array(patchList))
    return quantizedPatchList
    

