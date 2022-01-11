import cv2
import torch
import numpy as np
import os
import json
import pathlib
from sklearn.cluster import MiniBatchKMeans
import datetime
import warnings
from torch._C import device
from utils.pytorch_msssim import MS_SSIM_Loss, SSIM_Loss

import utils.autoencoder_boilerplate as ae

COLOR_MAPPING = cv2.COLOR_BGR2HSV
INVERSE_COLOR_MAPPING = cv2.COLOR_HSV2BGR

ColorMappingGDict = {
    'RGB': {'mapping': cv2.COLOR_BGR2RGB, 'inverse': cv2.COLOR_RGB2BGR}, 
    'HSV': {'mapping': cv2.COLOR_BGR2HSV, 'inverse': cv2.COLOR_HSV2BGR}, 
    'LAB': {'mapping': cv2.COLOR_BGR2LAB, 'inverse': cv2.COLOR_LAB2BGR}
    }


def flattenListOfLists(t):
    return [item for sublist in t for item in sublist]

def dataLoadWrapper(patches, batchSize=1024): #dataloader length is number of images/batchsize
    batchSize = batchSize if batchSize <= len(patches) else len(patches)
    print(f'Creating Dataloader with batchSize {batchSize}')
    return torch.utils.data.DataLoader(ae.PatchDataset(patches), batch_size=batchSize, shuffle=False,num_workers=0, pin_memory=True)


def loadValidationImages(color, timepoints = [3]):
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
                            patches,mapping,resolution,tileCount = sliceImage(imagePath, color)
                            folderDict[timepoint][0].append(patches)
                            folderDict[timepoint][1].append(mapping)
                            folderDict[timepoint][2].append(resolution)
                            folderDict[timepoint][3].append(folder+"/"+file)
                            folderDict[timepoint][4].append(tileCount)

            returnDict[folder] = folderDict

    return returnDict


def loadImages(color, fileName='/0-B01.png', baseDir='data/train/', size=None, completeData=False): #so far, this only loads the images of one camera in the training set
    print(f'Loading images into {color} space...')
    fileName = fileName.split('/')[-1]
    patches_list = []
    mapping_list = []
    resolution_list = []
    image_name_list = []
    tileCount_list = []
    for root, dirs, files in os.walk(baseDir, topdown=False):
        for name in dirs:
            for file in os.listdir(os.path.abspath(os.path.join(root, name))):
                takeFile = False
                if completeData and file.endswith(".png"):
                    takeFile = True
                elif not completeData and fileName in file:
                    takeFile = True
                if takeFile:
                    sub_folder_image_name = f'{name}/{file}'
                    img_path = baseDir + sub_folder_image_name
                    slice_img_path = os.path.join(pathlib.Path().resolve(), img_path)
                    patches,mapping,resolution, tileCount = sliceImage(slice_img_path, color)
                    patches_list.append(patches)
                    mapping_list.append(mapping)
                    resolution_list.append(resolution)
                    image_name_list.append(sub_folder_image_name)
                    tileCount_list.append(tileCount)
    if type(size) == int:
        return patches_list[0:size], mapping_list[0:size], resolution_list[0:size], image_name_list[0:size], tileCount_list[0:size]
    else:
        return patches_list, mapping_list, resolution_list, image_name_list, tileCount_list

def sliceImage(imagepath, color):
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

            if cv2.countNonZero(patch[::,0]) > 0: #discard all completely black patches
                tileCount += 1
                patch = cv2.cvtColor(patch, ColorMappingGDict[color]['mapping'])
                patches.append(patch)
                mapping.append((xCoord,yCoord))

    return patches,mapping,image.shape,tileCount

# always returns BGR image
def convertTensorToImage(tensor, color):
    img = tensor.detach().numpy()
    img = np.moveaxis(img,0,2)
    img *= 255
    img = img.astype(np.uint8)
    bgrImg = cv2.cvtColor(img, ColorMappingGDict[color]['inverse'])
    return bgrImg

def pickBestDevice():
    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    return device

def puzzleBackTogether(atEachBatch,atEachPatch,atEachImage,dataloader,resolutionsList,mappingsList,tileCountsList,imageNamesList,canvasCount, model, color):

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
            location = mappingsList[currentImageIndex][currentTileIndex]
            patchArray = atEachPatch(patchTensor,reconstructedPatchTensor, color)

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

def loadModel(name='models/testmodel.txt', optimizer=None):
    global device
    model = ae.Autoencoder()
    checkpoint = torch.load(name, map_location=device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        print(f'Trying to load legacy model {name}. Loading model without checkpoint, lastEpoch, lossPerEpoch and optimizer')
        model.load_state_dict(checkpoint)
        return model, None
    model.to(device)
    lastEpoch = checkpoint['epoch']
    lossPerEpoch = checkpoint['metricDict']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, checkpoint, lastEpoch, lossPerEpoch, optimizer
    return model, checkpoint, lastEpoch, lossPerEpoch

def loadMetrics(metricFileName):
    with open(metricFileName) as json_file:
        metricDict = json.load(json_file)
    return metricDict

def saveTrainMetrics(fileName, metricDict):
    with open(fileName, 'w') as fp:
        json.dump(metricDict, fp)

def createModelOptimizer(loss='mse'):
    model = ae.Autoencoder()
    model.to(device)
    if loss == 'mse':
        criterion = torch.nn.MSELoss()
        print(f'Using mse loss')
    elif loss == 'ssim':
        criterion = SSIM_Loss(data_range=1.0, win_size=5,  size_average=True, channel=3)
        print(f'Using ssim loss')
    else:
        criterion = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)
        print(f'Using mse ms_ssim')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    return model, criterion, optimizer

# stores the results and metrics after training
def storeModelResults(modelpath, lossPerEpoch, trainTime, preprDict, model, epoch, optimizer, batchSize, color, lossFunc):
    print(f'Saving model...')
    if 'miscInfo' in lossPerEpoch:
        trainTime = trainTime + lossPerEpoch['miscInfo']['trainTime']
        for prep, val in preprDict.items():
            if lossPerEpoch['miscInfo']['preprocessing'][prep] is not val:
                oldVal = lossPerEpoch['miscInfo']['preprocessing'][prep]
                warnings.warn(f'Mismatch with existing preprocessing metrics. {prep} was {oldVal} but now is {val}', stacklevel=2)
            if lossPerEpoch['miscInfo']['batchSize'] is not batchSize:
                oldBatchSize = lossPerEpoch['miscInfo']['batchSize']
                warnings.warn(f'Mismatch with existing batchSize metric. batchSize was {oldBatchSize} but now is {batchSize}. Old batchSize will be overwritten', stacklevel=2)
            if lossPerEpoch['miscInfo']['color'] is not color:
                oldColor = lossPerEpoch['miscInfo']['color']
                warnings.warn(f'Mismatch with existing color metric. color was {oldColor} but now is {color}. Old color will be overwritten', stacklevel=2)
    lossPerEpoch['miscInfo'] = {
        'trainTime': trainTime, 
        'preprocessing': preprDict, 
        'modelName': modelpath.split('/')[-1], 
        'trainDate': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
        'batchSize': batchSize, 
        'color': color, 
        'lossFunc': lossFunc}
    metricFileName = f'models/metrics_{ modelpath.split("/")[-1].replace(".txt", "") }.json'
    saveTrainMetrics(metricFileName, lossPerEpoch)
    print(f'{trainTime = }min')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metricDict': lossPerEpoch
            }, modelpath)
    print(f'Done saving model to: {modelpath} and metrics to: {metricFileName}')

# applies blur to a single patch
def blurMapWrapper(patch):
    blurredPatch = cv2.GaussianBlur(patch, (5,5), cv2.BORDER_DEFAULT)
    return blurredPatch

# effifciently applies blur to list of patches
def blurPatches(patchList, color):
    vFuncBlur = np.vectorize(blurMapWrapper, signature='(m,n,c)->(m,n,c)')
    blurredPatchList = vFuncBlur(np.array(patchList))
    return blurredPatchList

def quantizeMapWrapper(patch, color):
    bgrPatch = patch
    bgrPatch = cv2.cvtColor(patch, ColorMappingGDict[color]['inverse'])
    rgbPatch = cv2.cvtColor(bgrPatch, cv2.COLOR_BGR2RGB)
    (h, w) = bgrPatch.shape[:2]
    labPatch = cv2.cvtColor(rgbPatch, cv2.COLOR_RGB2LAB)
    labPatch = labPatch.reshape((labPatch.shape[0] * labPatch.shape[1], 3))
    clt = MiniBatchKMeans(16)
    labels = clt.fit_predict(labPatch)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h, w, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    quant = cv2.cvtColor(quant, ColorMappingGDict[color]['mapping'])
    return quant

def quantizePatches(patchList, color):
    quantizedPatchList = np.array([quantizeMapWrapper(x, color) for x in patchList])
    return quantizedPatchList
    

