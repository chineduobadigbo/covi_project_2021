import torch
import numpy as np
import cv2
import os
import argparse
import time
import datetime
import matplotlib.pyplot as plt
import colorsys
from pathlib import Path

#our own modules
import utils.utils as utils
SAVE_PATH = 'data/train_images_mse/'
SAVE_PATH = 'data/train_images_mse/'
VAL_SAVE_PATH = 'data/val_train_images_mse/'
BASE_TRAIN_PATH = 'data/train/'

imageOrder = ['original', 'rebuilt', 'diffed', 'mse']

device = utils.pickBestDevice()

#forward pass has to be done for each batch
def atEachBatch(model, batch):
    return model(batch)

#receives the original and reconstructed patch tensors,
#and returns a list of nparray patches which has to be the same length that you specify when calling puzzleBackTogether
#this allows you to do different computations per patch that then get merged into one final image
def atEachPatch(patchTensor,reconstructedPatchTensor):
    originalPatch = utils.convertTensorToImage(patchTensor)
    reconstructedPatch = utils.convertTensorToImage(reconstructedPatchTensor)
    # diffPatch = np.subtract(utils.blurMapWrapper(originalPatch),reconstructedPatch)
    diffPatch = np.subtract(originalPatch,reconstructedPatch)
    diffPatchHSV = cv2.cvtColor(diffPatch, cv2.COLOR_RGB2HSV)
    mse = np.mean(np.square(diffPatchHSV[:,:,0]))*2
    msePatch = np.ones(diffPatch.shape)*(mse,mse,100)
    return [originalPatch, reconstructedPatch, diffPatch,msePatch]

# receives a list of puzzled together images and the name of the current original image
def atEachImage(canvasArray,imageName):
    print(f'{imageName = }')
    sub_folder = imageName.split('/')[0]
    imgName = imageName.split('/')[-1]
    writeDir = SAVE_PATH+'{}'.format(sub_folder)
    Path(writeDir).mkdir(parents=True, exist_ok=True)
    plt.rcParams['figure.figsize'] = (15,5)
    for i, (canvas, imgType) in enumerate(zip(canvasArray, imageOrder)):
        cv2.imwrite(writeDir+'/'+f'{imgType}_{imgName}', canvas)
        plt.subplot(1,len(canvasArray),i+1)
        plt.axis('off')
        plt.grid(visible=None)
        plt.imshow(canvas)
    plt.show()

def loadModel(modelpath):
    print('Loading model...')
    model = utils.loadModel(modelpath)
    model.to(device)
    patchesList, mappingsList, resolutionsList, imageNamesList, tileCountsList = utils.loadImages(file_name='/4-B02.png', baseDir=BASE_TRAIN_PATH, size=5)
    prepPatches = applyPreprocessingFuncs(utils.flattenListOfLists(patchesList))
    dataloader = utils.dataLoadWrapper(prepPatches)
    print(f'Number of patch batches: {len(dataloader)}')

    #I wrote this really weird wrapper to puzzle the patches back together to an image
    #it takes the above defined functions as arguments and merges the patches to the whole image again (for n images)
    utils.puzzleBackTogether(atEachBatch,atEachPatch,atEachImage,dataloader,resolutionsList,mappingsList,tileCountsList,imageNamesList,4, model)
    return model 

# loops over a list of provided functions 
# which apply some preprocessing on the patchesList
def applyPreprocessingFuncs(patchesList):
    preprocessingList = [utils.blurPatches, utils.quantizePatches]
    for prepr in preprocessingList:
        patchesList = prepr(patchesList)
    return patchesList

def trainModel(modelpath, epochs, batchSize, validate=False):
    print('Training model...')
    patchesList, mappingsList, resolutionsList, imageNamesList, tileCountsList = utils.loadImages()
    flattenedpatchesList = utils.flattenListOfLists(patchesList)
    batchSize = len(flattenedpatchesList) if batchSize == -1 else batchSize
    print(f'{batchSize = }')
    dataloader = utils.dataLoadWrapper(flattenedpatchesList, batchSize)
    if validate:
        valpatchesList, valmappingsList, valresolutionsList, valimageNamesList, valtileCountsList = utils.loadImages(file_name='/4-B02.png', base_dir='data/validation/', size=5)
        valdataloader = utils.dataLoadWrapper(utils.flattenListOfLists(valpatchesList), batchSize)
    else:
        meanValLoss = 0.0
    model, criterion, optimizer = utils.createModelOptimizer()
    lossPerEpoch = {}
    start = time.time()
    for e in range(epochs):
        trainOutputStr = ''
        epochTrainLoss = 0.0
        model.train()
        for (item) in dataloader:
            item = item.to(device)
            recon = model(item)
            loss = criterion(recon, item)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epochTrainLoss += loss.item()
        meanTrainLoss = epochTrainLoss / len(dataloader)
        trainOutputStr = f'Epoch: {e}, Train Loss: {(meanTrainLoss):.4f}'
        # validation loop
        valOutputStr = ''
        if validate:
            epochValidLoss = 0.0
            model.eval()
            with torch.no_grad():
                for (valItem) in valdataloader:
                    valItem = valItem.to(device)
                    recon = model (valItem)
                    loss = criterion (recon, valItem)
                    epochValidLoss += loss.item()
            meanValLoss = epochValidLoss / len(valdataloader)
            valOutputStr = f' Validation Loss: {(meanValLoss):.4f}'
        print(trainOutputStr+valOutputStr)
        lossPerEpoch[e] = {'train_loss': meanTrainLoss, 'valid_loss': meanValLoss}
    trainTime = (time.time()-start)/60
    lossPerEpoch['miscInfo'] = {'trainTime': trainTime}
    print(f'{trainTime = }min')
    torch.save(model.state_dict(), modelpath)
    loadModel(modelpath)
    return model, lossPerEpoch

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    parser = argparse.ArgumentParser(description='Train autoencoder')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--model', default='models/testmodel.txt', type=str, help='Path to the model file')
    parser.add_argument('--batchSize', default=1024, type=int, help='Batch size during training and validation. Set to -1 to take the complete trainingset size')
    parser.add_argument('--validate', default=False, action='store_true', help='Whether validation should be done during training')
    parser.add_argument('--train', dest='train', default=False, action='store_true', help='Dont load the model, train it instead')
    args = parser.parse_args()

    if args.train:
        model, lossPerEpoch = trainModel(args.model, args.epochs, args.batchSize, validate=args.validate)
        metricFileName = 'models/metrics.json'
        utils.saveTrainMetrics(metricFileName, lossPerEpoch)
    else:
        loadModel(args.model)
