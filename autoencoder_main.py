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
import timeit
import json
import datetime
import sys
import signal
# import colour

from torch.utils import data

#our own modules
import utils.utils as utils
import utils.boundingbox as bb

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
def atEachPatch(patchTensor,reconstructedPatchTensor, color):
    originalPatch = utils.quantizeMapWrapper(utils.convertTensorToImage(patchTensor, color), color)
    reconstructedPatch = utils.blurMapWrapper(utils.convertTensorToImage(reconstructedPatchTensor, color))
    diffPatch = np.subtract(originalPatch,reconstructedPatch)
    originalPatchLAB = cv2.cvtColor(originalPatch, cv2.COLOR_BGR2LAB)
    reconstructedPatchLAB = cv2.cvtColor(reconstructedPatch, cv2.COLOR_BGR2LAB)
    # 3D LAB color histogram with 8 bins per channel, yielding a 512-dim feature vector
    originalHist = cv2.calcHist([originalPatchLAB], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    originalHist = cv2.normalize(originalHist, originalHist).flatten()
    reconstructedHist = cv2.calcHist([reconstructedPatchLAB], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    reconstructedHist = cv2.normalize(reconstructedHist, reconstructedHist).flatten()


    #mse = np.mean(colour.delta_E(cv2.cvtColor(originalPatchLAB.astype(np.float32) / 255, cv2.COLOR_RGB2LAB), cv2.cvtColor(reconstructedPatchLAB.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)))**2
    # histDiff = ((cv2.compareHist(originalHist, reconstructedHist, cv2.HISTCMP_CHISQR))+1)**2
    histDiff = ((cv2.compareHist(originalHist, reconstructedHist, cv2.HISTCMP_CHISQR)))*100
    
    # mse = np.square(np.mean(diffPatchLAB[:,:,1]))#+np.square(np.mean(originalLAB[:,:,2]))
    #print(mse)
    # mse = np.mean(np.square(diffPatchHSV[:,:,0]))
    # print(f'{location}, hist_diff: {histDiff}')
    mse = histDiff
    msePatch = np.ones(originalPatch.shape)*(mse,mse,mse)
    return [originalPatch, reconstructedPatch, diffPatch,msePatch]

# receives a list of puzzled together images and the name of the current original image
def atEachImage(canvasArray,imageName):
    print(f'{imageName = }')
    sub_folder = imageName.split('/')[0]
    imgName = imageName.split('/')[-1]
    writeDir = SAVE_PATH+sub_folder
    Path(writeDir).mkdir(parents=True, exist_ok=True)
    plt.rcParams['figure.figsize'] = (15,5)
    for i, (canvas, imgType) in enumerate(zip(canvasArray, imageOrder)):
        cv2.imwrite(writeDir+'/'+f'{imgType}_{imgName}', canvas)

def loadModel(modelpath, preprDict, color, official=False):
    print('Loading model...')
    model, *_ = utils.loadModel(modelpath)
    model.to(device)

    if official:
        print('Loading official validation set...')
        bbmodel = bb.getModel()
        boundingBoxes = {}
        global middleImage
        middleImage = None
        def customAtEachImage(canvasArray,imageName):
            print(f'{imageName = }')
            originalImage = canvasArray[0]
            recreatedImage = canvasArray[1]
            mseImage = canvasArray[3]
            diffImage = canvasArray[2]
            #originalLAB = cv2.cvtColor(originalImage, cv2.COLOR_RGB2HSV)
            #completeMSE = np.square(np.mean(originalLAB[:,:,1]))#+np.square(np.mean(originalLAB[:,:,2]))
            #mseImage -= int(completeMSE)
            reducedMSE = bb.reduceDimensionality(mseImage)
            reducedMSE *= reducedMSE
            reducedMSE *= reducedMSE
            #cv2.imshow('mse', bb.increaseDimensionality(reducedMSE))
            boundingbox = bb.getBoundingBox(bbmodel, reducedMSE)
            #print(boundingbox)

            boundingBoxes[imageName] = boundingbox
            if imageName.endswith("B01.png"):
                global middleImage
                middleImage = originalImage
            
            cv2.imshow('image', originalImage)
            cv2.imshow('mse', diffImage)
            #bb.showImageWithBoundingBox(originalImage, boundingbox)
            cv2.waitKey(0)

        folderDict = utils.loadValidationImages(color)
        validationDict = {}
        for datapoint,_ in folderDict.items():
            print(datapoint)
            for timepoint,_ in folderDict[datapoint].items(): #for now we only compute the middle timepoint(3)
                #print("\t",timepoint)
                patchesList, mappingsList, resolutionsList, imageNamesList, tileCountsList = folderDict[datapoint][timepoint]
                prepPatches = applyPreprocessingFuncs(utils.flattenListOfLists(patchesList), preprDict, color)
                dataloader = utils.dataLoadWrapper(prepPatches)
                #print(f'Number of patch batches: {len(dataloader)}')
                utils.puzzleBackTogether(atEachBatch,atEachPatch,customAtEachImage,dataloader,resolutionsList,mappingsList,tileCountsList,imageNamesList,4, model, color)
                combinedBoxes = bb.combineBoundingBoxes(datapoint,boundingBoxes,middleImage)
                validationDict[datapoint] = combinedBoxes
                boundingBoxes = {}
                middleImage = None
        
        with open('validation_boxes.json', 'w') as fp:
            json.dump(validationDict, fp)
        print("FINISHED")

    else:
        patchesList, mappingsList, resolutionsList, imageNamesList, tileCountsList = utils.loadImages(color, fileName='/4-B02.png', baseDir=BASE_TRAIN_PATH, size=5)
        prepPatches = applyPreprocessingFuncs(utils.flattenListOfLists(patchesList), preprDict, color)
        dataloader = utils.dataLoadWrapper(prepPatches)
        print(f'Number of patch batches: {len(dataloader)}')

        #I wrote this really weird wrapper to puzzle the patches back together to an image
        #it takes the above defined functions as arguments and merges the patches to the whole image again (for n images)
        utils.puzzleBackTogether(atEachBatch,atEachPatch,atEachImage,dataloader,resolutionsList,mappingsList,tileCountsList,imageNamesList,4, model, color)
        return model 

# loops over a list of provided functions 
# which apply some preprocessing on the patchesList
def applyPreprocessingFuncs(patchesList, preprDict, color):
    preprMapDict = {'blur': utils.blurPatches, 'quantize': utils.quantizePatches} # holds the corresponding function for each preprocessing step
    # build list with functions based on which prepr steps are True
    preprocessingList = []
    for step, val in preprDict.items():
        if val:
            print(f'{step} was added to preprocessing pipeline')
            preprocessingList.append((step, preprMapDict[step]))
    # loop over chosen functions and and apply preprocessing
    # these loops coudl be reduced to a single loop, this allows for a nicer log output though
    if len(preprocessingList) == 0:
        print('No preprocessing steps selected')
    for step, preprFunc in preprocessingList:
        print(f'Applying {step}...')
        t1 = timeit.default_timer()
        patchesList = preprFunc(patchesList, color)
        elapsed = timeit.default_timer() - t1
        print(f'{step} took {elapsed}s for {len(patchesList)} patches')
    return patchesList

def trainModel(modelpath, epochs, batchSize, preprDict, color, validate=False, continueModel=True):
    print('Loading training images...')
    patchesList, mappingsList, resolutionsList, imageNamesList, tileCountsList = utils.loadImages(color)
    flattenedpatchesList = utils.flattenListOfLists(patchesList)
    batchSize = len(flattenedpatchesList) if batchSize == -1 else batchSize
    prepPatches = applyPreprocessingFuncs(utils.flattenListOfLists(patchesList), preprDict, color)
    print(f'{batchSize = }')
    dataloader = utils.dataLoadWrapper(prepPatches, batchSize)
    if validate:
        print('Loading validation images...')
        valpatchesList, valmappingsList, valresolutionsList, valimageNamesList, valtileCountsList = utils.loadImages(color, fileName='/4-B02.png', baseDir='data/train/', size=5)
        valPrepPatches = applyPreprocessingFuncs(utils.flattenListOfLists(valpatchesList), preprDict, color)
        valdataloader = utils.dataLoadWrapper(valPrepPatches, batchSize)
    else:
        meanValLoss = 0.0
    model, criterion, optimizer = utils.createModelOptimizer()
    existingModel = Path(modelpath)
    lossPerEpoch = {}
    
    # check if model already exists an if training should be continued
    continueOutput = f'Continueing training if model {modelpath} already exists' if continueModel else f'Not continueing training if model {modelpath} already exists'
    print(continueOutput)
    lastEpoch = 0
    if existingModel.is_file() and continueModel:
        print(f'Loading existing model {modelpath}')
        model, checkpoint, lastEpoch, lossPerEpoch, optimizer = utils.loadModel(modelpath, optimizer=optimizer)
        lastEpoch += 1
    signal.signal(signal.SIGINT, signal.default_int_handler)
    print('Training model...')
    start = timeit.default_timer()
    try:
        for e in range(lastEpoch, epochs+lastEpoch):
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
    except KeyboardInterrupt:
        print(e)
        trainTime = (timeit.default_timer()-start)/60
        utils.storeModelResults(modelpath, lossPerEpoch, trainTime, preprDict, model, e, optimizer, batchSize, color)
        sys.exit(0)
    trainTime = (timeit.default_timer()-start)/60
    utils.storeModelResults(modelpath, lossPerEpoch, trainTime, preprDict, model, e, optimizer, batchSize, color)
    loadModel(modelpath, preprDict, color)
    return model, lossPerEpoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train autoencoder')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--model', default='models/testmodel.pt', type=str, help='Path to the model file')
    parser.add_argument('--color', default='RGB', type=str, help='The color space used for training (RGB, HSV, LAB)')
    parser.add_argument('--batchSize', default=-1, type=int, help='Batch size during training and validation. Set to -1 to take the complete trainingset size')
    parser.add_argument('--validate', default=False, action='store_true', help='Whether validation should be done during training')
    parser.add_argument('--train', dest='train', default=False, action='store_true', help='Dont load the model, train it instead')
    parser.add_argument('--blur', dest='blur', default=False, action='store_true', help='Apply blur during preprocessing')
    parser.add_argument('--quantize', dest='quantize', default=False, action='store_true', help='Apply quantization during preprocessing')
    parser.add_argument('--official', dest='official', default=False, action='store_true', help='Use the official validation dataset, compute bounding boxes and save them')
    parser.add_argument('--continueModel', dest='continueModel', default=False, action='store_true', help='Continue training if model already exists')
    args = parser.parse_args()
    preprDict = {'blur': args.blur, 'quantize': args.quantize}
    if args.train:
        model, lossPerEpoch = trainModel(args.model, args.epochs, args.batchSize, preprDict, args.color, validate=args.validate, continueModel=args.continueModel)
    else:
        loadModel(args.model, preprDict, args.color, official=args.official)
