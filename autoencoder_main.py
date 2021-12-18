import torch
import numpy as np
import cv2
import os
import argparse
import time
import datetime
import matplotlib.pyplot as plt
import colorsys

#our own modules
import utils.utils as utils
import utils.autoencoder_boilerplate as ae

device = utils.pickBestDevice()

def loadModel(modelpath):
    print("Loading model...")
    model = utils.loadModel(modelpath)
    model.to(device)
    patchesList, mappingsList, resolutionsList, imageNamesList, tileCountsList = utils.loadImages(file_name='/0-B02.png', size=5)
    dataloader = utils.dataLoadWrapper(utils.flattenListOfLists(patchesList))
    print("Number of patch batches: ", len(dataloader))

    #forward pass has to be done for each batch
    def atEachBatch(batch):
        return model(batch)

    #receives the original and reconstructed patch tensors,
    #and returns a list of nparray patches which has to be the same length that you specify when calling puzzleBackTogether
    #this allows you to do different computations per patch that then get merged into one final image
    def atEachPatch(patchTensor,reconstructedPatchTensor):
        originalPatch = utils.convertTensorToImage(patchTensor)
        reconstructedPatch = utils.convertTensorToImage(reconstructedPatchTensor)
        diffPatch = np.subtract(originalPatch,reconstructedPatch)
        diffPatchHSV = cv2.cvtColor(diffPatch, cv2.COLOR_RGB2HSV)
        mse = np.mean(np.square(diffPatchHSV[:,:,0]))*2
        msePatch = np.ones(diffPatch.shape)*(mse,mse,100)
        return [originalPatch, reconstructedPatch, diffPatch,msePatch]

    # receives a list of puzzled together images and the name of the current original image
    def atEachImage(canvasArray,imageName):
        plt.rcParams["figure.figsize"] = (15,5)
        for i in range(len(canvasArray)):
            plt.subplot(1,len(canvasArray),i+1)
            plt.axis('off')
            plt.grid(visible=None)
            plt.imshow(canvasArray[i])
        plt.show()
    #I wrote this really weird wrapper to puzzle the patches back together to an image
    #it takes the above defined functions as arguments and merges the patches to the whole image again (for n images)
    utils.puzzleBackTogether(atEachBatch,atEachPatch,atEachImage,dataloader,resolutionsList,mappingsList,tileCountsList,imageNamesList,4)
    return model 

#validationInterval denotes how many epoches need to pass for one validation loop to occur
def trainModel(modelpath, epochs, batchSize, validate=False):
    print("Training model...")
    patchesList, mappingsList, resolutionsList, imageNamesList, tileCountsList = utils.loadImages()
    flattenedPatchList = utils.flattenListOfLists(patchesList)
    batchSize = len(flattenedPatchList) if batchSize == -1 else batchSize
    print(f'{batchSize = }')
    dataloader = utils.dataLoadWrapper(flattenedPatchList, batchSize)
    if validate:
        valpatchesList, valmappingsList, valresolutionsList, valimageNamesList, valtileCountsList = utils.loadImages(file_name='/4-B02.png', base_dir='data/validation/', size=5)
        valdataloader = utils.dataLoadWrapper(utils.flattenListOfLists(valpatchesList), batchSize)
    else:
        meanValLoss = 0.0
    model = ae.Autoencoder()
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model. parameters (), lr=1e-3, weight_decay=1e-5)
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
            del item
            del recon
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epochTrainLoss += loss.item()
        meanTrainLoss = epochTrainLoss / len(dataloader)
        trainOutputStr = f'Epoch: {e}, Train Loss: {(meanTrainLoss):.4f}'
        # validation looü
        valOutputStr = ''
        if validate:
            epochValidLoss = 0.0
            model.eval()
            with torch.no_grad():
                for val_img in valdataloader:
                    val_img = val_img.to(device)
                    recon = model (val_img)
                    loss = criterion (recon, val_img)
                    del val_img
                    del recon
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    epochValidLoss += loss.item()
            meanValLoss = epochValidLoss / len(valdataloader)
            valOutputStr = f' Validation Loss: {(meanValLoss):.4f}'
        print(trainOutputStr+valOutputStr)
        lossPerEpoch[e] = {'train_loss': meanTrainLoss, 'valid_loss': meanValLoss}
    print("training time: {}min".format((time.time()-start)/60))
    torch.save(model.state_dict(), modelpath)
    loadModel(modelpath)
    return model, lossPerEpoch

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    parser = argparse.ArgumentParser(description='Train autoencoder')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--model', default="models/testmodel.txt", type=str, help='Path to the model file')
    parser.add_argument('--batchSize', default=1024, type=int, help='Batch size during training and validation. Set to -1 to take the complete trainingset size')
    parser.add_argument('--validate', default=False, action='store_true', help='Whether validation should be done during training')
    parser.add_argument('--train', dest='train', default=False, action='store_true', help='Dont load the model, train it instead')
    args = parser.parse_args()

    if args.train:
        trainModel(args.model, args.epochs, args.batchSize, validate=args.validate)
    else:
        loadModel(args.model)
