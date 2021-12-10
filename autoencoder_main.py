import torch
import numpy as np
import cv2
import os
import argparse
import time
import datetime

#our own modules
import utils.utils as utils
import utils.autoencoder_boilerplate as ae

device = utils.pickBestDevice()

def loadModel(modelpath):
    print("Loading model...")
    model = utils.loadModel(modelpath)
    model.to(device)
    patchesList, mappingsList, resolutionsList, imageNamesList, tileCountsList = utils.loadImages()
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
        msePatch = np.ones(diffPatch.shape)*np.mean(np.square(diffPatch))
        return [originalPatch, reconstructedPatch, diffPatch,msePatch]

    # receives a list of puzzled together images and the name of the current original image
    def atEachImage(canvasArray,imageName):
        cv2.destroyAllWindows()
        for i in range(len(canvasArray)):
            canvas = canvasArray[i]
            cv2.imshow(imageName+"-"+str(i),canvas)
        cv2.waitKey(0)

    #I wrote this really weird wrapper to puzzle the patches back together to an image
    #it takes the above defined functions as arguments and merges the patches to the whole image again (for n images)
    utils.puzzleBackTogether(atEachBatch,atEachPatch,atEachImage,dataloader,resolutionsList,mappingsList,tileCountsList,imageNamesList,4)
            



def trainModel(modelpath, epochs):
    print("Training model...")
    patchesList, mappingsList, resolutionsList, imageNamesList, tileCountsList = utils.loadImages()
    dataloader = utils.dataLoadWrapper(utils.flattenListOfLists(patchesList))
    model = ae.Autoencoder()
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model. parameters (), lr=1e-3, weight_decay=1e-5)

    start = time.time()
    for e in range(epochs):
        epochTrainLoss = 0.0
        for (item) in dataloader:
            item.to(device)
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

        print("Epoch: ", e, " Loss: ", epochTrainLoss)

    print("training time: {}min".format((time.time()-start)/60))
    torch.save(model.state_dict(), modelpath)
    loadModel(modelpath)





if __name__ == "__main__":
    os.system('clear')
    parser = argparse.ArgumentParser(description='Train autoencoder')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--model', default="models/testmodel.txt", type=str, help='Path to the model file')
    parser.add_argument('--train', dest='train', default=False, action='store_true', help='Dont load the model, train it instead')
    args = parser.parse_args()

    if args.train:
        trainModel(args.model,args.epochs)
    else:
        loadModel(args.model)
