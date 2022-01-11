import cv2
import torch
import numpy as np
import os
import pathlib
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchinfo import summary
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.geometry import Point
from shapely.ops import cascaded_union, unary_union
from itertools import combinations
import scipy.stats as stats
import random


class BBDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self): #Denotes the total number of samples
        return len(self.X)

    def __getitem__(self, index): #Generates one sample of data
        tX = self.XTransform(self.X[index])
        #ty = self.yransform(self.y[index])
        return tX.float(), self.y[index]

    XTransform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        ])
    
    yTransform = T.Compose([
        T.ToTensor(),
        ])



class BBNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
                #the stride parameter is describing how much the resoltion decreases in each step
                nn.Conv2d(1, 8, 3, stride=2, padding=1), # N, 8, 8, 8
                nn.ReLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1), # N, 16, 4, 4
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(16*4*4, 32),
                nn.Sigmoid(),
                nn.Linear(32, 4),
                nn.Sigmoid()
            )

    def forward(self, X):
        return self.model(X)

def generateTrainingImage(x,y,w,h,flipX,flipY,treeX,treeY,treeWidth,treeHeight,blank):
    #print("x: {}, y: {}, w: {}, h: {}, r: {}".format(x,y,w,h,r))

    canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)
    if blank:
        return canvas, (float(0),float(0),float(0),float(0))
    else:
        human = cv2.imread("./utils/human.png")
        treetop = cv2.imread("./utils/treetop2.png")
        transformed = cv2.resize(human, (w,h), interpolation = cv2.INTER_AREA)
        transformedTreetop = cv2.resize(treetop, (treeWidth,treeHeight), interpolation = cv2.INTER_AREA)
        if flipX:
            transformed = cv2.flip(transformed, 1)
        if flipY:
            transformed = cv2.flip(transformed, 0)


        canvas = superimposeImage(canvas, transformedTreetop, treeX, treeY, 0.05)
        canvas[x:x+transformed.shape[0], y:y+transformed.shape[1]] = transformed
        #cv2.rectangle(canvas, (y,x), (y+transformed.shape[1],x+transformed.shape[0]), (0,255,0), 2)
        #cv2.rectangle(canvas, (100,200),(150,250), (0,255,0), 2)
        #cv2.imshow("human", canvas)
        #cv2.waitKey(0)

        cWidth = canvas.shape[1]
        cHeight = canvas.shape[0]

        #bounding box in form (x,y,w,h,p) p denotes wheter a human is present on the training image
        return canvas, (float(x)/cWidth,float(y)/cHeight,float(x+transformed.shape[0])/cWidth,float(y+transformed.shape[1])/cHeight)

def superimposeImage(canvas,image,x,y, intensity):
    canvasWidth = canvas.shape[1]
    canvasHeight = canvas.shape[0]
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]


    availableWidth = min(canvasWidth-x, imageWidth)
    availableHeight = min(canvasHeight-y, imageHeight)

    image = image*intensity
    image = image.astype(np.uint8)

    canvas[y:y+availableHeight, x:x+availableWidth] += image[0:availableHeight, 0:availableWidth]
    return canvas


def generateTrainingBatch(batchSize,split):

    data = []
    labels = []

    maxWidth = 120
    maxHeight = 120
    print("starting to generate training data")
    for i in range(int(batchSize*split)):
        x = np.random.randint(192+64, 832-64-maxWidth)
        y = np.random.randint(0, 1024-maxHeight) #these weird numbers are picked because the training data is black on top and on the bottom
        w = np.random.randint(32, maxWidth) #and x and y are flipped for some reason...
        h = np.random.randint(32, maxHeight)
        flipX = bool(random.getrandbits(1))
        flipY = bool(random.getrandbits(1))
        treeX = np.random.randint(0, 1024)
        treeY = np.random.randint(0, 1024)
        treeWidth = np.random.randint(500, 700)
        treeHeight = np.random.randint(500, 700)

        canvas, boundingbox = generateTrainingImage(x,y,w,h,flipX,flipY,treeX,treeY,treeWidth,treeHeight,False)
        sliced = sliceAndComputeFakeMSE(canvas)
        reduced = reduceDimensionality(sliced)
        data.append(reduced)
        labels.append(boundingbox)
        #cv2.imshow("reduced", reduced)
        #showImageWithBoundingBox(sliced, boundingbox)

    for i in range(int(batchSize*(1-split))):
        canvas, boundingbox = generateTrainingImage(0,0,0,0,False,False,treeX,treeY,treeWidth,treeHeight,True)
        sliced = sliceAndComputeFakeMSE(canvas)
        reduced = reduceDimensionality(sliced)
        data.append(reduced)
        labels.append(boundingbox)

    print("finished!")

    return data, labels

def showImageWithBoundingBox(image, boundingbox, original=None): #original denotes the bounding box label
    x1,y1,x2,y2 = boundingbox
    cWidth = image.shape[1]
    cHeight = image.shape[0]
    cv2.rectangle(image, (int(y1*cWidth),int(x1*cHeight)), (int(y2*cWidth),int(x2*cHeight)), (0,255,0), 2)
    if(original is not None):
        x1,y1,x2,y2 = original
        cv2.rectangle(image, (int(y1*cWidth),int(x1*cHeight)), (int(y2*cWidth),int(x2*cHeight)), (255,0,0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)

def reduceDimensionality(image):
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
        image = image / 255.0
    greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(greyscale, (16,16), interpolation = cv2.INTER_NEAREST)

def increaseDimensionality(tensor, isTensor = True):
    if isTensor:
        img = tensor.detach().numpy()
        img = np.moveaxis(img,0,2)
    else:
        img = tensor
    print("IMAGE SHAPE: {}".format(img.shape))
    #img = tensor
    img *= 255
    img = img.astype(np.uint8)
    img = cv2.resize(img, (1024,1024), interpolation = cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def sliceAndComputeFakeMSE(image):
    patchSize = (64,64)

    bottomBorder = 3
    topBorder = 12

    imgResolution = image.shape[:-1]
    patchCount = (int(imgResolution[0]/patchSize[0]),int(imgResolution[1]/patchSize[1]))
    for x in range(patchCount[0]):
        for y in range(patchCount[1]):
            xCoord = x*patchSize[0]
            yCoord = y*patchSize[1]
            patch = image[xCoord:xCoord+patchSize[0],yCoord:yCoord+patchSize[1],:]
            computedMSE = np.mean(np.square(patch[:,:,:]))*2

            a, b = 20, 255
            mu, sigma = 34, 15
            dist = stats.truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
            # #print(dist.rvs())
            # if mse <= 10:
            #     mse += np.random.random()*50 #add noise to pathes where no human is present
            # else:
            #     mse -= np.random.random()*10 #remove noise from pathes where human is present

            if computedMSE <= 10:
                mse = dist.rvs()
            else:
                mse = min(computedMSE + (dist.rvs()*1.3),140)


            if x <= bottomBorder or x >= topBorder:
                mse = 0
            #print("MSE: {}".format(mse))
            msebuffer = np.ones(patch.shape)*(mse,mse,mse)
            image[xCoord:xCoord+patchSize[0],yCoord:yCoord+patchSize[1],:] = msebuffer

    #cv2.imshow("fakeMSE", image)
    #cv2.waitKey(0)
    return image
    

def trainModel(dataloader):
    model = BBNet()
    summary(model, input_size=(1024,1, 16, 16))
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    for epoch in range(15000):
        for (X,y) in dataloader:
            optimizer.zero_grad()
            output = model(X).double()
            #print(output.shape)

            loss = criterion(output, y)

            # displayable = increaseDimensionality(X[0])
            # bb = y[0].detach().numpy()
            # showImageWithBoundingBox(displayable, output[0].detach().numpy())
            #print(loss)
            #print("output: ",output)
            #print("y: ",y)
            loss.backward()
            optimizer.step()
        print("epoch: {}, loss: {}".format(epoch, loss))

    #save model
    torch.save(model.state_dict(), "./utils/bbmodel.txt")

def testModel(dataloader):
    model = BBNet()
    model.load_state_dict(torch.load("./utils/bbmodel.txt"))
    model.eval()
    for (X,y) in dataloader:
        output = model(X).double()
        for i in range(len(X)):

            print(X[i].mean())
            displayable = increaseDimensionality(X[i])
            bb_output = output[i].detach().numpy()
            bb_original = y[i].detach().numpy()
            showImageWithBoundingBox(displayable, bb_output, bb_original)


def getModel():
    model = BBNet()
    model.load_state_dict(torch.load("./utils/bbmodel.txt"))
    model.eval()
    return model

def getBoundingBox(model, reducedImage):
    dataset = BBDataset(np.array([reducedImage]),np.array([[0,0,1,1]]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True,num_workers=0, pin_memory=True)
    for (X,y) in dataloader:
        output = model(X).double()
        return (output[0].detach().numpy())


def combineBoundingBoxes(dataDir,boxes,middleImage):

    debug = True

    returnList = []

    #warp the bounding boxes based on the homographies
    homographiesPath = "./data/validation/"+dataDir+"/homographies.json"
    with open(homographiesPath,) as f:   
        homographies = json.load(f)

        warpedPolygons = []
        for boxName in boxes.keys():
            imageTag = boxName.split("/")[1]
            imageTag = imageTag.split(".")[0]
            homography = np.array(homographies[imageTag])
            x1,y1,x2,y2 = boxes[boxName]

            #disregard boxes that are too small since they correspond to "no box" in the bb regressor
            boxArea = (x2-x1)*(y2-y1) 
            if boxArea > 0.005:
                ul = np.array([x1,y1,1])
                dr = np.array([x2,y2,1])
                ur = np.array([x2,y1,1])
                dl = np.array([x1,y2,1])

                polygon = np.array([ul,ur,dr,dl])
                cWidth = middleImage.shape[1]
                cHeight = middleImage.shape[0]
                polygon[:,0] = polygon[:,0]*cWidth
                polygon[:,1] = polygon[:,1]*cHeight
                warped = np.zeros((4,3))
                for i in range(4):
                    warped[i] = np.dot(homography,polygon[i])

                warpedPolygons.append(Polygon(warped))
                if debug:
                    middleImage = drawPolygon(middleImage,warped)


    #find groups with overlapping bounding boxes
    roiAreas = unary_union(warpedPolygons)
    if roiAreas.type != "MultiPolygon":
        roiAreas = MultiPolygon([roiAreas])

    groups = []
    for area in roiAreas:
        areaPolys = []
        for poly in warpedPolygons:
            if area.intersects(poly):
                areaPolys.append(poly)
        groups.append(areaPolys)
        nparea = np.array(list(zip(area.exterior.coords.xy)))
        middleImage = drawPolygon(middleImage,nparea.T,color=(0,0,255),thickness=4)

    #TODO: Something does not work correctly here, not all intersections with an group are found
    #merge find the intersection areas between groups which is used for the final bounding box
    for group in groups:
        if(len(group) == 1):
            intersections = MultiPolygon([group[0]])
            #npintersection = np.array(list(zip(group[0].exterior.coords.xy)))

        else:
            print("GROUP LENGTH: ",len(group))
            intersections = unary_union(
                [a.intersection(b) for a, b in combinations(group, 2)]
            )
            if intersections.type != "MultiPolygon":
                intersections = MultiPolygon([intersections])
            #npintersection = np.array(list(zip(intersection[0].exterior.coords.xy)))


        for intersection in intersections:
            npintersection = np.array(list(zip(intersection.exterior.coords.xy)))

            maxX = int(np.amax(npintersection[0]))
            minX = int(np.amin(npintersection[0]))
            maxY = int(np.amax(npintersection[1]))
            minY = int(np.amin(npintersection[1]))
            pt1 = (minX,minY)
            pt2 = (maxX,maxY)
            width = maxX-minX
            height = maxY-minY

            returnList.append((minX,minY,width,height))

            if debug:
                middleImage = drawPolygon(middleImage,npintersection.T,color=(0,0,255),thickness=4)
                cv2.rectangle(middleImage, pt1, pt2, (255,255,0), 4)

    if debug:
        cv2.imshow("debug",middleImage)
        cv2.waitKey(0)

    return returnList




def drawPolygon(image, points, color=(255,0,0), thickness=2):
    if points.shape[1] >= 3:
        points = points[:,0:2]
    # cWidth = image.shape[1]
    # cHeight = image.shape[0]
    # points[:,0] = points[:,0]*cWidth
    # points[:,1] = points[:,1]*cHeight
    points = points.astype(np.int32)

    cv2.polylines(image, [points], 1, color, thickness)
    return image

    # cv2.imshow("image", image)
    # cv2.waitKey(0)
        


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    parser = argparse.ArgumentParser(description='Train autoencoder')
    parser.add_argument('--train', action='store_true', help='Train model')
    args = parser.parse_args()  

    if args.train:
        print("Training model")
        data, labels = generateTrainingBatch(1024*8,1)
        dataset = BBDataset(np.array(data),np.array(labels))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True,num_workers=0, pin_memory=True)
        print("dataset generated")
        trainModel(dataloader)
    else:
        print("Testing model")
        data, labels = generateTrainingBatch(128,1)
        #dataset = BBDataset(np.array(data),np.array(labels))
        #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True,num_workers=0, pin_memory=True)
        #print("dataset generated")
        #testModel(dataloader)
        model = getModel()
        for reduced,label in zip(data,labels):
            bb = getBoundingBox(model,reduced)
            showImageWithBoundingBox(increaseDimensionality(reduced,False),bb,label)
    