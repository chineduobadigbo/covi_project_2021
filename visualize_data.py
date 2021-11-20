import cv2
import json
import os
import numpy as np

BASE_VAL_PATH = "data/validation/"


def combineImages(foldername):
    folderpath = BASE_VAL_PATH+foldername
    f = open(folderpath+"/homographies.json",)
    homographies = json.load(f)
    f.close()
    print(folderpath)

    image = cv2.imread(folderpath+"/3-B01.png")
    resultImg = np.zeros(image.shape)

    for root, dirs, files in os.walk(folderpath, topdown=False):
        for name in files:
            print(name[0])
            if name != "labels.json" and name[0] == "3":
                image = cv2.imread(folderpath+"/"+name)
                homography = np.array(homographies[name.replace(".png","")])
                warped_image = cv2.warpPerspective(image,homography,image.shape[:2])
                resultImg = cv2.addWeighted(warped_image, 0.1, resultImg, 1, 0.0, dtype=cv2.CV_32F).astype(np.uint8)

    return resultImg
    #cv2.imshow("combined",resultImg)
    #cv2.waitKey(0)

def showImageWithAnnotations(foldername,withCombined):
    folderpath = BASE_VAL_PATH+foldername
    if not withCombined:
        image = cv2.imread(folderpath+"/3-B01.png")
    else:
        image = combineImages(foldername)
    f = open(folderpath+"/labels.json",)
    gtlabels = json.load(f)
    f.close()
    for bb in gtlabels:
        x,y,w,h = bb
        cv2.rectangle(image,(x, y),(x+w, y+h),(0,0,255),5) 

    cv2.imshow(foldername,image)
    cv2.waitKey(0)


if __name__ == "__main__":
    for root, dirs, files in os.walk(BASE_VAL_PATH, topdown=False):
        for name in dirs:
            print(name)
            showImageWithAnnotations(name,True)