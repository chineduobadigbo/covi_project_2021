import cv2
import json
import os
import numpy as np
from pathlib import Path
import argparse
import copy

BASE_VAL_PATH = "data/validation/"
SAVE_PATH = "data/validation_with_labels/"

def combineImages(foldername):
    folderpath = BASE_VAL_PATH+foldername
    with open(folderpath+"/homographies.json",) as f:   
        homographies = json.load(f)
    with open(folderpath+"/labels.json",) as f:   
        labels = json.load(f)

    image = cv2.imread(folderpath+"/3-B01.png")
    resultImg = np.zeros(image.shape)

    for root, dirs, files in os.walk(folderpath, topdown=False):
        for name in files:
            if ".json" not in name and name[0] == "3":
                image = cv2.imread(folderpath+"/"+name)
                print('Image Name: {}'.format(folderpath+"/"+name))
                homography = np.array(homographies[name.replace(".png","")])
                warped_image = cv2.warpPerspective(image,homography,image.shape[:2])
                image_with_labels = add_labels(warped_image, labels)
                resultImg = cv2.addWeighted(warped_image, 0.1, resultImg, 1, 0.0, dtype=cv2.CV_32F).astype(np.uint8)
                write_dir = SAVE_PATH+foldername
                print(write_dir)
                # create dir if not exists
                Path(write_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(write_dir+'/'+name, image_with_labels)

    return resultImg
    #cv2.imshow("combined",resultImg)
    #cv2.waitKey(0)

def add_labels(image, labels):
    cp_image = copy.deepcopy(image)
    for bb in labels:
        x,y,w,h = bb
        cv2.rectangle(cp_image,(x, y),(x+w, y+h),(0,0,255),5) 
    return cp_image

def showImageWithAnnotations(foldername,withCombined):
    folderpath = BASE_VAL_PATH+foldername
    if not withCombined:
        image = cv2.imread(folderpath+"/3-B01.png")
    else:
        image = combineImages(foldername)
    with open(folderpath+"/labels.json",) as f:   
        gtlabels = json.load(f)
    print('Labels: {}'.format(gtlabels))
    for bb in gtlabels:
        x,y,w,h = bb
        cv2.rectangle(image,(x, y),(x+w, y+h),(0,0,255),5) 


    if SHOW_IMAGES:
        cv2.imshow(foldername,image)
        cv2.waitKey(0)
    write_dir = SAVE_PATH+foldername
    cv2.imwrite(write_dir+'/merged.png', image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Validation Data. Results are written to "{}"'.format(SAVE_PATH))
    parser.add_argument('--show', dest='show', default=False, action='store_true',
                    help='show the images instead of just saving them')
    args = parser.parse_args()
    global SHOW_IMAGES
    SHOW_IMAGES = args.show
    for root, dirs, files in os.walk(BASE_VAL_PATH, topdown=False):
        for name in dirs:
            print(name)
            showImageWithAnnotations(name,True)