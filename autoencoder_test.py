from PIL.Image import SAVE
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU, Sigmoid
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import os
import sys
from pathlib import Path
import argparse
import copy
import time
import datetime
import pathlib
import itertools
import matplotlib
import copy

BASE_TRAIN_PATH = "data/train/"
SAVE_PATH = "data/train_images_mse/"
VAL_SAVE_PATH = "data/val_train_images_mse/"
if torch.cuda.is_available():
    print('Current device: {}'.format(torch.cuda.current_device()))
    print('Device Name: {}'.format(torch.cuda.get_device_name(0)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class PatchDataset(Dataset): #a dataset object has to be definied that specifies how PyTorch can access the training data

    def __init__(self, X):
        'Initialization'
        self.X = X
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.X[index]
        X = self.transform(image)
        return X

    transform = T.Compose([
        T.ToPILImage(),
        #T.Resize(image_size),
        T.ToTensor()])

class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            #the stride parameter is describing how much the resoltion decreases in each step
            nn. Conv2d(3, 16, 3, stride=2, padding=1), # N, 16, 32, 32
            nn. ReLU() ,
            nn. Conv2d(16, 32, 3 , stride=2, padding=1), # N, 32, 16, 16
            nn. ReLU(),
            nn. Conv2d(32, 64, 3 , stride=2, padding=1), # N, 64, 8, 8
            nn. ReLU(),
            #in the last step, we make the stride equal to the image resolution, thus reducing it to just one pixel
            nn. Conv2d(64, 128, 8) # N, 128, 1, 1 -> this results a simple 128d vector
        )

        self.decoder = nn.Sequential(
            nn. ConvTranspose2d(128, 64, 8), # N, 64, 8, 8
            nn. ReLU() ,
            nn. ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # N, 64, 8, 8
            nn. ReLU() ,
            nn. ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # N, 32, 16, 16
            nn. ReLU() ,
            nn. ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), # N, 3, 64, 64
            nn. Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def trainEncoder(data_loader, val_data_loader=None):
    start = time.time()
    model = Autoencoder()
    if torch.cuda.is_available():
        model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model. parameters (), lr=1e-3, weight_decay=1e-5)
    loss_per_epoch = {}
    for epoch in range (num_epochs) :
        epoch_train_loss = 0.0
        model.train()
        for (img) in data_loader:
            #print(img.shape)
            #img = img. reshape (-1, 28*28)
            img = img.to(device)
            recon = model (img)
            loss = criterion (recon, img)
            del img
            del recon
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        if val_data_loader != None:
            epoch_valid_loss = 0.0
            model.eval()
            with torch.no_grad():
                for val_img in val_data_loader:
                    val_img = val_img.to(device)
                    recon = model (val_img)
                    loss = criterion (recon, val_img)
                    del val_img
                    del recon
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    epoch_valid_loss += loss.item()
        if val_data_loader != None:
            print (f' Epoch: {epoch+1}, Train Loss: {(epoch_train_loss / len(data_loader)):.4f}, Validation Loss: {(epoch_valid_loss / len(val_data_loader)):.4f}')
            loss_per_epoch[epoch] = {'train_loss': epoch_train_loss / len(data_loader), 'valid_loss': epoch_valid_loss / len(val_data_loader)}
        else:
            print (f' Epoch: {epoch+1}, Train Loss: {(epoch_train_loss / len(data_loader)):.4f}')

    print("training time: {}min".format((time.time()-start)/60))
    return model, loss_per_epoch

def sliceImage(imagepath):
    image = cv2.imread(imagepath)
    patchSize = (64,64)
    imgResolution = image.shape[:-1]
    patchCount = (int(imgResolution[0]/patchSize[0]),int(imgResolution[1]/patchSize[1]))
    print(patchCount)
    patches = []
    mapping = [] #stores the positions for each patch on the original image
    for x in range(patchCount[0]):
        for y in range(patchCount[1]):
            xCoord = x*patchSize[0]
            yCoord = y*patchSize[1]
            patch = image[xCoord:xCoord+patchSize[0],yCoord:yCoord+patchSize[1],:]
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            #patch = patch.astype(np.float32)
            #patch /= 255. #convert image to float, so that every value is between 0 an 1
            if cv2.countNonZero(patch[::,0]) > 0: #discard all completely black patches
                patches.append(patch)
                mapping.append((xCoord,yCoord))
                # print(patch)
                # cv2.imshow(imagepath,patch)
                # cv2.waitKey(100)
    
    return patches,mapping,image.shape


def loadModel(name='testmodel.txt'):
    model = Autoencoder()
    model.load_state_dict(torch.load(name, map_location=device))
    return model

def convert_hsv(img):

    return img
    if len(img.shape) > 3:
        resh_img = np.moveaxis(img, 1, -1)
        print(resh_img.shape)
        new_list = np.zeros(resh_img.shape)
        for i in range(resh_img.shape[0]):
            new_list[i] = cv2.cvtColor(np.float32(resh_img[i]), cv2.COLOR_RGB2HSV)
        return new_list
    try:
        return cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2HSV)
    except:
        return cv2.cvtColor(np.float32(np.moveaxis(img, 0, 2)), cv2.COLOR_RGB2HSV)

def convert_rbg(img):
    return cv2.cvtColor(np.float32(img), cv2.COLOR_HSV2RGB)

# img can be list
def reshape_imgs(img):
    if img.shape[-1] != 3:
        if len(img.shape) > 3:
            resh_img = np.moveaxis(img, 1, -1)
        else:
            resh_img = np.moveaxis(img, 0, -1)
    else:
        resh_img = img
    return resh_img

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def compareImages(patches_list,model,mapping_list,resolution_list, image_name_list, save_folder):
    if torch.cuda.is_available():
        model.cuda()
    global_mse = []
    canvas_list = []
    rebuilt_list = []
    diffed_list = []
    nprecon_list = []
    new_patches = []
    hsv_compare_index = 0 # HUE channel
    print('Paches len {}'.format(len(patches_list)))
    print('mappings len {}'.format(len(mapping_list)))
    print('resolutions len {}'.format(len(resolution_list)))
    for dataloader, mapping, resolution in zip(patches_list, mapping_list, resolution_list):
        canvas = np.zeros(resolution)
        rebuilt = np.zeros(resolution)
        diffed = np.zeros(resolution)
        for (img) in dataloader:
            npimg = reshape_imgs(img.numpy())
            img = img.to(device)
            with torch.no_grad():
                recon = model (img)
            nprecon = reshape_imgs(recon.detach().cpu().numpy())
            mses = []
            for i in range(len(img)):

                location=mapping[i]

                patch=npimg[i]
                #patch = np.moveaxis(patch, 0, 2)
                canvas[location[0]:location[0]+patch.shape[0],location[1]:location[1]+patch.shape[1],:]=patch

                reconpatch = nprecon[i]
                #reconpatch = np.moveaxis(reconpatch, 0, 2)

                mse = normalize_data(np.square(np.subtract(convert_hsv(patch)[:,:,hsv_compare_index], convert_hsv(reconpatch)[:,:,hsv_compare_index]))).mean()
                mses.append(mse)
            canvas_list.append(canvas)
            rebuilt_list.append(rebuilt)
            global_mse.append(np.mean(mses))
            nprecon_list.append(nprecon)
            new_patches.append(patch)
            diffed_list.append(diffed)
    mean_mse=np.mean(global_mse)
    print(global_mse)
    print(mean_mse)
    save_image_id = 0
    for dataloader, mapping, canvas, rebuilt, nprecon, patch, diffed,  image_folder_name in zip(patches_list, mapping_list, canvas_list, rebuilt_list, nprecon_list, new_patches, diffed_list, image_name_list):
        for (img) in dataloader:
            rebuilt_no_vis = copy.deepcopy(rebuilt)
            print('Img len: {}'.format(len(img)))
            for i in range(len(img)):

                location=mapping[i]

                reconpatch = nprecon[i]
                rebuilt_no_vis[location[0]:location[0]+reconpatch.shape[0],location[1]:location[1]+reconpatch.shape[1],:] = reconpatch
                mse = normalize_data(np.square(np.subtract(convert_hsv(patch)[:,:,hsv_compare_index], convert_hsv(reconpatch.copy())[:,:,hsv_compare_index]))).mean()
                msediff = np.power(np.subtract(mean_mse,mse),2)*10
                #print('MSE Diff: {}'.format(msediff))
                #print(msediff)
                orig_patch = canvas[location[0]:location[0]+patch.shape[0],location[1]:location[1]+patch.shape[1],:]
                diffed[location[0]:location[0]+reconpatch.shape[0],location[1]:location[1]+reconpatch.shape[1],:] = cv2.subtract(np.float64(orig_patch), np.float64(reconpatch))
                reconpatch[:,:,2] +=msediff
                #canvas[location[0]:location[0]+patch.shape[0],location[1]:location[1]+patch.shape[1],0]+=msediff
                rebuilt[location[0]:location[0]+reconpatch.shape[0],location[1]:location[1]+reconpatch.shape[1],:]=reconpatch

            cv2.imshow("original",canvas)
            cv2.imshow("reconstruction",rebuilt)
            cv2.waitKey(2)
            sub_folder = image_folder_name.split('/')[0]
            img_name = image_folder_name.split('/')[-1]
            write_dir = save_folder+'{}'.format(sub_folder)
            print(write_dir)
            # create dir if not exists
            Path(write_dir).mkdir(parents=True, exist_ok=True)
            conv_range_canvas = canvas * 255
            conv_range_rebuilt = rebuilt * 255
            conv_range_diffed = diffed * 255
            converted_range_rebuilt_no_vis = rebuilt_no_vis * 255
            cv2.imwrite(write_dir+'/'+"original_{}".format(img_name), conv_range_canvas)
            cv2.imwrite(write_dir+'/'+"orig_hue_{}".format(img_name), convert_hsv(conv_range_canvas)[:,:,0])
            cv2.imwrite(write_dir+'/'+"orig_saturation_{}".format(img_name), convert_hsv(conv_range_canvas)[:,:,1])
            cv2.imwrite(write_dir+'/'+"orig_value_{}".format(img_name), convert_hsv(conv_range_canvas)[:,:,2])
            cv2.imwrite(write_dir+'/'+"rebuilt_hue_{}".format(img_name), convert_hsv(converted_range_rebuilt_no_vis)[:,:,0])
            cv2.imwrite(write_dir+'/'+"rebuilt_saturation_{}".format(img_name), convert_hsv(converted_range_rebuilt_no_vis)[:,:,1])
            cv2.imwrite(write_dir+'/'+"rebuilt_value_{}".format(img_name), convert_hsv(converted_range_rebuilt_no_vis)[:,:,2])
            cv2.imwrite(write_dir+'/'+"rebuilt_{}".format(img_name), conv_range_rebuilt)
            cv2.imwrite(write_dir+'/'+"diffed_{}".format(img_name), conv_range_diffed)
            save_image_id += 1
    
def flatten_list_of_lists(t):
    return [item for sublist in t for item in sublist]

def data_load_wrapper(train_patches):
    return torch.utils.data.DataLoader(PatchDataset(train_patches), batch_size=len(train_patches), shuffle=False,num_workers=0, pin_memory=True)

def get_dataloader_list(patches_list):
    dataloaders = []
    for patch in patches_list:
        dataloaders.append(data_load_wrapper(patch))
    return dataloaders            

def load_train_data(file_name='/0-B01.png', size=None):
    patches_list = []
    mapping_list = []
    resolution_list = []
    image_name_list = []
    for root, dirs, files in os.walk(BASE_TRAIN_PATH, topdown=False):
        for name in dirs:
            sub_folder_image_name = name +file_name
            img_path = BASE_TRAIN_PATH + sub_folder_image_name
            slice_img_path = os.path.join(pathlib.Path().resolve(), img_path)
            patches,mapping,resolution = sliceImage(slice_img_path)
            patches_list.append(patches)
            mapping_list.append(mapping)
            resolution_list.append(resolution)
            image_name_list.append(sub_folder_image_name)
    if type(size) == int:
        return patches_list[0:size], mapping_list[0:size], resolution_list[0:size], image_name_list[0:size]
    else:
        return patches_list, mapping_list, resolution_list, image_name_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train autoencoder')
    parser.add_argument('--epochs', default=15, type=int,
                    help='Number of training epochs')
    parser.add_argument('--train', dest='train', default=False, action='store_true',
                    help='Dont train the model, load it instead')
    args = parser.parse_args()
    global num_epochs
    num_epochs = args.epochs
    print(pathlib.Path().resolve())
    patches_list, mapping_list, resolution_list, image_name_list = load_train_data(file_name='/0-B01.png')
    val_patches_list, val_mapping_list, val_resolution_list, val_image_name_list = load_train_data(file_name='/4-B02.png', size=5)
    print('Len Patch List: {}'.format(len(patches_list)))
    train_patches = flatten_list_of_lists(patches_list)
    print('Len Train Patches: {}'.format(len(train_patches)))
    val_patches = flatten_list_of_lists(val_patches_list)
    print('Len Validation Patches: {}'.format(len(val_patches)))
    train_dataloader = data_load_wrapper(train_patches)
    val_dataloader = data_load_wrapper(val_patches)

    if args.train:
        model, loss_metrics = trainEncoder(train_dataloader, val_dataloader)
        with open('{}epochs_metrics_{}.json'.format(num_epochs, datetime.datetime.now().strftime("%m_%d_%Y_%H%M%S")), 'w') as fp:
            json.dump(loss_metrics, fp)
        torch.save(model.state_dict(), "{}epochs_testmodel.txt".format(num_epochs))
    else:
        model = loadModel(name='500epochs_model.txt')
    compareImages(get_dataloader_list(patches_list),model,mapping_list,resolution_list, image_name_list, SAVE_PATH)
    compareImages(get_dataloader_list(val_patches_list),model,val_mapping_list,val_resolution_list, val_image_name_list, VAL_SAVE_PATH)

