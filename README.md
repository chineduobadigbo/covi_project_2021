# covi_project_2021


CLI:
usage: autoencoder_main.py [-h] [--epochs EPOCHS] [--model MODEL] [--color COLOR] [--lossFunc LOSSFUNC] [--batchSize BATCHSIZE] [--validate] [--train] [--blur]
                           [--quantize] [--official] [--continueModel] [--completeData] [--combineImages] [--outputErrType OUTPUTERRTYPE]


optional arguments:
  -h, --help            		show this help message and exit
  --epochs EPOCHS       	Number of training epochs (default: 100)
  --model MODEL         	Path to the model file (default: models/testmodel_rgb.pt)
  --color COLOR         		The color space used for training (RGB, HSV, LAB) (default: RGB)
  --lossFunc LOSSFUNC   	The loss used for training (default: ssim)
  --batchSize BATCHSIZE
                        Batch size during training and validation. Set to -1 to take the complete trainingset size (default: -1)
  --validate            		Whether validation should be done during training (default: False)
  --train               		Dont load the model, train it instead (default: False)
  --blur                		Apply blur during preprocessing (default: False)
  --quantize            		Apply quantization during preprocessing (default: False)
  --official            		Use the official validation dataset, compute bounding boxes and           save them (default: False)
  --continueModel       		Continue training if model already exists (default: False)
  --completeData        		Use the complete trainingdata (default: False)
  --combineImages       	Combine images based on homographies (default: False)
  --outputErrType OUTPUTERRTYPE
                        		The error func that gets used for creating the output patches (ssim, mse or custom) (default: ssim)

Methods: 

trainModel(modelpath, epochs, batchSize, preprDict, color, outputErrType, validate=False, continueModel=True, completeData=True, lossFunc='mse', combineImages = False)

Takes necessary hyperparameters and runs through the training loop and stores the results to a checkpoint file. 

applyPreprocessingFuncs(patchesList, preprDict, color)

Loops over provided preprocessing functions and applies them to image patches

utils.loadImages(color, fileName='/0-B01.png', baseDir='data/train/', size=None, completeData=False, combined = False)

Loads the training images (either complete set or subset into memory and divides them into patches.

utils.puzzleBackTogether(atEachBatch,atEachPatch,atEachImage,dataloader,resolutionsList,mappingsList,tileCountsList,imageNamesList,canvasCount, model, color, outputErrType)

Takes a list of image patches and computes the model output, error and difference for each patch and then puts them back into complete images and saves them to disc. 

autoencoder_boilerplate.Autoencoder(nn.Module)

The Autoencoder model definition and architecture. 

pytorch_mssim.SSIM_Loss(SSIM)

Defines the Structural Similarity Index Loss function to be used during pytorch training.

boundingbox.drawPolygon(image, points, color=(255,0,0), thickness=2)

Draws the bounding box given and image and coordinates. 

boundingbox.getBoundingBox(model, reducedImage)

Returns the bounding box prediction of the boundingbox regression model
