import os
import numpy as np
import time
import sys
import logging
import argparse

from ChexnetTrainer import ChexnetTrainer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, "0" to  "7" 
os.environ["CUDA_VISIBLE_DEVICES"]="7"

#-------------------------------------------------------------------------------- 
def main (args):

    if args.trainable:
        runTrain(args)
    else:
        runTest(args)
  
#--------------------------------------------------------------------------------   

def runTrain(args):

	save_loss = os.path.join(args.defaultPath, 'loss.csv'))
	if not os.path.exists(save_loss):
		fcsv = open(save_loss, "w")
		fcsv.write("Epoch,train_loss,val_loss\n")
		fcsv.close()

    fname = os.path.join(args.defaultPath, (os.path.splitext(os.path.basename(args.train_file))[0]+'.log'))
    logging.basicConfig(filename=fname, filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
    
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    path = args.defaultPath
    pathDirData = args.img_dir
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains triplet [path to image, abnormality name ,output vector]
    #---- Example: images_011/00027736_001.png, No_finding, 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = args.train_file
    pathFileVal = args.val_file
    pathFileTest = args.test_file
    
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnArchitecture = DENSENET121
    nnIsTrained = True
    nnClassCount = args.class_count
    
    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = args.batch_size
    trMaxEpoch = args.epochs
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = args.img_size
    imgtransCrop = args.crop_size
        
    pathModel = path + '/models/m-' + str(timestampLaunch) + '.pth.tar'
    
    logging.info('Training NN architecture = {0}'.format(nnArchitecture))
    logging.info('Model Save Path:{0}'.format(pathModel))
    ChexnetTrainer.train(path, pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)
    
    logging.info('Testing the trained model')
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

def runTest(args):

    fname = os.path.join(args.defaultPath, (os.path.splitext(os.path.basename(args.trained_model))[0]+'.log'))
    logging.basicConfig(filename=fname, filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
    
    pathDirData = args.img_dir
    pathFileTest = args.test_file
    nnArchitecture = 'DENSE-NET-121'
    nnIsTrained = False
    nnClassCount = args.class_count
    trBatchSize = args.batch_size
    imgtransResize = args.img_size
    imgtransCrop = args.crop_size
    
    pathModel = args.trained_model
    
    timestampLaunch = ''
    
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch NIH-CXR Chexnet Training')
    parser.add_argument('--defaultPath', '-dp', metavar='PATH', default='/workspace/data/chexnet/')
    parser.add_argument('--img_dir', default='/workspace/data/NIHChestXray14', type=str)
    parser.add_argument('--train_file', default='dataset/NIH-CXR-train.csv', type=str)
    parser.add_argument('--val_file', default='dataset/NIH-CXR-val.csv', type=str)
    parser.add_argument('--test_file', default='dataset/NIH-CXR-test.csv', type=str)
    parser.add_argument('--img_size', '-sz', default=256, type=int)
    parser.add_argument('--crop_size', '-cs', default=224, type=int)
    parser.add_argument('--batch_size', '-bs', default=32, type=int)
    parser.add_argument('--class_count', '-c', default=15, type=int)
    parser.add_argument('--epochs', '-e', default=20, type=int)
    parser.add_argument('--trainable', default=True, type=bool)
    parser.add_argument('--trained_model', default='/workspace/data/chexnet/test.pth.tar', type=str)

    args = parser.parse_args()
    main(args)
