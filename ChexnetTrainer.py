import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import roc_auc_score

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201

from CXR_Data_Generator import DatasetGenerator


#-------------------------------------------------------------------------------- 

class ChexnetTrainer ():

    #---- Train the densenet network 
    #---- path - path to the directory containing main file
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def train (path, pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint):

        
        #-------------------- SETTINGS: SAVE LOSSES
        save_loss = os.path.join(path, 'loss.csv'))
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(in_features=1, nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(in_features=1, nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(in_features=1, nnClassCount, nnIsTrained).cuda()
                
        #-------------------- SETTINGS: DATA TRANSFORMS
        
        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(transforms.Grayscale(num_output_channels=1)) 
        transformSequence=transforms.Compose(transformList)

        #-------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DataGenerator(img_dir=pathDirData, split_file=pathFileTrain, transform=transformSequence, ext="")
        datasetVal =   DataGenerator(img_dir=pathDirData, split_file=pathFileTrain, transform=transformSequence, ext="")
              
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False)
        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
                
        #-------------------- SETTINGS: LOSS
        loss = torch.nn.BCELoss(size_average = True)
        
        lossMIN = 100000
        
        #---- Load checkpoint 
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            lossMIN = modelCheckpoint['best_loss']

        
        #---- TRAIN THE NETWORK
        
        for epochID in range (0, trMaxEpoch):
            
            # Calculating Start time for current epoch
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
            logging.info('Start time for {0} epoch,\t{1}'.format((epochID+1), str(timestampSTART)))
            
            # Training and evaluating the model for current epoch                       
            lossTrain = ChexnetTrainer.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            lossVal = ChexnetTrainer.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            
            # Formating loss for current epoch as string
            txt = "[" + str(epochID) + "/" + str(trMaxEpoch) + "],"  + str(lossTrain + "," + str(lossVal) + "\n"
            
            # Calculating End time for current epoch
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            logging.info('End time for {0} epoch,\t{1}'.format((epochID+1), str(timestampEND)))
            
            # Using scheduler to change the learning rate based on validation loss
            scheduler.step(lossVal)
            
            # Saving the current model 
            torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 
            			path + 'model/m-' + str(launchTimestamp) + '.pth.tar')
            
            # Writing Loss for each epoch to CSV file
            fcsv = open(save_loss, "a")
            fcsv.write(txt)
            fcsv.close()
            
            # Comparing the previous saved model with current and saving the best model
            if lossVal < lossMIN:
                lossMIN = lossVal   
                logging.info ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 
                			path + 'model/m-' + str(launchTimestamp) + '_best.pth.tar')
            else:
                logging.info ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
                     
    #-------------------------------------------------------------------------------- 
    
    # Defining Training Step for each epoch  
    def epochTrain (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        lossTrain = 0
        lossTrainNorm = 0
        model.train()
        
        for batchID, (input, target) in enumerate (dataLoader):
                        
            target = target.cuda(non_blocking = True)
                 
            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)         
            varOutput = model(varInput)
            
            lossvalue = loss(varOutput, varTarget)
                       
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
            lossTrain += lossvalue.item()
            lossTrainNorm += 1
            
        outLoss = lossTrain / lossTrainNorm
        
        return outLoss
                    
    #-------------------------------------------------------------------------------- 

    # Defining Evaluation Step for each epoch    
    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.eval ()
        
        lossVal = 0
        lossValNorm = 0
        
        for i, (input, target) in enumerate (dataLoader):
            
            target = target.cuda(non_blocking=True)
                 
            varInput = torch.autograd.Variable(input, volatile=True)
            varTarget = torch.autograd.Variable(target, volatile=True)    
            varOutput = model(varInput)
            
            losstensor = loss(varOutput, varTarget)
            
            lossVal += losstensor.item()
            lossValNorm += 1
            
        outLoss = lossVal / lossValNorm
        
        return outLoss
               
    #--------------------------------------------------------------------------------     
     
    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
        return outAUROC
        
        
    #--------------------------------------------------------------------------------  
    
    #---- Test the trained network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def test (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
        
        
        CLASS_NAMES = [ "Cardiomegaly" ,"Emphysema" ,"Effusion" ,"No_Finding" ,"Hernia" ,"Infiltration" ,"Mass" ,"Nodule" ,"Atelectasis" ,"Pneumothorax" ,
        				"Pleural_Thickening" ,"Pneumonia" ,"Fibrosis" ,"Edema" ,"Consolidation"]
        
        cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(in_features=1, nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(in_features=1, nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(in_features=1, nnClassCount, nnIsTrained).cuda()
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        
        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(transforms.Grayscale(num_output_channels=1))
        transformSequence=transforms.Compose(transformList)
        
        datasetTest = DataGenerator(img_dir=pathDirData, split_file=pathFileTest, transform=transformSequence, ext="")
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, shuffle=False)
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        model.eval()
        
        # Defining Testing steps
        for i, (input, target) in enumerate(dataLoaderTest):
            
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0)
            
            bs, c, h, w = input.size()
            
            varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda(), volatile=True)
            
            out = model(varInput)
            outMean = torch.nn.Softmax(dim=1)(out)
            
            outPRED = torch.cat((outPRED, outMean.data), 0)

		# Calculating classwise AUROC for test dataset
        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        
        # Calculating mean AUROC of all the class for test dataset
        aurocMean = np.array(aurocIndividual).mean()
        
        logging.info('AUROC mean,\t{0:.3f}'.format(aurocMean))
        
        for i in range (0, len(aurocIndividual)):
            logging.info("{0},\t{1:.3f}".format(CLASS_NAMES[i],aurocIndividual[i]))
        
     
        return
#-------------------------------------------------------------------------------- 





