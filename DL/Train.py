print("The way you run teh file is: python filename GPUID Datasetname")
# print("Datatype options are: Deepsigweb, Deepsigcode,'clean','AWGNOnly','ClockOnly','FadingOnly','All'")
import sys

# Inserting the paths below since the train file and model definition files are in the below folders.
sys.path.insert(1, '/home/vesathya/ModulationClassification/ModelClassDefinitions/')
sys.path.insert(1, '/home/vesathya/ModulationClassification/Mainfiles_Sep2020/')
import numpy as np
import pickle
import datetime

import torch.nn as nn
from GRU_AMC import GRU_AMC
import torch
from Train_LRScheduler import train_model
from sklearn.model_selection import train_test_split
from ModulationPredictionCNN128_Nov9th2021 import ModulationPredictionCNN128_Nov9th2021
import os
import matplotlib.pyplot as plt
import logging

modulationTypes = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", "BFM", "DSBAM"]

label_dict = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4, 'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7,
              'PAM4': 8, 'QPSK': 9}

TrainParams = {}
DatasetParams = {}
L_range = [128]
BS_range = [512,128,32]
LR_range = [10 ** -3]  # [float(sys.argv[3])]#[10**-2,10**-3,10**-4,10**-5,10**-6]
# Since we are using a LR scheduler, we can start with a LR of 10**-2
initializer_list = ['Xavier']  # ['Xavier','Orthogonal']
TrainParams['num_epochs'] = 200
TrainParams['n_epochs_earlystop'] = 16
TrainParams['test_bactchsize'] = 1024
TrainParams['LRScheduler_stepsize'] = 8  # These many iterations are done for every step
TrainParams['LRSchedulerGamma'] = 0.1  # Every r itertaions, the LR is reduced by a multiplicative factor of LRSchedulerDecay
weight_decay = TrainParams['weight_decay'] = 5e-4
TrainParams['optimizer_type'] = 'Adam'
TrainParams['validation_size'] = 0.2
TrainParams['clip'] = 5

#DatasetParams['DeepsigFiletype'] = 'codeGenerated'#website
DatasetParams['SNRrange'] = np.arange(-20, 21, 2)
print("SNR range is :", DatasetParams['SNRrange'])
DatasetParams['Modulationtypes'] = ['QAM16', 'QAM64', '8PSK', 'WBFM', 'BPSK', 'CPFSK', 'AM-DSB', 'GFSK', 'PAM4', 'QPSK']
DatasetParams['NumClasses'] = len(DatasetParams['Modulationtypes'])
print("Modulation types used are: ",DatasetParams['Modulationtypes'])
DatasetParams['datatype'] = sys.argv[2]  # 'old'
DatasetParams['NumFrames'] = 2000
print("DatasetParams['NumFrames']: ",DatasetParams['NumFrames'])
# DatasetParams['CFOmaxdev'] = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
TrainParams['criterion'] = nn.CrossEntropyLoss()
TrainParams['computing_device'] = torch.device("cuda")
#print("GPU used is: ", sys.argv[1])
print("screen session used is: ", os.system("echo $STY"))
modelloc_lastpart = sys.argv[3]

Savemodelfile_location = '/home/vesathya/ModulationClassification/Aug2020' + \
   '/Code/JournalPaperSNRPartitioning/CFOSweep/models/Journal/Nov11th/'+ modelloc_lastpart + '/'
print("Savemodelfile_location: ", Savemodelfile_location)
datafilelocation = '/home/vesathya/ModulationClassification/Aug2020/Code/JournalPaperSNRPartitioning/'+\
'CFOSweep/DatasetGeneration_GnuRadio/TWC_Dataset/'
datafilename = datafilelocation + DatasetParams['datatype']




class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        pass
            
                    
                    
logfilename = Savemodelfile_location+'Datasettype_'+DatasetParams['datatype']+'_logs.txt'
f = open(logfilename, 'w')
backup = sys.stdout
sys.stdout = Tee(sys.stdout, f)
            
for initialize_type in initializer_list:
    for L in L_range:
        for BS in BS_range:
            for LR in LR_range:
                TrainParams['initialize_type'] = initialize_type
                TrainParams['L'] = L
                TrainParams['BS'] = BS
                TrainParams['LR'] = LR
                
                Savemodelfilename = 'LRScheduled_' + str(LR) + '_BS_' + str(BS) + \
                                    TrainParams['initialize_type'] + TrainParams['optimizer_type'] + '_L_' + str(L) + \
                                    '_dataIs_' + DatasetParams['datatype']

                '''
                Load dataset from the pickle file. The data is in a dictionary format with keys corresponding 
                to modulation and SNR. Every dict item in the dictionary contains X items per mod per SNR.
                (X=500 for simulated data)
                '''

                print("Dataset Filename is:  ", datafilename)
                f = open(datafilename, 'rb')
                dataset = pickle.load(f, encoding='latin1')
                f.close()

                # read the keys - snrs and mods.
                snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], dataset.keys())))), [1, 0])
                #snrs = list(range(-20, 21, 2))
                X = []
                lbl = []
                for mod in mods:
                    if mod in DatasetParams['Modulationtypes']:
                        for snr in snrs:
                            if snr in DatasetParams['SNRrange']:
                                X.append(dataset[(mod, snr)][0:DatasetParams['NumFrames']])
                                for i in range(DatasetParams['NumFrames']):  lbl.append((mod, snr))
            
            
                X = np.vstack(X)
                label_val = list(map(lambda x: lbl[x][0], range(len(lbl))))
                label = list(map(lambda x: label_dict[x], label_val))
                label = np.array(label)
                data = X[:, :, 0:L]
                print('data.shape: ', data.shape)
                del dataset, X  # deleting large arrays to free up space in RAM.
                modelclass = ModulationPredictionCNN128_Nov9th2021


                def loadSplitTrain(modelclass, Savemodelfile_location, Savemodelfilename, data, label, TrainParams):

                    model = modelclass(DatasetParams['NumClasses'])
                    x_train, x_test, y_train, y_test = train_test_split(data, label,
                                                                        test_size=TrainParams['validation_size'] \
                                                                        , random_state=1)
                    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                                      test_size=TrainParams['validation_size'],
                                                                      random_state=1)
                    train_set = {'data': torch.tensor(x_train).float(), 'labels': torch.tensor(y_train).float()}
                    val_set = {'data': torch.tensor(x_val).float(), 'labels': torch.tensor(y_val).float()}
                    print(x_train.shape, x_val.shape, x_test.shape)
                    del data

                    ############ ############### Train Model ########################## ##########
                    ############ ############ ############ ############ ############ ############
                    model_file = Savemodelfile_location + Savemodelfilename + '_model.pt'
                    print('model file name is: ', model_file)

                    print("Training parameters: ", TrainParams)
                    model1, Loss, Accuracy = train_model(model, model_file, train_set, val_set, TrainParams)
                    # Save Loss and accuracy plots

                    # %matplotlib inline
                    plt.figure(1)
                    epochs = [i for i in range(len(Loss['train']))]
                    plt.plot(epochs, Loss['train'])
                    plt.plot(epochs, Loss['valid'])
                    plt.title('')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend(['Training', 'Validation'])
                    plt.savefig(Savemodelfile_location + Savemodelfilename + 'model_loss.png')
                    plt.clf()
                    # plt.show()

                    plt.figure(2)
                    plt.plot(epochs, Accuracy['train'])
                    plt.plot(epochs, Accuracy['valid'])
                    plt.title('')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.legend(['Training', 'Validation'])
                    plt.savefig(Savemodelfile_location + Savemodelfilename + 'model_acc.png')
                    plt.clf()

                    outfile1 = open(Savemodelfile_location + Savemodelfilename + '_Params.txt', 'at')
                    outfile1.write(str(TrainParams))
                    outfile1.write("\n \n")
                    outfile1.write(str(DatasetParams))
                    outfile1.close()

                loadSplitTrain(modelclass, Savemodelfile_location, Savemodelfilename, data, label, TrainParams)
f.close()
