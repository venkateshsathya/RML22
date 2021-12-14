%reset -f
import sys
sys.path.insert(1, '/home/vesathya/ModulationClassification/ModelClassDefinitions/')
sys.path.insert(1, '/home/vesathya/ModulationClassification/Mainfiles_Sep2020/')
import numpy as np
import pickle
from ModulationPredictionCNN128_Nov9th2021 import ModulationPredictionCNN128_Nov9th2021
import torch
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
# modulationTypes = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", "BFM", "DSBAM"]
label_dict = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4, 'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7,
              'PAM4': 8, 'QPSK': 9}
TestParams = {}
DatasetParams = {}

TestParams['validation_size'] = 0.2
TestParams['test_BS'] = 1024
#TrainParams['clip'] = 5
DatasetParams['SNRrange'] = np.arange(-20, 21, 2)
DatasetParams['Modulationtypes'] = ['QAM16', 'QAM64', '8PSK', 'WBFM', 'BPSK', 'CPFSK', 'AM-DSB', \
                                    'GFSK', 'PAM4', 'QPSK']
DatasetParams['NumClasses'] = len(DatasetParams['Modulationtypes'])
print("Modulation types used are: ",DatasetParams['Modulationtypes'])
DatasetParams['frameLength'] = 128
DatasetParams['NumFrames'] = 2000
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # sys.argv[2]#"0"
TestParams['computing_device'] = torch.device("cuda")
# AMC2021_8samplespersymbol_All.01A
# datasettype_list = ['Deepsigcode','clean','AWGNOnly','ClockOnly','FadingOnly','All']
# oldinfomodelLocation = '/home/vesathya/ModulationClassification/Aug2020/Code/JournalPaperSNRPartitioning/CFOSweep/'+\
# 'models/Journal/InformationSourceTests/Oct28th/'
newmodelLocation = '/home/vesathya/ModulationClassification/Aug2020/Code/JournalPaperSNRPartitioning/CFOSweep/'+\
'models/Journal/Nov11th/sps2_5datasets/'
modelfile_list = {}


modelfile_list = dict({'AMC2021clean.01A':newmodelLocation+\
'LRScheduled_0.001_BS_512XavierAdam_L_128_dataIs_AMC2021clean.01A_model.pt',\
'AMC2021AWGNOnly.01A':\
newmodelLocation+'LRScheduled_0.001_BS_32XavierAdam_L_128_dataIs_AMC2021AWGNOnly.01A_model.pt',\
'AMC2021ClockOnly.01A':\
newmodelLocation+'LRScheduled_0.001_BS_32XavierAdam_L_128_dataIs_AMC2021ClockOnly.01A_model.pt',\
'AMC2021FadingOnly.01A':newmodelLocation+\
'LRScheduled_0.001_BS_128XavierAdam_L_128_dataIs_AMC2021FadingOnly.01A_model.pt',\
'AMC2021All.01A':newmodelLocation+'LRScheduled_0.001_BS_128XavierAdam_L_128_dataIs_AMC2021All.01A_model.pt'})


datasettype_list = modelfile_list.keys()




accuracy = {}
dataset = {}

def calculate_accuracy(model, data, label, batch_size, computing_device):
    # print(data.shape)
    model.eval()
    with torch.no_grad():
        n_samples = data.shape[0]
        n_minibatch = int((n_samples + batch_size - 1) / batch_size)
        # print("minibatch value is ", n_minibatch)
        accuracy = 0
        I = np.arange(n_samples)
        for i in range(n_minibatch):
            idx = I[batch_size * i:min(batch_size * (i + 1), n_samples)]
            dt = data[idx].to(computing_device)
            # print(idx, label)
            lbl = label[idx].numpy()
            output = model(dt)
            output = output.cpu().numpy()
            output = np.argmax(output, axis=1)

            accuracy += np.sum(output == lbl)

        return accuracy / n_samples

# Loading all data as a one time step. Since the dataset size is not too large, this helps save test time.

datafilelocation = "/home/vesathya/ModulationClassification/Aug2020/Code/JournalPaperSNRPartitioning/" + \
"CFOSweep/DatasetGeneration_GnuRadio/TWC_Dataset/"
for datatype in datasettype_list:
    datafilename = datafilelocation + datatype #'AMC2021' + '_8samplespersymbol_' + datatype+ '.01A'
    f = open(datafilename, 'rb')
    dataset[datatype] = pickle.load(f, encoding='latin1')
    f.close()

for mod in DatasetParams['Modulationtypes']:
    for snr in DatasetParams['SNRrange']:

        for trainmodeltype in datasettype_list:

            # Load pre-trained model
            modelclass = ModulationPredictionCNN128_Nov9th2021
            if 'model' in globals():
                del model  # deleting it so that pre-loaded model in previous loop iteration doesnt not mess up this one.
            model = modelclass(DatasetParams['NumClasses'])
            modelfile = modelfile_list[trainmodeltype]
            model.load_state_dict(torch.load(modelfile))
            #print(modelfile)
            model = model.to(TestParams['computing_device'])
            #print("Model on CUDA?", next(model.parameters()).is_cuda)

            for testdatatype in datasettype_list:
#                 data = dataset[testdatatype][(mod, snr)]
                data = dataset[testdatatype][(mod, snr)][0:DatasetParams['NumFrames']]
                label = label_dict[mod]*np.ones((data.shape[0],))
                x_train, x_test, y_train, y_test = train_test_split(data, label,
                                                                    test_size=TestParams['validation_size'],
                                                                    random_state=1)
                test_data = torch.tensor(x_test).float()
                test_label = torch.tensor(y_test).float()
                #print(test_data.shape, test_label.shape)

                accuracy[(trainmodeltype, testdatatype, mod,snr)] = calculate_accuracy(model, test_data, test_label,
                                                                                    TestParams['test_BS'],
                                                                                    TestParams['computing_device'])

accfilename = newmodelLocation + 'AccuracypermodperSNR_2spsNov15th.pickle'
file_handle = open(accfilename, "wb")
pickle.dump(accuracy, file_handle)
file_handle.close()


