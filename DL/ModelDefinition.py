# %reset -f
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
#import torch.optim as optim

class ModelDefinition(nn.Module):
    """ A one dimensional convolutional neural network model.

    Consists of six Conv1d layers, followed by max pooling
    and one fully-connected (FC) layer:

    conv1 -> conv2 -> conv3 -> conv4 con5 -> conv6 -> fc  (outputs)

    Make note:
    - Inputs are expected to be 2 channel signals of length 1024

    """

    def __init__(self,num_classes, activation='relu'):
        super(ModelDefinition, self).__init__()
        self.activation = activation
        conv_kernel_size = 3
        conv_padding = 1
        maxpool_kernel_size = 2
        maxpool_padding = 0
        maxpool_stride = 2
        # conv1
        in_dim = 2
        out_dim = 64
        dropout_perc=0.5
        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=conv_kernel_size, \
                               padding=conv_padding)
        self.conv1_norm = nn.BatchNorm1d(out_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=maxpool_stride, \
                                  padding = maxpool_padding)  
#         self.dropout1 = nn.Dropout(dropout_perc)
        # conv2
        in_dim = out_dim
        out_dim = 64
        self.conv2 = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=conv_kernel_size, \
                               padding=conv_padding)
        self.conv2_norm = nn.BatchNorm1d(out_dim)
        self.pool2 = nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=maxpool_stride, \
                                  padding = maxpool_padding)
#         self.dropout2 = nn.Dropout(dropout_perc)
        
        #conv4
        in_dim = out_dim
        out_dim = 32
        self.conv3 = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=conv_kernel_size, \
                               padding=conv_padding)
        self.conv3_norm = nn.BatchNorm1d(out_dim)
        self.pool3 = nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=maxpool_stride, \
                                  padding = maxpool_padding)
#         self.dropout3 = nn.Dropout(dropout_perc)
        
        # Define fully connected layer:
        self.dropout = nn.Dropout(dropout_perc)
        self.fc1 = nn.Linear(in_features=512, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        #self.softmax = nn.Softmax(11)# this is not being used
        
        #self.dropout = nn.Dropout(dropout_perc)
        #print(dropout_perc)
        # Initialize weights
        torch_init.xavier_normal_(self.conv1.weight)
        torch_init.xavier_normal_(self.conv2.weight)
        torch_init.xavier_normal_(self.conv3.weight)
        torch_init.xavier_normal_(self.fc1.weight)
        torch_init.xavier_normal_(self.fc2.weight)


    def forward(self, batch):
        """Pass the batch of images through each layer of the network, applying
        non-linearities after each layer.

        Params:
        -------
        - batch: (Tensor) An input batch of images
        Returns:
        --------
        - logits: (Variable) The output of the network
        """
        if (self.activation == 'relu' ):
            # Apply convolutions with relu activation
            temp11 = self.conv1(batch)
            temp12 = self.conv1_norm(temp11)
            temp13 = func.relu(temp12)
            batch = self.pool1(temp13)
#             print(temp11.shape,temp12.shape,temp13.shape,batch.shape)
            batch = self.pool2(func.relu(self.conv2_norm(self.conv2(batch))))
#             print(batch.shape)
            batch = self.pool3(func.relu(self.conv3_norm(self.conv3(batch))))
#             print(batch.shape)
        elif (self.activation == 'tanh'):
            # Apply convolutions with tanh activation
            batch = self.pool1(torch.tanh(self.conv1_norm(self.conv1(batch))))
            batch = self.pool2(torch.tanh(self.conv2_norm(self.conv2(batch))))
            batch = self.pool3(torch.tanh(self.conv3_norm(self.conv3(batch))))
        elif (self.activation == 'sigmoid'):
            # Apply convolutions with sigmoid activation
            batch = self.pool1(func.sigmoid(self.conv1_norm(self.conv1(batch))))
            batch = self.pool2(func.sigmoid(self.conv2_norm(self.conv2(batch))))
            batch = self.pool3(func.sigmoid(self.conv3_norm(self.conv3(batch))))

        batch = batch.view(-1, self.num_flat_features(batch))
        batch = self.dropout(self.fc1(batch))
        batch = (self.fc2(batch))
        return batch

    def num_flat_features(self, inputs):

        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s
        return num_features
    
