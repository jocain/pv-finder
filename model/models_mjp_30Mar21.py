import torch
from torch import nn
import numpy as np

## This model is designed as a "perturbative" model.The feature set passed 
## in (initial neuron values) is of the form (X,x,y) where X is the original 
## KDE and x and y are the values of x and y at each z where the KDE is 
## maximum. These feature sets will be divided into two parts (X) and (x,y) 
## and each of these will be run through some convolutional layers to produce
## 4000 bin tensors. Then, the element-wise multiplication of the feature 
## sets will be passed through a final convolutional layer. The hope is that 
## the learning from the (X) features can start from a previously trained 
## model with the same structure that works well. Then, the model will learn 
## a filter that will pass most learned features with essentially no change, 
## but will sometimes "mask out" regions where we see that changes in (x,y) 
## appear to flag the presence of false positives in the original approach.
## 
## With luck, this will allow the algorithm to reduce the number of false 
## positives for a fixed efficiency, so improve overall performance relative 
## to the same architecture processing only (X).

'''
In order to better streamline the model naming conventions, "tags" will be
used. These tags will represent important characteristics of the model so
that an at-a-glance understanding of the model can be formed. Between each 
tag there will be a _ to indicate a new tag being referenced.

These tags include:
#S - number of skip connections, if any
#L - number of layers
BN - Batch Normalization
ACN - AllCNN "family" of models
#i# - model step number out of total steps (e.g. 3i4 means model step 3 of 4)
RC# - reduced channel size, followed by the iteration number (i.e. this
    number is used to differentiate models that are different only in
    their channel size)
IC# - increased channel size, followed by the iteration number
RK# - reduced kernel size, followed by the iteration number
IK# - increased kernel size, followed by the iteration number
C - concatenation of perturbative and non-perturbative layers at the end
BM - benchmark; if changes are made to future models, it will be tagged
    based on changes made in reference to this model (locally; file scope)

The tag hierarchy will be of the format:
BM_ACN_#_P_#L_#S_BN_RC#_IC#_RK#_IK#_C

All model files from 30Jan21 and on will use this format when they have 
"mjp" in them.

NOTE: All models in this notebook (except one) use Batch Normalization. 
This was not the case in previous notebooks but is now. Here is why: Batch 
Normalization reduces covariance shift from layer to layer, which is 
especially needed in Deep Learning. It also makes training more efficient.
See the link below for more information.
https://towardsdatascience.com/understanding-batch-normalization-for-neural-networks-1cd269786fa6

'''

class Conv(nn.Sequential):
    ## convolution => BatchNorm => Dropout => LeakyReLU
    def __init__(self, INC, OUTC, k_size=15, drop_rate=.15):
        super(Conv, self).__init__(
            nn.Conv1d(in_channels=INC, out_channels=OUTC, kernel_size=k_size, stride=1, padding=(k_size-1)//2),
            nn.BatchNorm1d(OUTC),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(0.01)
        )

class Conv_No_BN(nn.Sequential):
    ## convolution => BatchNorm => Dropout => LeakyReLU
    def __init__(self, INC, OUTC, k_size=15, drop_rate=.15):
        super(Conv, self).__init__(
            nn.Conv1d(in_channels=INC, out_channels=OUTC, kernel_size=k_size, stride=1, padding=(k_size-1)//2),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(0.01)
        )





'''
Benchmark network architectures with the following attributes:
1. Three feature set using X, x, y.
2. 6 layer convolutional architecture for X and Xsq feature set.
3. 4 layer conovlutional architecture for x and y feature set.
4. Concatenates two feature sets and passes through a convolutional layer.
5. LeakyRELU activation used for convolutional layer.
6. Softplus activation used for output.
7. Channel count follows the format:    20-10-10-10-1-1 (X), 20-10-10-1 (x, y), 20-1 (X, x, y)
8. Kernel size follows the format:      25-15-15-15-15-91 (X), 25-15-15-91 (x, y),  91  (X, x, y)
'''
class BM_ACN_1i4_6L(nn.Module):
    # creates model architecture
    def __init__(self):
        super(BM_ACN_1i4_6L, self).__init__()

        self.conv1 = Conv_No_BN(1, 20, k_size=25)
        self.conv2 = Conv_No_BN(20, 10)
        self.conv3 = Conv_No_BN(10, 10)
        self.conv4 = Conv_No_BN(10, 10)
        self.conv5 = Conv_No_BN(10, 1, k_size=5, drop_rate=.35)
        self.fc1 =  nn.Linear(in_features=4000 * 1, out_features=4000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x = x.view(x.shape[0], x.shape[-1])
        x = self.fc1(x)

        x = torch.nn.Softplus()(x)

        return x

class BM_ACN_2i4_6L(nn.Module):
    '''
    This is used to pretrain the X feature set
    '''
    def __init__(self):
        super(BM_ACN_2i4_6L, self).__init__()

        self.conv1 = Conv_No_BN(1, 20, k_size=25)
        self.conv2 = Conv_No_BN(20, 10)
        self.conv3 = Conv_No_BN(10, 10)
        self.conv4 = Conv_No_BN(10, 10)
        self.conv5 = Conv_No_BN(1, 1, k_size=5)
        self.conv6 = Conv_No_BN(1, 1, k_size=91)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        ## Remove empty middle shape diminsion
        ## Reshape conv6 output to work as output 
        ## to the softplus activation        
        x = x.view(x.shape[0], x.shape[-1])
        x = torch.nn.Softplus()(x)

        return x

class BM_ACN_3i4_P_6L(nn.Module):
    '''
    This is used to pretrain the (x,y) feature set
    '''
    def __init__(self):
        super(BM_ACN_3i4_P_6L, self).__init__()

        self.conv1 = Conv_No_BN(1, 20, k_size=25)
        self.conv2 = Conv_No_BN(20, 10)
        self.conv3 = Conv_No_BN(10, 10)
        self.conv4 = Conv_No_BN(10, 10)
        self.conv5 = Conv_No_BN(1, 1, k_size=5)
        self.conv6 = Conv_No_BN(1, 1, k_size=91)

        # Perturbation layers
        # Notice how there are two less layers in "perturbative" 
        ## compared to "non-perturbative"
        self.pConv1 = Conv_No_BN(2, 20, k_size=25)
        self.pConv2 = Conv_No_BN(10, 10)
        self.pConv3 = Conv_No_BN(10, 1)
        self.pFC1 = nn.Linear(in_features=4000 * 1, out_features=4000)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        x0 = self.conv1(x0)
        x0 = self.conv2(x0)
        x0 = self.conv3(x0)
        x0 = self.conv4(x0)
        x0 = self.conv5(x0)
        x0 = self.conv6(x0)
        
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        
        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)

        # Remove empty middle shape diminsion
        x1 = x1.view(x1.shape[0], x1.shape[-1])

        x1 = self.pFC(x1)

        ## 30/3/2021: Take the product of the two feature sets as an 
        ## output layer. Concatenation has been tried and passed 
        ## through another convolutional layer, but it showed 
        ## little/no difference in performance. This uses less memory.
        neuronValues = torch.nn.Softplus()(x0 * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

class BM_ACN_4i4_P_6L_C(nn.Module):
    def __init__(self):
        super(BM_ACN_4i4_P_6L_C, self).__init__()

        self.conv1 = Conv_No_BN(1, 20, k_size=25)
        self.conv2 = Conv_No_BN(20, 10)
        self.conv3 = Conv_No_BN(10, 10)
        self.conv4 = Conv_No_BN(10, 10)
        self.conv5 = Conv_No_BN(1, 1, k_size=5)
        self.conv6 = Conv_No_BN(1, 1, k_size=91)

        # Perturbation layers
        # Notice how there are two less layers in "perturbative" 
        ## compared to "non-perturbative"
        self.pConv1 = Conv_No_BN(2, 20, k_size=25)
        self.pConv2 = Conv_No_BN(10, 10)
        self.pConv3 = Conv_No_BN(10, 1)
        self.pConv4 = Conv_No_BN(1, 1, k_size=91)

        self.largeConv = Conv_No_BN(2, 1, k_size=91)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        x0 = self.conv1(x0)
        x0 = self.conv2(x0)
        x0 = self.conv3(x0)
        x0 = self.conv4(x0)
        x0 = self.conv5(x0)
        x0 = self.conv6(x0)
        
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        
        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)
        x1 = self.pConv4(x1)

        ## 30/3/2021: Take the product of the two feature sets as an 
        ## output layer. Concatenation has been tried and passed 
        ## through another convolutional layer, but it showed 
        ## little/no difference in performance. This uses less memory.

        # Run concatenated "perturbative" and "non-perturbative" features through a convolutional layer,
        # then softplus for output layer. This should, maybe, work better than just taking the product of 
        # the two feature sets as an output layer (which is what used to be done). 
        x0_and_x1 = self.largeConv(torch.cat([x0, x1], 1))
        x0_and_x1 = x0_and_x1.view(x0_and_x1.shape[0], x0_and_x1.shape[-1])
        neuronValues = torch.nn.Softplus()(x0_and_x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

class BM_ACN_4i4_P_6L(nn.Module):
    def __init__(self):
        super(BM_ACN_4i4_P_6L, self).__init__()
        self.conv1 = Conv_No_BN(1, 20, k_size=25)
        self.conv2 = Conv_No_BN(20, 10)
        self.conv3 = Conv_No_BN(10, 10)
        self.conv4 = Conv_No_BN(10, 10)
        self.conv5 = Conv_No_BN(1, 1, k_size=5)
        self.conv6 = Conv_No_BN(1, 1, k_size=91)

        # Perturbation layers
        # Notice how there are two less layers in "perturbative" 
        ## compared to "non-perturbative"
        self.pConv1 = Conv_No_BN(2, 20, k_size=25)
        self.pConv2 = Conv_No_BN(10, 10)
        self.pConv3 = Conv_No_BN(10, 1)
        self.pConv4 = Conv_No_BN(1, 1, k_size=91)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        x0 = self.conv1(x0)
        x0 = self.conv2(x0)
        x0 = self.conv3(x0)
        x0 = self.conv4(x0)
        x0 = self.conv5(x0)
        x0 = self.conv6(x0)
        
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        
        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)
        x1 = self.pConv4(x1)

        x1 = x1.view(x1.shape[0], x1.shape[-1])

        neuronValues = torch.nn.Softplus()(x0 * x1)

        return neuronValues


'''
Benchmark network architectures with the following attributes:
1. Same as benchmark [BM] model but with BatchNormalization.
'''
class ACN_1i4_6L_BN(nn.Module):
    # creates model architecture
    def __init__(self):
        super(ACN_1i4_6L_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(10, 1, k_size=5, drop_rate=.35)
        self.fc1 =  nn.Linear(in_features=4000 * 1, out_features=4000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x = x.view(x.shape[0], x.shape[-1])
        x = self.fc1(x)

        x = torch.nn.Softplus()(x)

        return x

class ACN_2i4_6L_BN(nn.Module):
    '''
    This is used to pretrain the X feature set
    '''
    def __init__(self):
        super(ACN_2i4_6L_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(1, 1, k_size=5)
        self.conv6 = Conv(1, 1, k_size=91)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        ## Remove empty middle shape diminsion
        ## Reshape conv6 output to work as output 
        ## to the softplus activation        
        x = x.view(x.shape[0], x.shape[-1])
        x = torch.nn.Softplus()(x)

        return x

class ACN_3i4_P_6L_BN(nn.Module):
    '''
    This is used to pretrain the (x,y) feature set
    '''
    def __init__(self):
        super(ACN_3i4_P_6L_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(1, 1, k_size=5)
        self.conv6 = Conv(1, 1, k_size=91)

        # Perturbation layers
        # Notice how there are two less layers in "perturbative" 
        ## compared to "non-perturbative"
        self.pConv1 = Conv(2, 20, k_size=25)
        self.pConv2 = Conv(10, 10)
        self.pConv3 = Conv(10, 1)
        self.pFC1 = nn.Linear(in_features=4000 * 1, out_features=4000)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        x0 = self.conv1(x0)
        x0 = self.conv2(x0)
        x0 = self.conv3(x0)
        x0 = self.conv4(x0)
        x0 = self.conv5(x0)
        x0 = self.conv6(x0)
        
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        
        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)

        # Remove empty middle shape diminsion
        x1 = x1.view(x1.shape[0], x1.shape[-1])

        x1 = self.pFC(x1)

        ## 30/3/2021: Take the product of the two feature sets as an 
        ## output layer. Concatenation has been tried and passed 
        ## through another convolutional layer, but it showed 
        ## little/no difference in performance. This uses less memory.
        neuronValues = torch.nn.Softplus()(x0 * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

class ACN_4i4_P_6L_BN_C(nn.Module):
    def __init__(self):
        super(ACN_4i4_P_6L_BN_C, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(1, 1, k_size=5)
        self.conv6 = Conv(1, 1, k_size=91)

        # Perturbation layers
        # Notice how there are two less layers in "perturbative" 
        ## compared to "non-perturbative"
        self.pConv1 = Conv(2, 20, k_size=25)
        self.pConv2 = Conv(10, 10)
        self.pConv3 = Conv(10, 1)
        self.pConv4 = Conv(1, 1, k_size=91)

        self.largeConv = Conv(2, 1, k_size=91)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        x0 = self.conv1(x0)
        x0 = self.conv2(x0)
        x0 = self.conv3(x0)
        x0 = self.conv4(x0)
        x0 = self.conv5(x0)
        x0 = self.conv6(x0)
        
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        
        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)
        x1 = self.pConv4(x1)

        ## 30/3/2021: Take the product of the two feature sets as an 
        ## output layer. Concatenation has been tried and passed 
        ## through another convolutional layer, but it showed 
        ## little/no difference in performance. This uses less memory.

        # Run concatenated "perturbative" and "non-perturbative" features through a convolutional layer,
        # then softplus for output layer. This should, maybe, work better than just taking the product of 
        # the two feature sets as an output layer (which is what used to be done). 
        x0_and_x1 = self.largeConv(torch.cat([x0, x1], 1))
        x0_and_x1 = x0_and_x1.view(x0_and_x1.shape[0], x0_and_x1.shape[-1])
        neuronValues = torch.nn.Softplus()(x0_and_x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues
    
class BM_ACN_4i4_P_6L_BN(nn.Module):
    def __init__(self):
        super(BM_ACN_4i4_P_6L_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(1, 1, k_size=5)
        self.conv6 = Conv(1, 1, k_size=91)

        # Perturbation layers
        # Notice how there are two less layers in "perturbative" 
        ## compared to "non-perturbative"
        self.pConv1 = Conv(2, 20, k_size=25)
        self.pConv2 = Conv(10, 10)
        self.pConv3 = Conv(10, 1)
        self.pConv4 = Conv(1, 1, k_size=91)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        x0 = self.conv1(x0)
        x0 = self.conv2(x0)
        x0 = self.conv3(x0)
        x0 = self.conv4(x0)
        x0 = self.conv5(x0)
        x0 = self.conv6(x0)
        
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        
        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)
        x1 = self.pConv4(x1)

        x1 = x1.view(x1.shape[0], x1.shape[-1])

        neuronValues = torch.nn.Softplus()(x0 * x1)

        return neuronValues


'''
Modified network architecture of benchmark with the following attributes:
NOTE: All attributes shared with benchmark are omitted
1. Batch Normalization in each layer
2. One skip connection added
3. Channel count follows the format:    16-9-9-9-1-1 (X), 16-9-9-01 (x, y), 20-1 (X, x, y)
4. Kernel size follows the format:      25-15-15-15-15-91 (X), 25-15-15-91 (x, y),  91  (X, x, y)
5. SimpleCNN5Layer, TwoFeature_CNN6Layer_A, and All_CNN6Layer_A are intermediate models to obtain well-trained models to use for the perturbative model.
'''
class ACN_1i4_6L_BN_RC1(nn.Module):
    # creates model architecture
    def __init__(self):
        super(ACN_1i4_6L_BN_RC1, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 9)
        self.conv3 = Conv(9, 9)
        self.conv4 = Conv(9, 9)
        self.conv5 = Conv(9, 1, drop_rate=.35)
        self.fc1 =  nn.Linear(in_features=4000 * 1, out_features=4000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x = x.view(x.shape[0], x.shape[-1])
        x = self.fc1(x)

        x = torch.nn.Softplus()(x)

        return x
    
class ACN_2i4_6L_1S_BN_RC1(nn.Module):
    '''
    This is used to pretrain the X feature set
    '''
    def __init__(self):
        super(ACN_2i4_6L_1S_BN_RC1, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 9)
        self.conv3 = Conv(9+20, 9)
        self.conv4 = Conv(9, 9)
        self.conv5 = Conv(9, 1)
        self.conv6 = Conv(1, 1, k_size=91)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x2,x1], 1))
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x = self.conv6(x5)

        ## Remove empty middle shape diminsion
        ## Reshape conv6 output to work as output 
        ## to the softplus activation        
        x = x.view(x.shape[0], x.shape[-1])
        x = torch.nn.Softplus()(x)

        return x

class ACN_3i4_P_6L_1S_BN_RC1(nn.Module):
    '''
    This is used to pretrain the (x,y) feature set
    '''
    def __init__(self):
        super(ACN_3i4_P_6L_1S_BN_RC1, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 9)
        self.conv3 = Conv(9+20, 9)
        self.conv4 = Conv(9, 9)
        self.conv5 = Conv(9, 1)
        self.conv6 = Conv(1, 1, k_size=91)

        # Perturbation layers
        # Notice how there are two less layers in "perturbative" 
        ## compared to "non-perturbative"
        self.pConv1 = Conv(2, 20, k_size=25)
        self.pConv2 = Conv(20, 10)
        self.pConv3 = Conv(10, 1)
        self.pFC1 = nn.Linear(in_features=4000 * 1, out_features=4000)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        x01 = self.conv1(x0)
        x2 = self.conv2(x01)
        x3 = self.conv3(torch.cat([x2,x01], 1))
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x0 = self.conv6(x5)
        
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        
        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)

        # Remove empty middle shape diminsion
        x1 = x1.view(x1.shape[0], x1.shape[-1])

        x1 = self.pFC(x1)

        ## 30/3/2021: Take the product of the two feature sets as an 
        ## output layer. Concatenation has been tried and passed 
        ## through another convolutional layer, but it showed 
        ## little/no difference in performance. This uses less memory.
        neuronValues = torch.nn.Softplus()(x0 * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

class ACN_4i4_P_6L_1S_BN_RC1(nn.Module):
    ## This is the perturbative model
    def __init__(self):
        super(BM_ACN_4i4_P_6L_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 9)
        self.conv3 = Conv(9+20, 9)
        self.conv4 = Conv(9, 9)
        self.conv5 = Conv(9, 1)
        self.conv6 = Conv(1, 1, k_size=91)

        # Perturbation layers
        # Notice how there are two less layers in "perturbative" 
        ## compared to "non-perturbative"
        self.pConv1 = Conv(2, 20, k_size=25)
        self.pConv2 = Conv(20, 10)
        self.pConv3 = Conv(10, 1)
        self.pConv4 = Conv(1, 1, k_size=91)

    def forward(self, neuronValues):

        ## Since there is no way to exclude Xsq from being loaded in the notebook (that I know of) this
        ## needs to be changed to accomadate for this fact. This would be done with the following code:

        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X (excludes Xsq)
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        x01 = self.conv1(x0)
        x2 = self.conv2(x01)
        x3 = self.conv3(torch.cat([x2, x01], 1))
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x0 = self.conv6(x5)
        
        x0 = x0.view(x0.shape[0], x0.shape[-1])
        
        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)
        x1 = self.pConv4(x1)

        x1 = x1.view(x1.shape[0], x1.shape[-1])

        neuronValues = torch.nn.Softplus()(x0 * x1)

        return neuronValues


'''
Modified network architecture of benchmark with the following attributes:
NOTE: All attributes shared with benchmark are omitted
1. Batch Normalization in each layer
2. Three feature set using X, x, y.
3. 8 layer convolutional architecture for X and Xsq feature set.
4. 4 layer conovlutional architecture for x and y feature set.
5. Takes element-wise product of the two feature sets for softplus.
6. Channel count follows the format:    01-20-10-10-10-10-10-01-01 (X), 02-10-01-01 (x, y)
7. Kernel size follows the format:      25-15-15-15-15-15-05-91 (X),    25-15-15-91 (x, y)
8. 3 skip connections, located at layers 3, 5, and 7
'''
class ACN_1i4_8L_3S_BN(nn.Module):
    # creates model architecture
    def __init__(self):
        super(ACN_1i4_8L_3S_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(10+10, 10)
        self.conv6 = Conv(10, 10)
        self.conv7 = Conv(10+10, 1, k_size=5, drop_rate=.35)
        self.fc1 =  nn.Linear(in_features=4000 * 1, out_features=4000)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x2, x1], 1))
        x4 = self.conv4(x3)
        x5 = self.conv5(torch.cat([x4, x3], 1))
        x6 = self.conv6(x5)
        x7 = self.conv7(torch.cat([x6, x5], 1))

        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x = x7.view(x7.shape[0], x7.shape[-1])
        x = self.fc1(x)

        x = torch.nn.Softplus()(x)

        return x
                                
class ACN_2i4_8L_3S_BN(nn.Module):
    '''
    This is used to pretrain the X feature set
    '''
    # creates model architecture
    def __init__(self):
        super(ACN_2i4_8L_3S_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(10+10, 10)
        self.conv6 = Conv(10, 10)
        self.conv7 = Conv(10+10, 1, k_size=5)
        self.conv8 = Conv(1, 1, k_size=91)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x2, x1], 1))
        x4 = self.conv4(x3)
        x5 = self.conv5(torch.cat([x4, x3], 1))
        x6 = self.conv6(x5)
        x7 = self.conv7(torch.cat([x6, x5], 1))

        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x8 = self.conv8(x7)
        x = x8.view(x8.shape[0], x8.shape[-1])

        x = torch.nn.Softplus()(x)

        return x

class ACN_3i4_P_8L_3S_BN(nn.Module):
    '''
    This is used to pretrain the (x,y) feature set
    '''
    def __init__(self):
        super(ACN_3i4_P_8L_3S_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(10+10, 10)
        self.conv6 = Conv(10, 10)
        self.conv7 = Conv(10+10, 1, k_size=5)
        self.conv8 = Conv(1, 1, k_size=91)

        self.pConv1 = Conv(2, 20, k_size=25)
        self.pConv2 = Conv(20, 10)
        self.pConv3 = Conv(10, 1)
        self.pFC = nn.Linear(in_features=4000 * 1, out_features=4000)

    def forward(self, x):
        x01 = self.conv1(x)
        x2 = self.conv2(x01)
        x3 = self.conv3(torch.cat([x2, x01], 1))
        x4 = self.conv4(x3)
        x5 = self.conv5(torch.cat([x4, x3], 1))
        x6 = self.conv6(x5)
        x7 = self.conv7(torch.cat([x6, x5], 1))
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x8 = self.conv8(x7)
        x0 = x8.view(x8.shape[0], x8.shape[-1])

        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x1 = x1.view(x1.shape[0], x1.shape[-1])
        x1 = self.pFC(x1)

        neuronValues = torch.nn.Softplus()(x0 * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

class ACN_4i4_P_8L_3S_BN(nn.Module):
    def __init__(self):
        super(ACN_4i4_P_8L_3S_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(10+10, 10)
        self.conv6 = Conv(10, 10)
        self.conv7 = Conv(10+10, 1, k_size=5)
        self.conv8 = Conv(1, 1, k_size=91)

        self.pConv1 = Conv(2, 20, k_size=25)
        self.pConv2 = Conv(20, 10)
        self.pConv3 = Conv(10, 1)
        self.pConv4 = Conv(1, 1, k_size=91)

    def forward(self, x):
        x01 = self.conv1(x)
        x2 = self.conv2(x01)
        x3 = self.conv3(torch.cat([x2, x01], 1))
        x4 = self.conv4(x3)
        x5 = self.conv5(torch.cat([x4, x3], 1))
        x6 = self.conv6(x5)
        x7 = self.conv7(torch.cat([x6, x5], 1))
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x8 = self.conv8(x7)
        x0 = x8.view(x8.shape[0], x8.shape[-1])

        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)
        x1 = self.pConv4(x1)
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x1 = x1.view(x1.shape[0], x1.shape[-1])

        neuronValues = torch.nn.Softplus()(x0 * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues


'''
Modified network architecture of benchmark with the following attributes:
NOTE: All attributes shared with benchmark are omitted
1. Batch Normalization in each layer
2. Three feature set using X, x, y.
3. 10 layer convolutional architecture for X feature set.
4. 4 layer conovlutional architecture for x and y feature set.
5. Takes element-wise product of the two feature sets for final layer.
6. Channel count follows the format:    01-20-10-10-10-10-07-05-01-01 (X), 20-10-10-01 (x, y),  20-01 (X, x, y)
7. Kernel size follows the format:      25-15-15-15-15-15-09-05-91 (X),    25-15-15-91 (x, y),  25-91 (X, x, y)
8. 4 skip connections, located at layers 3,5,7,9
'''
class ACN_1i4_10L_4S_BN(nn.Module):
    def __init__(self):
        super(ACN_1i4_10L_4S_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(10+10, 10)
        self.conv6 = Conv(10, 10)
        self.conv7 = Conv(10+10, 7)
        self.conv8 = Conv(7, 5, k_size=9)
        self.conv9 = Conv(5+7, 1, k_size=5, drop_rate=.35)
        self.fc1 =  nn.Linear(in_features=4000 * 1, out_features=4000)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x2, x1], 1))
        x4 = self.conv4(x3)
        x5 = self.conv5(torch.cat([x4, x3], 1))
        x6 = self.conv6(x5)
        x7 = self.conv7(torch.cat([x6, x5], 1))
        x8 = self.conv8(x7)
        x9 = self.conv9(torch.cat([x8, x7], 1))
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x = x9.squeeze(1)
        x = self.fc1(x)

        x = torch.nn.Softplus()(x)

        return x
                                   
class ACN_2i4_10L_4S_BN(nn.Module):
    '''
    This is used to pretrain the X feature set
    '''
    def __init__(self):
        super(ACN_2i4_10L_4S_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(10+10, 10)
        self.conv6 = Conv(10, 10)
        self.conv7 = Conv(10+10, 7)
        self.conv8 = Conv(7, 5, k_size=9)
        self.conv9 = Conv(5+7, 1, k_size=5, drop_rate=.35)
        self.conv10 = Conv(1, 1, k_size=91)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x2, x1], 1))
        x4 = self.conv4(x3)
        x5 = self.conv5(torch.cat([x4, x3], 1))
        x6 = self.conv6(x5)
        x7 = self.conv7(torch.cat([x6, x5], 1))
        x8 = self.conv8(x7)
        x9 = self.conv9(torch.cat([x8, x7], 1))
        x10 = self.conv10(x9)
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x = x10.squeeze(1)

        x = torch.nn.Softplus()(x)

        return x

class ACN_3i4_P_10L_4S_BN(nn.Module):
    '''
    This is used to pretrain the (x,y) feature set
    '''
    def __init__(self):
        super(ACN_3i4_P_10L_4S_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(10+10, 10)
        self.conv6 = Conv(10, 10)
        self.conv7 = Conv(10+10, 7)
        self.conv8 = Conv(7, 5, k_size=9)
        self.conv9 = Conv(5+7, 1, k_size=5)
        self.conv10 = Conv(1, 1, k_size=91)

        self.pConv1 = Conv(2, 20, k_size=25)
        self.pConv2 = Conv(20, 10)
        self.pConv3 = Conv(10, 1)
        self.pFC = nn.Linear(in_features=4000 * 1, out_features=4000)

    def forward(self, neuronValues):
        
        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        x01 = self.conv1(x0)
        x2 = self.conv2(x01)
        x3 = self.conv3(torch.cat([x2, x01], 1))
        x4 = self.conv4(x3)
        x5 = self.conv5(torch.cat([x4, x3], 1))
        x6 = self.conv6(x5)
        x7 = self.conv7(torch.cat([x6, x5], 1))
        x8 = self.conv8(x7)
        x9 = self.conv9(torch.cat([x8, x7], 1))
        x10 = self.conv10(x9)
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x0 = x10.view(x10.shape[0], x10.shape[-1])

        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x1 = x1.view(x1.shape[0], x1.shape[-1])
        x1 = self.pFC(x1)

        neuronValues = torch.nn.Softplus()(x0 * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

        x = torch.nn.Softplus()(x)

        return x

class ACN_4i4_P_10L_4S_BN(nn.Module):
    def __init__(self):
        super(ACN_4i4_P_10L_4S_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(10+10, 10)
        self.conv6 = Conv(10, 10)
        self.conv7 = Conv(10+10, 7)
        self.conv8 = Conv(7, 5, k_size=9)
        self.conv9 = Conv(5+7, 1, k_size=5)
        self.conv10 = Conv(1, 1, k_size=91)

        self.pConv1 = Conv(2, 20, k_size=25)
        self.pConv2 = Conv(20, 10)
        self.pConv3 = Conv(10, 1)
        self.pConv4 = Conv(1, 1, k_size=91)

    def forward(self, neuronValues):
        
        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        x01 = self.conv1(x0)
        x2 = self.conv2(x01)
        x3 = self.conv3(torch.cat([x2, x01], 1))
        x4 = self.conv4(x3)
        x5 = self.conv5(torch.cat([x4, x3], 1))
        x6 = self.conv6(x5)
        x7 = self.conv7(torch.cat([x6, x5], 1))
        x8 = self.conv8(x7)
        x9 = self.conv9(torch.cat([x8, x7], 1))
        x10 = self.conv10(x9)
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x0 = x10.view(x10.shape[0], x10.shape[-1])

        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)
        x1 = self.pConv4(x1)
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x1 = x1.view(x1.shape[0], x1.shape[-1])

        neuronValues = torch.nn.Softplus()(x0 * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

        x = torch.nn.Softplus()(x)

        return x

    
'''
Modified network architecture of benchmark with the following attributes:
NOTE: All attributes shared with benchmark are omitted
1. Batch Normalization in each layer
2. Three feature set using X, x, y.
3. 10 layer convolutional architecture for X feature set.
4. 4 layer conovlutional architecture for x and y feature set.
5. Takes element-wise product of the two feature sets for final layer.
6. Channel count follows the format:    01-20-10-10-10-10-07-05-01-01 (X), 20-10-10-01 (x, y),  20-01 (X, x, y)
7. Kernel size follows the format:      25-15-15-15-15-15-09-05-91 (X),    25-15-15-91 (x, y),  25-91 (X, x, y)
8. 4 skip connections, located at layers 3,5,7,9
9. Batch Normalization used on input layer
'''
class ACN_1i4_10L_4S_BN_NI(nn.Module):
    def __init__(self):
        super(ACN_1i4_10L_4S_BN_NI, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(10+10, 10)
        self.conv6 = Conv(10, 10)
        self.conv7 = Conv(10+10, 7)
        self.conv8 = Conv(7, 5, k_size=9)
        self.conv9 = Conv(5+7, 1, k_size=5, drop_rate=.35)
        self.fc1 =  nn.Linear(in_features=4000 * 1, out_features=4000)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x2, x1], 1))
        x4 = self.conv4(x3)
        x5 = self.conv5(torch.cat([x4, x3], 1))
        x6 = self.conv6(x5)
        x7 = self.conv7(torch.cat([x6, x5], 1))
        x8 = self.conv8(x7)
        x9 = self.conv9(torch.cat([x8, x7], 1))
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x = x9.view(x9.shape[0], x9.shape[-1])
        x = self.fc1(x)

        x = torch.nn.Softplus()(x)

        return x
                                   
class ACN_2i4_10L_4S_BN_NI(nn.Module):
    '''
    This is used to pretrain the X feature set
    '''
    def __init__(self):
        super(ACN_2i4_10L_4S_BN_NI, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(10+10, 10)
        self.conv6 = Conv(10, 10)
        self.conv7 = Conv(10+10, 7)
        self.conv8 = Conv(7, 5, k_size=9)
        self.conv9 = Conv(5+7, 1, k_size=5)
        self.conv10 = Conv(1, 1, k_size=91)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x2, x1], 1))
        x4 = self.conv4(x3)
        x5 = self.conv5(torch.cat([x4, x3], 1))
        x6 = self.conv6(x5)
        x7 = self.conv7(torch.cat([x6, x5], 1))
        x8 = self.conv8(x7)
        x9 = self.conv9(torch.cat([x8, x7], 1))
        x10 = self.conv10(x9)
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x = x10.view(x10.shape[0], x10.shape[-1])

        x = torch.nn.Softplus()(x)

        return x

class ACN_3i4_P_10L_4S_BN_NI(nn.Module):
    '''
    This is used to pretrain the (x,y) feature set
    '''
    def __init__(self):
        super(ACN_3i4_P_10L_4S_BN_NI, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(10+10, 10)
        self.conv6 = Conv(10, 10)
        self.conv7 = Conv(10+10, 7)
        self.conv8 = Conv(7, 5, k_size=9)
        self.conv9 = Conv(5+7, 1, k_size=5)
        self.conv10 = Conv(1, 1, k_size=91)

        self.pConv1 = Conv(2, 20, k_size=25)
        self.pConv2 = Conv(20, 10)
        self.pConv3 = Conv(10, 1)
        self.pFC = nn.Linear(in_features=4000 * 1, out_features=4000)

    def forward(self, neuronValues):
        
        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y
        
        x0 = nn.BatchNorm1D(x0)
        x1 = nn.BatchNorm1D(x1)

        x01 = self.conv1(x0)
        x2 = self.conv2(x01)
        x3 = self.conv3(torch.cat([x2, x01], 1))
        x4 = self.conv4(x3)
        x5 = self.conv5(torch.cat([x4, x3], 1))
        x6 = self.conv6(x5)
        x7 = self.conv7(torch.cat([x6, x5], 1))
        x8 = self.conv8(x7)
        x9 = self.conv9(torch.cat([x8, x7], 1))
        x10 = self.conv10(x9)
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x0 = x10.view(x10.shape[0], x10.shape[-1])

        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x1 = x1.view(x1.shape[0], x1.shape[-1])
        x1 = self.pFC(x1)

        neuronValues = torch.nn.Softplus()(x0 * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

        x = torch.nn.Softplus()(x)

        return x

class ACN_4i4_P_10L_4S_BN_NI(nn.Module):
    def __init__(self):
        super(ACN_4i4_P_10L_4S_BN_NI, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10, 10)
        self.conv5 = Conv(10+10, 10)
        self.conv6 = Conv(10, 10)
        self.conv7 = Conv(10+10, 7)
        self.conv8 = Conv(7, 5, k_size=9)
        self.conv9 = Conv(5+7, 1, k_size=5)
        self.conv10 = Conv(1, 1, k_size=91)

        self.pConv1 = Conv(2, 20, k_size=25)
        self.pConv2 = Conv(20, 10)
        self.pConv3 = Conv(10, 1)
        self.pConv4 = Conv(1, 1, k_size=91)

    def forward(self, neuronValues):
        
        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        x0 = nn.BatchNorm1D(x0)
        x1 = nn.BatchNorm1D(x1)
        
        x01 = self.conv1(x0)
        x2 = self.conv2(x01)
        x3 = self.conv3(torch.cat([x2, x01], 1))
        x4 = self.conv4(x3)
        x5 = self.conv5(torch.cat([x4, x3], 1))
        x6 = self.conv6(x5)
        x7 = self.conv7(torch.cat([x6, x5], 1))
        x8 = self.conv8(x7)
        x9 = self.conv9(torch.cat([x8, x7], 1))
        x10 = self.conv10(x9)
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x0 = x10.view(x10.shape[0], x10.shape[-1])

        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)
        x1 = self.pConv4(x1)
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x1 = x1.view(x1.shape[0], x1.shape[-1])

        neuronValues = torch.nn.Softplus()(x0 * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

        x = torch.nn.Softplus()(x)

        return x
    

'''
Modified network architecture of benchmark with the following attributes:
NOTE: All attributes shared with benchmark are omitted
1. Batch Normalization in each layer
2. Three feature set using X, x, y.
3. 8 layer convolutional architecture for X feature set.
4. 4 layer conovlutional architecture for x and y feature set.
5. Takes element-wise product of the two feature sets for softplus.
6. DenseNet skip connections; each layer's input are the features of each previous layers' outputs
7. Channel count follows the format:    01-20-10-10-10-10-10-01-01 (X), 02-10-01-01 (x, y)
8. Kernel size follows the format:      25-15-15-15-15-15-05-91 (X),    25-15-15-91 (x, y)
'''
class ACN_1i4_8L_DenseNet_BN(nn.Module):
    def __init__(self):
        super(ACN_1i4_8L_DenseNet_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10+10+20, 10)
        self.conv5 = Conv(10+10+10+20, 10)
        self.conv6 = Conv(10+10+10+10+20, 10)
        self.conv7 = Conv(10+10+10+10+10+20, 1, k_size=5, drop_rate=.35)
        self.fc1 =  nn.Linear(in_features=4000 * 1, out_features=4000)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], 1))
        x4 = self.conv4(torch.cat([x1,x2,x3], 1))
        x5 = self.conv5(torch.cat([x1,x2,x3,x4], 1))
        x6 = self.conv6(torch.cat([x1,x2,x3,x4,x5], 1))
        x7 = self.conv7(torch.cat([x1,x2,x3,x4,x5,x6], 1))
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x = x7.view(x7.shape[0], x7.shape[-1])
        x = self.fc1(x)

        x = torch.nn.Softplus()(x)

        return x
                
class ACN_2i4_8L_DenseNet_BN(nn.Module):
    '''
    This is used to pretrain the X feature set
    '''
    def __init__(self):
        super(ACN_2i4_8L_DenseNet_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10+10+20, 10)
        self.conv5 = Conv(10+10+10+20, 10)
        self.conv6 = Conv(10+10+10+10+20, 10)
        self.conv7 = Conv(10+10+10+10+10+20, 1, k_size=5)
        self.conv8 = Conv(1+10+10+10+10+10+20, 1, k_size=91)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], 1))
        x4 = self.conv4(torch.cat([x1,x2,x3], 1))
        x5 = self.conv5(torch.cat([x1,x2,x3,x4], 1))
        x6 = self.conv6(torch.cat([x1,x2,x3,x4,x5], 1))
        x7 = self.conv7(torch.cat([x1,x2,x3,x4,x5,x6], 1))
        x8 = self.conv8(torch.cat([x1,x2,x3,x4,x5,x6,x7],1))
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x = x7.view(x7.shape[0], x7.shape[-1])

        x = torch.nn.Softplus()(x)

        return x

class ACN_3i4_8L_DenseNet_BN(nn.Module):
    '''
    This is used to pretrain the (x,y) feature set
    '''
    def __init__(self):
        super(ACN_3i4_8L_DenseNet_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10+10+20, 10)
        self.conv5 = Conv(10+10+10+20, 10)
        self.conv6 = Conv(10+10+10+10+20, 10)
        self.conv7 = Conv(10+10+10+10+10+20, 1, k_size=5)
        self.conv8 = Conv(1+10+10+10+10+10+20, 1, k_size=91)

        self.pConv1 = Conv(2, 20, k_size=25)
        self.pConv2 = Conv(20, 10)
        self.pConv3 = Conv(10, 1)
        self.pFC = nn.Linear(in_features=4000 * 1, out_features=4000)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        x01 = self.conv1(x0)
        x2 = self.conv2(x01)
        x3 = self.conv3(torch.cat([x01, x2], 1))
        x4 = self.conv4(torch.cat([x01,x2,x3], 1))
        x5 = self.conv5(torch.cat([x01,x2,x3,x4], 1))
        x6 = self.conv6(torch.cat([x01,x2,x3,x4,x5], 1))
        x7 = self.conv7(torch.cat([x01,x2,x3,x4,x5,x6], 1))
        x8 = self.conv8(torch.cat([x01,x2,x3,x4,x5,x6,x7],1))
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x0 = x7.view(x7.shape[0], x7.shape[-1])

        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x1 = x1.view(x1.shape[0], x1.shape[-1])
        x1 = self.pFC(x1)

        neuronValues = torch.nn.Softplus()(x0 * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

        x = torch.nn.Softplus()(x)

        return x

class ACN_4i4_8L_DenseNet_BN(nn.Module):
    def __init__(self):
        super(ACN_4i4_8L_DenseNet_BN, self).__init__()

        self.conv1 = Conv(1, 20, k_size=25)
        self.conv2 = Conv(20, 10)
        self.conv3 = Conv(10+20, 10)
        self.conv4 = Conv(10+10+20, 10)
        self.conv5 = Conv(10+10+10+20, 10)
        self.conv6 = Conv(10+10+10+10+20, 10)
        self.conv7 = Conv(10+10+10+10+10+20, 1, k_size=5)
        self.conv8 = Conv(1+10+10+10+10+10+20, 1, k_size=91)

        self.pConv1 = Conv(2, 20, k_size=25)
        self.pConv2 = Conv(20, 10)
        self.pConv3 = Conv(10, 1)
        self.pConv4 = Conv(1, 1, k_size=91)

    def forward(self, neuronValues):

        ## in the method definition, neuronValues corresponds to (X,x,y)
        ## here, we will use the name x0 to denote the (X) feature set and
        ## the name x1 to denote the (x,y) feature set
        x0 = neuronValues[:, 0:1, :]  ## picks out the 0 feature set, X
        x1 = neuronValues[:, 2:4, :]  ## picks out the 2 & 3 feature sets, x & y

        x01 = self.conv1(x0)
        x2 = self.conv2(x01)
        x3 = self.conv3(torch.cat([x01, x2], 1))
        x4 = self.conv4(torch.cat([x01,x2,x3], 1))
        x5 = self.conv5(torch.cat([x01,x2,x3,x4], 1))
        x6 = self.conv6(torch.cat([x01,x2,x3,x4,x5], 1))
        x7 = self.conv7(torch.cat([x01,x2,x3,x4,x5,x6], 1))
        x8 = self.conv8(torch.cat([x01,x2,x3,x4,x5,x6,x7],1))
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x0 = x7.view(x7.shape[0], x7.shape[-1])

        x1 = self.pConv1(x1)
        x1 = self.pConv2(x1)
        x1 = self.pConv3(x1)
        x1 = self.pConv4(x1)
        ## Remove empty middle shape diminsion
        ## reshape conv6 output to work as output 
        ## to the softplus activation
        x1 = x1.view(x1.shape[0], x1.shape[-1])

        neuronValues = torch.nn.Softplus()(x0 * x1)
        neuronValues = neuronValues.squeeze()

        return neuronValues

        x = torch.nn.Softplus()(x)

        return x
    