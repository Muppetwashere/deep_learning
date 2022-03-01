from cv2 import multiply
import torch
import torch.nn as nn
import  numpy as np

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()
        self.linear1 = torch.nn.Linear(36,10)
        self.linear2 = torch.nn.Linear(36,1)
        self.sequential = torch.nn.Sequential(
            nn.Linear(36,16),
            nn.Linear(16,1),
)


    def forward(self, x):
        fwl_mult_factors = x.permute(2,0,1) # reshape to be able to select fwl shape is [36,16,306]
        fwl_mult_factors = fwl_mult_factors [-1] # select fwl (the 36th parameter), shape is [16,306]
        fwl_mult_factors = fwl_mult_factors.unsqueeze(2) # add a 1 dim as dimension 2, shape here is [16,306,1]
        #x = self.linear1(x) # shape of x is [16,306,36] -> [16,306,10], the last minibatch only has 10 instead of 16
        k_coefficient = self.sequential(x) # shape here is [16,306,1]
        x = fwl_mult_factors * k_coefficient # this multiplies term-to-term, shape here is [16,306,1] -> [16,306,1]
        x = torch.sum(x,dim=1) #shape here is [16,1]
        return x


class LinearModel_mean(torch.nn.Module):

    def __init__(self):
        super(LinearModel_mean, self).__init__()
        self.sequential = torch.nn.Sequential(
            nn.Linear(8,20),
            nn.SELU(),
            nn.Linear(20,8),
            nn.SELU(),
            nn.Linear(8,1),
            nn.Sigmoid(),
)


    def forward(self, x):
        x = self.sequential(x)
        return x

class Linear_Model_param_Antonin(torch.nn.Module):

    def __init__(self):
        super(Linear_Model_param_Antonin, self).__init__()
        self.sequential = torch.nn.Sequential(
            nn.Linear(36*306,16),
            nn.SELU(),
            nn.Linear(16,4),
            nn.SELU(),
            nn.Linear(4,1),
            nn.Sigmoid(),
)


    def forward(self, x):
        x = x.reshape(x.size()[0],-1) #reshaping from [16,306,36] to [16,306*36], the last minibatch only has 10 instead of 16
        x = self.sequential(x)
        x = x.reshape(x.size()[0])
        return x

class Faulty_ExpertModel(torch.nn.Module):

    def __init__(self):
        super(ExpertModel, self).__init__()
        self.linear1 = torch.nn.Linear(36,10)
        self.linear2 = torch.nn.Linear(36,1)
        self.sequential = torch.nn.Sequential(
            nn.Linear(36,16),
            nn.ReLU(),
            nn.Linear(16,4),
            nn.Sigmoid(),
            nn.Linear(4,1)
)


    def forward(self, x):
        fwl_mult_factors = x.permute(2,0,1) # reshape to be able to select fwl shape is [36,16,306]
        fwl_mult_factors = fwl_mult_factors [-1] # select fwl (the 36th parameter), shape is [16,306]
        fwl_mult_factors = fwl_mult_factors.unsqueeze(2) # add a 1 dim as dimension 2, shape here is [16,306,1]
        #x = self.linear1(x) # shape of x is [16,306,36] -> [16,306,10], the last minibatch only has 10 instead of 16
        k_coefficient = self.sequential(x) # shape here is [16,306,1]
        x = fwl_mult_factors * k_coefficient # this multiplies term-to-term, shape here is [16,306,1] -> [16,306,1]
        x = torch.sum(x,dim=1) #shape here is [16,1]
        return x

class ExpertModel(torch.nn.Module):

    def __init__(self):
        super(ExpertModel, self).__init__()
        self.linear1 = torch.nn.Linear(36,10)
        self.linear2 = torch.nn.Linear(36,1)
        self.sequential = torch.nn.Sequential(
            nn.Linear(36,16),
            nn.SELU(),
            nn.Linear(16,4),
            nn.ReLU(),
            nn.Linear(4,1),
            nn.Sigmoid(),
)


    def forward(self, x):
        fwl_mult_factors = x.permute(2,0,1) # reshape to be able to select fwl shape is [36,16,306]
        fwl_mult_factors = fwl_mult_factors [-1] # select fwl (the 36th parameter), shape is [16,306]
        fwl_mult_factors = fwl_mult_factors.unsqueeze(2) # add a 1 dim as dimension 2, shape here is [16,306,1]
        #x = self.linear1(x) # shape of x is [16,306,36] -> [16,306,10], the last minibatch only has 10 instead of 16
        k_coefficient = self.sequential(x) # shape here is [16,306,1]
        x = fwl_mult_factors * k_coefficient # this multiplies term-to-term, shape here is [16,306,1] -> [16,306,1]
        x = torch.sum(x,dim=1) #shape here is [16,1]
        return x

class ExpertModel_with_loop(torch.nn.Module):

    def __init__(self):
        super(ExpertModel_with_loop, self).__init__()
        self.linear1 = torch.nn.Linear(36,10)
        self.linear2 = torch.nn.Linear(36,1)
        self.sequential = torch.nn.Sequential(
            nn.Linear(36,16),
            nn.SELU(),
            nn.Linear(16,4),
            nn.ReLU(),
            nn.Linear(4,1),
            nn.Sigmoid(),
)


    def forward(self, x):
        fwl_mult_factors = x.permute(2,0,1) # reshape to be able to select fwl shape is [36,16,306]
        fwl_mult_factors = fwl_mult_factors [-1] # select fwl (the 36th parameter), shape is [16,306]
        fwl_mult_factors = fwl_mult_factors.unsqueeze(2) # add a 1 dim as dimension 2, shape here is [16,306,1]
        #x = self.linear1(x) # shape of x is [16,306,36] -> [16,306,10], the last minibatch only has 10 instead of 16
        k_coefficient = self.sequential(x) # output shape here is [16,306,1]
        sm = [fwl_mult_factors.select(1,i) * k_coefficient.select(1,i) for i in range(0,fwl_mult_factors.shape[1]-1)] # this multiplies term-to-term, shape here is [16,306,1] -> [16,306,1]
        x = sum(sm) #shape here is [16,1]
        return x


class ExpertModel_No_Fwl(torch.nn.Module):

    def __init__(self):
        super(ExpertModel_No_Fwl, self).__init__()
        self.linear1 = torch.nn.Linear(36,10)
        self.linear2 = torch.nn.Linear(36,1)
        self.sequential = torch.nn.Sequential(
            nn.Linear(36,16),
            nn.SELU(),
            nn.Linear(16,4),
            nn.ReLU(),
            nn.Linear(4,1),
            nn.Sigmoid(),
)


    def forward(self, x):
        k_coefficient = self.sequential(x) # shape here is [16,306,1]
        x = k_coefficient # this multiplies term-to-term, shape here is [16,306,1] -> [16,306,1]
        x = torch.sum(x,dim=1) #shape here is [16,1]
        return x