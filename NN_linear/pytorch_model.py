import torch

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()
        self.linear1 = torch.nn.Linear(36*306,10)
        self.linear2 = torch.nn.Linear(10,1)


    def forward(self, x):
        x = x.reshape(x.size()[0],-1) #reshaping from [16,306,36] to [16,306*36], the last minibatch only has 10 instead of 16
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.reshape(x.size()[0])
        return x


