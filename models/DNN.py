import torch
import torch.nn as nn
from torch.autograd import Variable
import copy


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)
class DNN(nn.Module):
    def __init__(self, hidden_dim):
        super(DNN, self).__init__()

        self.fn = nn.Linear(12, 1)
        self.fis = nn.Linear(12, 1)
        self.time = nn.Linear(12, 1)
        self.dh = nn.Linear(12 , 1)
        initialize_weight(self.fn)
        self.sig = nn.Sigmoid()


    def forward(self, x,use_EM,pinn_flag,device=0):
        #[广义幂律:'GYML'; 半衰期：'HLR';ACT-R: 'ACTR']
        # 材料难度 a
        # a = x[:, [5, 11]]
        a = x
        # 时间 系数t
        # t = x[:, [0, 3, 4, 6, 9, 10]]
        t = x
        # 复习 系数t
        # d = x[:, [1, 2, 7, 8]]
        d = x
        if use_EM==True:
            if pinn_flag == 'HLR':
                #半衰期公式
                h = self.fn(x)
                out = torch.pow(2,-abs(h.squeeze()))
                return out
            elif pinn_flag== 'Wickelgren':
                a = self.fis(a)
                t = self.time(t)
                d = self.dh(d)
                out = d + torch.pow(abs(t), -a)
                # return out
            elif pinn_flag=='ACT-R':
                a = self.fis(a)
                t = self.time(t)
                d = self.dh(d)
                out = a + torch.log(torch.pow(abs(t), -d))
            elif pinn_flag == 'nom':
                out = self.fn(x)


        return self.sig(out)