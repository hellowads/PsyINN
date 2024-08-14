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
            elif pinn_flag== 'Wickelgren':
                a = self.fis(a)
                t = self.time(t)
                d = self.dh(d)
                out = d * torch.pow(abs(t), -a)
            elif pinn_flag=='ACT-R':
                a = self.fis(a)
                t = self.time(t)
                d = self.dh(d)
                out = a + torch.log(torch.pow(abs(t), -d))
            elif pinn_flag == 'nom':
                out = self.fn(x)
        else:
            if pinn_flag == 'HLR':
                #半衰期公式
                para = torch.tensor([ 0.0014, -0.0293, -0.0618,  0.1118, -0.1761, -0.1530,  0.2011,  0.5866,-0.6918, -0.0431, -0.4883,  0.0720]).to(device)
                para = torch.tensor(
                    [-0.2721, -0.6535, -0.2572,  0.8952, -0.1660,  0.8374, -0.3589,  0.8494,-0.8019,  0.0433, -0.3157, -0.1558]).to(device)
                # para[abs(para)<0.01]=0
                # para = torch.tensor([ 0.1027, -0.8600, -0.6949,  0.0862,  0.0287,  0.6136,  0.0699,  0.9844,-1.0948, -0.0241, -0.0053,  0.0188]).to(device)
                # out = torch.pow(2, -abs(x @ para +0.1368))
                out = torch.pow(2, -abs(x @ para +0.1354))
            elif pinn_flag== 'Wickelgren':
                da = torch.tensor([ 3.1370e-06, -1.5929e-05,  1.0641e-05,  5.4830e-06,  1.0921e-05,-9.3148e-06,  2.7098e-04,  9.7749e-02,  3.1829e-01, -3.2587e-03,9.9155e-02,  2.3024e-01]).to(device)
                dt = torch.tensor([-1.8212e-05,  2.2046e-05,  3.5368e-05, -7.0456e-05, -6.6306e-06,8.9821e-08,  2.4691e-01,  1.2114e+00,  2.2885e-02,  2.8536e-02,8.9629e-02,  5.1038e-01]).to(device)
                dd = torch.tensor([ 1.4424e-05, -1.1965e-05,  1.9291e-05, -2.3580e-06, -3.5014e-05, 1.9003e-05,  8.0112e-02,  2.0807e-01,  5.9315e-01,  1.4771e-02, 4.6308e-02,  2.8631e-01]).to(device)
                da = torch.tensor(
                    [ 0.0072,  0.0069, -0.0009, -0.1503, -0.0555,  0.0029, -0.3077,  0.2852, 0.6305,  0.0094, -0.0240, -0.1346]).to(device)
                dt = torch.tensor(
                    [ 0.0389,  0.0620,  0.0019, -0.1978, -0.2704,  0.2406,  0.2656,  0.6063,-0.8075,  0.0170, -0.0120,  0.0323]).to(
                    device)
                dd = torch.tensor(
                    [-0.1051, -0.0507, -0.0032,  0.0804, -0.0176, -0.0432,  0.1237, -0.9348, 0.3024,  0.0668, -0.0215,  0.1262]).to(device)

                out = (d.squeeze() @ dd + 0.7294) * torch.pow((abs(t @ dt +0.4150)), -(a @ da +0.2573))
                #best
                # out = (d.squeeze() @ dd + 0.4548) * torch.pow((abs(t @ dt + 0.2320)), -(a @ da + 0.4281))

            elif pinn_flag=='ACT-R':
                da = torch.tensor([ 0.0945, -0.1213,  0.0936, -0.0283,  0.0315, -0.1790,  0.2398,  0.0463,0.3815,  0.0508,  0.1675,  0.1459]).to(device)
                dt = torch.tensor([ 0.2645, -0.3485,  0.0567,  0.0650, -0.0568, -0.3495, -0.2434, -0.8413,0.0607, -0.0288, -0.2858, -0.4504]).to(device)
                dd = torch.tensor([-0.0728,  0.0341,  0.1388, -0.0922,  0.0585,  0.0152, -0.0151,  0.3831,0.2693,  0.0626,  0.2422,  0.4119]).to(device)
                out = (a @ da +0.2659) + torch.log(torch.pow(abs(t @ dt -0.1169), -(d @ dd +0.2759)))
                da = torch.tensor(
                    [ 0.0082, -0.0021,  0.0055,  0.0094, -0.0012, -0.0106,  0.2644, -0.4956, 0.0461,  0.0287, -0.0173,  0.1003]).to(device)
                dt = torch.tensor(
                    [ 0.0068, -0.0087,  0.0140,  0.0088,  0.0033, -0.0093, -0.2821, -1.0167,0.8402, -0.0259,  0.0168, -0.0986]).to(device)
                dd = torch.tensor(
                    [ 0.0080, -0.0024,  0.0043,  0.0089, -0.0017, -0.0100, -0.1557,  0.3690,0.3804, -0.0110, -0.0128,  0.0868]).to(device)
                out = (a @ da + 0.4453) + torch.log(torch.pow(abs(t @ dt -0.2955), -(d @ dd + 0.3654)))
            elif pinn_flag == 'nom':
                para = torch.tensor([-0.0772,  0.0458,  0.1078,  0.0994, -0.3042, -0.4247, -0.0677,  0.0449,0.0288,  0.0420,  0.0715, -0.0289]).to(device)
                out = x @ para + 0.7439

        # out = d*torch.pow(abs(t),-a)

        #ACT-R
        # out = a + torch.log(torch.pow(abs(t),-d))


        return out