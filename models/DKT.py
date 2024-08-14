import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)
class DKT(nn.Module):
    def __init__(self,n_question,p_num,embed_l,embed_p,  input_dim, hidden_dim, layer_num, output_dim):
        super(DKT, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.output_dim = output_dim
        self.lin = nn.Linear(input_dim+embed_p+embed_l,256)
        self.encoder = nn.LSTM(256, hidden_dim, layer_num,dropout=0.3)
        self.decoder = nn.LSTM(256, hidden_dim, layer_num)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()
        self.relu =  nn.ReLU()
        self.q_embed = nn.Embedding(n_question, embed_l, padding_idx=0)
        self.p_embed = nn.Embedding(p_num, embed_p, padding_idx=0)
        self.att = nn.Linear(hidden_dim,hidden_dim)
        self.att2 = nn.Linear(hidden_dim, hidden_dim)
        self.batchnorm = nn.BatchNorm2d(6,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.tanh = nn.Tanh()
        self.out = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.final= nn.Linear(10, 1)
        self.sig = nn.Sigmoid()
        initialize_weight(self.lin)
        initialize_weight(self.fc)

        self.fn = nn.Linear(12, 1)
        self.fis = nn.Linear(12, 1)
        self.time = nn.Linear(12, 1)
        self.dh = nn.Linear(12, 1)

        self.fn1 = nn.Linear(6, 1)
        self.fis2 = nn.Linear(6, 1)
        self.time3 = nn.Linear(6, 1)
        self.dh4 = nn.Linear(6, 1)

    def forward(self, x,targets,device=0):
        q_data = x[:, :, 4].unsqueeze(-1)
        q_embed_data = self.q_embed(q_data.to(dtype=torch.long)).squeeze()  # input : [batch_size, len_seq, embedding_dim]
        # user embeding
        p_data = x[:, :, 0].unsqueeze(-1)
        p_embed_data = self.p_embed(p_data.to(dtype=torch.long)).squeeze()
        # x = x[:,:,[0,1,2,3,4,5,6,7,8,11]]
        rnn_input = torch.cat([q_embed_data, p_embed_data, x],dim=2)
        rnn_input2 = torch.cat([self.p_embed(targets[:,0].to(dtype=torch.long)).squeeze(), self.q_embed(targets[:,4].to(dtype=torch.long)).squeeze(), targets], dim=1)
        h0 = Variable(torch.zeros(self.layer_num, x.size(1), self.hidden_dim)).to(device)
        c0 = Variable(torch.zeros(self.layer_num, x.size(1), self.hidden_dim)).to(device)
        encoder_input = rnn_input
        decoder_input = rnn_input2.unsqueeze(0)
        encoder_input = self.lin(encoder_input)
        decoder_input = self.lin(decoder_input)
        out1,(hn,cn) = self.encoder(encoder_input, (h0,c0))
        out2, _ = self.decoder(decoder_input,(hn,cn))
        #GRU 模块
        outall = out2#torch.cat([out1,out2],axis=0).transpose(0, 1).reshape(out1.shape[1],-1)
        # out2 =  sum(self.sig(self.att(torch.cat([out1,out2]))) * self.relu(self.att2(torch.cat([out1,out2]))))
        res = self.out(outall)
        z = copy.copy(res)
        res = self.final(res)
        # res = torch.pow(2,-x[-1,:,3]/res.squeeze())
        res = self.sig(res)
        return res.squeeze()