import copy
import json
import os
import pickle
import time
from typing import Dict, Optional, List

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import random
from .data import get_dataloaders, PredictorDataset
from .evaluate import evaluate
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True,allow_unused=True )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)
def create_pad_mask(t, pad):
    mask = (t == pad).unsqueeze(-2)
    return mask
class SRTrainer(object):
    def __init__(self,
                 model: nn.Module,
                 loss: nn.Module,
                 scaler,
                 device: torch.device,
                 optimizer,
                 reg_weight_decay: float,
                 reg_norm: int,
                 max_grad_norm: Optional[float] or Optional[str]):
        self.model = model.to(device)
        self.loss = loss.to(device)
        self.optimizer = optimizer
        self.scaler = scaler
        self.clip = max_grad_norm
        self.device = device
        self.norm = reg_norm
        self.weight_decay = reg_weight_decay

    def find_ux(self,inputs, targets,trainer):
        trainer.model.eval()
        trainer.optimizer.zero_grad()
        targets = targets.to(self.device)
        inputs = inputs.to(self.device).requires_grad_()
        predicts= trainer._run(inputs)
        ux_all = gradients(predicts, inputs)
        uh_last = ux_all[:, -1, [ 3, 5, 6, 8, 9, 10]]
        h_last = inputs[:, -1, [ 3, 5, 6, 8, 9, 10]]
        h_last = torch.cat([uh_last, h_last], dim=-1).requires_grad_()
        # trainer.optimizer.zero_grad()
        return h_last.detach()

    def train(self, inputs, targets,trainer,use_EM,pinn_flag):
        self.model.train()
        self.optimizer.zero_grad()
        # h_last= self.find_ux(inputs, targets,trainer).requires_grad_()
        trainer.model.eval()
        trainer.optimizer.zero_grad()
        targets = targets.to(self.device)
        inputs = inputs.to(self.device).requires_grad_()
        pre= trainer._run(inputs)
        # ux_all = gradients(pre, inputs)
        ux_all = torch.autograd.grad(pre, inputs, grad_outputs=torch.ones_like(pre),
                                   create_graph=True,
                                   only_inputs=True,allow_unused=True )[0]
        uh_last = ux_all[:, -1, [3, 5, 6, 8, 9, 10]]
        h_last = inputs[:, -1, [3, 5, 6, 8, 9, 10]]
        h_last = torch.cat([uh_last, h_last], dim=-1).requires_grad_()
        h_last = torch.tensor(h_last,requires_grad=True)
        h = h_last.to(self.device)
        predict = self._run(h,use_EM,pinn_flag)
        loss_function = nn.MSELoss()
        loss1 = loss_function(pre.to(self.device).squeeze(), targets.to(self.device).squeeze())
        loss2 = loss_function(predict.to(self.device).squeeze(), targets.to(self.device).squeeze())  +loss1
        reg = get_regularization(self.model, weight_decay=self.weight_decay, p=self.norm) #加入正则项
        (loss2+reg).backward()
        if self.clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        self.optimizer.step()
        return predict

    def predict(self, inputs, targets,trainer,use_EM,pinn_flag):
        self.model.eval()
        h_last = self.find_ux(inputs, targets, trainer)
        return self._run(h_last,use_EM,pinn_flag)

    def _run(self, inputs,use_EM,pinn_flag):
        #inputs[..., 0] = self.scaler.transform(inputs[..., 0], 0.0)
        outputs = self.model(inputs,use_EM,pinn_flag)
        return outputs  #self.scaler.inverse_transform(outputs, 0.0)

    def load_state_dict(self, model_state_dict, optimizer_state_dict):
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.model = self.model.to(self.device)
        set_device_recursive(self.optimizer.state, self.device)
class OursTrainer(object):
    def __init__(self,
                 model: nn.Module,
                 loss: nn.Module,
                 scaler,
                 device: torch.device,
                 optimizer,
                 reg_weight_decay: float,
                 reg_norm: int,
                 max_grad_norm: Optional[float] or Optional[str]):
        self.model = model.to(device)
        self.loss = loss.to(device)
        self.optimizer = optimizer
        self.scaler = scaler
        self.clip = max_grad_norm
        self.device = device
        self.norm = reg_norm
        self.weight_decay = reg_weight_decay

    def train(self, inputs, targets,srtrainer,use_EM,pinn_flag):
        self.model.train()
        self.optimizer.zero_grad()
        targets = targets.to(self.device)
        inputs = inputs.to(self.device).requires_grad_()
        predicts = self._run(inputs)
        if pinn_flag != 'No':
            ux_all = gradients(predicts, inputs)
            uh_last = ux_all[:, -1, [3, 5, 6, 8, 9, 10]]
            h_last = inputs[:, -1, [3, 5, 6, 8, 9, 10]]
            h_last = torch.cat([uh_last, h_last], dim=-1)
            pinnpre = srtrainer._run(h_last, use_EM, pinn_flag)
            loss_function = nn.MSELoss()
            losspinn = loss_function(pinnpre, targets)
            loss = self.loss(predicts.to(self.device).squeeze(), targets.to(self.device).squeeze()) + losspinn
        else:
            loss = self.loss(predicts.to(self.device).squeeze(), targets.to(self.device).squeeze())
        reg = get_regularization(self.model, weight_decay=self.weight_decay, p=self.norm) #加入正则项
        (loss).backward()
        # if self.clip is not None:
        #     nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        return predicts

    def predict(self, inputs):
        inputs = inputs.to(self.device).requires_grad_()
        self.model.eval()
        return self._run(inputs)

    def _run(self, inputs):
        inputs = inputs.transpose(0, 1).to(self.device)
        outputs = self.model(inputs[:-1, :, :], inputs[-1, :, :])
        inputs = inputs.transpose(0, 1)
        return outputs  #self.scaler.inverse_transform(outputs, 0.0)

    def load_state_dict(self, model_state_dict, optimizer_state_dict):
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.model = self.model.to(self.device)
        set_device_recursive(self.optimizer.state, self.device)
        '''
        print('Resume:load_state_dict__________________________')
        #print model's state_dict
        print('Model.state_dict:')
        for param_tensor in self.model.state_dict():
            #打印 key value字典
            print(param_tensor,'\t',self.model.state_dict()[param_tensor].size())
 
        #print optimizer's state_dict
        print('Optimizer,s state_dict:')
        for var_name in self.optimizer.state_dict():
            print(var_name,'\t',self.optimizer.state_dict()[var_name])
        '''



def train_model(
        datasets: Dict[str, PredictorDataset],
        batch_size: int,
        folder: str,
        trainer: OursTrainer,
        scheduler,
        epochs: int,
        early_stop_steps: int ,
min,max,
srtrainer,
use_EM ,
    pinn_flag , # [广义幂律:'GYML'; 半衰期：'HLR';ACT-R: 'ACTR']
    EM_epoch,
replace,scheduler_sr
):#Optional[int]
    datasets['train'].inputs = (datasets['train'].inputs - np.array(min))/(np.array(max)-np.array(min))
    datasets['val'].inputs = (datasets['val'].inputs - np.array(min))/(np.array(max)-np.array(min))
    datasets['test'].inputs = (datasets['test'].inputs - np.array(min))/(np.array(max)-np.array(min))
    data_loaders = get_dataloaders(datasets, batch_size)

    save_path = os.path.join(folder, 'best_model.pkl')
    begin_epoch = 0

    if os.path.exists(save_path):
        save_dict = torch.load(save_path)

        trainer.load_state_dict(save_dict['model_state_dict'], save_dict['optimizer_state_dict'])

        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1
    else:
        save_dict = dict()
        best_val_loss = float('inf')
        begin_epoch = 0

    phases = ['train', 'val', 'test']

    writer = SummaryWriter(folder)

    since = time.perf_counter()

    print(trainer.model)
    print(f'Trainable parameters: {get_number_of_parameters(trainer.model)}.')
    xs =[]
    xs2 = []
    sr_values = list()
    fis_values= list()
    time_values= list()
    dh_values= list()
    value = list()
    torch.backends.cudnn.enabled = False
    trainer.model.to(0)
    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):

            running_metrics = dict()
            running_metrics1 = dict()
            running_metrics_union = dict()
            for phase in phases:
                steps, predicts, targets,predict_uion = 0, list(), list(), list()
                predicts1, targets1 ,targets_union= list(),list(),list()
                for i in range(EM_epoch-1):
                    if phase != 'train':
                        break
                    for x, y in tqdm(data_loaders[phase], f'{phase.capitalize():5} {epoch}'):
                        if phase == 'train':
                            y_ = trainer.train(x, y,srtrainer,use_EM,pinn_flag)
                        else:
                            with torch.no_grad():
                                y_ = trainer.predict(x)
                        # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                        torch.cuda.empty_cache()
                for x, y in tqdm(data_loaders[phase], f'{phase.capitalize():5} {epoch}'):
                    # 归一化处理

                    targets.append(y.numpy().copy())
                    if phase == 'train':
                        y_ = trainer.train(x, y,srtrainer,use_EM,pinn_flag)
                    else:
                        with torch.no_grad():
                            y_ = trainer.predict(x)

                    predicts.append(y_.detach().cpu().numpy())

                    # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                    torch.cuda.empty_cache()
                # [0.1027, -0.8600, -0.6949, 0.0862, 0.0287, 0.6136, 0.0699, 0.9844,       -1.0948, -0.0241, -0.0053, 0.0188]
                if use_EM:
                    for i in range(EM_epoch-1):
                        if phase != 'train':
                            break
                        for x, y in tqdm(data_loaders[phase], f'{phase.capitalize():5} {epoch}'):
                            if phase == 'train':
                                y_ = srtrainer.train(x, y, trainer,use_EM,pinn_flag)
                            else:
                                y_ = srtrainer.predict(x, y, trainer,use_EM,pinn_flag)
                            torch.cuda.empty_cache()
                    for x, y in tqdm(data_loaders[phase], f'{phase.capitalize():5} {epoch}'):
                        # 归一化处理
                        targets1.append(y.numpy().copy())
                        if phase == 'train':
                            y_ = srtrainer.train(x, y,trainer,use_EM,pinn_flag)
                        else:
                            y_ = srtrainer.predict(x, y,trainer,use_EM,pinn_flag)

                        predicts1.append(y_.detach().cpu().numpy())


                        #For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                        torch.cuda.empty_cache()


                # SRTrainer.model.fn.weight[SRTrainer.model.fn.weight>0 and SRTrainer.model.fn.weight<0.05]=0
                # SRTrainer.model.fn.weight[SRTrainer.model.fn.weight < 0 and SRTrainer.model.fn.weight > -0.05] = 0
                # 性能
                running_metrics[phase] = evaluate(np.concatenate(predicts), np.concatenate(targets))
                if use_EM:
                    running_metrics1[phase] = evaluate(np.concatenate(predicts1), np.concatenate(targets1))
                    # with torch.no_grad():
                    #     # if epoch > 5:
                    #     #     pass
                    #     #     # use_EM=False
                    #     #     # replace = 'best_replace'
                    #     if phase == 'train':
                    #         if replace == 'replace':
                    #             sr_values.append(
                    #                 [copy.deepcopy(torch.tensor(running_metrics1['train']['loss']).to(0)),copy.deepcopy(srtrainer.model.fn.weight),copy.deepcopy(srtrainer.model.fn.bias)])
                    #             if len(sr_values) > 10:
                    #                 value = list()
                    #                 sr_values = sorted(sr_values)
                    #                 sr_values = sr_values[0:-1]
                    #                 for i in sr_values:
                    #                     value.append(i[1])
                    #         else:
                    #             if pinn_flag == 'HLR':
                    #                 sr_values.append([copy.deepcopy(torch.tensor(running_metrics1['train']['loss']).to(0)),copy.deepcopy(srtrainer.model.fn.weight),copy.deepcopy(srtrainer.model.fn.bias)])
                    #                 value = list()
                    #                 bais = list()
                    #                 if len(sr_values) > 4:
                    #
                    #                     sr_values = sorted(sr_values)
                    #                     sr_values = sr_values[0:-1]
                    #                 else:
                    #                     sr_values = sorted(sr_values)
                    #
                    #
                    #                 for i in sr_values:
                    #                     bais.append(i[2])
                    #                     value.append(i[1])
                    #                 sofmax = torch.nn.Softmax(dim=0)
                    #                 sig = sofmax(torch.stack(value).squeeze(1))
                    #                 sig_b= sofmax(torch.stack(bais))
                    #                 z = torch.zeros_like(sr_values[0][1])
                    #                 b = torch.zeros_like(sr_values[0][2])
                    #                 for i in range(len(value)):
                    #                     z += sr_values[i][1] * sig[i]
                    #                     b +=sr_values[i][2] * sig_b[i]
                    #                 if replace == 'best_replace':
                    #                     srtrainer.model.fn.weight[:] = sr_values[0][1]
                    #                     srtrainer.model.fn.bais = sr_values[0][2]
                    #                 elif replace == 'union':
                    #                     srtrainer.model.fn.weight[:] = z
                    #                     srtrainer.model.fn.bias[:] = b
                    #                 elif replace == 'replace':
                    #                     srtrainer.model.fn.weight[:] = srtrainer.model.fn.weight[:]
                    #             else:
                    #
                    #                     fis_values.append([copy.deepcopy(torch.tensor(running_metrics1['train']['loss']).to(0)), copy.deepcopy(srtrainer.model.fis.weight),copy.deepcopy(srtrainer.model.fis.bias)])
                    #                     time_values.append([copy.deepcopy(torch.tensor(running_metrics1['train']['loss']).to(0)), copy.deepcopy(srtrainer.model.time.weight),copy.deepcopy(srtrainer.model.time.bias)])
                    #                     dh_values.append([copy.deepcopy(torch.tensor(running_metrics1['train']['loss']).to(0)), copy.deepcopy(srtrainer.model.dh.weight),copy.deepcopy(srtrainer.model.dh.bias)])
                    #                     value = list()
                    #                     loss = list()
                    #                     bias = list()
                    #                     value2 = list()
                    #                     bias2 = list()
                    #                     loss2 = list()
                    #                     value3 = list()
                    #                     bias3 = list()
                    #                     loss3 = list()
                    #                     if len(fis_values) > 4:
                    #                         fis_values = sorted(fis_values)
                    #                         fis_values = fis_values[0:-1]
                    #                         time_values = sorted(time_values)
                    #                         time_values = time_values[0:-1]
                    #                         dh_values = sorted(dh_values)
                    #                         dh_values = dh_values[0:-1]
                    #                     else:
                    #                         fis_values = sorted(fis_values)
                    #                         time_values = sorted(time_values)
                    #                         dh_values = sorted(dh_values)
                    #                     for i in fis_values:
                    #                         value.append(i[1])
                    #                         bias.append(i[2])
                    #                         loss.append(1.0/i[0])
                    #                     for i in time_values:
                    #                         value2.append(i[1])
                    #                         bias2.append(i[2])
                    #                         loss2.append(1.0/i[0])
                    #                     for i in dh_values:
                    #                         value3.append(i[1])
                    #                         bias3.append(i[2])
                    #                         loss3.append(1.0/i[0])
                    #                     sofmax = torch.nn.Softmax(dim=0)
                    #
                    #                     if replace != 'losssoft':
                    #                         sig = sofmax(torch.stack(value).squeeze(1))
                    #                         sig2 = sofmax(torch.stack(value2).squeeze(1))
                    #                         sig3 = sofmax(torch.stack(value3).squeeze(1))
                    #                         sig_b = sofmax(torch.stack(bias).squeeze(1))
                    #                         sig2_b = sofmax(torch.stack(bias2).squeeze(1))
                    #                         sig3_b = sofmax(torch.stack(bias3).squeeze(1))
                    #                     else:
                    #                         sig =sofmax(torch.stack(loss))
                    #                         sig2 = sofmax(torch.stack(loss2))
                    #                         sig3 = sofmax(torch.stack(loss3))
                    #                         sig_b = sofmax(torch.stack(loss))
                    #                         sig2_b = sofmax(torch.stack(loss2))
                    #                         sig3_b = sofmax(torch.stack(loss3))
                    #                     z = torch.zeros_like(value[0])
                    #                     z2 = torch.zeros_like(value2[0])
                    #                     z3 = torch.zeros_like(value3[0])
                    #                     b = torch.zeros_like(bias[0])
                    #                     b2 = torch.zeros_like(bias2[0])
                    #                     b3 = torch.zeros_like(bias3[0])
                    #                     for i in range(len(value)):
                    #                         z += value[i] * sig[i]
                    #                         z2 += value2[i] * sig2[i]
                    #                         z3 += value3[i] * sig3[i]
                    #                         b +=bias[i] * sig_b[i]
                    #                         b2 += bias2[i] * sig2_b[i]
                    #                         b3 += bias3[i] * sig3_b[i]
                    #                     if replace == 'best_replace':
                    #                         srtrainer.model.fis.weight = nn.Parameter(value[0])
                    #                         srtrainer.model.time.weight =nn.Parameter(value2[0])
                    #                         srtrainer.model.dh.weight = nn.Parameter(value3[0])
                    #                         srtrainer.model.fis.bias = nn.Parameter(bias[0])
                    #                         srtrainer.model.time.bias = nn.Parameter(bias2[0])
                    #                         srtrainer.model.dh.bias = nn.Parameter(bias3[0])
                    #                     elif replace == 'union':
                    #                         srtrainer.model.fis.weight[:] = z
                    #                         srtrainer.model.time.weight[:] = z2
                    #                         srtrainer.model.dh.weight[:] = z3
                    #                         srtrainer.model.fis.bias[:] = b
                    #                         srtrainer.model.time.bias[:] = b2
                    #                         srtrainer.model.dh.bias[:] = b3
                    #                     elif replace == 'sum':
                    #                         srtrainer.model.fis.weight[:] = sum(value)/len(value)
                    #                         srtrainer.model.time.weight[:] = sum(value2)/len(value2)
                    #                         srtrainer.model.dh.weight[:] = sum(value3)/len(value3)
                    #                         srtrainer.model.fis.bias[:] = sum(bias)/len(bias)
                    #                         srtrainer.model.time.bias[:] = sum(bias2)/len(bias2)
                    #                         srtrainer.model.fis.bias[:] = sum(bias3)/len(bias3)
                    #                     elif replace == 'losssoft':
                    #                         srtrainer.model.fn.weight[:] = z
                    #                         srtrainer.model.time.weight[:] = z2
                    #                         srtrainer.model.fis.weight[:] = z3
                    #                         srtrainer.model.fn.bias[:] = b
                    #                         srtrainer.model.time.bias[:] = b2
                    #                         srtrainer.model.fis.bias[:] = b3

                if phase == 'val':
                    if running_metrics['val']['loss'] < best_val_loss:
                        best_val_loss = running_metrics['val']['loss']
                        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
                        save_dict.update(
                            model_state_dict=copy.deepcopy(trainer.model.state_dict()),
                            epoch=epoch,
                            best_val_loss=best_val_loss,
                            optimizer_state_dict=copy.deepcopy(trainer.optimizer.state_dict()))
                        # save_model(save_path, **save_dict)
                        torch.save(copy.deepcopy(trainer.model.state_dict()), save_path)
                        print(f'A better model at epoch {epoch} recorded.')
                    elif early_stop_steps is not None and epoch - save_dict['epoch'] > early_stop_steps:
                        raise ValueError('Early stopped.')
            scheduler.step()
            scheduler_sr.step()
            loss_dict = {f'model {phase} loss: ': running_metrics[phase] for phase in phases}
            if use_EM:
                with torch.no_grad():
                    if replace == 'replace':
                        sr_values.append(
                            [copy.deepcopy(torch.tensor(running_metrics1['train']['loss']).to(0)),
                             copy.deepcopy(srtrainer.model.fn.weight), copy.deepcopy(srtrainer.model.fn.bias)])
                        if len(sr_values) > 10:
                            value = list()
                            sr_values = sorted(sr_values)
                            sr_values = sr_values[0:-1]
                            for i in sr_values:
                                value.append(i[1])
                    else:
                        if pinn_flag == 'HLR':
                            sr_values.append([copy.deepcopy(torch.tensor(running_metrics1['train']['loss']).to(0)),
                                              copy.deepcopy(srtrainer.model.fn.weight),
                                              copy.deepcopy(srtrainer.model.fn.bias)])
                            value = list()
                            bais = list()
                            if len(sr_values) > 4:

                                sr_values = sorted(sr_values)
                                sr_values = sr_values[0:-1]
                            else:
                                sr_values = sorted(sr_values)

                            for i in sr_values:
                                bais.append(i[2])
                                value.append(i[1])
                            sofmax = torch.nn.Softmax(dim=0)
                            sig = sofmax(torch.stack(value).squeeze(1))
                            sig_b = sofmax(torch.stack(bais))
                            z = torch.zeros_like(sr_values[0][1])
                            b = torch.zeros_like(sr_values[0][2])
                            for i in range(len(value)):
                                z += sr_values[i][1] * sig[i]
                                b += sr_values[i][2] * sig_b[i]
                            if replace == 'best_replace':
                                srtrainer.model.fn.weight[:] = sr_values[0][1]
                                srtrainer.model.fn.bais = sr_values[0][2]
                            elif replace == 'union':
                                srtrainer.model.fn.weight[:] = z
                                srtrainer.model.fn.bias[:] = b
                            elif replace == 'replace':
                                srtrainer.model.fn.weight[:] = srtrainer.model.fn.weight[:]
                        else:

                            fis_values.append([copy.deepcopy(torch.tensor(running_metrics1['val']['loss']).to(0)),
                                               copy.deepcopy(srtrainer.model.fis.weight),
                                               copy.deepcopy(srtrainer.model.fis.bias)])
                            time_values.append(
                                [copy.deepcopy(torch.tensor(running_metrics1['val']['loss']).to(0)),
                                 copy.deepcopy(srtrainer.model.time.weight),
                                 copy.deepcopy(srtrainer.model.time.bias)])
                            dh_values.append([copy.deepcopy(torch.tensor(running_metrics1['val']['loss']).to(0)),
                                              copy.deepcopy(srtrainer.model.dh.weight),
                                              copy.deepcopy(srtrainer.model.dh.bias)])
                            value = list()
                            loss = list()
                            bias = list()
                            value2 = list()
                            bias2 = list()
                            loss2 = list()
                            value3 = list()
                            bias3 = list()
                            loss3 = list()
                            if len(fis_values) > 3:
                                fis_values = sorted(fis_values)
                                fis_values = fis_values[0:-1]
                                time_values = sorted(time_values)
                                time_values = time_values[0:-1]
                                dh_values = sorted(dh_values)
                                dh_values = dh_values[0:-1]
                            else:
                                fis_values = sorted(fis_values)
                                time_values = sorted(time_values)
                                dh_values = sorted(dh_values)
                            for i in fis_values:
                                value.append(i[1])
                                bias.append(i[2])
                                loss.append(1.0 / i[0])
                            for i in time_values:
                                value2.append(i[1])
                                bias2.append(i[2])
                                loss2.append(1.0 / i[0])
                            for i in dh_values:
                                value3.append(i[1])
                                bias3.append(i[2])
                                loss3.append(1.0 / i[0])
                            sofmax = torch.nn.Softmax(dim=0)

                            if replace != 'losssoft':
                                sig = sofmax(torch.stack(value).squeeze(1))
                                sig2 = sofmax(torch.stack(value2).squeeze(1))
                                sig3 = sofmax(torch.stack(value3).squeeze(1))
                                sig_b = sofmax(torch.stack(bias).squeeze(1))
                                sig2_b = sofmax(torch.stack(bias2).squeeze(1))
                                sig3_b = sofmax(torch.stack(bias3).squeeze(1))
                            else:
                                sig = sofmax(torch.stack(loss))
                                sig2 = sofmax(torch.stack(loss2))
                                sig3 = sofmax(torch.stack(loss3))
                                sig_b = sofmax(torch.stack(loss))
                                sig2_b = sofmax(torch.stack(loss2))
                                sig3_b = sofmax(torch.stack(loss3))
                            z = torch.zeros_like(value[0])
                            z2 = torch.zeros_like(value2[0])
                            z3 = torch.zeros_like(value3[0])
                            b = torch.zeros_like(bias[0])
                            b2 = torch.zeros_like(bias2[0])
                            b3 = torch.zeros_like(bias3[0])
                            for i in range(len(value)):
                                z += value[i] * sig[i]
                                z2 += value2[i] * sig2[i]
                                z3 += value3[i] * sig3[i]
                                b += bias[i] * sig_b[i]
                                b2 += bias2[i] * sig2_b[i]
                                b3 += bias3[i] * sig3_b[i]
                            if replace == 'best_replace':
                                srtrainer.model.fis.weight[:] = nn.Parameter(value[0])
                                srtrainer.model.time.weight[:] = nn.Parameter(value2[0])
                                srtrainer.model.dh.weight[:] = nn.Parameter(value3[0])
                                srtrainer.model.fis.bias[:] = nn.Parameter(bias[0])
                                srtrainer.model.time.bias[:] = nn.Parameter(bias2[0])
                                srtrainer.model.dh.bias[:] = nn.Parameter(bias3[0])
                            elif replace == 'union':
                                srtrainer.model.fis.weight[:] = z
                                srtrainer.model.time.weight[:] = z2
                                srtrainer.model.dh.weight[:] = z3
                                srtrainer.model.fis.bias[:] = b
                                srtrainer.model.time.bias[:] = b2
                                srtrainer.model.dh.bias[:] = b3
                            elif replace == 'sum':
                                srtrainer.model.fis.weight[:] = sum(value) / len(value)
                                srtrainer.model.time.weight[:] = sum(value2) / len(value2)
                                srtrainer.model.dh.weight[:] = sum(value3) / len(value3)
                                srtrainer.model.fis.bias[:] = sum(bias) / len(bias)
                                srtrainer.model.time.bias[:] = sum(bias2) / len(bias2)
                                srtrainer.model.fis.bias[:] = sum(bias3) / len(bias3)
                            elif replace == 'losssoft':
                                srtrainer.model.fn.weight[:] = z
                                srtrainer.model.time.weight[:] = z2
                                srtrainer.model.fis.weight[:] = z3
                                srtrainer.model.fn.bias[:] = b
                                srtrainer.model.time.bias[:] = b2
                                srtrainer.model.fis.bias[:] = b3
                            elif replace == 'rand':
                                n = random.randint(0, len(fis_values)-1)
                                srtrainer.model.fis.weight[:] = nn.Parameter(value[n])
                                srtrainer.model.time.weight[:] = nn.Parameter(value2[n])
                                srtrainer.model.dh.weight[:] = nn.Parameter(value3[n])
                                srtrainer.model.fis.bias[:] = nn.Parameter(bias[n])
                                srtrainer.model.time.bias[:] = nn.Parameter(bias2[n])
                                srtrainer.model.dh.bias[:] = nn.Parameter(bias3[n])
                if use_EM:
                    targets2 = list()
                    predicts2=list()
                    for x, y in tqdm(data_loaders['test']):
                        # 归一化处理
                        targets2.append(y.numpy().copy())
                        if phase == 'train':
                            y_ = srtrainer.train(x, y, trainer, use_EM, pinn_flag)
                        else:
                            y_ = srtrainer.predict(x, y, trainer, use_EM, pinn_flag)

                        predicts2.append(y_.detach().cpu().numpy())
                    running_metricsa = evaluate(np.concatenate(predicts2), np.concatenate(targets2))
                    print(running_metricsa)

                loss_dict.update({f'SR {phase} loss: ': running_metrics1[phase] for phase in phases})
                if pinn_flag == 'HLR' or pinn_flag == 'nom':
                    sr_value = 'sr: '+str(srtrainer.model.fn.weight)+str(srtrainer.model.fn.bias)
                else:
                    sr_value = 'fis: '+str(srtrainer.model.fis.weight)+str(srtrainer.model.fis.bias) + '   time:' +str(srtrainer.model.time.weight)+str(srtrainer.model.time.bias)+'   dh:'+str(srtrainer.model.dh.weight)+str(srtrainer.model.dh.bias)
                loss_dict.update({'sr_value: ': sr_value})
            txt_file = open("log.txt", "w")
            txt_file.write(f"{epoch}\n")
            for i in loss_dict.items():
                txt_file.write(f"{i}\n")
                print(i)

            loss_dict = {f'{phase} loss': running_metrics[phase].pop('loss') for phase in phases}

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(loss_dict['train'])
                else:
                    scheduler.step()

            writer.add_scalars('Loss', loss_dict, global_step=epoch)
            for metric in running_metrics['train'].keys():
                for phase in phases:
                    for key, val in running_metrics[phase].items():
                        writer.add_scalars(f'{metric}/{key}', {f'{phase}': val}, global_step=epoch)

    except (ValueError, KeyboardInterrupt) as e:
        print(e)
    time_elapsed = time.perf_counter() - since
    print(f"cost {time_elapsed} seconds")
    print(f'The best adaptor and model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')


def test_model(
        datasets: Dict[str, PredictorDataset],
        batch_size: int,
        trainer: OursTrainer,
        folder: str):
    dataloaders = get_dataloaders(datasets, batch_size)

    saved_path = os.path.join(folder, 'best_model.pkl')

    saved_dict = torch.load(saved_path)
    trainer.model.load_state_dict(saved_dict['model_state_dict'])

    predictions, running_targets = list(), list()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloaders['test'], 'Test model'):
            running_targets.append(targets.numpy().copy())
            predicts = trainer.predict(inputs)
            predictions.append(predicts.cpu().numpy())

    # 性能
    predictions, running_targets = np.concatenate(predictions), np.concatenate(running_targets)
    scores = evaluate(predictions, running_targets)
    scores.pop('loss')
    print('test results:')
    print(json.dumps(scores, cls=JsonEncoder, indent=4))
    with open(os.path.join(folder, 'test-scores.json'), 'w+') as f:
        json.dump(scores, f, cls=JsonEncoder, indent=4)

    np.savez(os.path.join(folder, 'test-results.npz'), predictions=predictions, targets=running_targets)


def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


def get_number_of_parameters(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def get_scheduler(name, optimizer, **kwargs):
    return getattr(optim.lr_scheduler, name)(optimizer, **kwargs)


def get_optimizer(name: str, parameters, **kwargs):
    return getattr(optim, name)(parameters, **kwargs)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def set_device_recursive(var, device):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = set_device_recursive(var[key], device)
        else:
            try:
                var[key] = var[key].to(device)
            except AttributeError:
                pass
    return var


def set_requires_grad(models: List[nn.Module], required: bool):
    for model in models:
        for param in model.parameters():
            param.requires_grad_(required)


def get_regularization(model: nn.Module, weight_decay: float, p: float = 2.0):
    weight_list = list(filter(lambda item: 'weight' in item[0], model.named_parameters()))
    return weight_decay * sum(torch.norm(w, p=p) for name, w in weight_list)
