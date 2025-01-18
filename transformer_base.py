import argparse
import json
import os
import shutil
import torch
import yaml
import utils
import models
import random
import numpy as np


from utils import get_datasets
import torch.nn as nn


def train(_config,args,resume: bool = False, test: bool = False):
    # print(json.dumps(_config, indent=4))
    device = torch.device(_config['device'])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)
    device = torch.device(0)
    replace = args.replace
    dataset = args.dataset
    optimizer_name = _config['optimizer']['name']
    scheduler_name = _config['scheduler']['name']
    loss = nn.MSELoss()
    loss.to(device)
    use_EM = args.use_EM
    pinn_flag = args.pinn_flag  # [广义幂律:'Wickelgren'; 半衰期：'HLR';ACT-R: 'ACT-R']
    EM_epoch = args.EM_epoch

    model = models.DKT(**_config['model']['DKT'])

    sr_model = models.DNN(**_config['model']['DNN'])

    optimizer = utils.get_optimizer(optimizer_name, model.parameters(), **_config['optimizer'][optimizer_name])
    optimizer_sr = utils.get_optimizer(optimizer_name, sr_model.parameters(), **_config['optimizer_sr'][optimizer_name])
    scheduler = None
    if scheduler_name is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        scheduler_sr = torch.optim.lr_scheduler.StepLR(optimizer_sr, step_size=5, gamma=0.5)
        # scheduler = utils.get_scheduler(scheduler_name, optimizer, **_config['scheduler'][scheduler_name])
        # scheduler_sr = utils.get_scheduler(scheduler_name, optimizer_sr, **_config['scheduler'][scheduler_name])
    save_folder = os.path.join('saves', dataset,pinn_flag+f'_EM{use_EM}_fix_pinn')
    if not resume and not test:
        shutil.rmtree(save_folder, ignore_errors=True)
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.yaml'), 'w+') as _f:
        yaml.safe_dump(_config, _f)

    datasets = get_datasets(dataset)
    min,max = datasets['train'].__minmax__()
    scaler = 0#utils.ZScoreScaler(datasets['train'].mean, datasets['train'].std)
    trainer = utils.OursTrainer(model, loss, scaler, device, optimizer, **_config['trainer'])
    srtrainer = utils.SRTrainer(sr_model, loss, scaler, device, optimizer_sr, **_config['trainer'])
    if not test:
        utils.train_model(
            datasets=datasets,
            batch_size=_config['data']['batch-size'],
            folder=save_folder,
            trainer=trainer,
            scheduler=scheduler,
            epochs=_config['epochs'],
            early_stop_steps=_config['early_stop_steps'],
            min=min,max=max,srtrainer=srtrainer,
            use_EM=use_EM,
            pinn_flag = pinn_flag,  # [广义幂律:'GYML'; 半衰期：'HLR';ACT-R: 'ACTR']
            EM_epoch = EM_epoch,
            replace=replace,
            scheduler_sr=scheduler_sr
        )






