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
    print(json.dumps(config, indent=4))
    device = torch.device(_config['device'])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)
    device = torch.device(0)
    replace = args.replace
    dataset = _config['data']['dataset']
    optimizer_name = _config['optimizer']['name']
    scheduler_name = _config['scheduler']['name']
    loss = torch.nn.MSELoss()
    loss.to(device)
    use_EM = args.use_EM
    pinn_flag = args.pinn_flag  # [广义幂律:'Wickelgren'; 半衰期：'HLR';ACT-R: 'ACT-R']
    EM_epoch = args.EM_epoch
    model_name = 'DKT'
    if model_name == 'DKT':
        model = models.DKT(**_config['model']['DKT'])

    sr_name = 'DNN'
    sr_model = models.DNN(**config['model']['DNN'])
    optimizer = utils.get_optimizer(optimizer_name, model.parameters(), **_config['optimizer'][optimizer_name])
    optimizer_sr = utils.get_optimizer(optimizer_name, sr_model.parameters(), **_config['optimizer_sr'][optimizer_name])
    scheduler = None
    if scheduler_name is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        scheduler_sr = torch.optim.lr_scheduler.StepLR(optimizer_sr, step_size=5, gamma=0.5)
        # scheduler = utils.get_scheduler(scheduler_name, optimizer, **_config['scheduler'][scheduler_name])
        # scheduler_sr = utils.get_scheduler(scheduler_name, optimizer_sr, **_config['scheduler'][scheduler_name])
    save_folder = os.path.join('saves', dataset,model_name+pinn_flag+f'_EM{use_EM}_fix_pinn')
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
            epochs=config['epochs'],
            early_stop_steps=config['early_stop_steps'],
            min=min,max=max,srtrainer=srtrainer,
            use_EM=use_EM,
        pinn_flag = pinn_flag,  # [广义幂律:'GYML'; 半衰期：'HLR';ACT-R: 'ACTR']
        EM_epoch = EM_epoch,
            replace=replace,
            scheduler_sr=scheduler_sr
        )
#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ## 2
    parser.add_argument('--seed', type=int, default=3)#2 3 4
    parser.add_argument('--config', type=str, default="config",
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='if to resume a trained model?')
    parser.add_argument('--test', action='store_true', default=False,
                        help='if in the test mode?')
    parser.add_argument('--name', type=str, default="stage",
                        help='The name of the folder where the model is stored.')
    parser.add_argument('--use_EM', type=bool, default=True)
    #  [广义幂律:'Wickelgren'; 半衰期：'HLR';ACT-R: 'ACT-R';'do_nothing:':'No','nom']
    parser.add_argument('--pinn_flag', type=str, default='Wickelgren')
    parser.add_argument('--EM_epoch', type=int, default="1")
    ## 最优情况 best_replace  融合 union  直接替换replace losssoft sum  rand
    parser.add_argument('--replace', type=str, default="rand")
    args = parser.parse_args()
    seed_everything(args.seed)
    with open(os.path.join('config', f'{args.config}.yaml')) as f:
        config = yaml.safe_load(f)
        config['name'] = args.name
    if args.resume:
        print(f'Resume to {config["name"]}.')
        train(config, resume=True)
    elif args.test:
        print(f'Test {config["name"]}.')
        train(config, test=True)
    else:
        train(config,args)


