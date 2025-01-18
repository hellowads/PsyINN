
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
import transformer_base as runfile
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ## 3
    parser.add_argument('--seed', type=int, default=3)

    parser.add_argument('--config', type=str, default="config",
                        help='Configuration filename for restoring the model.')

    parser.add_argument('--resume', action='store_true', default=False,
                        help='if to resume a trained model?')
    parser.add_argument('--test', action='store_true', default=False,
                        help='if in the test mode?')
    parser.add_argument('--name', type=str, default="stage",
                        help='The name of the folder where the model is stored.')

    parser.add_argument('--use_EM', type=bool, default=True)

    #  [Wickelgren: 'Wickelgren'; HLR function：'HLR'; ACT-R: 'ACT-R'; 'do_nothing:':'No']
    parser.add_argument('--pinn_flag', type=str, default='Wickelgren')

    parser.add_argument('--EM_epoch', type=int, default="1")

    ## BestSelect: best_replace  ReplaceSelect: replace  RandomSelect rand
    parser.add_argument('--replace', type=str, default="rand")

    # duolingguo/en_to_de ; duolingguo/en_to_es  ; duolingguo/all_data
    parser.add_argument('--dataset', type=str, default="duolingguo/en_to_de")


    args = parser.parse_args()
    seed_everything(args.seed)
    with open(os.path.join('config', f'{args.config}.yaml')) as f:
        config = yaml.safe_load(f)
        config['name'] = args.name
    if args.resume:
        print(f'Resume to {config["name"]}.')
        runfile.train(config, resume=True)
    elif args.test:
        print(f'Test {config["name"]}.')
        runfile.train(config, test=True)
    else:
        runfile.train(config,args)


