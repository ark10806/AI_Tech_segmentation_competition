import os
import time
import yaml
import torch
import wandb
import data
import logging
import argparse
import numpy as np
from network.hrnetv2 import get_seg_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from lib.scheduler import CosineAnnealingWarmupRestarts

import torch.optim as optim


import lib.utils as utils
import lib.loss as loss
from lib.utils import seed_everything

def train_one_epoch():
    pass

def train():
    parser = argparse.ArgumentParser(description='Dobby Team')
    parser.add_argument('--expr_name', type=str)
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--decay', default=1e-7, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--workers', default=4, type=int)
    args = parser.parse_args()
    print(args)

    wandb.init(config=args, project="Seg_psc", name=args.expr_name, save_code=True)
    
    seed_everything(args.seed)

    dataroot = '/opt/ml/segmentation/input/data'
    train_loader, val_loader = data.get_dataloader(dataroot, 'train', args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_path = './configs/hrnet_seg_ocr.yaml'
    with open(config_path) as f:
        cfg = yaml.load(f)
        print(cfg)
        # exit()
    model = get_seg_model(cfg)

    model = model.to(device)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
    n_iter = (2617//args.batch_size)
    print(n_iter)
    # exit()
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=n_iter*10, cycle_mult=1.0, max_lr=args.lr, min_lr=1e-7, warmup_steps=n_iter, gamma=0.7)

    criterion = loss.DiceCELoss(weight=None)

    logger = logging.getLogger("Segmentation")
    logger.setLevel(logging.INFO)
    logger_dir = f'./logs/'
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)
    file_handler = logging.FileHandler(os.path.join(logger_dir, f'{args.expr_name}.log'))
    logger.addHandler(file_handler)
    
    best_loss = float('INF')
    best_mIoU = 0
    for epoch in range(args.epochs):
        train_loss, train_mIoU, val_loss, val_mIoU = utils.train_valid(
            epoch,
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            logger, 
            device,
        )
        print()

        utils.save_model(model, version=args.expr_name, epoch=epoch, save_type='current')
        if best_loss > val_loss:
            logger.info(f"Best loss {best_loss:.5f} -> {val_loss:.5f}")
            best_loss = val_loss
            utils.save_model(model, version=args.expr_name, epoch=epoch, save_type='loss')

        if best_mIoU < val_mIoU:
            logger.info(f"Best mIoU {best_mIoU:.5f} -> {val_mIoU:.5f}")
            best_mIoU = val_mIoU
            utils.save_model(model, version=args.expr_name, epoch=epoch, save_type='mIoU')

if __name__ == '__main__':
    past = {'h': (time.localtime().tm_hour+9)%24, 'm': time.localtime().tm_min}
    print(past)
    train()
    end = {'h': (time.localtime().tm_hour+9)%24, 'm': time.localtime().tm_min}

    print(f"startTime:\t {past['h']} : {past['m']}")
    print(f"endTime:\t {end['h']} : {end['m']}")
    m = (end['h'] - past['h']) * 60 + end['m'] - past['m']
    print(f'span:\t\t {m//60}hours {m%60}min')