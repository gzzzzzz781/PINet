# -*- coding:utf-8 -*-
import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio
from dataset.dataset_sig17 import SIG17_Training_Dataset, SIG17_Validation_Dataset
from models.loss import L1MuLoss, JointReconPerceptualLoss
from utils.utils import *

from models.PINet import HDRTransformer

def get_args():
    parser = argparse.ArgumentParser(description='HDR-Transformer',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_dir", type=str, default="./data",
                        help='dataset directory'),
    parser.add_argument('--patch_size', type=int, default=128),
    parser.add_argument("--sub_set", type=str, default='sig17_training_crop128_stride64_augment',
                        help='dataset directory')
    parser.add_argument('--logdir', type=str, default='./checkpoints',
                        help='target log directory')
    parser.add_argument('--num_workers', type=int, default=2, metavar='N',
                        help='number of workers to fetch data (default: 2)')
    parser.add_argument('--resume', type=str, default=False,
                        help='load model from a .pth file')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=443, metavar='S',
                        help='random seed (default: 443)')
    parser.add_argument('--init_weights', action='store_true', default=True,
                        help='init model weights')
    parser.add_argument('--loss_func', type=int, default=0,
                        help='loss functions for training')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--lr_decay_interval', type=int, default=100,
                        help='decay learning rate every N epochs(default: 100)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='start epoch of training (default: 0)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='training batch size (default: 8)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 1)')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    return parser.parse_args()

def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    total_loss = 0.0
    with tqdm(total=train_loader.__len__()) as pbar:
        for batch_idx, batch_data in enumerate(train_loader):
            data_time.update(time.time() - end)
            batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), \
                                                 batch_data['input2'].to(device)
            label = batch_data['label'].to(device)
            pred = model(batch_ldr0, batch_ldr1, batch_ldr2)
            loss = criterion(pred, label)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:3f})\t'
                      'Data: {data_time.val:.3f} ({data_time.avg:3f})'.format(
                    epoch,
                    batch_idx * args.batch_size,
                    len(train_loader.dataset),
                    100. * batch_idx * args.batch_size / len(train_loader.dataset),
                    loss.item(),
                    batch_time=batch_time,
                    data_time=data_time
                ))
                print("avg_loss:",total_loss/(batch_idx+1))
            pbar.set_postfix(loss=float(loss.cpu()), epoch=epoch)
            pbar.update(1)

def validation(args, model, device, val_loader, optimizer, epoch, criterion, cur_psnr):
    model.eval()

    val_psnr = AverageMeter()
    val_mu_psnr = AverageMeter()
    val_loss = AverageMeter()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), batch_data['input2'].to(device)
            label = batch_data['label'].to(device)
            pred = model(batch_ldr0, batch_ldr1, batch_ldr2)
            loss = criterion(pred, label)

            psnr = batch_psnr(pred, label, 1.0)
            mu_psnr = batch_psnr_mu(pred, label, 1.0)
            val_psnr.update(psnr.item())
            val_mu_psnr.update(mu_psnr.item())
            val_loss.update(loss.item())
            print(batch_idx)

    print('Validation set: Average Loss: {:.4f}'.format(val_loss.avg))
    print('Validation set: Average PSNR: {:.4f}, mu_law: {:.4f}'.format(val_psnr.avg, val_mu_psnr.avg))

    save_dict = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(save_dict, os.path.join(args.logdir, 'val_latest_checkpoint.pth'))
    if val_mu_psnr.avg > cur_psnr[0]:
        torch.save(save_dict, os.path.join(args.logdir, 'best_checkpoint.pth'))
        cur_psnr[0] = val_mu_psnr.avg
        with open(os.path.join(args.logdir, 'best_checkpoint.json'), 'w') as f:
            f.write('best epoch:' + str(epoch) + '\n')
            f.write('Validation set: Average PSNR: {:.4f}, PSNR_mu_law: {:.4f}\n'.format(val_psnr.avg, val_mu_psnr.avg))

def main():
    args = get_args()
    cur_psnr = [-1.0]

    if args.seed is not None:
        set_random_seed(args.seed)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = HDRTransformer()
    model = model.to(device)

    loss_dict = {
        0: L1MuLoss,
        1: JointReconPerceptualLoss,
        }
    criterion = loss_dict[args.loss_func]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)

    if args.resume:
        if os.path.isfile(args.resume):
            print("===> Loading checkpoint from: {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("===> Loaded checkpoint: epoch {}".format(checkpoint['epoch']))
        else:
            print("===> No checkpoint is founded at {}.".format(args.resume))

    train_dataset = SIG17_Training_Dataset(root_dir=args.dataset_dir, sub_set=args.sub_set, is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataset = SIG17_Validation_Dataset(root_dir=args.dataset_dir, is_training=False, crop=False, crop_size=256)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    dataset_size = len(train_loader.dataset)
    print(f'''===> Start training HDR-Transformer

        Dataset dir:     {args.dataset_dir}
        Subset:          {args.sub_set}
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Loss function:   {args.loss_func}
        Learning rate:   {args.lr}
        Training size:   {dataset_size}
        Device:          {device.type}
        ''')

    for epoch in range(args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        validation(args, model, device, val_loader, optimizer, epoch, criterion, cur_psnr)

if __name__ == '__main__':
    main()
