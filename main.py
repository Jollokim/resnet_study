import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from timm import create_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import modules.models
from modules.dataset import TinyImageNetDataset, create_dataloader
from modules.engine import test_accuracy, train_one_epoch


# function for defining all the commandline parameters
def get_args_parser():
    parser = argparse.ArgumentParser('Main', add_help=False)

    # Model mode
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'pass'], required=True,
                        help='train or test a model')

    # Model settings
    parser.add_argument('--model', type=str,
                        help='Name of model', required=True)
    parser.add_argument('--pretrained_weights', type=str,
                        help='the path to pretrained weights file')

    # Dataset folder paths
    parser.add_argument('--train_folder', type=str, help='The train root folder')

    parser.add_argument('--valid_folder', type=str, help='The valid root folder')

    parser.add_argument('--test_folder', type=str, help='The test root folder')

    # Dataloader settings
    parser.add_argument('--batch_size', type=int, default=32,
                        help='number of samples per iteration in the epoch')
    parser.add_argument('--num_workers', default=5, type=int)

    # optimizer settings
    parser.add_argument('--lr', type=float, default=0.001,
                        help='The learning rate')

    # trainng related parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train for')

    # logging related parameters
    # Run settings, also defines folder where the log and weights will be stored
    parser.add_argument('--name', type=str, help='Name of run')
    
    # logging path, the logging path referes to folder of some particular runs, the run name defines the folder which will hold that particular run
    parser.add_argument('--log_path', type=str, default=r'logs_and_weights/', help='Number of epochs to train for')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite log if already exist for the name of run.')

    # validation settings
    parser.add_argument('--no_validate', action='store_true', help='Should model be validated')
    parser.add_argument('--valid_every', type=int, default=10,
                        help='validate every x epoch')

    return parser



def main(args):
    # arg_parser = argparse.ArgumentParser('Main', parents=[get_args_parser()])
    # args = arg_parser.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(f'{args.log_path}/{args.name}')

    # setting the device to do stuff on
    print('Training on GPU:', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(args.model).to(device)

    summary(model, input_size=(3, 64, 64))

    print('lr', args.lr)

    def training():
        dataloader_train, _ = create_dataloader(args.train_folder, args)

        if not args.no_validate:
            dataloader_valid, dataset_valid = create_dataloader(args.valid_folder, args)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'max', 0.95, 3, 0.005, verbose=True)

        mx_acc = 0
        best_epoch = 0
        for epoch in range(1, args.epochs+1):
            loss = train_one_epoch(model, dataloader_train, optimizer, epoch, device)
            writer.add_scalar('Mean loss over epoch', loss, epoch)

            if not args.no_validate:
                acc_current = test_accuracy(model, dataset_valid, device)
                
                print('Acc:', acc_current)
                writer.add_scalar('Accuracy over epoch', acc_current, epoch)

                if acc_current > mx_acc:
                    if os.path.exists(f'{args.log_path}/{args.name}/epoch{best_epoch}.pt'):
                        os.remove(f'{args.log_path}/{args.name}/epoch{best_epoch}.pt')

                    best_epoch = epoch
                    mx_acc = acc_current

                    torch.save(model.state_dict(), f'{args.log_path}/{args.name}/epoch{best_epoch}.pt')

            scheduler.step(acc_current)
            writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)

        torch.save(model.state_dict(), f'{args.log_path}/{args.name}/epochs.pt')

            
            

    def testing():
        pass
    
    training() if args.mode == 'train' else testing()


if __name__ == '__main__':
    # creates commandline parser
    arg_parser = argparse.ArgumentParser('Alexnet study with different filters', parents=[get_args_parser()])
    args = arg_parser.parse_args()

    # passes the commandline argument to the main function
    main(args)
