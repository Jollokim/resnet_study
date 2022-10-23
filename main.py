import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from timm import create_model
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import modules.models
from modules.dataset import TinyImageNetDataset, create_dataloader
from modules.engine import test_accuracy, train_one_epoch, f1Score


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

    # trainng related parameters
    parser.add_argument('--epochs', type=int, default=30,
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

    # optimizer settings
    parser.add_argument('--lr', type=float, default=0.001,
                        help='The learning rate')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='')

    # scheduler settings
    parser.add_argument('--start_factor', type=float, default=1,
                        help='')
    parser.add_argument('--end_factor', type=float, default=0.5,
                        help='')

    # seed
    parser.add_argument('--seed', type=int, default=1,
                        help='')


    return parser



def main(args):
    # arg_parser = argparse.ArgumentParser('Main', parents=[get_args_parser()])
    # args = arg_parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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
            _, dataset_valid = create_dataloader(args.valid_folder, args)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
        scheduler = LinearLR(optimizer, args.start_factor, args.end_factor, args.epochs, verbose=True)

        mx_acc = 0
        best_epoch = 0
        early_stopping_c = 0
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

                    early_stopping_c = 0

                    args.pretrained_weights = f'{args.log_path}/{args.name}/epoch{best_epoch}.pt'
                    torch.save(model.state_dict(), f'{args.log_path}/{args.name}/epoch{best_epoch}.pt')
                else:
                    early_stopping_c += 1
                    if early_stopping_c == 3:
                        print('No improvement! Stopping.')
                        return


            writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)
            scheduler.step()

        torch.save(model.state_dict(), f'{args.log_path}/{args.name}/end.pt')

    def testing():
        _, dataset_test = create_dataloader(args.test_folder, args)

        # TODO: load model weights
        model.load_state_dict(torch.load(args.pretrained_weights))
        acc = test_accuracy(model, dataset_test, device)
        f1Score = f1Score(model, dataset_test, device)

        print('test accuracy:', acc)
        print('f1 score:', f1Score)

        with open(f'{args.log_path}/{args.name}/test_results.txt', 'a') as f:
            f.write(str(acc))



    
    training() if args.mode == 'train' else testing()


if __name__ == '__main__':
    # creates commandline parser
    arg_parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = arg_parser.parse_args()

    # passes the commandline argument to the main function
    main(args)
