import torch
import argparse

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
    parser.add_argument('--train_csv', type=str, help='The train csv')
    parser.add_argument('--train_folder', type=str,
                        help='The train root folder')

    parser.add_argument('--valid_csv', type=str, help='The valid csv')
    parser.add_argument('--valid_folder', type=str,
                        help='The valid root folder')

    parser.add_argument('--test_csv_seen', type=str, help='The seen test csv')
    parser.add_argument('--test_folder_seen', type=str,
                        help='The seen test root folder')

    # Dataloader settings
    parser.add_argument('--batch_size', type=int, default=32,
                        help='number of samples per iteration in the epoch')
    parser.add_argument('--num_workers', default=10, type=int)

    # optimizer settings
    parser.add_argument('--lr', type=float, default=0.005,
                        help='The learning rate')

    # trainng related parameters
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train for')

    # logging related parameters
    # Run settings, also defines folder where the log and weights will be stored
    parser.add_argument('--name', type=str, help='Name of run')
    # logging path, the logging path referes to folder of some particular runs, the run name defines the folder which will hold that particular run
    parser.add_argument('--log_path', type=str, default=r'logs_and_weights/', help='Number of epochs to train for')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite log if already exist for the name of run.')

    # validation settings
    parser.add_argument('--no_validate', action='store_false', help='Should model be validated')
    parser.add_argument('--valid_every', type=int, default=10,
                        help='validate every x epoch')

    return parser


def main():
    # arg_parser = argparse.ArgumentParser('Main', parents=[get_args_parser()])
    # args = arg_parser.parse_args()

    print('Code for starting the training')

    

if __name__ == '__main__':
    main()