# modified version of one of my earlier works: https://github.com/Jollokim/Alexnet-edgemaps-vs-RGB/blob/main/train_valid_split_NI.py
import argparse
import os
import shutil

import numpy as np

from sklearn.model_selection import train_test_split

from tqdm import tqdm

"""
    A script for splitting natural images (NI) dataset into train, valid and test sets.
    This wasn't done initial
    train: 70% valid: 20% test: 10%
    randomly picking images for each set

    exanple run:
        python train_valid_test_split.py --root_dir image_data/all_label/ --output_dir image_data/example_split --train_size 0.7 --valid_size 0.1 --stratify
"""


def get_args_parser():
    parser = argparse.ArgumentParser('NI train valid test splitter', add_help=False)

    # directory parameters:
    parser.add_argument('--root_dir', type=str, required=True, help='root dir of NI dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='output dir')

    # training and testing size parameters
    parser.add_argument('--train_size', type=float, required=True, default=0.7, help='percentage of dataset which will be assigned to training. (between 0 and 1)')
    parser.add_argument('--valid_size', type=float, required=True, default=0.1, help='percentage of dataset which will be assigned to validation. (between 0 and 1)')

    # seed parameters
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # stratify, each dataset has approximately the same ammount class samples
    parser.add_argument('--stratify', action='store_true', help='stratify dataset or not')
    return parser


def main(args):
    # defining different path where images can be found and where the split should end up
    root_dir = args.root_dir
    output_dir = args.output_dir

    # seed
    seed = args.seed

    # stratify
    stratify = args.stratify

    # sizes of train, valid and test
    train_size = args.train_size
    valid_size = args.valid_size
    test_size = 1 - (train_size + valid_size)

    print('root_dir', root_dir)
    print('output_dir', output_dir)
    print('seed', seed)
    print('stratify', stratify)
    print('train_size', train_size)
    print('valid_size', valid_size)
    print('test_size', test_size)

    sets = ['train', 'valid', 'test']

    # getting classes from dataset
    labels = os.listdir(root_dir)

    # list for holding image paths and correspond class
    img_list = []
    label_list = []

    # fills above list by going through each class folder
    for label in labels:
        for img in os.listdir(f'{root_dir}/{label}'):
            img_list.append(f'{img}')
            label_list.append(label)

    # Turning the above list into numpy arrays, to speed things a little
    X = np.array(img_list)
    y = np.array(label_list)

    print(X.shape)

    if stratify:
        stratify_labels = y

    # splits dataset into 70% train and 30% valid/test
    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size=(valid_size + test_size), random_state=seed, stratify=stratify_labels)

    if stratify:
        stratify_labels = y_valid_test

    # then splits the 30% valid/test to 20% valid and 10% test. Done by making 66% of the 30% valid/test the valid and
    # the rest 33% of the 30% valid/test makes the test 10%
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=(test_size / (valid_size + test_size)), random_state=seed, stratify=stratify_labels)

    if not os.path.exists(f'{output_dir}'):
        os.mkdir(f'{output_dir}')

    for s in sets:
        # creates the train, valid, test split inside the new dataset folder
        if not os.path.exists(f'{output_dir}/{s}'):
            os.mkdir(f'{output_dir}/{s}')

            # creates class folders inside the set folder
            for label in labels:
                os.mkdir(f'{output_dir}/{s}/{label}')

    # copies images to their corresponding class folder of the train split
    for i in tqdm(range(len(X_train))):
        shutil.copyfile(f'{root_dir}/{y_train[i]}/{X_train[i]}', f'{output_dir}/train/{y_train[i]}/{X_train[i]}')

    # copies images to their corresponding class folder of the valid split
    for i in tqdm(range(len(X_valid))):
        shutil.copyfile(f'{root_dir}/{y_valid[i]}/{X_valid[i]}', f'{output_dir}/valid/{y_valid[i]}/{X_valid[i]}')

    # copies images to their corresponding class folder of the test split
    for i in tqdm(range(len(X_test))):
        shutil.copyfile(f'{root_dir}/{y_test[i]}/{X_test[i]}', f'{output_dir}/test/{y_test[i]}/{X_test[i]}')


if __name__ == '__main__':
    # creates commandline parser
    parser = argparse.ArgumentParser('Creates tiny_imagenet train valid test split', parents=[get_args_parser()])
    args = parser.parse_args()

    # passes the commandline arguments to the create_variation function
    main(args)