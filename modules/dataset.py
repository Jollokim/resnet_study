# modified version of one of my earlier works: https://github.com/Jollokim/Alexnet-edgemaps-vs-RGB/blob/main/datasets.py
import os
import torch

import pandas as pd
import numpy as np
import cv2 as cv

from torch.utils.data import Dataset
from torchvision.transforms import transforms


"""
Datasets follow this structure:
    Dataset 
        train
            class1
                img1
                img2
                ...
            class2
                img11
                ...
            class3
                img21
                ...
            ...
        valid
            class1
            ...
        test
            class1
            ...
    
    When the below dataset class loads, it only loads one of the train, valid or test split.
"""


# custom dataset class for loading the datasets used in this project
class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir):
        # the path to the particular train, test or valid split inside of the dataset
        self.root_dir = root_dir

        # initializing the transform method, more on this below
        self.transform = transforms.ToTensor()

        # list for holding path to image
        self.img_path_list = []

        # list for holding the class by name
        self.label_list = []

        # list for holding the class as one hot vector
        self.target_list = []

        # getting classes by listing the different class folders, see above structure of datasets
        self.classes = os.listdir(root_dir)

        # counter of classes, used to decide the index of 1 for one hot vector
        count = 0

        for label in self.classes:

            # getting all image path from inside its class folder
            for img in os.listdir(f'{root_dir}/{label}'):
                # appending image path and its class by name to each of their list
                self.img_path_list.append(f'{root_dir}/{label}/{img}')
                self.label_list.append(label)

                # creates the one hot vector for that class. Here count defines whch index gets the 1
                target = np.eye(1, len(self.classes), count)

                # appends one hot vector
                self.target_list.append(target)

            count += 1

        # pandas dataframe for easier access to data
        self.data = pd.DataFrame()

        # the 3 columns for each of the 3 created above
        self.data['img'] = self.img_path_list
        self.data['target'] = self.target_list
        self.data['label'] = self.label_list

    # method for extracting one sample and corresponding target. Used by dataloader during batch creation
    def __getitem__(self, index):

        # label of this index
        label = self.data.iloc[index, 2]

        # reads image to numpy array H x W x C (Height, Width, Channels)
        img = cv.imread(self.data.iloc[index, 0])

        # converts target np array to a tensor
        y = torch.tensor(self.data.iloc[index, 1]).view(-1)

        # transforms image to C x H x W and scales values to interval [0, 1]
        X = self.transform(img)

        # returns everything as floats, instead of double. For making the loss calculator happy
        return {'image': img, 'X': X.float(), 'y': y.float(), 'label': label}

    def __len__(self):
        return len(self.data)


# Testing the dataset works with the dataloader class works
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = TinyImageNetDataset('image_data/train_label')

    d = dataset.__getitem__(0)
    print(d['image'].shape)
    print(d['X'].shape)
    print(d['y'].shape)
    print(d['y'])
    print(d['label'])

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    for batch in dataloader:
        print(batch['X'].shape, batch['y'].shape)
        break