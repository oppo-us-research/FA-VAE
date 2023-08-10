"""
* Copyright (c) 2023 OPPO. All rights reserved.
* SPDX-License-Identifier: MIT
* For full license text, see LICENSE.txt file in the repo root
"""

"""
This script is used to create dataloaders for training FA-VAE
"""

import pickle as pk
import torch
import torchvision.transforms as T
from PIL import Image
from .statistic import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GeneralDataset(torch.utils.data.Dataset):
#   'Characterizes a dataset for PyTorch'
    def __init__(self, resolution, train=True, val=False, train_file=None, test_file=None):

        if train:
            with open(train_file, "rb") as input_file:
                self.names_dict = pk.load(input_file)
                
        if val:
            with open(test_file, "rb") as input_file:
                self.names_dict = pk.load(input_file)

        self.transform = T.Compose([
            T.Resize((resolution,resolution)),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.clip_transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=clip_mean, std=clip_std)
        ])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.names_dict)

    def load_image(self, name):
        try:
            image = Image.open(name)
            if not image.mode == "RGB":
                image = image.convert("RGB")

            return image
        except:
            return None

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        name = self.names_dict[index]
        img = self.load_image(name)
        if img is None:
            return self.__getitem__(index+1)
        ori_img = self.transform(img)

        return ori_img


def load_data(args):
    train_loader = None
    test_loader = None
    if args.train_file is not None:
        train_set = GeneralDataset(resolution=args.resolution, train=True, val=False, train_file=args.train_file, test_file=None)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        print("\nLoaded the train set length {}, dataloader length {}".format(len(train_set), len(train_loader)))

    if args.test_file is not None:
        test_set = GeneralDataset(resolution=args.resolution, train=False, val=True, train_file=None, test_file=args.test_file)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print("\nLoaded the test set length {}, dataloader length {}".format(len(test_set), len(test_loader)))

    return train_loader, test_loader
