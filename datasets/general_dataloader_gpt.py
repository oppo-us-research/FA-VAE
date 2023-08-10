"""
* Copyright (c) 2023 OPPO. All rights reserved.
* SPDX-License-Identifier: MIT
* For full license text, see LICENSE.txt file in the repo root
"""


"""
This script is to build dataloaders for training CAT 
"""

import pickle as pk
import torch
import os
import torchvision.transforms as T
from PIL import Image
from .statistic import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(42)

class GeneralDataset(torch.utils.data.Dataset):
    def __init__(self, resolution, train=True, val=False, train_file=None, val_file=None, text_cond=False, text_tok_cond=False):

        self.text_cond = text_cond
        self.text_tok_cond = text_tok_cond

        if train:
            with open(train_file, "rb") as input_file:
                self.names_dict = pk.load(input_file)
                
        if val:
            with open(val_file, "rb") as input_file:
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
        name = self.names_dict[index][0]

        if self.text_cond or self.text_tok_cond:
            caption = self.names_dict[index][1]

            if not os.path.exists(name):
                print("\n this image seems to not exist...", name)

        img = self.load_image(name)

        if img is None:
            print("\nerror_loading the image, ", name)
            return self.__getitem__(index+1)

        ori_img = self.transform(img)

        if self.text_cond or self.text_tok_cond:

            clip_img = self.clip_transform(img)

            return ori_img, clip_img, caption


def load_data(args):
    train_loader = None
    val_loader = None

    if args.train_file is not None:

        train_set = GeneralDataset(resolution=args.resolution, train=True, val=False, train_file=args.train_file, val_file=None, text_cond=args.txt_cond, text_tok_cond=args.txt_tok_cond)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        print("\nLoaded the train set length {}, dataloader length {}".format(len(train_set), len(train_loader)))

    if args.val_file is not None:

        val_set = GeneralDataset(resolution=args.resolution, train=False, val=True, train_file=None, val_file=args.val_file, text_cond=args.txt_cond, text_tok_cond=args.txt_tok_cond)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print("\nLoaded the val set length {}, dataloader length {}".format(len(val_set), len(val_loader)))

    return train_loader, val_loader
