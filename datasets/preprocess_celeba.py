"""
* Copyright (c) 2023 OPPO. All rights reserved.
* SPDX-License-Identifier: MIT
* For full license text, see LICENSE.txt file in the repo root
"""

## preprocess datasets for training

import os
import json
import pickle as pk
import pandas as pd
import glob
import argparse


def create_caption_ind_celeba(ind_list, data_path):
    ind_cap_dict = []
    ind_dict = []

    for index in ind_list:
        caption_path = os.path.join(data_path, "celeba-caption", "{}.txt".format(index))
        with open(caption_path, 'r') as f:
            caption_list = f.readlines()
        for caption in caption_list:
            ind_cap_dict.append([os.path.join(data_path, 'CelebA-HQ-img', '{}.jpg'.format(index)), caption])
        
        ind_dict.append(os.path.join(data_path, 'CelebA-HQ-img', '{}.jpg'.format(index)))

    return ind_cap_dict, ind_dict


def create_train_splits_celeba(data_path):

    map_list = pd.read_csv(os.path.join(data_path, 'CelebA-HQ-to-CelebA-mapping.txt'), 
                            delim_whitespace=True, header=0)
    
    part_list = pd.read_csv(os.path.join(data_path, 'list_eval_partition.txt'),
                            delim_whitespace=True, header=None,
                            names=["orig_file", "label"])

    ind_list = pd.merge(map_list, part_list, left_on=['orig_file'], right_on=['orig_file'])
    
    train_list = ind_list[ind_list['label']==0]['idx'].values.tolist()
    val_list = ind_list[ind_list['label']==1]['idx'].values.tolist()
    test_list = ind_list[ind_list['label']==2]['idx'].values.tolist()


    train_cap_dict, train_ind_dict = create_caption_ind_celeba(train_list, data_path)
    val_cap_dict, val_ind_dict = create_caption_ind_celeba(val_list, data_path)
    test_cap_dict, test_ind_dict = create_caption_ind_celeba(test_list, data_path)

    os.makedirs('pkl_files', exist_ok=True)

    with open('pkl_files/celeba_train_w_cap.pkl', 'wb') as f:
        pk.dump(train_cap_dict, f)
    with open('pkl_files/celeba_val_w_cap.pkl', 'wb') as f:
        pk.dump(val_cap_dict, f)
    with open('pkl_files/celeba_test_w_cap.pkl', 'wb') as f:
        pk.dump(test_cap_dict, f)

    with open('pkl_files/celeba_train.pkl', 'wb') as f:
        pk.dump(train_ind_dict, f)
    with open('pkl_files/celeba_val.pkl', 'wb') as f:
        pk.dump(val_ind_dict, f)
    with open('pkl_files/celeba_test.pkl', 'wb') as f:
        pk.dump(test_ind_dict, f)

    print("train set: {}, val set: {}, test set: {}".format(len(train_ind_dict), len(val_ind_dict), len(test_ind_dict)))
    print("with captions, train set: {}, val set: {}, test set: {}".format(len(train_cap_dict), len(val_cap_dict), len(test_cap_dict)))


def create_train_splits_ffhq(data_path):
    f = open(os.path.join(data_path, 'ffhq-dataset-v2.json'))
    data = json.load(f)

    train_dict = []
    val_dict = []
    for key in data:
        values = data[key]
        key_name = os.path.join(data_path, values["image"]['file_path'])
        if values['category'] == 'validation':
            val_dict.append(key_name)

        elif values['category'] == 'training':
            train_dict.append(key_name)

        else:
            print("found a new category, ", values['category'])


    print("\nfinished preprocessing ffhq training set... length = ", len(train_dict))

    os.makedirs('pkl_files', exist_ok=True)
    with open('pkl_files/ffhq_train.pkl', 'wb') as f:
        pk.dump(train_dict, f)

    print("\nfinished preprocessing celeba test set... length = ", len(val_dict))

    with open('pkl_files/ffhq_test.pkl', 'wb') as f:
        pk.dump(val_dict, f)


def preprocess_imagenet_helper(train_path):
    values_dict = []
    class_dirs = glob.glob(train_path+'/*')
    for class_dir in class_dirs:
        class_imgs = glob.glob(os.path.join(train_path, class_dir, '*'))
        for class_img in class_imgs:
            img_path = os.path.join(train_path, class_dir, class_img)
            values_dict.append(img_path)

    return values_dict


# this function constructs pkl files for Imagenet without captions
def create_train_splits_imagenet(data_root):

    train_path = os.path.join(data_root, "train")
    val_path = os.path.join(data_root, "val")

    values_dict = preprocess_imagenet_helper(train_path)
    print("\ntotal {} training images".format(len(values_dict)))

    os.makedirs('pkl_files', exist_ok=True)
    with open('pkl_files/imagenet_train.pkl', 'wb') as f:
        pk.dump(values_dict, f)

    values_dict = preprocess_imagenet_helper(val_path)
    print("\ntotal {} training images".format(len(values_dict)))

    with open('pkl_files/imagenet_val.pkl', 'wb') as f:
        pk.dump(values_dict, f)



def main(args):
    if args.dataset=='celeba':
        create_train_splits_celeba(args.data_path)
    elif args.dataset=='ffhq':
        create_train_splits_ffhq(args.data_path)
    elif args.dataset=='imagenet':
        create_train_splits_imagenet(args.data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['celeba', 'ffhq', 'imagenet'], help="the name of dataset")
    parser.add_argument("--data_path", type=str, help="dataset directory path")

    args = parser.parse_args()
    main(args)