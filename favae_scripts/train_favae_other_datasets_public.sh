"""
* Copyright (c) 2023 OPPO. All rights reserved.
* SPDX-License-Identifier: MIT
* For full license text, see LICENSE.txt file in the repo root
"""


############# [For Table 1 row 3]  1. FA-VAE on FFHQ #############
# torchrun --nnodes=1 --nproc_per_node=1 train_favae.py --ds whatever --batch_size 2 --print_steps 1000 --img_steps 10000 \
#                         --codebook_size 2048 --disc_start_epochs 20 --embed_dim 256 --use_l2_quantizer --use_cosine_sim --num_groups 32 \
#                         --with_fcm --ffl_weight 1.0 --use_same_conv_gauss --DSL_weight_features 0.01 --gaussian_kernel 9 --dsl_init_sigma 3.0 \
#                         --codebook_weight 1.0 --perceptual_weight 1.0 --disc_weight 0.25 --base_lr 2.0e-6 \
#                         --train_file ffhq_train.pkl --test_file ../datasets/pkl_files/ffhq_test.pkl \


############# [For Table 1 last row] FA-VAE on ImageNet, with patchGAN discriminator, f=16 #############
# torchrun --nnodes=1 --nproc_per_node=1 train_favae.py --ds whatever --batch_size 1 --print_steps 5000 --img_steps 20000 \
#     --codebook_size 16384 --disc_start_epochs 20 --embed_dim 256 --use_l2_quantizer --use_cosine_sim --num_groups 32 \
#     --with_fcm --ffl_weight 1.0 --use_same_conv_gauss --DSL_weight_features 0.01 --gaussian_kernel 3 --dsl_init_sigma 3.0 \
#     --codebook_weight 1.0 --perceptual_weight 1.0 --disc_weight 0.75 --use_patch_discriminator --disc_n_layers 2 --base_lr 2.0e-6 \
#     --train_file ../datasets/pkl_files/imagenet_train.pkl --test_file ../datasets/pkl_files/imagenet_test.pkl \


############# [For Table 1 row 6] 7. FA-VAE on ImageNet, f=4 #############
############# used projection by mistake, set --codebook_dim to 256
# torchrun --nnodes=1 --nproc_per_node=1 train_favae.py --ds whatever --batch_size 1 --print_steps 5000 --img_steps 20000 --downsample_factor 4\
#     --codebook_size 8192 --disc_start_epochs 5 --embed_dim 3 --use_l2_quantizer --use_cosine_sim --num_groups 3 --codebook_dim 256 \
#     --with_fcm --ffl_weight 1.0 --use_same_conv_gauss --DSL_weight_features 0.01 --gaussian_kernel 3 --dsl_init_sigma 3.0 \
#     --codebook_weight 1.0 --perceptual_weight 1.0 --disc_weight 0.75 --base_lr 2.0e-6 \
#     --train_file ../datasets/pkl_files/imagenet_train.pkl --test_file ../datasets/pkl_files/imagenet_test.pkl \