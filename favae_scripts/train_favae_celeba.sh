"""
* Copyright (c) 2023 OPPO. All rights reserved.
* SPDX-License-Identifier: MIT
* For full license text, see LICENSE.txt file in the repo root
"""

############# 1.[For Table 2 row 4] FA-VAE: FCM + FFL #############
# torchrun --nnodes=1 --nproc_per_node=1 train_favae.py --ds whatever --batch_size 1 --print_steps 100 --img_steps 600 \
#                         --codebook_size 1024 --disc_start_epochs 1 --embed_dim 256 --use_l2_quantizer --use_cosine_sim \
#                         --with_fcm --ffl_weight 1.0 --use_ffl_with_fcm \
#                         --codebook_weight 1.0 --perceptual_weight 1.0 --disc_weight 0.75 --base_lr 2.0e-6 \
#                         --train_file ../datasets/pkl_files/celeba_train.pkl --test_file ../datasets/pkl_files/celeba_test.pkl \
#                         --resume --resume_path expe_1.pt

# ckpts at: fa-vae-ckpts/expe_1


############# 2. [For Table 2 row 5] FA-VAE: FCM + FFLAll (FFL in the 4 levels of encoder and decoder) #############
# torchrun --nnodes=1 --nproc_per_node=1 train_favae.py --ds whatever --batch_size 1 --print_steps 100 --img_steps 600 \
#                         --codebook_size 1024 --disc_start_epochs 1 --embed_dim 256 --use_l2_quantizer --use_cosine_sim \
#                         --with_fcm --use_ffl_with_fcm --ffl_weight 1.0 --DSL_weight_features 0.01 \
#                         --codebook_weight 1.0 --perceptual_weight 1.0 --disc_weight 0.75 --base_lr 2.0e-6 \
#                         --train_file ../datasets/pkl_files/celeba_train.pkl --test_file ../datasets/pkl_files/celeba_test.pkl \
#                         --resume --resume_path expe_2.pt

# ckpts at: fa-vae-ckpts/expe_2


############# 3. [For Table 2 row 6] FA-VAE: FCM + SL (mu=9, sigma=3) #############
############# a small typo in the paper, in fact we used mu=5 in the experiment
# torchrun --nnodes=1 --nproc_per_node=1 train_favae.py --ds whatever --batch_size 1 --print_steps 100 --img_steps 600 \
#                         --codebook_size 1024 --disc_start_epochs 1 --embed_dim 256 --use_l2_quantizer --use_cosine_sim \
#                         --with_fcm --use_ffl_with_fcm --ffl_weight 1.0 --SL_weight 0.01 --gaussian_kernel 5 --gaussian_sigma 3 \
#                         --codebook_weight 1.0 --perceptual_weight 1.0 --disc_weight 0.75 --base_lr 2.0e-6 \
#                         --train_file ../datasets/pkl_files/celeba_train.pkl --test_file ../datasets/pkl_files/celeba_test.pkl \
#                         --resume --resume_path expe_3.pt

# ckpts at: fa-vae-ckpts/expe_3

########## the batch size for V-100 is usually 8 for the experiments below ###########

############# 4. [For Table 2 row 7] FA-VAE: FCM (CONV) + non pair-wise DSL (mu=9, initial sigma=3) #############
# torchrun --nnodes=1 --nproc_per_node=1 train_favae.py --ds whatever --batch_size 1 --print_steps 100 --img_steps 600 \
#                         --codebook_size 1024 --disc_start_epochs 1 --embed_dim 256 --use_l2_quantizer --use_cosine_sim \
#                         --with_fcm --ffl_weight 1.0 --use_non_pair_conv --DSL_weight_features 0.01 --gaussian_kernel 9 --dsl_init_sigma 3.0  \
#                         --codebook_weight 1.0 --perceptual_weight 1.0 --disc_weight 0.75 --base_lr 2.0e-6 \
#                         --train_file ../datasets/pkl_files/celeba_train.pkl --test_file ../datasets/pkl_files/celeba_test.pkl \
#                         --resume --resume_path expe_4.pt


# ckpts at: fa-vae-ckpts/expe_4


############# 5. [For Table 2 row 8] FA-VAE: FCM (Res) + non pair-wise DSL (mu=9, initial sigma=3) #############
# torchrun --nnodes=1 --nproc_per_node=1 train_favae.py --ds whatever --batch_size 1 --print_steps 100 --img_steps 800 \
#                         --codebook_size 1024 --disc_start_epochs 1 --embed_dim 256 --use_l2_quantizer --use_cosine_sim \
#                          --with_fcm --ffl_weight 1.0 --use_gauss_resblock --DSL_weight_features 0.01 --gaussian_kernel 9 --dsl_init_sigma 3.0  \
#                         --codebook_weight 1.0 --perceptual_weight 1.0 --disc_weight 0.75 --base_lr 2.0e-6  \
#                         --train_file ../datasets/pkl_files/celeba_train.pkl --test_file ../datasets/pkl_files/celeba_test.pkl \
#                         --resume --resume_path expe_5.pt

# ckpts at: fa-vae-ckpts/expe_5


############# 6. [For Table 2 row 9] FA-VAE: FCM (Attn) + non pair-wise DSL (mu=9, initial sigma=3) #############
# torchrun --nnodes=1 --nproc_per_node=1 train_favae.py --ds whatever --batch_size 1 --print_steps 100 --img_steps 800 \
#                         --codebook_size 1024 --disc_start_epochs 1 --embed_dim 256 --use_l2_quantizer --use_cosine_sim \
#                         --with_fcm --ffl_weight 1.0 --use_gauss_attn --DSL_weight_features 0.01 --gaussian_kernel 9 --dsl_init_sigma 3.0  \
#                         --codebook_weight 1.0 --perceptual_weight 1.0 --disc_weight 0.75 --base_lr 2.0e-6  \
#                         --train_file ../datasets/pkl_files/celeba_train.pkl --test_file ../datasets/pkl_files/celeba_test.pkl \
#                         --resume --resume_path expe_6.pt

# ckpts at: fa-vae-ckpts/expe_6

############# 7. [For Table 2 row 10 to 14] FA-VAE: FCM (Resblock) + pair-wise DSL (mu={3,5,9,11,15}, initial sigma=3} #############
############# vary the --gaussian_kernel argument to change the kernel size
# torchrun --nnodes=1 --nproc_per_node=1 train_favae.py --ds whatever --batch_size 1 --print_steps 100 --img_steps 1600 \
#                         --codebook_size 1024 --disc_start_epochs 1 --embed_dim 256 --use_l2_quantizer --use_cosine_sim \
#                         --with_fcm --ffl_weight 1.0 --use_same_gauss_resblock --DSL_weight_features 0.01 --gaussian_kernel 15 --dsl_init_sigma 3.0 \
#                         --codebook_weight 1.0 --perceptual_weight 1.0 --disc_weight 0.75 --base_lr 2.0e-6 \
#                         --train_file ../datasets/pkl_files/celeba_train.pkl --test_file ../datasets/pkl_files/celeba_test.pkl \
#                         --resume --resume_path expe_7_mu15.pt

# ckpts at: fa-vae-ckpts/expe_7_mu3
# ckpts at: fa-vae-ckpts/expe_7_mu5
# ckpts at: fa-vae-ckpts/expe_7_mu9
# ckpts at: fa-vae-ckpts/expe_7_mu11
# ckpts at: fa-vae-ckpts/expe_7_mu15
