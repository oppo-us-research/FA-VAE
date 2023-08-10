###
# Copyright (c) 2023 OPPO. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###


torchrun --nnodes=1 --nproc_per_node=2 train_cat.py --codebook_size 1024 --embed_dim 256 \
                --enabled_warmup --ds $DS --print_steps 300 --img_steps 2000 \
                --batch_size 1 --txt_tok_cond --top_k 500 --top_p 0.95 \
                --clip vit-l-14 --n_cond_embed 768 \
                --train_file ../../fa-vae/datasets/pkl_files/celeba_train_w_cap.pkl \
                --val_file ../../fa-vae/datasets/pkl_files/celeba_test_w_cap.pkl \
                --use_l2_quantizer --use_cosine_sim --use_same_gauss_resblock  --gaussian_kernel 3 \
                --favae_ckpt expe_7_mu9.pt \