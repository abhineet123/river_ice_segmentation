<!-- MarkdownTOC -->

- [install](#install)
- [384](#384)
   - [10k/10k/600/loss_4       @ 384](#10k10k600loss_4__384)
      - [random_32       @ 10k/10k/600/loss_4/384](#random_32__10k10k600loss_4384)
         - [vis       @ random_32/10k/10k/600/loss_4/384](#vis__random_3210k10k600loss_4384)
   - [obsolete       @ 384](#obsolete__384)
- [512](#512)
   - [10k/10k/500/loss_4       @ 512](#10k10k500loss_4__512)
      - [random_32       @ 10k/10k/500/loss_4/512](#random_32__10k10k500loss_4512)
         - [vis       @ random_32/10k/10k/500/loss_4/512](#vis__random_3210k10k500loss_4512)
- [640](#640)
   - [10k/10k/400/loss_4       @ 640](#10k10k400loss_4__640)
      - [random_50       @ 10k/10k/400/loss_4/640](#random_50__10k10k400loss_4640)
         - [vis       @ random_50/10k/10k/400/loss_4/640](#vis__random_5010k10k400loss_4640)
         - [validation       @ random_50/10k/10k/400/loss_4/640](#validation__random_5010k10k400loss_4640)
- [1000](#1000)
   - [10k/10k/400/loss_4       @ 1000](#10k10k400loss_4__1000)
      - [random_50       @ 10k/10k/400/loss_4/1000](#random_50__10k10k400loss_41000)
         - [predict       @ random_50/10k/10k/400/loss_4/1000](#predict__random_5010k10k400loss_41000)
         - [vis       @ random_50/10k/10k/400/loss_4/1000](#vis__random_5010k10k400loss_41000)
         - [validation       @ random_50/10k/10k/400/loss_4/1000](#validation__random_5010k10k400loss_41000)
         - [video       @ random_50/10k/10k/400/loss_4/1000](#video__random_5010k10k400loss_41000)
            - [YUN00001_1920x1080_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/400/loss_4/1000](#yun00001_1920x108001799_1000_1000_1000_1000__videorandom_5010k10k400loss_41000)
            - [YUN00002_1920x1080_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/400/loss_4/1000](#yun00002_1920x108001799_1000_1000_1000_1000__videorandom_5010k10k400loss_41000)
            - [YUN00002_2000_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/400/loss_4/1000](#yun00002_200001799_1000_1000_1000_1000__videorandom_5010k10k400loss_41000)
            - [20160122_YUN00002_700_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/400/loss_4/1000](#20160122_yun00002_70001799_1000_1000_1000_1000__videorandom_5010k10k400loss_41000)
            - [20161201_YUN00002_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/400/loss_4/1000](#20161201_yun0000201799_1000_1000_1000_1000__videorandom_5010k10k400loss_41000)
            - [20170114_YUN00005_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/400/loss_4/1000](#20170114_yun0000501799_1000_1000_1000_1000__videorandom_5010k10k400loss_41000)
   - [10k/10k/800/loss_4       @ 1000](#10k10k800loss_4__1000)
      - [random_50       @ 10k/10k/800/loss_4/1000](#random_50__10k10k800loss_41000)
         - [predict       @ random_50/10k/10k/800/loss_4/1000](#predict__random_5010k10k800loss_41000)
         - [vis       @ random_50/10k/10k/800/loss_4/1000](#vis__random_5010k10k800loss_41000)
         - [validation       @ random_50/10k/10k/800/loss_4/1000](#validation__random_5010k10k800loss_41000)
         - [video       @ random_50/10k/10k/800/loss_4/1000](#video__random_5010k10k800loss_41000)
            - [YUN00001_1920x1080_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/800/loss_4/1000](#yun00001_1920x108001799_1000_1000_1000_1000__videorandom_5010k10k800loss_41000)
            - [YUN00002_1920x1080_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/800/loss_4/1000](#yun00002_1920x108001799_1000_1000_1000_1000__videorandom_5010k10k800loss_41000)
            - [YUN00001_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/800/loss_4/1000](#yun0000101799_1000_1000_1000_1000__videorandom_5010k10k800loss_41000)
            - [YUN00002_2000_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/800/loss_4/1000](#yun00002_200001799_1000_1000_1000_1000__videorandom_5010k10k800loss_41000)
            - [20160122_YUN00002_700_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/800/loss_4/1000](#20160122_yun00002_70001799_1000_1000_1000_1000__videorandom_5010k10k800loss_41000)
            - [20161201_YUN00002_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/800/loss_4/1000](#20161201_yun0000201799_1000_1000_1000_1000__videorandom_5010k10k800loss_41000)
            - [20170114_YUN00005_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/800/loss_4/1000](#20170114_yun0000501799_1000_1000_1000_1000__videorandom_5010k10k800loss_41000)
   - [10k/10k/all/loss_4       @ 1000](#10k10kallloss_4__1000)
- [horse](#horse)
- [800](#800)
   - [10k/10k/1       @ 800](#10k10k1__800)
      - [loss_3       @ 10k/10k/1/800](#loss_3__10k10k1800)
      - [loss_4       @ 10k/10k/1/800](#loss_4__10k10k1800)
   - [10k/10k/50       @ 800](#10k10k50__800)
   - [10k/10k/100       @ 800](#10k10k100__800)
      - [loss_0       @ 10k/10k/100/800](#loss_0__10k10k100800)
      - [loss_2       @ 10k/10k/100/800](#loss_2__10k10k100800)
      - [loss_3       @ 10k/10k/100/800](#loss_3__10k10k100800)
      - [loss_4       @ 10k/10k/100/800](#loss_4__10k10k100800)
   - [10k/10k/200/loss_4       @ 800](#10k10k200loss_4__800)
      - [training       @ 10k/10k/200/loss_4/800](#training__10k10k200loss_4800)
         - [prediction       @ training/10k/10k/200/loss_4/800](#prediction__training10k10k200loss_4800)
      - [random       @ 10k/10k/200/loss_4/800](#random__10k10k200loss_4800)
         - [prediction       @ random/10k/10k/200/loss_4/800](#prediction__random10k10k200loss_4800)
   - [vis       @ 800](#vis__800)
            - [video       @ vis/800](#video__vis800)
            - [YUN00001_0_239_800_800_800_800       @ vis/800](#yun000010239_800_800_800_800__vis800)
            - [YUN00002_2000_0_1799_800_800_800_800       @ vis/800](#yun00002_200001799_800_800_800_800__vis800)
            - [20160122_YUN00002_700_0_1799_800_800_800_800       @ vis/800](#20160122_yun00002_70001799_800_800_800_800__vis800)
            - [20161201_YUN00002_0_1799_800_800_800_800       @ vis/800](#20161201_yun0000201799_800_800_800_800__vis800)
            - [20170114_YUN00005_0_1799_800_800_800_800       @ vis/800](#20170114_yun0000501799_800_800_800_800__vis800)
            - [zip       @ vis/800](#zip__vis800)
   - [10k/10k/all       @ 800](#10k10kall__800)
      - [loss_4       @ 10k/10k/all/800](#loss_4__10k10kall800)
   - [15k/10k       @ 800](#15k10k__800)
   - [8k/8k/all/loss_4/augmented       @ 800](#8k8kallloss_4augmented__800)
      - [50       @ 8k/8k/all/loss_4/augmented/800](#50__8k8kallloss_4augmented800)
         - [vis       @ 50/8k/8k/all/loss_4/augmented/800](#vis__508k8kallloss_4augmented800)
         - [validation       @ 50/8k/8k/all/loss_4/augmented/800](#validation__508k8kallloss_4augmented800)
      - [4       @ 8k/8k/all/loss_4/augmented/800](#4__8k8kallloss_4augmented800)
         - [vis       @ 4/8k/8k/all/loss_4/augmented/800](#vis__48k8kallloss_4augmented800)
         - [no_aug       @ 4/8k/8k/all/loss_4/augmented/800](#no_aug__48k8kallloss_4augmented800)
         - [stitched       @ 4/8k/8k/all/loss_4/augmented/800](#stitched__48k8kallloss_4augmented800)
      - [8       @ 8k/8k/all/loss_4/augmented/800](#8__8k8kallloss_4augmented800)
      - [vis       @ 8k/8k/all/loss_4/augmented/800](#vis__8k8kallloss_4augmented800)
         - [no_aug       @ vis/8k/8k/all/loss_4/augmented/800](#no_aug__vis8k8kallloss_4augmented800)
         - [stitched       @ vis/8k/8k/all/loss_4/augmented/800](#stitched__vis8k8kallloss_4augmented800)
      - [16       @ 8k/8k/all/loss_4/augmented/800](#16__8k8kallloss_4augmented800)
      - [vis       @ 8k/8k/all/loss_4/augmented/800](#vis__8k8kallloss_4augmented800-1)
         - [no_aug       @ vis/8k/8k/all/loss_4/augmented/800](#no_aug__vis8k8kallloss_4augmented800-1)
         - [stitched       @ vis/8k/8k/all/loss_4/augmented/800](#stitched__vis8k8kallloss_4augmented800-1)
      - [24       @ 8k/8k/all/loss_4/augmented/800](#24__8k8kallloss_4augmented800)
      - [vis       @ 8k/8k/all/loss_4/augmented/800](#vis__8k8kallloss_4augmented800-2)
         - [no_aug       @ vis/8k/8k/all/loss_4/augmented/800](#no_aug__vis8k8kallloss_4augmented800-2)
         - [stitched       @ vis/8k/8k/all/loss_4/augmented/800](#stitched__vis8k8kallloss_4augmented800-2)
      - [32       @ 8k/8k/all/loss_4/augmented/800](#32__8k8kallloss_4augmented800)
      - [vis       @ 8k/8k/all/loss_4/augmented/800](#vis__8k8kallloss_4augmented800-3)
         - [no_aug       @ vis/8k/8k/all/loss_4/augmented/800](#no_aug__vis8k8kallloss_4augmented800-3)
         - [stitched       @ vis/8k/8k/all/loss_4/augmented/800](#stitched__vis8k8kallloss_4augmented800-3)
- [800_retraining](#800_retraining)
   - [4__non_aug       @ 800_retraining](#4__non_aug__800_retraining)
   - [4       @ 800_retraining](#4__800_retraining)
      - [loss_1       @ 4/800_retraining](#loss_1__4800_retraining)
      - [12_layers       @ 4/800_retraining](#12_layers__4800_retraining)
      - [vis       @ 4/800_retraining](#vis__4800_retraining)
         - [loss_1       @ vis/4/800_retraining](#loss_1__vis4800_retraining)
      - [vis_no_aug       @ 4/800_retraining](#vis_no_aug__4800_retraining)
         - [loss_1       @ vis_no_aug/4/800_retraining](#loss_1__vis_no_aug4800_retraining)
      - [vis_stitched       @ 4/800_retraining](#vis_stitched__4800_retraining)
      - [vis_no_aug__4_49       @ 4/800_retraining](#vis_no_aug_449__4800_retraining)
   - [8       @ 800_retraining](#8__800_retraining)
      - [loss_1       @ 8/800_retraining](#loss_1__8800_retraining)
         - [vis       @ loss_1/8/800_retraining](#vis__loss_18800_retraining)
         - [vis_no_aug       @ loss_1/8/800_retraining](#vis_no_aug__loss_18800_retraining)
         - [vis_stitched       @ loss_1/8/800_retraining](#vis_stitched__loss_18800_retraining)
   - [16       @ 800_retraining](#16__800_retraining)
      - [vis       @ 16/800_retraining](#vis__16800_retraining)
      - [vis_no_aug       @ 16/800_retraining](#vis_no_aug__16800_retraining)
      - [vis_stitched       @ 16/800_retraining](#vis_stitched__16800_retraining)
   - [24       @ 800_retraining](#24__800_retraining)
      - [vis       @ 24/800_retraining](#vis__24800_retraining)
      - [vis_no_aug       @ 24/800_retraining](#vis_no_aug__24800_retraining)
      - [vis_stitched       @ 24/800_retraining](#vis_stitched__24800_retraining)
   - [32       @ 800_retraining](#32__800_retraining)
      - [vis       @ 32/800_retraining](#vis__32800_retraining)
      - [vis_no_aug       @ 32/800_retraining](#vis_no_aug__32800_retraining)
      - [vis_stitched       @ 32/800_retraining](#vis_stitched__32800_retraining)
      - [YUN00001       @ 32/800_retraining](#yun00001__32800_retraining)
      - [YUN00001_3600       @ 32/800_retraining](#yun00001_3600__32800_retraining)
      - [vis_png       @ 32/800_retraining](#vis_png__32800_retraining)
         - [20160122_YUN00002_700_2500       @ vis_png/32/800_retraining](#20160122_yun00002_700_2500__vis_png32800_retraining)
         - [20160122_YUN00020_2000_3800       @ vis_png/32/800_retraining](#20160122_yun00020_2000_3800__vis_png32800_retraining)
         - [20161203_Deployment_1_YUN00001_900_2700       @ vis_png/32/800_retraining](#20161203_deployment1yun00001_900_2700__vis_png32800_retraining)
         - [20161203_Deployment_1_YUN00002_1800       @ vis_png/32/800_retraining](#20161203_deployment1yun00002_1800__vis_png32800_retraining)
         - [YUN00001_3600       @ vis_png/32/800_retraining](#yun00001_3600__vis_png32800_retraining)
         - [20161203_Deployment_1_YUN00001_900_2700       @ vis_png/32/800_retraining](#20161203_deployment1yun00001_900_2700__vis_png32800_retraining-1)
         - [20161203_Deployment_1_YUN00002_1800       @ vis_png/32/800_retraining](#20161203_deployment1yun00002_1800__vis_png32800_retraining-1)
      - [fixed_lr       @ 32/800_retraining](#fixed_lr__32800_retraining)
      - [loss_1       @ 32/800_retraining](#loss_1__32800_retraining)
      - [18_layers       @ 32/800_retraining](#18_layers__32800_retraining)
      - [21_layers       @ 32/800_retraining](#21_layers__32800_retraining)
      - [21_layers__640       @ 32/800_retraining](#21_layers__640__32800_retraining)
      - [24_layers__640       @ 32/800_retraining](#24_layers__640__32800_retraining)
- [800_retraining_fixed_indices](#800_retraining_fixed_indices)
   - [4__non_aug       @ 800_retraining_fixed_indices](#4__non_aug__800_retraining_fixed_indices)
      - [2       @ 4__non_aug/800_retraining_fixed_indices](#2__4__non_aug800_retraining_fixed_indices)
      - [10       @ 4__non_aug/800_retraining_fixed_indices](#10__4__non_aug800_retraining_fixed_indices)
      - [100       @ 4__non_aug/800_retraining_fixed_indices](#100__4__non_aug800_retraining_fixed_indices)
      - [1000       @ 4__non_aug/800_retraining_fixed_indices](#1000__4__non_aug800_retraining_fixed_indices)
   - [4       @ 800_retraining_fixed_indices](#4__800_retraining_fixed_indices)
      - [2       @ 4/800_retraining_fixed_indices](#2__4800_retraining_fixed_indices)
      - [5       @ 4/800_retraining_fixed_indices](#5__4800_retraining_fixed_indices)
      - [10       @ 4/800_retraining_fixed_indices](#10__4800_retraining_fixed_indices)
      - [100       @ 4/800_retraining_fixed_indices](#100__4800_retraining_fixed_indices)
      - [vis       @ 4/800_retraining_fixed_indices](#vis__4800_retraining_fixed_indices)
      - [vis_no_aug       @ 4/800_retraining_fixed_indices](#vis_no_aug__4800_retraining_fixed_indices)
      - [vis_stitched       @ 4/800_retraining_fixed_indices](#vis_stitched__4800_retraining_fixed_indices)
      - [1K       @ 4/800_retraining_fixed_indices](#1k__4800_retraining_fixed_indices)
      - [5K       @ 4/800_retraining_fixed_indices](#5k__4800_retraining_fixed_indices)
   - [16       @ 800_retraining_fixed_indices](#16__800_retraining_fixed_indices)
      - [100       @ 16/800_retraining_fixed_indices](#100__16800_retraining_fixed_indices)
      - [1K       @ 16/800_retraining_fixed_indices](#1k__16800_retraining_fixed_indices)
      - [5K       @ 16/800_retraining_fixed_indices](#5k__16800_retraining_fixed_indices)
- [640_retraining](#640_retraining)
   - [4       @ 640_retraining](#4__640_retraining)
      - [sel_2       @ 4/640_retraining](#sel_2__4640_retraining)
      - [sel_10       @ 4/640_retraining](#sel_10__4640_retraining)
      - [sel_100       @ 4/640_retraining](#sel_100__4640_retraining)
      - [sel_1000       @ 4/640_retraining](#sel_1000__4640_retraining)
      - [sel_10000       @ 4/640_retraining](#sel_10000__4640_retraining)

<!-- /MarkdownTOC -->

<a id="install"></a>
# install

pip2 install h5py==2.8.0rc1
pip3 install h5py==2.8.0rc1

<a id="384"></a>
# 384

python3 densenet_train.py --train_images=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --train_labels=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --height=384 --width=384 --index_percent=5 --n_classes=3 --start_id=0 --end_id=5 --n_epochs=1000 --log_dir=log/ratio_5_0_0_384 --gpu_id=2 --max_indices=10000 --min_indices=10000

python3 densenet_train.py --train_images=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --train_labels=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --height=384 --width=384 --index_percent=5 --n_classes=3 --start_id=0 --end_id=5 --n_epochs=1000 --log_dir=log/ratio_5_0_0_384 --gpu_id=2 --max_indices=10000 --min_indices=10000

<a id="10k10k600loss_4__384"></a>
## 10k/10k/600/loss_4       @ 384

<a id="random_32__10k10k600loss_4384"></a>
### random_32       @ 10k/10k/600/loss_4/384

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_31_384_384_25_100_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_31_384_384_25_100_rot_15_345_4_flip/labels --height=384 --width=384 --index_percent=50 --n_classes=3 --start_id=-1 --end_id=600 --n_epochs=100000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=99 --save_stitched=1 --loss_type=4 --eval_every=10

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --height=384 --width=384 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_384_random_601_4/weights/ --save_path=log/50_10000_10000_384_random_601_4/predict/training_32_49_384_384_25_100_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/validation_0_20_384_384_384_384/images --height=384 --width=384 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_384_random_601_4/weights/ --save_path=log/50_10000_10000_384_random_601_4/predict/validation_0_20_384_384_384_384 --save_stitched=1 --gpu_id=2 --loss_type=4

<a id="vis__random_3210k10k600loss_4384"></a>
#### vis       @ random_32/10k/10k/600/loss_4/384

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_384_384_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_384_384_64_256_rot_15_345_4_flip/labels --seg_path=log/50_10000_10000_384_random_401_4/predict/training_32_49_384_384_64_256_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_384_384_64_256_rot_15_345_4_flip/vis_50_10000_10000_384_random_401_4 --n_classes=3 --start_id=0 --end_id=-1

<a id="obsolete__384"></a>
## obsolete       @ 384

python3 densenet_train.py --train_images=/data/617/images/training/images --train_labels=/data/617/images/training/labels --train_images_ext=tif --train_labels_ext=tif --index_percent=50 --n_classes=3 --start_id=0 --end_id=0 --n_epochs=1000 --log_dir=log/ratio_5_0_0_384

python3 densenet_train.py --train_images=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --train_labels=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --index_percent=50 --n_classes=3 --start_id=0 --end_id=0 --n_epochs=1000 --log_dir=log/ratio_5_0_0_384 --gpu_id=2


<a id="512"></a>
# 512

<a id="10k10k500loss_4__512"></a>
## 10k/10k/500/loss_4       @ 512

<a id="random_32__10k10k500loss_4512"></a>
### random_32       @ 10k/10k/500/loss_4/512

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/labels --height=512 --width=512 --index_percent=50 --n_classes=3 --start_id=-1 --end_id=500 --n_epochs=100000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=99 --save_stitched=1 --loss_type=4 --eval_every=10

CUDA_VISIBLE_DEVICES=0 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels --height=512 --width=512 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_512_random_501_4/weights/ --save_path=log/50_10000_10000_512_random_501_4/predict/training_32_49_512_512_25_100_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

CUDA_VISIBLE_DEVICES=0 python3 densenet_predict.py --images_path=/data/617/images/validation_0_20_512_512_512_512/images --height=512 --width=512 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_512_random_501_4/weights/ --save_path=log/50_10000_10000_512_random_501_4/predict/validation_0_20_512_512_512_512 --save_stitched=1 --gpu_id=2 --loss_type=4

<a id="vis__random_3210k10k500loss_4512"></a>
#### vis       @ random_32/10k/10k/500/loss_4/512

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_512_512_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_512_512_64_256_rot_15_345_4_flip/labels --seg_path=log/50_10000_10000_512_random_401_4/predict/training_32_49_512_512_64_256_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_512_512_64_256_rot_15_345_4_flip/vis_50_10000_10000_512_random_401_4 --n_classes=3 --start_id=0 --end_id=-1

<a id="640"></a>
# 640

<a id="10k10k400loss_4__640"></a>
## 10k/10k/400/loss_4       @ 640

<a id="random_50__10k10k400loss_4640"></a>
### random_50       @ 10k/10k/400/loss_4/640

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_49_640_640_64_256_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_49_640_640_64_256_rot_15_345_4_flip/labels --height=640 --width=640 --index_percent=50 --n_classes=3 --start_id=-1 --end_id=400 --n_epochs=100000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=99 --save_stitched=1 --loss_type=4 --eval_every=10


<a id="vis__random_5010k10k400loss_4640"></a>
#### vis       @ random_50/10k/10k/400/loss_4/640


CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --height=640 --width=640 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_640_random_401_4/weights/ --save_path=log/50_10000_10000_640_random_401_4/predict/training_32_49_640_640_64_256_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/50_10000_10000_640_random_401_4/predict/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/vis_50_10000_10000_640_random_401_4 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation__random_5010k10k400loss_4640"></a>
#### validation       @ random_50/10k/10k/400/loss_4/640

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/validation_0_20_640_640_640_640/images --height=640 --width=640 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_640_random_401_4/weights/ --save_path=log/50_10000_10000_640_random_401_4/predict/validation_0_20_640_640_640_640 --save_stitched=1 --gpu_id=2 --loss_type=4


<a id="1000"></a>
# 1000

<a id="10k10k400loss_4__1000"></a>
## 10k/10k/400/loss_4       @ 1000

<a id="random_50__10k10k400loss_41000"></a>
### random_50       @ 10k/10k/400/loss_4/1000

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_49_1000_1000_100_400_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_49_1000_1000_100_400_rot_15_345_4_flip/labels --height=1000 --width=1000 --index_percent=50 --n_classes=3 --start_id=-1 --end_id=400 --n_epochs=100000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=99 --save_stitched=1 --loss_type=4 --eval_every=10

<a id="predict__random_5010k10k400loss_41000"></a>
#### predict       @ random_50/10k/10k/400/loss_4/1000

CUDA_VISIBLE_DEVICES=0 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_1000_1000_100_400_rot_15_345_4_flip/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_401_4/weights/ --save_path=log/50_10000_10000_1000_random_401_4/predict/training_32_49_1000_1000_100_400_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

<a id="vis__random_5010k10k400loss_41000"></a>
#### vis       @ random_50/10k/10k/400/loss_4/1000

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_1000_1000_100_400_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_1000_1000_100_400_rot_15_345_4_flip/labels --seg_path=log/50_10000_10000_1000_random_401_4/predict/training_32_49_1000_1000_100_400_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_1000_1000_100_400_rot_15_345_4_flip/vis_50_10000_10000_1000_random_401_4 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation__random_5010k10k400loss_41000"></a>
#### validation       @ random_50/10k/10k/400/loss_4/1000


CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/validation_0_20_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_401_4/weights/ --save_path=log/50_10000_10000_1000_random_401_4/predict/validation_0_20_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4

zr 50_10000_10000_1000_random_401_4_validation_0_20_1000_1000_1000_1000 50_10000_10000_1000_random_401_4/predict/validation_0_20_1000_1000_1000_1000

<a id="video__random_5010k10k400loss_41000"></a>
#### video       @ random_50/10k/10k/400/loss_4/1000

<a id="yun00001_1920x108001799_1000_1000_1000_1000__videorandom_5010k10k400loss_41000"></a>
##### YUN00001_1920x1080_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/400/loss_4/1000

CUDA_VISIBLE_DEVICES=0 python3 densenet_predict.py --images_path=/data/617/images/YUN00001_1920x1080_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_401_4/weights/ --save_path=log/50_10000_10000_1000_random_401_4/predict/YUN00001_1920x1080_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_1920x1080/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_401_4/predict/YUN00001_1920x1080_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_401_4/predict/YUN00001_1920x1080_0_1799_1000_1000_1000_1000_stitched patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=1

python3 ~/PTF/imgSeqToVideo.py src_path=log/50_10000_10000_1000_random_401_4/predict/YUN00001_1920x1080_0_1799_1000_1000_1000_1000_stitched save_path=log/50_10000_10000_1000_random_401_4/predict/YUN00001_1920x1080_0_1799_1000_1000_1000_1000_stitched.mkv width=1920 height=1080 show_img=0

<a id="yun00002_1920x108001799_1000_1000_1000_1000__videorandom_5010k10k400loss_41000"></a>
##### YUN00002_1920x1080_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/400/loss_4/1000

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/YUN00002_1920x1080_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_401_4/weights/ --save_path=log/50_10000_10000_1000_random_401_4/predict/YUN00002_1920x1080_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/YUN00002_1920x1080/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_401_4/predict/YUN00002_1920x1080_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_401_4/predict/YUN00002_1920x1080_0_1799_1000_1000_1000_1000_stitched patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=1

python3 ~/PTF/imgSeqToVideo.py src_path=log/50_10000_10000_1000_random_401_4/predict/YUN00002_1920x1080_0_1799_1000_1000_1000_1000_stitched save_path=log/50_10000_10000_1000_random_401_4/predict/YUN00002_1920x1080_0_1799_1000_1000_1000_1000_stitched.mkv width=1920 height=1080 show_img=0


<a id="yun00002_200001799_1000_1000_1000_1000__videorandom_5010k10k400loss_41000"></a>
##### YUN00002_2000_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/400/loss_4/1000

CUDA_VISIBLE_DEVICES=0 python3 densenet_predict.py --images_path=/data/617/images/YUN00002_2000_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_401_4/weights/ --save_path=log/50_10000_10000_1000_random_401_4/predict/YUN00002_2000_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/YUN00002_2000/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_401_4/predict/YUN00002_2000_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_401_4/predict/YUN00002_2000_0_1799_1000_1000_1000_1000_stitched patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=0.5

python3 ~/PTF/imgSeqToVideo.py src_path=log/50_10000_10000_1000_random_401_4/predict/YUN00002_2000_0_1799_1000_1000_1000_1000_stitched save_path=log/50_10000_10000_1000_random_401_4/predict/YUN00002_2000_0_1799_1000_1000_1000_1000_stitched.mkv width=1920 height=1080 show_img=0


<a id="20160122_yun00002_70001799_1000_1000_1000_1000__videorandom_5010k10k400loss_41000"></a>
##### 20160122_YUN00002_700_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/400/loss_4/1000

CUDA_VISIBLE_DEVICES=0 python3 densenet_predict.py --images_path=/data/617/images/20160122_YUN00002_700_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_401_4/weights/ --save_path=log/50_10000_10000_1000_random_401_4/predict/20160122_YUN00002_700_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00002_700/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_401_4/predict/20160122_YUN00002_700_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_401_4/predict/20160122_YUN00002_700_0_1799_1000_1000_1000_1000_stitched patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=0.5

<a id="20161201_yun0000201799_1000_1000_1000_1000__videorandom_5010k10k400loss_41000"></a>
##### 20161201_YUN00002_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/400/loss_4/1000

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/20161201_YUN00002_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_401_4/weights/ --save_path=log/50_10000_10000_1000_random_401_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20161201_YUN00002/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_401_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_401_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000_stitched patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=0.5

<a id="20170114_yun0000501799_1000_1000_1000_1000__videorandom_5010k10k400loss_41000"></a>
##### 20170114_YUN00005_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/400/loss_4/1000

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/20170114_YUN00005_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_401_4/weights/ --save_path=log/50_10000_10000_1000_random_401_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20170114_YUN00005/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_401_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_401_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000_stitched patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=0.5


<a id="10k10k800loss_4__1000"></a>
## 10k/10k/800/loss_4       @ 1000

<a id="random_50__10k10k800loss_41000"></a>
### random_50       @ 10k/10k/800/loss_4/1000

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_49_1000_1000_100_800_rot_15_345_4_flip/images --height=1000 --width=1000 --index_percent=50 --n_classes=3 --start_id=-1 --end_id=800 --n_epochs=100000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_images=/data/617/images/training_32_49_1000_1000_100_400_rot_15_345_4_flip/images --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=10

<a id="predict__random_5010k10k800loss_41000"></a>
#### predict       @ random_50/10k/10k/800/loss_4/1000

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_1000_1000_100_400_rot_15_345_4_flip/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_801_4/weights/ --save_path=log/50_10000_10000_1000_random_801_4/predict/training_32_49_1000_1000_100_400_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

<a id="vis__random_5010k10k800loss_41000"></a>
#### vis       @ random_50/10k/10k/800/loss_4/1000

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_1000_1000_100_400_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_1000_1000_100_400_rot_15_345_4_flip/labels --seg_path=log/50_10000_10000_1000_random_801_4/predict/training_32_49_1000_1000_100_400_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_1000_1000_100_400_rot_15_345_4_flip/vis_50_10000_10000_1000_random_801_4 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation__random_5010k10k800loss_41000"></a>
#### validation       @ random_50/10k/10k/800/loss_4/1000


CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/validation_0_20_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_801_4/weights/ --save_path=log/50_10000_10000_1000_random_801_4/predict/validation_0_20_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4

zr 50_10000_10000_1000_random_801_4_validation_0_20_1000_1000_1000_1000 50_10000_10000_1000_random_801_4/predict/validation_0_20_1000_1000_1000_1000

<a id="video__random_5010k10k800loss_41000"></a>
#### video       @ random_50/10k/10k/800/loss_4/1000

<a id="yun00001_1920x108001799_1000_1000_1000_1000__videorandom_5010k10k800loss_41000"></a>
##### YUN00001_1920x1080_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/800/loss_4/1000

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/YUN00001_1920x1080_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_801_4/weights/ --save_path=log/50_10000_10000_1000_random_801_4/predict/YUN00001_1920x1080_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_1920x1080/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_801_4/predict/YUN00001_1920x1080_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_801_4/predict/YUN00001_1920x1080_0_1799_1000_1000_1000_1000_stitched patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=mkv resize_factor=1 patch_ext=jpg del_patch_seq=1

python3 ~/PTF/imgSeqToVideo.py src_path=log/50_10000_10000_1000_random_801_4/predict/YUN00001_1920x1080_0_1799_1000_1000_1000_1000_stitched save_path=log/50_10000_10000_1000_random_801_4/predict/YUN00001_1920x1080_0_1799_1000_1000_1000_1000_stitched.mkv width=1920 height=1080 show_img=0

<a id="yun00002_1920x108001799_1000_1000_1000_1000__videorandom_5010k10k800loss_41000"></a>
##### YUN00002_1920x1080_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/800/loss_4/1000

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/YUN00002_1920x1080_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_801_4/weights/ --save_path=log/50_10000_10000_1000_random_801_4/predict/YUN00002_1920x1080_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/YUN00002_1920x1080/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_801_4/predict/YUN00002_1920x1080_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_801_4/predict/YUN00002_1920x1080_0_1799_1000_1000_1000_1000_stitched patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=1 patch_ext=jpg del_patch_seq=1

python3 ~/PTF/imgSeqToVideo.py src_path=log/50_10000_10000_1000_random_801_4/predict/YUN00002_1920x1080_0_1799_1000_1000_1000_1000_stitched save_path=log/50_10000_10000_1000_random_801_4/predict/YUN00002_1920x1080_0_1799_1000_1000_1000_1000_stitched.mkv width=1920 height=1080 show_img=0


<a id="yun0000101799_1000_1000_1000_1000__videorandom_5010k10k800loss_41000"></a>
##### YUN00001_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/800/loss_4/1000

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/YUN00001_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_801_4/weights/ --save_path=log/50_10000_10000_1000_random_801_4/predict/YUN00001_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/YUN00001/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_801_4/predict/YUN00001_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_801_4/predict/YUN00001_0_1799_1000_1000_1000_1000_stitched patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=mkv resize_factor=0.5 patch_ext=jpg del_patch_seq=1

<a id="yun00002_200001799_1000_1000_1000_1000__videorandom_5010k10k800loss_41000"></a>
##### YUN00002_2000_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/800/loss_4/1000

CUDA_VISIBLE_DEVICES=0 python3 densenet_predict.py --images_path=/data/617/images/YUN00002_2000_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_801_4/weights/ --save_path=log/50_10000_10000_1000_random_801_4/predict/YUN00002_2000_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/YUN00002_2000/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_801_4/predict/YUN00002_2000_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_801_4/predict/YUN00002_2000_0_1799_1000_1000_1000_1000_stitched patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=0.5 patch_ext=jpg del_patch_seq=1

python3 ~/PTF/imgSeqToVideo.py src_path=log/50_10000_10000_1000_random_801_4/predict/YUN00002_2000_0_1799_1000_1000_1000_1000_stitched save_path=log/50_10000_10000_1000_random_801_4/predict/YUN00002_2000_0_1799_1000_1000_1000_1000_stitched.mkv width=1920 height=1080 show_img=0


<a id="20160122_yun00002_70001799_1000_1000_1000_1000__videorandom_5010k10k800loss_41000"></a>
##### 20160122_YUN00002_700_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/800/loss_4/1000

CUDA_VISIBLE_DEVICES=0 python3 densenet_predict.py --images_path=/data/617/images/20160122_YUN00002_700_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_801_4/weights/ --save_path=log/50_10000_10000_1000_random_801_4/predict/20160122_YUN00002_700_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00002_700/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_801_4/predict/20160122_YUN00002_700_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_801_4/predict/20160122_YUN00002_700_0_1799_1000_1000_1000_1000_stitched patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=0.5 patch_ext=jpg del_patch_seq=1

<a id="20161201_yun0000201799_1000_1000_1000_1000__videorandom_5010k10k800loss_41000"></a>
##### 20161201_YUN00002_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/800/loss_4/1000

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/20161201_YUN00002_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_801_4/weights/ --save_path=log/50_10000_10000_1000_random_801_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20161201_YUN00002/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_801_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_801_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000_stitched patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=0.5 patch_ext=jpg del_patch_seq=1

<a id="20170114_yun0000501799_1000_1000_1000_1000__videorandom_5010k10k800loss_41000"></a>
##### 20170114_YUN00005_0_1799_1000_1000_1000_1000       @ video/random_50/10k/10k/800/loss_4/1000

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/20170114_YUN00005_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_801_4/weights/ --save_path=log/50_10000_10000_1000_random_801_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20170114_YUN00005/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_801_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_801_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000_stitched patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=0.5 patch_ext=jpg del_patch_seq=1


<a id="10k10kallloss_4__1000"></a>
## 10k/10k/all/loss_4       @ 1000

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_49_1000_1000_100_400_rot_15_345_4_flip/images --height=1000 --width=1000 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=100000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_images=/data/617/images/training_32_49_1000_1000_100_400_rot_15_345_4_flip/images --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=10


<a id="horse"></a>
# horse

python3 densenet_train.py --train_images=/data/617/images/il_horse/images/ --train_labels=/data/617/images/il_horse/labels --height=675 --width=900 --index_percent=100 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=1000 --log_dir=log/il_horse --gpu_id=2 --max_indices=1000000 --min_indices=10000 --eval_every=100




<a id="800"></a>
# 800

<a id="10k10k1__800"></a>
## 10k/10k/1       @ 800

<a id="loss_3__10k10k1800"></a>
### loss_3       @ 10k/10k/1/800

python3 densenet_train.py --train_images=/data/617/images/training_0_49_800_800_100_200/images --train_labels=/data/617/images/training_0_49_800_800_100_200/labels --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=1 --end_id=1 --n_epochs=100000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=1 --test_end_id=1 --save_stitched=1 --loss_type=3 --eval_every=100

python3 densenet_predict.py --images_path=/data/617/images/training_0_49_800_800_100_200/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=50 --weights_path=log/ratio_50_10k_10k_800_0_49/weights/ --save_path=log/ratio_50_10k_10k_800_0_49/test --save_stitched=1 --gpu_id=2 

<a id="loss_4__10k10k1800"></a>
### loss_4       @ 10k/10k/1/800

python3 densenet_train.py --train_images=/data/617/images/training_0_49_800_800_100_200/images --train_labels=/data/617/images/training_0_49_800_800_100_200/labels --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=1 --end_id=1 --n_epochs=100000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=1 --test_end_id=1 --save_stitched=1 --loss_type=4 --eval_every=10


<a id="10k10k50__800"></a>
## 10k/10k/50       @ 800

python3 densenet_train.py --train_images=/data/617/images/training_0_49_800_800_100_200/images --train_labels=/data/617/images/training_0_49_800_800_100_200/labels --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=50 --n_epochs=100000 --log_dir=log/ratio_50_10k_10k_800_0_49 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=0 --save_stitched=1

python3 densenet_predict.py --images_path=/data/617/images/training_0_49_800_800_100_200/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=50 --weights_path=log/ratio_50_10k_10k_800_0_49/weights/ --save_path=log/ratio_50_10k_10k_800_0_49/test --save_stitched=1 --gpu_id=2 

<a id="10k10k100__800"></a>
## 10k/10k/100       @ 800

<a id="loss_0__10k10k100800"></a>
### loss_0       @ 10k/10k/100/800

python3 densenet_train.py --train_images=/data/617/images/training_0_49_800_800_100_200/images --train_labels=/data/617/images/training_0_49_800_800_100_200/labels --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=99 --n_epochs=100000 --log_dir=log/ratio_50_10k_10k_800_0_99 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=0 --save_stitched=1

python3 densenet_predict.py --images_path=/data/617/images/training_0_49_800_800_100_200/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=50 --weights_path=log/ratio_50_10k_10k_800_0_49/weights/ --save_path=log/ratio_50_10k_10k_800_0_49/test --save_stitched=1 --gpu_id=2 

<a id="loss_2__10k10k100800"></a>
### loss_2       @ 10k/10k/100/800

python3 densenet_train.py --train_images=/data/617/images/training_0_49_800_800_100_200/images --train_labels=/data/617/images/training_0_49_800_800_100_200/labels --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=99 --n_epochs=100000 --log_dir=log/ratio_50_10k_10k_800_0_99 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=0 --save_stitched=1 --loss_type=2

<a id="loss_3__10k10k100800"></a>
### loss_3       @ 10k/10k/100/800

python3 densenet_train.py --train_images=/data/617/images/training_0_49_800_800_100_200/images --train_labels=/data/617/images/training_0_49_800_800_100_200/labels --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=99 --n_epochs=100000 --log_dir=log/ratio_50_10k_10k_800_0_99_loss_3 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=0 --save_stitched=1 --loss_type=3

<a id="loss_4__10k10k100800"></a>
### loss_4       @ 10k/10k/100/800

python3 densenet_train.py --train_images=/data/617/images/training_0_49_800_800_100_200/images --train_labels=/data/617/images/training_0_49_800_800_100_200/labels --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=1 --end_id=99 --n_epochs=100000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=1 --test_end_id=99 --save_stitched=1 --loss_type=4 --eval_every=10

python3 densenet_predict.py --images_path=/data/617/images/training_0_49_800_800_100_200/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_800_1_99_4/weights/ --save_path=log/50_10000_10000_800_1_99_4/predict/training_0_49_800_800_100_200 --save_stitched=1 --gpu_id=2 --loss_type=4

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/validation_0_20_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_800_1_99_4/weights/ --save_path=log/50_10000_10000_800_1_99_4/predict/validation_0_20_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

<a id="10k10k200loss_4__800"></a>
## 10k/10k/200/loss_4       @ 800

<a id="training__10k10k200loss_4800"></a>
### training       @ 10k/10k/200/loss_4/800

python3 densenet_train.py --train_images=/data/617/images/training_0_49_800_800_100_200/images --train_labels=/data/617/images/training_0_49_800_800_100_200/labels --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=199 --n_epochs=100000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=199 --save_stitched=1 --loss_type=4 --eval_every=10

<a id="prediction__training10k10k200loss_4800"></a>
#### prediction       @ training/10k/10k/200/loss_4/800

CUDA_VISIBLE_DEVICES=0 python3 densenet_predict.py --images_path=/data/617/images/training_0_49_800_800_100_200/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_800_0_199_4/weights/model.ckpt-0 --save_path=log/50_10000_10000_800_0_199_4/test --save_stitched=1 --gpu_id=2 --loss_type=4

CUDA_VISIBLE_DEVICES=0 python3 densenet_predict.py --images_path=/data/617/images/validation_0_20_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_800_0_199_4/weights/ --save_path=log/50_10000_10000_800_0_199_4/test --save_stitched=1 --gpu_id=2 

<a id="random__10k10k200loss_4800"></a>
### random       @ 10k/10k/200/loss_4/800

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_49_800_800_100_200/images --train_labels=/data/617/images/training_0_49_800_800_100_200/labels --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=-1 --end_id=199 --n_epochs=100000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=-1 --test_end_id=9 --save_stitched=1 --loss_type=4 --eval_every=10 --load_weights=1

<a id="prediction__random10k10k200loss_4800"></a>
#### prediction       @ random/10k/10k/200/loss_4/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_0_49_800_800_100_200/images --labels_path=/data/617/images/training_0_49_800_800_100_200/labels --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_800_random_200_4/weights/ --save_path=log/50_10000_10000_800_random_200_4/predict/training_0_49_800_800_100_200 --save_stitched=1 --gpu_id=2 --loss_type=4

CUDA_VISIBLE_DEVICES=0 python3 densenet_predict.py --images_path=/data/617/images/validation_0_20_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_800_random_200_4/weights/ --save_path=log/50_10000_10000_800_random_200_4/predict/validation_0_20_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

<a id="vis__800"></a>
## vis       @ 800

python3 visDataset.py --images_path=/data/617/images/training_0_49_800_800_100_200/images --labels_path=/data/617/images/training_0_49_800_800_100_200/labels --seg_path=indicator_learning/log/50_10000_10000_800_random_200_4/predict/training_0_49_800_800_100_200/raw --save_path=/data/617/images/training_0_49_800_800_100_200/vis_50_10000_10000_800_random_200_4 --n_classes=3 --start_id=0 --end_id=-1

<a id="video__vis800"></a>
##### video       @ vis/800

<a id="yun000010239_800_800_800_800__vis800"></a>
##### YUN00001_0_239_800_800_800_800       @ vis/800

CUDA_VISIBLE_DEVICES=0 python3 densenet_predict.py --images_path=/data/617/images/YUN00001_0_239_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_800_random_200_4/weights/ --save_path=log/50_10000_10000_800_random_200_4/predict/YUN00001_0_239_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001/images img_ext=jpg  patch_seq_path=log/50_10000_10000_800_random_200_4/predict/YUN00001_0_239_800_800_800_800 stitched_seq_path=log/50_10000_10000_800_random_200_4/predict/YUN00001_0_239_800_800_800_800_stitched patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0

50_10000_10000_800_random_200_4_predict_YUN00001_0_239_800_800_800_800_stitched
50_10000_10000_800_random_200_4_predict_YUN00001_0_239_800_800_800_800_stitched_grs_201804241144.zip
50_10000_10000_800_random_200_4_predict_YUN00001_0_239_800_800_800_800_stitched_grs_201804241159.zip

<a id="yun00002_200001799_800_800_800_800__vis800"></a>
##### YUN00002_2000_0_1799_800_800_800_800       @ vis/800

CUDA_VISIBLE_DEVICES=0 python3 densenet_predict.py --images_path=/data/617/images/YUN00002_2000_0_1799_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_800_random_200_4/weights/ --save_path=log/50_10000_10000_800_random_200_4/predict/YUN00002_2000_0_1799_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/YUN00002_2000/images img_ext=jpg  patch_seq_path=log/50_10000_10000_800_random_200_4/predict/YUN00002_2000_0_1799_800_800_800_800 stitched_seq_path=log/50_10000_10000_800_random_200_4/predict/YUN00002_2000_0_1799_800_800_800_8000_stitched patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=0.5 patch_ext=jpg

<a id="20160122_yun00002_70001799_800_800_800_800__vis800"></a>
##### 20160122_YUN00002_700_0_1799_800_800_800_800       @ vis/800

CUDA_VISIBLE_DEVICES=0 python3 densenet_predict.py --images_path=/data/617/images/20160122_YUN00002_700_0_1799_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_800_random_200_4/weights/ --save_path=log/50_10000_10000_800_random_200_4/predict/20160122_YUN00002_700_0_1799_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00002_700/images img_ext=jpg  patch_seq_path=log/50_10000_10000_800_random_200_4/predict/20160122_YUN00002_700_0_1799_800_800_800_800 stitched_seq_path=log/50_10000_10000_800_random_200_4/predict/20160122_YUN00002_700_0_1799_800_800_800_8000_stitched patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=0.5 patch_ext=jpg

python3 ~/PTF/imgSeqToVideo.py src_path=log/50_10000_10000_800_random_200_4/predict/20160122_YUN00002_700_0_1799_800_800_800_8000_stitched save_path=log/50_10000_10000_800_random_200_4/predict/20160122_YUN00002_700_0_1799_800_800_800_8000_stitched.mkv width=1920 height=1080 show_img=0 del_src=1


<a id="20161201_yun0000201799_800_800_800_800__vis800"></a>
##### 20161201_YUN00002_0_1799_800_800_800_800       @ vis/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/20161201_YUN00002_0_1799_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_800_random_200_4/weights/ --save_path=log/50_10000_10000_800_random_200_4/predict/20161201_YUN00002_0_1799_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20161201_YUN00002/images img_ext=jpg  patch_seq_path=log/50_10000_10000_800_random_200_4/predict/20161201_YUN00002_0_1799_800_800_800_800 stitched_seq_path=log/50_10000_10000_800_random_200_4/predict/20161201_YUN00002_0_1799_800_800_800_8000_stitched patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=0.5

<a id="20170114_yun0000501799_800_800_800_800__vis800"></a>
##### 20170114_YUN00005_0_1799_800_800_800_800       @ vis/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/20170114_YUN00005_0_1799_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_800_random_200_4/weights/ --save_path=log/50_10000_10000_800_random_200_4/predict/20170114_YUN00005_0_1799_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20170114_YUN00005/images img_ext=jpg  patch_seq_path=log/50_10000_10000_800_random_200_4/predict/20170114_YUN00005_0_1799_800_800_800_800 stitched_seq_path=log/50_10000_10000_800_random_200_4/predict/20170114_YUN00005_0_1799_800_800_800_8000_stitched patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=0.5


<a id="zip__vis800"></a>
##### zip       @ vis/800

50_10000_10000_800_random_200_4_predict_training_0_49_800_800_100_200_1_50
50_10000_10000_800_random_200_4_predict_validation_0_20_800_800_800_800

zrbj 50_10000_10000_800_random_200_4_predict_validation_0_20_800_800_800_800 log/50_10000_10000_800_random_200_4/predict/validation_0_20_800_800_800_800/img_XXX_* 1 10

zrbj 50_10000_10000_800_random_200_4_predict_training_0_49_800_800_100_200 log/50_10000_10000_800_random_200_4/predict/training_0_49_800_800_100_200/img_XXX_* 1 10

<a id="10k10kall__800"></a>
## 10k/10k/all       @ 800

<a id="loss_4__10k10kall800"></a>
### loss_4       @ 10k/10k/all/800

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_49_800_800_100_200/images --train_labels=/data/617/images/training_0_49_800_800_100_200/labels --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=100000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=99 --save_stitched=1 --loss_type=4 --eval_every=1

<a id="15k10k__800"></a>
## 15k/10k       @ 800

python3 densenet_train.py --test_images=/data/617/images/training_0_49_800_800_100_200/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/ratio_50_1k_10k_800_0_0 --gpu_id=2 --eval_every=10


<a id="8k8kallloss_4augmented__800"></a>
## 8k/8k/all/loss_4/augmented       @ 800

<a id="50__8k8kallloss_4augmented800"></a>
### 50       @ 8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_49_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_49_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=8000 --min_indices=8000 --test_start_id=0 --test_end_id=99 --save_stitched=1 --loss_type=4 --eval_every=10

<a id="vis__508k8kallloss_4augmented800"></a>
#### vis       @ 50/8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_random_401_4/weights/ --save_path=log/50_8000_8000_800_random_401_4/predict/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --seg_path=log/50_8000_8000_800_random_401_4/predict/training_32_49_800_800_80_320_rot_15_345_4_flip/raw --n_classes=3 --start_id=0 --end_id=-1

<a id="validation__508k8kallloss_4augmented800"></a>
#### validation       @ 50/8k/8k/all/loss_4/augmented/800


CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/validation_0_20_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_random_401_4/weights/ --save_path=log/50_8000_8000_800_random_401_4/predict/validation_0_20_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

<a id="4__8k8kallloss_4augmented800"></a>
### 4       @ 8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=8000 --min_indices=8000 --test_start_id=0 --test_end_id=99 --save_stitched=1 --loss_type=4 --eval_every=10

<a id="vis__48k8kallloss_4augmented800"></a>
#### vis       @ 4/8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_0_320_4/weights/ --save_path=log/50_8000_8000_800_0_320_4/predict/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --seg_path=log/50_8000_8000_800_0_320_4/predict/training_32_49_800_800_80_320_rot_15_345_4_flip/raw --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug__48k8kallloss_4augmented800"></a>
#### no_aug       @ 4/8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_0_320_4/weights/ --save_path=log/50_8000_8000_800_0_320_4/predict/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/50_8000_8000_800_0_320_4/predict/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__48k8kallloss_4augmented800"></a>
#### stitched       @ 4/8k/8k/all/loss_4/augmented/800

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/50_8000_8000_800_0_320_4/predict/training_32_49_800_800_800_800 stitched_seq_path=log/50_8000_8000_800_0_320_4/predict/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/50_8000_8000_800_0_320_4/predict/training_32_49/raw --n_classes=3 --start_id=0 --end_id=-1

<a id="8__8k8kallloss_4augmented800"></a>
### 8       @ 8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_7_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_7_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=8000 --min_indices=8000 --test_start_id=0 --test_end_id=99 --save_stitched=1 --loss_type=4 --eval_every=10

<a id="vis__8k8kallloss_4augmented800"></a>
### vis       @ 8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_0_431_4/weights/ --save_path=log/50_8000_8000_800_0_431_4/predict/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --seg_path=log/50_8000_8000_800_0_431_4/predict/training_32_49_800_800_80_320_rot_15_345_4_flip/raw--n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug__vis8k8kallloss_4augmented800"></a>
#### no_aug       @ vis/8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_0_431_4/weights/ --save_path=log/50_8000_8000_800_0_431_4/predict/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/50_8000_8000_800_0_431_4/predict/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__vis8k8kallloss_4augmented800"></a>
#### stitched       @ vis/8k/8k/all/loss_4/augmented/800

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/50_8000_8000_800_0_431_4/predict/training_32_49_800_800_800_800 stitched_seq_path=log/50_8000_8000_800_0_431_4/predict/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/50_8000_8000_800_0_431_4/predict/training_32_49/raw --n_classes=3 --start_id=0 --end_id=-1

<a id="16__8k8kallloss_4augmented800"></a>
### 16       @ 8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_15_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_15_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=8000 --min_indices=8000 --test_start_id=0 --test_end_id=99 --save_stitched=1 --loss_type=4 --eval_every=10 --load_weights=0

<a id="vis__8k8kallloss_4augmented800-1"></a>
### vis       @ 8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_0_920_4/weights/ --save_path=log/50_8000_8000_800_0_920_4/predict/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --seg_path=log/50_8000_8000_800_0_920_4/predict/training_32_49_800_800_80_320_rot_15_345_4_flip/raw --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug__vis8k8kallloss_4augmented800-1"></a>
#### no_aug       @ vis/8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_0_920_4/weights/ --save_path=log/50_8000_8000_800_0_920_4/predict/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/50_8000_8000_800_0_920_4/predict/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__vis8k8kallloss_4augmented800-1"></a>
#### stitched       @ vis/8k/8k/all/loss_4/augmented/800

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/50_8000_8000_800_0_920_4/predict/training_32_49_800_800_800_800 stitched_seq_path=log/50_8000_8000_800_0_920_4/predict/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/50_8000_8000_800_0_920_4/predict/training_32_49/raw --n_classes=3 --start_id=0 --end_id=-1

<a id="24__8k8kallloss_4augmented800"></a>
### 24       @ 8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_23_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_23_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=8000 --min_indices=8000 --test_start_id=0 --test_end_id=99 --save_stitched=1 --loss_type=4 --eval_every=10 --load_weights=1

<a id="vis__8k8kallloss_4augmented800-2"></a>
### vis       @ 8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_0_1505_4/weights/ --save_path=log/50_8000_8000_800_0_1505_4/predict/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --seg_path=log/50_8000_8000_800_0_1505_4/predict/training_32_49_800_800_80_320_rot_15_345_4_flip/raw --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug__vis8k8kallloss_4augmented800-2"></a>
#### no_aug       @ vis/8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_0_1505_4/weights/ --save_path=log/50_8000_8000_800_0_1505_4/predict/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/50_8000_8000_800_0_1505_4/predict/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__vis8k8kallloss_4augmented800-2"></a>
#### stitched       @ vis/8k/8k/all/loss_4/augmented/800

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/50_8000_8000_800_0_1505_4/predict/training_32_49_800_800_800_800 stitched_seq_path=log/50_8000_8000_800_0_1505_4/predict/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/50_8000_8000_800_0_1505_4/predict/training_32_49/raw --n_classes=3 --start_id=0 --end_id=-1

<a id="32__8k8kallloss_4augmented800"></a>
### 32       @ 8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_31_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_31_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=8000 --min_indices=8000 --test_start_id=0 --test_end_id=99 --save_stitched=1 --loss_type=4 --eval_every=10 --preload_images=0  --load_weights=1

<a id="vis__8k8kallloss_4augmented800-3"></a>
### vis       @ 8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_0_1586_4/weights/ --save_path=log/50_8000_8000_800_0_1586_4/predict/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --seg_path=log/50_8000_8000_800_0_1586_4/predict/training_32_49_800_800_80_320_rot_15_345_4_flip/raw --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug__vis8k8kallloss_4augmented800-3"></a>
#### no_aug       @ vis/8k/8k/all/loss_4/augmented/800

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_8000_8000_800_0_1586_4/weights/ --save_path=log/50_8000_8000_800_0_1586_4/predict/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --seg_path=log/50_8000_8000_800_0_1586_4/predict/training_32_49_800_800_800_800/raw --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__vis8k8kallloss_4augmented800-3"></a>
#### stitched       @ vis/8k/8k/all/loss_4/augmented/800

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/50_8000_8000_800_0_1586_4/predict/training_32_49_800_800_800_800 stitched_seq_path=log/50_8000_8000_800_0_1586_4/predict/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/50_8000_8000_800_0_1586_4/predict/training_32_49/raw --n_classes=3 --start_id=0 --end_id=-1


<a id="800_retraining"></a>
# 800_retraining

<a id="4__non_aug__800_retraining"></a>
## 4__non_aug       @ 800_retraining

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_3_800_800_800_800/images --train_labels=/data/617/images/training_0_3_800_800_800_800/labels --test_images=/data/617/images/training_32_49_800_800_800_800/images --test_labels=/data/617/images/training_32_49_800_800_800_800/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --load_weights=0 --log_dir=log/rt2_training_0_3_800_800_800_800 --lr_dec_rate=0.90 --lr_dec_epochs=10 --psi_act_type=1

<a id="4__800_retraining"></a>
## 4       @ 800_retraining

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --load_weights=0 --log_dir=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.90 --lr_dec_epochs=10 --psi_act_type=1

<a id="loss_1__4800_retraining"></a>
### loss_1       @ 4/800_retraining

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=1 --eval_every=1 --load_weights=0 --log_dir=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1

<a id="12_layers__4800_retraining"></a>
### 12_layers       @ 4/800_retraining

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --load_weights=0 --log_dir=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --n_layers=12


<a id="vis__4800_retraining"></a>
### vis       @ 4/800_retraining

CUDA_VISIBLE_DEVICES=2 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu/weights_acc/ --save_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu/predict_acc/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --save_seg=0 --psi_act_type=1

<a id="loss_1__vis4800_retraining"></a>
#### loss_1       @ vis/4/800_retraining

CUDA_VISIBLE_DEVICES=2 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_1_elu/weights_acc/ --save_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_1_elu/predict_acc/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --save_seg=0 --loss_type=1 --psi_act_type=1

<a id="vis_no_aug__4800_retraining"></a>
### vis_no_aug       @ 4/800_retraining

CUDA_VISIBLE_DEVICES=2 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu/weights_acc/ --save_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu/predict_acc/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --psi_act_type=1

<a id="loss_1__vis_no_aug4800_retraining"></a>
#### loss_1       @ vis_no_aug/4/800_retraining

CUDA_VISIBLE_DEVICES=2 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_1_elu/weights/ --save_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_1_elu/predict/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=1 --psi_act_type=1

<a id="vis_stitched__4800_retraining"></a>
### vis_stitched       @ 4/800_retraining

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels patch_seq_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu/predict_acc/training_32_49_800_800_800_800 stitched_seq_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu/predict_acc/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels patch_seq_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_1_elu/predict/training_32_49_800_800_800_800 stitched_seq_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_1_elu/predict/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4/predict/training_32_49/raw  --n_classes=3 --start_id=0 --end_id=-1

<a id="vis_no_aug_449__4800_retraining"></a>
### vis_no_aug__4_49       @ 4/800_retraining

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_4_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu/weights_acc/ --save_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu/predict_acc/training_4_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_4_49_800_800_800_800/labels --psi_act_type=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels patch_seq_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu/predict_acc/training_4_49_800_800_800_800 stitched_seq_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_320_4_elu/predict_acc/training_4_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png del_patch_seq=0


<a id="8__800_retraining"></a>
## 8       @ 800_retraining

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_7_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_7_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --load_weights=1 --log_dir=log/rt2_training_0_7_800_800_80_320_rot_15_345_4_flip_95_10 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1

<a id="loss_1__8800_retraining"></a>
### loss_1       @ 8/800_retraining

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_7_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_7_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=1 --eval_every=1 --load_weights=1 --log_dir=log/training_0_7_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1

<a id="vis__loss_18800_retraining"></a>
#### vis       @ loss_1/8/800_retraining

CUDA_VISIBLE_DEVICES=2 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_7_800_800_80_320_rot_15_345_4_flip_95_10_50_10000_10000_800_0_581_4/weights_acc/ --save_path=log/rt2_training_0_7_800_800_80_320_rot_15_345_4_flip_95_10_50_10000_10000_800_0_581_4/predict_acc/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --save_seg=0 --psi_act_type=1

<a id="vis_no_aug__loss_18800_retraining"></a>
#### vis_no_aug       @ loss_1/8/800_retraining

CUDA_VISIBLE_DEVICES=2 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_7_800_800_80_320_rot_15_345_4_flip_95_10_50_10000_10000_800_0_581_4/weights_acc/ --save_path=log/rt2_training_0_7_800_800_80_320_rot_15_345_4_flip_95_10_50_10000_10000_800_0_581_4/predict_acc/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --psi_act_type=1

<a id="vis_stitched__loss_18800_retraining"></a>
#### vis_stitched       @ loss_1/8/800_retraining

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels patch_seq_path=log/rt2_training_0_7_800_800_80_320_rot_15_345_4_flip_95_10_50_10000_10000_800_0_581_4/predict_acc/training_32_49_800_800_800_800 stitched_seq_path=log/rt2_training_0_7_800_800_80_320_rot_15_345_4_flip_95_10_50_10000_10000_800_0_581_4/predict_acc/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0


<a id="16__800_retraining"></a>
## 16       @ 800_retraining

CUDA_VISIBLE_DEVICES=2 python3 densenet_train.py --train_images=/data/617/images/training_0_15_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_15_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --load_weights=1 --log_dir=log/rt2_training_0_15_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1

<a id="vis__16800_retraining"></a>
### vis       @ 16/800_retraining

CUDA_VISIBLE_DEVICES=2 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_10_4/weights_acc/ --save_path=log/rt2_training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_10_4/predict_acc/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --save_seg=0 --psi_act_type=1

<a id="vis_no_aug__16800_retraining"></a>
### vis_no_aug       @ 16/800_retraining

CUDA_VISIBLE_DEVICES=2 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_10_4/weights_acc/ --save_path=log/rt2_training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_10_4/predict_acc/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --psi_act_type=1

<a id="vis_stitched__16800_retraining"></a>
### vis_stitched       @ 16/800_retraining

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels patch_seq_path=log/rt2_training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_10_4/predict_acc/training_32_49_800_800_800_800 stitched_seq_path=log/rt2_training_0_15_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_10_4/predict_acc/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0


<a id="24__800_retraining"></a>
## 24       @ 800_retraining

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_23_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_23_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --load_weights=1 --log_dir=log/rt2_training_0_23_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1


<a id="vis__24800_retraining"></a>
### vis       @ 24/800_retraining

CUDA_VISIBLE_DEVICES=2 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4_elu/weights_acc/ --save_path=log/rt2_training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4_elu/predict_acc/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --save_seg=0 --psi_act_type=1

<a id="vis_no_aug__24800_retraining"></a>
### vis_no_aug       @ 24/800_retraining

CUDA_VISIBLE_DEVICES=2 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4_elu/weights_acc/ --save_path=log/rt2_training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4_elu/predict_acc/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --psi_act_type=1

<a id="vis_stitched__24800_retraining"></a>
### vis_stitched       @ 24/800_retraining

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels patch_seq_path=log/rt2_training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4_elu/predict_acc/training_32_49_800_800_800_800 stitched_seq_path=log/rt2_training_0_23_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1274_4_elu/predict_acc/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

<a id="32__800_retraining"></a>
## 32       @ 800_retraining

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_31_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_31_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=1 --n_epochs=80000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=1 --save_stitched=1 --loss_type=4 --eval_every=1 --load_weights=1 --log_dir=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1

<a id="vis__32800_retraining"></a>
### vis       @ 32/800_retraining

CUDA_VISIBLE_DEVICES=2 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/weights_acc/ --save_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --save_seg=0 --psi_act_type=1

<a id="vis_no_aug__32800_retraining"></a>
### vis_no_aug       @ 32/800_retraining

CUDA_VISIBLE_DEVICES=2 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/weights_acc/ --save_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/training_32_49_800_800_800_800 --save_stitched=0 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --psi_act_type=1

<a id="vis_stitched__32800_retraining"></a>
### vis_stitched       @ 32/800_retraining

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels patch_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/training_32_49_800_800_800_800 stitched_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=1 img_ext=png del_patch_seq=0


<a id="yun00001__32800_retraining"></a>
### YUN00001       @ 32/800_retraining

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/YUN00001_0_8999_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/weights_acc/ --save_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/YUN00001_0_8999_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --psi_act_type=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001/images patch_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/YUN00001_0_8999_800_800_800_800 stitched_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/YUN00001 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=1 out_ext=mkv width=1920 height=1080

<a id="yun00001_3600__32800_retraining"></a>
### YUN00001_3600       @ 32/800_retraining

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/YUN00001_3600_0_3599_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/weights_acc/ --save_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/YUN00001_3600_0_3599_800_800_800_800 --save_stitched=0 --save_seg=1 --gpu_id=2 --loss_type=4 --psi_act_type=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images patch_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/YUN00001_3600_0_3599_800_800_800_800 stitched_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/YUN00001_3600 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg del_patch_seq=0 out_ext=mkv width=1920 height=1080

<a id="vis_png__32800_retraining"></a>
### vis_png       @ 32/800_retraining

<a id="20160122_yun00002_700_2500__vis_png32800_retraining"></a>
#### 20160122_YUN00002_700_2500       @ vis_png/32/800_retraining

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00002_700_2500/images patch_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/20160122_YUN00002_700_2500_0_1799_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/20160122_YUN00002_700_2500 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg del_patch_seq=0 out_ext=png

<a id="20160122_yun00020_2000_3800__vis_png32800_retraining"></a>
#### 20160122_YUN00020_2000_3800       @ vis_png/32/800_retraining

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00020_2000_3800/images patch_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/20160122_YUN00020_2000_3800_0_1799_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/20160122_YUN00020_2000_3800 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg del_patch_seq=0 out_ext=png


<a id="20161203_deployment1yun00001_900_2700__vis_png32800_retraining"></a>
#### 20161203_Deployment_1_YUN00001_900_2700       @ vis_png/32/800_retraining

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images patch_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/20161203_Deployment_1_YUN00001_900_2700_0_1799_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/20161203_Deployment_1_YUN00001_900_2700 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg del_patch_seq=0 out_ext=png


<a id="20161203_deployment1yun00002_1800__vis_png32800_retraining"></a>
#### 20161203_Deployment_1_YUN00002_1800       @ vis_png/32/800_retraining

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20161203_Deployment_1_YUN00002_1800/images patch_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/20161203_Deployment_1_YUN00002_1800_0_1799_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/20161203_Deployment_1_YUN00002_1800 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg del_patch_seq=0 out_ext=png

<a id="yun00001_3600__vis_png32800_retraining"></a>
#### YUN00001_3600       @ vis_png/32/800_retraining

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images patch_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/YUN00001_3600_0_3599_800_800_800_800/raw stitched_seq_path=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_50_10000_10000_800_0_1586_4_elu/predict_acc/YUN00001_3600 patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg del_patch_seq=0 out_ext=png

<a id="20161203_deployment1yun00001_900_2700__vis_png32800_retraining-1"></a>
#### 20161203_Deployment_1_YUN00001_900_2700       @ vis_png/32/800_retraining

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/20161203_Deployment_1_YUN00001_900_2700_0_1799_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/20161203_Deployment_1_YUN00001_900_2700_max_val_acc patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=png

<a id="20161203_deployment1yun00002_1800__vis_png32800_retraining-1"></a>
#### 20161203_Deployment_1_YUN00002_1800       @ vis_png/32/800_retraining

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20161203_Deployment_1_YUN00002_1800/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/20161203_Deployment_1_YUN00002_1800_0_1799_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/20161203_Deployment_1_YUN00002_1800_max_val_acc patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=png



<a id="fixed_lr__32800_retraining"></a>
### fixed_lr       @ 32/800_retraining

CUDA_VISIBLE_DEVICES=2 python3 densenet_train.py --train_images=/data/617/images/training_0_31_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_31_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --load_weights=1 --log_dir=log/flr_training_0_31_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=1.00 --lr_dec_epochs=0 --psi_act_type=1

<a id="loss_1__32800_retraining"></a>
### loss_1       @ 32/800_retraining

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_31_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_31_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=1 --eval_every=1 --load_weights=1 --log_dir=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_95_10 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1

<a id="18_layers__32800_retraining"></a>
### 18_layers       @ 32/800_retraining

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_31_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_31_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_800_800/images --test_labels=/data/617/images/training_32_49_800_800_800_800/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --load_weights=1 --log_dir=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_18 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --n_layers=18 --preload_images=0

<a id="21_layers__32800_retraining"></a>
### 21_layers       @ 32/800_retraining

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_31_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_31_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_800_800/images --test_labels=/data/617/images/training_32_49_800_800_800_800/labels  --height=800 --width=800 --index_percent=50 --n_classes=3 --start_id=0 --end_id=10 --n_epochs=80000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --load_weights=1 --log_dir=log/rt2_training_0_31_800_800_80_320_rot_15_345_4_flip_21 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --n_layers=21 --preload_images=0


<a id="21_layers__640__32800_retraining"></a>
### 21_layers__640       @ 32/800_retraining


CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_640_640_640_640/images --test_labels=/data/617/images/training_32_49_640_640_640_640/labels  --height=640 --width=640 --index_percent=50 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=64000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --load_weights=1 --log_dir=log/rt2_training_0_31_640_640_64_256_rot_15_345_4_flip_21 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --n_layers=21 --preload_images=0

<a id="24_layers__640__32800_retraining"></a>
### 24_layers__640       @ 32/800_retraining


CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_640_640_640_640/images --test_labels=/data/617/images/training_32_49_640_640_640_640/labels  --height=640 --width=640 --index_percent=50 --n_classes=3 --start_id=0 --end_id=10 --n_epochs=64000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --load_weights=1 --log_dir=log/rt2_training_0_31_640_640_64_256_rot_15_345_4_flip_24 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --n_layers=24 --preload_images=0

<a id="800_retraining_fixed_indices"></a>
# 800_retraining_fixed_indices

<a id="4__non_aug__800_retraining_fixed_indices"></a>
## 4__non_aug       @ 800_retraining_fixed_indices

<a id="2__4__non_aug800_retraining_fixed_indices"></a>
### 2       @ 4__non_aug/800_retraining_fixed_indices

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_3_800_800_800_800/images --train_labels=/data/617/images/training_0_3_800_800_800_800/labels --test_images=/data/617/images/training_32_49_800_800_800_800/images --test_labels=/data/617/images/training_32_49_800_800_800_800/labels  --height=800 --width=800 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=2 --min_indices=2 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_800_800_800_800 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0

mv rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_0_2_2_800_0_17_4_elu rt2_training_0_3_800_800_800_800_0_2_2_800_0_17_4_elu

<a id="10__4__non_aug800_retraining_fixed_indices"></a>
### 10       @ 4__non_aug/800_retraining_fixed_indices

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_3_800_800_800_800/images --train_labels=/data/617/images/training_0_3_800_800_800_800/labels --test_images=/data/617/images/training_32_49_800_800_800_800/images --test_labels=/data/617/images/training_32_49_800_800_800_800/labels  --height=800 --width=800 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=10 --min_indices=10 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_800_800_800_800 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0

mv rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_0_10_10_800_0_17_4_elu rt2_training_0_3_800_800_800_800_0_10_10_800_0_17_4_elu

<a id="100__4__non_aug800_retraining_fixed_indices"></a>
### 100       @ 4__non_aug/800_retraining_fixed_indices

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_3_800_800_800_800/images --train_labels=/data/617/images/training_0_3_800_800_800_800/labels --test_images=/data/617/images/training_32_49_800_800_800_800/images --test_labels=/data/617/images/training_32_49_800_800_800_800/labels  --height=800 --width=800 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=100 --min_indices=100 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_800_800_800_800 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0

mv rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_0_100_100_800_0_17_4_elu rt2_training_0_3_800_800_800_800_0_100_100_800_0_17_4_elu

<a id="1000__4__non_aug800_retraining_fixed_indices"></a>
### 1000       @ 4__non_aug/800_retraining_fixed_indices

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_3_800_800_800_800/images --train_labels=/data/617/images/training_0_3_800_800_800_800/labels --test_images=/data/617/images/training_32_49_800_800_800_800/images --test_labels=/data/617/images/training_32_49_800_800_800_800/labels  --height=800 --width=800 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=1000 --min_indices=1000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_800_800_800_800 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0

mv rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_0_1000_1000_800_0_17_4_elu rt2_training_0_3_800_800_800_800_0_1000_1000_800_0_17_4_elu


<a id="4__800_retraining_fixed_indices"></a>
## 4       @ 800_retraining_fixed_indices

<a id="2__4800_retraining_fixed_indices"></a>
### 2       @ 4/800_retraining_fixed_indices

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=2 --min_indices=2 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0

<a id="5__4800_retraining_fixed_indices"></a>
### 5       @ 4/800_retraining_fixed_indices

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=5 --min_indices=5 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0

<a id="10__4800_retraining_fixed_indices"></a>
### 10       @ 4/800_retraining_fixed_indices

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=10 --min_indices=10 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0

<a id="100__4800_retraining_fixed_indices"></a>
### 100       @ 4/800_retraining_fixed_indices

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=100 --min_indices=100 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0


<a id="vis__4800_retraining_fixed_indices"></a>
### vis       @ 4/800_retraining_fixed_indices

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_0_100_100_800_0_320_4_elu/weights_acc/ --save_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_0_100_100_800_0_320_4_elu/predict_acc/training_32_49_800_800_80_320_rot_15_345_4_flip --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --save_seg=0 --psi_act_type=1

<a id="vis_no_aug__4800_retraining_fixed_indices"></a>
### vis_no_aug       @ 4/800_retraining_fixed_indices

CUDA_VISIBLE_DEVICES=1 python3 densenet_predict.py --images_path=/data/617/images/training_32_49_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_0_100_100_800_0_320_4_elu/weights_acc/ --save_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_0_100_100_800_0_320_4_elu/predict_acc/training_32_49_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --labels_path=/data/617/images/training_32_49_800_800_800_800/labels --psi_act_type=1

<a id="vis_stitched__4800_retraining_fixed_indices"></a>
### vis_stitched       @ 4/800_retraining_fixed_indices

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels patch_seq_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_0_100_100_800_0_320_4_elu/predict_acc/training_32_49_800_800_800_800 stitched_seq_path=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip_0_100_100_800_0_320_4_elu/predict_acc/training_32_49/raw patch_height=800 patch_width=800 start_id=0 end_id=-1 show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png del_patch_seq=0

<a id="1k__4800_retraining_fixed_indices"></a>
### 1K       @ 4/800_retraining_fixed_indices

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=1000 --min_indices=1000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1

<a id="5k__4800_retraining_fixed_indices"></a>
### 5K       @ 4/800_retraining_fixed_indices

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_3_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=5000 --min_indices=5000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0

<a id="16__800_retraining_fixed_indices"></a>
## 16       @ 800_retraining_fixed_indices

<a id="100__16800_retraining_fixed_indices"></a>
### 100       @ 16/800_retraining_fixed_indices

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_15_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_15_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=100 --min_indices=100 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_15_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0

<a id="1k__16800_retraining_fixed_indices"></a>
### 1K       @ 16/800_retraining_fixed_indices

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_15_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_15_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=1000 --min_indices=1000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_15_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1

<a id="5k__16800_retraining_fixed_indices"></a>
### 5K       @ 16/800_retraining_fixed_indices

CUDA_VISIBLE_DEVICES=0 python3 densenet_train.py --train_images=/data/617/images/training_0_15_800_800_80_320_rot_15_345_4_flip/images --train_labels=/data/617/images/training_0_15_800_800_80_320_rot_15_345_4_flip/labels --test_images=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --test_labels=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels  --height=800 --width=800 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=5000 --min_indices=5000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_15_800_800_80_320_rot_15_345_4_flip --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0


<a id="640_retraining"></a>
# 640_retraining

<a id="4__640_retraining"></a>
## 4       @ 640_retraining

<a id="sel_2__4640_retraining"></a>
### sel_2       @ 4/640_retraining

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_3_640_640_640_640/images --train_labels=/data/617/images/training_0_3_640_640_640_640/labels --test_images=/data/617/images/training_32_49_640_640_640_640/images --test_labels=/data/617/images/training_32_49_640_640_640_640/labels  --height=640 --width=640 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=80000 --gpu_id=2 --max_indices=2 --min_indices=2 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_640_640_640_640 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0

<a id="sel_10__4640_retraining"></a>
### sel_10       @ 4/640_retraining

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_3_640_640_640_640/images --train_labels=/data/617/images/training_0_3_640_640_640_640/labels --test_images=/data/617/images/training_32_49_640_640_640_640/images --test_labels=/data/617/images/training_32_49_640_640_640_640/labels  --height=640 --width=640 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=1000 --gpu_id=2 --max_indices=10 --min_indices=10 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_640_640_640_640 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0

<a id="sel_100__4640_retraining"></a>
### sel_100       @ 4/640_retraining

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_3_640_640_640_640/images --train_labels=/data/617/images/training_0_3_640_640_640_640/labels --test_images=/data/617/images/training_32_49_640_640_640_640/images --test_labels=/data/617/images/training_32_49_640_640_640_640/labels  --height=640 --width=640 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=1000 --gpu_id=2 --max_indices=100 --min_indices=100 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_640_640_640_640 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0

<a id="sel_1000__4640_retraining"></a>
### sel_1000       @ 4/640_retraining

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_3_640_640_640_640/images --train_labels=/data/617/images/training_0_3_640_640_640_640/labels --test_images=/data/617/images/training_32_49_640_640_640_640/images --test_labels=/data/617/images/training_32_49_640_640_640_640/labels  --height=640 --width=640 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=1000 --gpu_id=2 --max_indices=1000 --min_indices=1000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_640_640_640_640 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0

<a id="sel_10000__4640_retraining"></a>
### sel_10000       @ 4/640_retraining

CUDA_VISIBLE_DEVICES=1 python3 densenet_train.py --train_images=/data/617/images/training_0_3_640_640_640_640/images --train_labels=/data/617/images/training_0_3_640_640_640_640/labels --test_images=/data/617/images/training_32_49_640_640_640_640/images --test_labels=/data/617/images/training_32_49_640_640_640_640/labels  --height=640 --width=640 --index_percent=0 --n_classes=3 --start_id=0 --end_id=-1 --n_epochs=1000 --gpu_id=2 --max_indices=10000 --min_indices=10000 --test_start_id=0 --test_end_id=-1 --save_stitched=1 --loss_type=4 --eval_every=1 --log_dir=log/rt2_training_0_3_640_640_640_640 --lr_dec_rate=0.95 --lr_dec_epochs=10 --psi_act_type=1 --load_weights=1 --preload_images=0






