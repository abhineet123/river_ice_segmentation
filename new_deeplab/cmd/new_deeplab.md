<!-- MarkdownTOC -->

- [build_data](#build_dat_a_)
    - [voc2012       @ build_data](#voc2012___build_data_)
    - [ade20k       @ build_data](#ade20k___build_data_)
- [hnasnet](#hnasnet_)
    - [atrous:6_12_18       @ hnasnet](#atrous_6_12_18___hnasne_t_)
        - [voc2012       @ atrous:6_12_18/hnasnet](#voc2012___atrous_6_12_18_hnasnet_)
        - [ade20k       @ atrous:6_12_18/hnasnet](#ade20k___atrous_6_12_18_hnasnet_)
    - [atrous_rates_12_24_36       @ hnasnet](#atrous_rates_12_24_36___hnasne_t_)
        - [voc2012       @ atrous_rates_12_24_36/hnasnet](#voc2012___atrous_rates_12_24_36_hnasne_t_)
        - [ade20k       @ atrous_rates_12_24_36/hnasnet](#ade20k___atrous_rates_12_24_36_hnasne_t_)
        - [ctc       @ atrous_rates_12_24_36/hnasnet](#ctc___atrous_rates_12_24_36_hnasne_t_)
            - [huh       @ ctc/atrous_rates_12_24_36/hnasnet](#huh___ctc_atrous_rates_12_24_36_hnasne_t_)
- [resnet_v1_101_beta](#resnet_v1_101_bet_a_)
    - [atrous_6_12_18       @ resnet_v1_101_beta](#atrous_6_12_18___resnet_v1_101_beta_)
        - [voc2012       @ atrous_6_12_18/resnet_v1_101_beta](#voc2012___atrous_6_12_18_resnet_v1_101_bet_a_)
        - [ade20k       @ atrous_6_12_18/resnet_v1_101_beta](#ade20k___atrous_6_12_18_resnet_v1_101_bet_a_)
    - [atrous_rates_12_24_36       @ resnet_v1_101_beta](#atrous_rates_12_24_36___resnet_v1_101_beta_)
        - [voc2012       @ atrous_rates_12_24_36/resnet_v1_101_beta](#voc2012___atrous_rates_12_24_36_resnet_v1_101_beta_)
        - [ade20k       @ atrous_rates_12_24_36/resnet_v1_101_beta](#ade20k___atrous_rates_12_24_36_resnet_v1_101_beta_)
- [640_hnasnet](#640_hnasnet_)
    - [no_pretrained       @ 640_hnasnet](#no_pretrained___640_hnasne_t_)
        - [32       @ no_pretrained/640_hnasnet](#32___no_pretrained_640_hnasne_t_)
            - [vis       @ 32/no_pretrained/640_hnasnet](#vis___32_no_pretrained_640_hnasnet_)
        - [4       @ no_pretrained/640_hnasnet](#4___no_pretrained_640_hnasne_t_)
            - [vis       @ 4/no_pretrained/640_hnasnet](#vis___4_no_pretrained_640_hnasne_t_)
    - [ade20k_pretrained       @ 640_hnasnet](#ade20k_pretrained___640_hnasne_t_)
        - [4       @ ade20k_pretrained/640_hnasnet](#4___ade20k_pretrained_640_hnasne_t_)
            - [vis       @ 4/ade20k_pretrained/640_hnasnet](#vis___4_ade20k_pretrained_640_hnasne_t_)
        - [4_non_aug       @ ade20k_pretrained/640_hnasnet](#4_non_aug___ade20k_pretrained_640_hnasne_t_)
            - [vis       @ 4_non_aug/ade20k_pretrained/640_hnasnet](#vis___4_non_aug_ade20k_pretrained_640_hnasne_t_)
            - [vis_4_49       @ 4_non_aug/ade20k_pretrained/640_hnasnet](#vis_4_49___4_non_aug_ade20k_pretrained_640_hnasne_t_)
        - [4_non_aug_sel_2       @ ade20k_pretrained/640_hnasnet](#4_non_aug_sel_2___ade20k_pretrained_640_hnasne_t_)
            - [vis       @ 4_non_aug_sel_2/ade20k_pretrained/640_hnasnet](#vis___4_non_aug_sel_2_ade20k_pretrained_640_hnasne_t_)
            - [vis_4_49       @ 4_non_aug_sel_2/ade20k_pretrained/640_hnasnet](#vis_4_49___4_non_aug_sel_2_ade20k_pretrained_640_hnasne_t_)
        - [4_non_aug_sel_10       @ ade20k_pretrained/640_hnasnet](#4_non_aug_sel_10___ade20k_pretrained_640_hnasne_t_)
            - [vis       @ 4_non_aug_sel_10/ade20k_pretrained/640_hnasnet](#vis___4_non_aug_sel_10_ade20k_pretrained_640_hnasnet_)
            - [vis_4_49       @ 4_non_aug_sel_10/ade20k_pretrained/640_hnasnet](#vis_4_49___4_non_aug_sel_10_ade20k_pretrained_640_hnasnet_)
        - [4_non_aug_sel_100       @ ade20k_pretrained/640_hnasnet](#4_non_aug_sel_100___ade20k_pretrained_640_hnasne_t_)
            - [vis       @ 4_non_aug_sel_100/ade20k_pretrained/640_hnasnet](#vis___4_non_aug_sel_100_ade20k_pretrained_640_hnasne_t_)
            - [vis_4_49       @ 4_non_aug_sel_100/ade20k_pretrained/640_hnasnet](#vis_4_49___4_non_aug_sel_100_ade20k_pretrained_640_hnasne_t_)
        - [4_non_aug_sel_1000       @ ade20k_pretrained/640_hnasnet](#4_non_aug_sel_1000___ade20k_pretrained_640_hnasne_t_)
            - [vis       @ 4_non_aug_sel_1000/ade20k_pretrained/640_hnasnet](#vis___4_non_aug_sel_1000_ade20k_pretrained_640_hnasnet_)
            - [vis_4_49       @ 4_non_aug_sel_1000/ade20k_pretrained/640_hnasnet](#vis_4_49___4_non_aug_sel_1000_ade20k_pretrained_640_hnasnet_)
        - [8       @ ade20k_pretrained/640_hnasnet](#8___ade20k_pretrained_640_hnasne_t_)
            - [vis       @ 8/ade20k_pretrained/640_hnasnet](#vis___8_ade20k_pretrained_640_hnasne_t_)
        - [16       @ ade20k_pretrained/640_hnasnet](#16___ade20k_pretrained_640_hnasne_t_)
            - [vis       @ 16/ade20k_pretrained/640_hnasnet](#vis___16_ade20k_pretrained_640_hnasnet_)
        - [24       @ ade20k_pretrained/640_hnasnet](#24___ade20k_pretrained_640_hnasne_t_)
            - [vis       @ 24/ade20k_pretrained/640_hnasnet](#vis___24_ade20k_pretrained_640_hnasnet_)
        - [32       @ ade20k_pretrained/640_hnasnet](#32___ade20k_pretrained_640_hnasne_t_)
            - [vis       @ 32/ade20k_pretrained/640_hnasnet](#vis___32_ade20k_pretrained_640_hnasnet_)
- [640_resnet_v1_101_beta](#640_resnet_v1_101_bet_a_)
    - [imagenet_pretrained       @ 640_resnet_v1_101_beta](#imagenet_pretrained___640_resnet_v1_101_beta_)
        - [4       @ imagenet_pretrained/640_resnet_v1_101_beta](#4___imagenet_pretrained_640_resnet_v1_101_beta_)
            - [vis       @ 4/imagenet_pretrained/640_resnet_v1_101_beta](#vis___4_imagenet_pretrained_640_resnet_v1_101_beta_)
        - [4_non_aug       @ imagenet_pretrained/640_resnet_v1_101_beta](#4_non_aug___imagenet_pretrained_640_resnet_v1_101_beta_)
            - [vis_4_49       @ 4_non_aug/imagenet_pretrained/640_resnet_v1_101_beta](#vis_4_49___4_non_aug_imagenet_pretrained_640_resnet_v1_101_beta_)
        - [4_non_aug_sel_2       @ imagenet_pretrained/640_resnet_v1_101_beta](#4_non_aug_sel_2___imagenet_pretrained_640_resnet_v1_101_beta_)
            - [vis_4_49       @ 4_non_aug_sel_2/imagenet_pretrained/640_resnet_v1_101_beta](#vis_4_49___4_non_aug_sel_2_imagenet_pretrained_640_resnet_v1_101_beta_)
        - [4_non_aug_sel_10       @ imagenet_pretrained/640_resnet_v1_101_beta](#4_non_aug_sel_10___imagenet_pretrained_640_resnet_v1_101_beta_)
            - [vis_4_49       @ 4_non_aug_sel_10/imagenet_pretrained/640_resnet_v1_101_beta](#vis_4_49___4_non_aug_sel_10_imagenet_pretrained_640_resnet_v1_101_bet_a_)
        - [4_non_aug_sel_100       @ imagenet_pretrained/640_resnet_v1_101_beta](#4_non_aug_sel_100___imagenet_pretrained_640_resnet_v1_101_beta_)
            - [vis_4_49       @ 4_non_aug_sel_100/imagenet_pretrained/640_resnet_v1_101_beta](#vis_4_49___4_non_aug_sel_100_imagenet_pretrained_640_resnet_v1_101_beta_)
        - [4_non_aug_sel_1000       @ imagenet_pretrained/640_resnet_v1_101_beta](#4_non_aug_sel_1000___imagenet_pretrained_640_resnet_v1_101_beta_)
            - [vis_4_49       @ 4_non_aug_sel_1000/imagenet_pretrained/640_resnet_v1_101_beta](#vis_4_49___4_non_aug_sel_1000_imagenet_pretrained_640_resnet_v1_101_bet_a_)
        - [8       @ imagenet_pretrained/640_resnet_v1_101_beta](#8___imagenet_pretrained_640_resnet_v1_101_beta_)
            - [vis       @ 8/imagenet_pretrained/640_resnet_v1_101_beta](#vis___8_imagenet_pretrained_640_resnet_v1_101_beta_)
        - [16       @ imagenet_pretrained/640_resnet_v1_101_beta](#16___imagenet_pretrained_640_resnet_v1_101_beta_)
            - [vis       @ 16/imagenet_pretrained/640_resnet_v1_101_beta](#vis___16_imagenet_pretrained_640_resnet_v1_101_bet_a_)
        - [24       @ imagenet_pretrained/640_resnet_v1_101_beta](#24___imagenet_pretrained_640_resnet_v1_101_beta_)
            - [vis       @ 24/imagenet_pretrained/640_resnet_v1_101_beta](#vis___24_imagenet_pretrained_640_resnet_v1_101_bet_a_)
        - [32       @ imagenet_pretrained/640_resnet_v1_101_beta](#32___imagenet_pretrained_640_resnet_v1_101_beta_)
            - [vis       @ 32/imagenet_pretrained/640_resnet_v1_101_beta](#vis___32_imagenet_pretrained_640_resnet_v1_101_bet_a_)
    - [ade20k_pretrained       @ 640_resnet_v1_101_beta](#ade20k_pretrained___640_resnet_v1_101_beta_)
        - [32       @ ade20k_pretrained/640_resnet_v1_101_beta](#32___ade20k_pretrained_640_resnet_v1_101_beta_)
            - [vis       @ 32/ade20k_pretrained/640_resnet_v1_101_beta](#vis___32_ade20k_pretrained_640_resnet_v1_101_bet_a_)
        - [4       @ ade20k_pretrained/640_resnet_v1_101_beta](#4___ade20k_pretrained_640_resnet_v1_101_beta_)
            - [vis       @ 4/ade20k_pretrained/640_resnet_v1_101_beta](#vis___4_ade20k_pretrained_640_resnet_v1_101_beta_)

<!-- /MarkdownTOC -->

<a id="build_dat_a_"></a>
# build_data

<a id="voc2012___build_data_"></a>
## voc2012       @ build_data-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 datasets/build_voc2012_data.py image_folder=/data/voc2012/JPEGImages semantic_segmentation_folder=/data/voc2012/SegmentationClass list_folder=/data/voc2012/ImageSets/Segmentation image_format="jpg" output_dir=/data/voc2012/tfrecord

<a id="ade20k___build_data_"></a>
## ade20k       @ build_data-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 datasets/build_ade20k_data.py train_image_folder=/data/ade20k/images/training/ train_image_label_folder=/data/ade20k/annotations/training/ val_image_folder=/data/ade20k/images/validation/ val_image_label_folder=/data/ade20k/annotations/validation/ output_dir=/data/ade20k/tfrecord

<a id="hnasnet_"></a>
# hnasnet

<a id="atrous_6_12_18___hnasne_t_"></a>
## atrous:6_12_18       @ hnasnet-->new_deeplab

<a id="voc2012___atrous_6_12_18_hnasnet_"></a>
### voc2012       @ atrous:6_12_18/hnasnet-->new_deeplab

python36 new_deeplab_train.py cfg=gpu:2,_hnas_:b2,_voc_

<a id="ade20k___atrous_6_12_18_hnasnet_"></a>
### ade20k       @ atrous:6_12_18/hnasnet-->new_deeplab

python36 new_deeplab_train.py cfg=gpu:2,_hnas_  
<a id="atrous_rates_12_24_36___hnasne_t_"></a>
## atrous_rates_12_24_36       @ hnasnet-->new_deeplab

<a id="voc2012___atrous_rates_12_24_36_hnasne_t_"></a>
### voc2012       @ atrous_rates_12_24_36/hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_train.py model_variant="nas_hnasnet" atrous_rates=12 atrous_rates=24 atrous_rates=36 output_stride=8 decoder_output_stride=4 train_crop_size="513,513" train_batch_size=2 dataset=pascal_voc_seg train_logdir=log/pascal_voc_seg/nas_hnasnet_atrous_rates_12_24_36 dataset_dir=/data/voc2012/tfrecord train_split=train num_clones=1 add_image_level_feature=0


<a id="ade20k___atrous_rates_12_24_36_hnasne_t_"></a>
### ade20k       @ atrous_rates_12_24_36/hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_train.py model_variant="nas_hnasnet" atrous_rates=12 atrous_rates=24 atrous_rates=36 output_stride=8 decoder_output_stride=4 train_crop_size="513,513" train_batch_size=4 dataset=ade20k train_logdir=log/ade20k/nas_hnasnet_atrous_rates_12_24_36 dataset_dir=/data/ade20k/tfrecord train_split=train num_clones=1 add_image_level_feature=0 min_resize_value=513 max_resize_value=513 resize_factor=16 

<a id="ctc___atrous_rates_12_24_36_hnasne_t_"></a>
### ctc       @ atrous_rates_12_24_36/hnasnet-->new_deeplab

<a id="huh___ctc_atrous_rates_12_24_36_hnasne_t_"></a>
#### huh       @ ctc/atrous_rates_12_24_36/hnasnet-->new_deeplab
python36 new_deeplab_train.py cfg=gpu:0,_hnas_:atrous-6_12_18,_ctc_:huh

<a id="resnet_v1_101_bet_a_"></a>
# resnet_v1_101_beta

<a id="atrous_6_12_18___resnet_v1_101_beta_"></a>
## atrous_6_12_18       @ resnet_v1_101_beta-->new_deeplab

<a id="voc2012___atrous_6_12_18_resnet_v1_101_bet_a_"></a>
### voc2012       @ atrous_6_12_18/resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_train.py model_variant=resnet_v1_101_beta atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size=513,513 train_batch_size=2 dataset=pascal_voc_seg train_logdir=log/pascal_voc_seg/resnet_v1_101_beta dataset_dir=/data/voc2012/tfrecord train_split=train num_clones=1

<a id="ade20k___atrous_6_12_18_resnet_v1_101_bet_a_"></a>
### ade20k       @ atrous_6_12_18/resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_train.py model_variant=resnet_v1_101_beta atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size="513,513" train_batch_size=4 dataset=ade20k train_logdir=log/ade20k/resnet_v1_101_beta dataset_dir=/data/ade20k/tfrecord train_split=train num_clones=1 min_resize_value=513 max_resize_value=513 resize_factor=16

<a id="atrous_rates_12_24_36___resnet_v1_101_beta_"></a>
## atrous_rates_12_24_36       @ resnet_v1_101_beta-->new_deeplab

<a id="voc2012___atrous_rates_12_24_36_resnet_v1_101_beta_"></a>
### voc2012       @ atrous_rates_12_24_36/resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_train.py model_variant=resnet_v1_101_beta atrous_rates=12 atrous_rates=24 atrous_rates=36 output_stride=8 decoder_output_stride=4 train_crop_size=513,513 train_batch_size=2 dataset=pascal_voc_seg train_logdir=log/pascal_voc_seg/resnet_v1_101_beta_atrous_rates_12_24_36 dataset_dir=/data/voc2012/tfrecord train_split=train num_clones=1

<a id="ade20k___atrous_rates_12_24_36_resnet_v1_101_beta_"></a>
### ade20k       @ atrous_rates_12_24_36/resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_train.py model_variant=resnet_v1_101_beta atrous_rates=12 atrous_rates=24 atrous_rates=36 output_stride=8 decoder_output_stride=4 train_crop_size="513,513" train_batch_size=4 dataset=ade20k train_logdir=log/ade20k/resnet_v1_101_beta_atrous_rates_12_24_36 dataset_dir=/data/ade20k/tfrecord train_split=train num_clones=1 min_resize_value=513 max_resize_value=513 resize_factor=16

<a id="640_hnasnet_"></a>
# 640_hnasnet

<a id="no_pretrained___640_hnasne_t_"></a>
## no_pretrained       @ 640_hnasnet-->new_deeplab

<a id="32___no_pretrained_640_hnasne_t_"></a>
### 32       @ no_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_train.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size=640,640 train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31 dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_31_640_640_64_256_rot_15_345_4_flip num_clones=1 add_image_level_feature=0

<a id="vis___32_no_pretrained_640_hnasnet_"></a>
#### vis       @ 32/no_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31 vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="4___no_pretrained_640_hnasne_t_"></a>
### 4       @ no_pretrained/640_hnasnet-->new_deeplab
CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_train.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size=640,640 train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3 dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_3_640_640_64_256_rot_15_345_4_flip num_clones=1 add_image_level_feature=0

<a id="vis___4_no_pretrained_640_hnasne_t_"></a>
#### vis       @ 4/no_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3 vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="ade20k_pretrained___640_hnasne_t_"></a>
## ade20k_pretrained       @ 640_hnasnet-->new_deeplab

<a id="4___ade20k_pretrained_640_hnasne_t_"></a>
### 4       @ ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_train.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size=640,640 train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_ade20k dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_3_640_640_64_256_rot_15_345_4_flip num_clones=1 add_image_level_feature=0 tf_initial_checkpoint=log/ade20k/nas_hnasnet/model.ckpt-352577 initialize_last_layer=0

<a id="vis___4_ade20k_pretrained_640_hnasne_t_"></a>
#### vis       @ 4/ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_ade20k/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_ade20k/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_ade20k/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_ade20k/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_ade20k/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1


<a id="4_non_aug___ade20k_pretrained_640_hnasne_t_"></a>
### 4_non_aug       @ ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_train.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size=640,640 train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_ade20k dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_3_640_640_640_640 num_clones=1 add_image_level_feature=0 tf_initial_checkpoint=log/ade20k/nas_hnasnet/model.ckpt-352577 initialize_last_layer=0

<a id="vis___4_non_aug_ade20k_pretrained_640_hnasne_t_"></a>
#### vis       @ 4_non_aug/ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_ade20k/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_ade20k/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_ade20k/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_ade20k/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_ade20k/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="vis_4_49___4_non_aug_ade20k_pretrained_640_hnasne_t_"></a>
#### vis_4_49       @ 4_non_aug/ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_ade20k/training_4_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_4_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_ade20k/training_4_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_ade20k/training_4_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_ade20k/training_4_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_ade20k/training_4_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="4_non_aug_sel_2___ade20k_pretrained_640_hnasne_t_"></a>
### 4_non_aug_sel_2       @ ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_train.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size=640,640 train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_2_ade20k dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_3_640_640_640_640_sel_2 num_clones=1 add_image_level_feature=0 tf_initial_checkpoint=log/ade20k/nas_hnasnet/model.ckpt-352577 initialize_last_layer=0

<a id="vis___4_non_aug_sel_2_ade20k_pretrained_640_hnasne_t_"></a>
#### vis       @ 4_non_aug_sel_2/ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_2_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_2_ade20k/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_2_ade20k/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_2_ade20k/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_2_ade20k/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_2_ade20k/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="vis_4_49___4_non_aug_sel_2_ade20k_pretrained_640_hnasne_t_"></a>
#### vis_4_49       @ 4_non_aug_sel_2/ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_2_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_2_ade20k/training_4_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_4_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_2_ade20k/training_4_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_2_ade20k/training_4_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_2_ade20k/training_4_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_2_ade20k/training_4_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1


<a id="4_non_aug_sel_10___ade20k_pretrained_640_hnasne_t_"></a>
### 4_non_aug_sel_10       @ ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_train.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size=640,640 train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_10_ade20k dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_3_640_640_640_640_sel_10 num_clones=1 add_image_level_feature=0 tf_initial_checkpoint=log/ade20k/nas_hnasnet/model.ckpt-352577 initialize_last_layer=0

<a id="vis___4_non_aug_sel_10_ade20k_pretrained_640_hnasnet_"></a>
#### vis       @ 4_non_aug_sel_10/ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_10_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_10_ade20k/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_10_ade20k/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_10_ade20k/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_10_ade20k/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_10_ade20k/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="vis_4_49___4_non_aug_sel_10_ade20k_pretrained_640_hnasnet_"></a>
#### vis_4_49       @ 4_non_aug_sel_10/ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_10_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_10_ade20k/training_4_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_4_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_10_ade20k/training_4_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_10_ade20k/training_4_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_10_ade20k/training_4_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_10_ade20k/training_4_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1


<a id="4_non_aug_sel_100___ade20k_pretrained_640_hnasne_t_"></a>
### 4_non_aug_sel_100       @ ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_train.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size=640,640 train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_100_ade20k dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_3_640_640_640_640_sel_100 num_clones=1 add_image_level_feature=0 tf_initial_checkpoint=log/ade20k/nas_hnasnet/model.ckpt-352577 initialize_last_layer=0

<a id="vis___4_non_aug_sel_100_ade20k_pretrained_640_hnasne_t_"></a>
#### vis       @ 4_non_aug_sel_100/ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_100_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_100_ade20k/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_100_ade20k/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_100_ade20k/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_100_ade20k/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_100_ade20k/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1


<a id="vis_4_49___4_non_aug_sel_100_ade20k_pretrained_640_hnasne_t_"></a>
#### vis_4_49       @ 4_non_aug_sel_100/ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_100_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_100_ade20k/training_4_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_4_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_100_ade20k/training_4_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_100_ade20k/training_4_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_100_ade20k/training_4_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_100_ade20k/training_4_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="4_non_aug_sel_1000___ade20k_pretrained_640_hnasne_t_"></a>
### 4_non_aug_sel_1000       @ ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_train.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size=640,640 train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_1000_ade20k dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_3_640_640_640_640_sel_1000 num_clones=1 add_image_level_feature=0 tf_initial_checkpoint=log/ade20k/nas_hnasnet/model.ckpt-352577 initialize_last_layer=0

<a id="vis___4_non_aug_sel_1000_ade20k_pretrained_640_hnasnet_"></a>
#### vis       @ 4_non_aug_sel_1000/ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_1000_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_1000_ade20k/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_1000_ade20k/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_1000_ade20k/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_1000_ade20k/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_1000_ade20k/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="vis_4_49___4_non_aug_sel_1000_ade20k_pretrained_640_hnasnet_"></a>
#### vis_4_49       @ 4_non_aug_sel_1000/ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_1000_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_1000_ade20k/training_4_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_4_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_1000_ade20k/training_4_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_1000_ade20k/training_4_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_1000_ade20k/training_4_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_3_non_aug_sel_1000_ade20k/training_4_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="8___ade20k_pretrained_640_hnasne_t_"></a>
### 8       @ ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_train.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size=640,640 train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_7_ade20k dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_7_640_640_64_256_rot_15_345_4_flip num_clones=1 add_image_level_feature=0 tf_initial_checkpoint=log/ade20k/nas_hnasnet/model.ckpt-352577 initialize_last_layer=0

<a id="vis___8_ade20k_pretrained_640_hnasne_t_"></a>
#### vis       @ 8/ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_7_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_7_ade20k/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_7_ade20k/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_7_ade20k/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_7_ade20k/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_7_ade20k/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1


<a id="16___ade20k_pretrained_640_hnasne_t_"></a>
### 16       @ ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_train.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size=640,640 train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_15_ade20k dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_15_640_640_64_256_rot_15_345_4_flip num_clones=1 add_image_level_feature=0 tf_initial_checkpoint=log/ade20k/nas_hnasnet/model.ckpt-352577 initialize_last_layer=0

<a id="vis___16_ade20k_pretrained_640_hnasnet_"></a>
#### vis       @ 16/ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_15_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_15_ade20k/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_15_ade20k/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_15_ade20k/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_15_ade20k/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_15_ade20k/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="24___ade20k_pretrained_640_hnasne_t_"></a>
### 24       @ ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_train.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size=640,640 train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_23_ade20k dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_23_640_640_64_256_rot_15_345_4_flip num_clones=1 add_image_level_feature=0 tf_initial_checkpoint=log/ade20k/nas_hnasnet/model.ckpt-352577 initialize_last_layer=0

<a id="vis___24_ade20k_pretrained_640_hnasnet_"></a>
#### vis       @ 24/ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_23_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_23_ade20k/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_23_ade20k/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_23_ade20k/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_23_ade20k/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_23_ade20k/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1


<a id="32___ade20k_pretrained_640_hnasne_t_"></a>
### 32       @ ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_train.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size=640,640 train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31_ade20k dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_31_640_640_64_256_rot_15_345_4_flip num_clones=1 add_image_level_feature=0 tf_initial_checkpoint=log/ade20k/nas_hnasnet/model.ckpt-209105 initialize_last_layer=0

<a id="vis___32_ade20k_pretrained_640_hnasnet_"></a>
#### vis       @ 32/ade20k_pretrained/640_hnasnet-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31_ade20k/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=50 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0 add_image_level_feature=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31_ade20k/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31_ade20k/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31_ade20k/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31_ade20k/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="640_resnet_v1_101_bet_a_"></a>
# 640_resnet_v1_101_beta

<a id="imagenet_pretrained___640_resnet_v1_101_beta_"></a>
## imagenet_pretrained       @ 640_resnet_v1_101_beta-->new_deeplab

<a id="4___imagenet_pretrained_640_resnet_v1_101_beta_"></a>
### 4       @ imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_train.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size="640,640" train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip tf_initial_checkpoint=pre_trained/resnet_v1_101/model.ckpt train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3 dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_3_640_640_64_256_rot_15_345_4_flip num_clones=1

<a id="vis___4_imagenet_pretrained_640_resnet_v1_101_beta_"></a>
#### vis       @ 4/imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3 vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=10 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="4_non_aug___imagenet_pretrained_640_resnet_v1_101_beta_"></a>
### 4_non_aug       @ imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_train.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size="640,640" train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip tf_initial_checkpoint=pre_trained/resnet_v1_101/model.ckpt train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_3_640_640_640_640 num_clones=1

<a id="vis_4_49___4_non_aug_imagenet_pretrained_640_resnet_v1_101_beta_"></a>
#### vis_4_49       @ 4_non_aug/imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug/training_4_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_4_49_640_640_640_640 vis_batch_size=10 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug/training_4_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug/training_4_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug/training_4_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug/training_4_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1


<a id="4_non_aug_sel_2___imagenet_pretrained_640_resnet_v1_101_beta_"></a>
### 4_non_aug_sel_2       @ imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_train.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size="640,640" train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip tf_initial_checkpoint=pre_trained/resnet_v1_101/model.ckpt train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_2 dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_3_640_640_640_640_sel_2 num_clones=1

<a id="vis_4_49___4_non_aug_sel_2_imagenet_pretrained_640_resnet_v1_101_beta_"></a>
#### vis_4_49       @ 4_non_aug_sel_2/imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_vis.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_2 vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_2/training_4_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_4_49_640_640_640_640 vis_batch_size=10 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_2/training_4_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_2/training_4_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_2/training_4_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_2/training_4_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1


<a id="4_non_aug_sel_10___imagenet_pretrained_640_resnet_v1_101_beta_"></a>
### 4_non_aug_sel_10       @ imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_train.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size="640,640" train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip tf_initial_checkpoint=pre_trained/resnet_v1_101/model.ckpt train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_10 dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_3_640_640_640_640_sel_10 num_clones=1

<a id="vis_4_49___4_non_aug_sel_10_imagenet_pretrained_640_resnet_v1_101_bet_a_"></a>
#### vis_4_49       @ 4_non_aug_sel_10/imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_vis.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_10 vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_10/training_4_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_4_49_640_640_640_640 vis_batch_size=10 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_10/training_4_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_10/training_4_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_10/training_4_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_10/training_4_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1


<a id="4_non_aug_sel_100___imagenet_pretrained_640_resnet_v1_101_beta_"></a>
### 4_non_aug_sel_100       @ imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_train.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size="640,640" train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip tf_initial_checkpoint=pre_trained/resnet_v1_101/model.ckpt train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_100 dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_3_640_640_640_640_sel_100 num_clones=1

<a id="vis_4_49___4_non_aug_sel_100_imagenet_pretrained_640_resnet_v1_101_beta_"></a>
#### vis_4_49       @ 4_non_aug_sel_100/imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_100 vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_100/training_4_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_4_49_640_640_640_640 vis_batch_size=10 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_100/training_4_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_100/training_4_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_100/training_4_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_100/training_4_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1


<a id="4_non_aug_sel_1000___imagenet_pretrained_640_resnet_v1_101_beta_"></a>
### 4_non_aug_sel_1000       @ imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_train.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size="640,640" train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip tf_initial_checkpoint=pre_trained/resnet_v1_101/model.ckpt train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_1000 dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_3_640_640_640_640_sel_1000 num_clones=1

<a id="vis_4_49___4_non_aug_sel_1000_imagenet_pretrained_640_resnet_v1_101_bet_a_"></a>
#### vis_4_49       @ 4_non_aug_sel_1000/imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_1000 vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_1000/training_4_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_4_49_640_640_640_640 vis_batch_size=10 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_1000/training_4_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_1000/training_4_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_1000/training_4_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_non_aug_sel_1000/training_4_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1



<a id="8___imagenet_pretrained_640_resnet_v1_101_beta_"></a>
### 8       @ imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_train.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size="640,640" train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip tf_initial_checkpoint=pre_trained/resnet_v1_101/model.ckpt train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_7 dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_7_640_640_64_256_rot_15_345_4_flip num_clones=1

<a id="vis___8_imagenet_pretrained_640_resnet_v1_101_beta_"></a>
#### vis       @ 8/imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_vis.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_7 vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_7/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=10 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_7/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_7/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_7/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_7/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="16___imagenet_pretrained_640_resnet_v1_101_beta_"></a>
### 16       @ imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_train.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size="640,640" train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip tf_initial_checkpoint=pre_trained/resnet_v1_101/model.ckpt train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_15 dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_15_640_640_64_256_rot_15_345_4_flip num_clones=1

<a id="vis___16_imagenet_pretrained_640_resnet_v1_101_bet_a_"></a>
#### vis       @ 16/imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_vis.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_15 vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_15/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=10 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_15/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_15/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_15/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_15/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="24___imagenet_pretrained_640_resnet_v1_101_beta_"></a>
### 24       @ imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_train.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size="640,640" train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip tf_initial_checkpoint=pre_trained/resnet_v1_101/model.ckpt train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_23 dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_23_640_640_64_256_rot_15_345_4_flip num_clones=1

<a id="vis___24_imagenet_pretrained_640_resnet_v1_101_bet_a_"></a>
#### vis       @ 24/imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_23 vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_23/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=10 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_23/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_23/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_23/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_23/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="32___imagenet_pretrained_640_resnet_v1_101_beta_"></a>
### 32       @ imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_train.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size="640,640" train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip tf_initial_checkpoint=pre_trained/resnet_v1_101/model.ckpt train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31 dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_31_640_640_64_256_rot_15_345_4_flip num_clones=1

<a id="vis___32_imagenet_pretrained_640_resnet_v1_101_bet_a_"></a>
#### vis       @ 32/imagenet_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31 vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=10 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1



<a id="ade20k_pretrained___640_resnet_v1_101_beta_"></a>
## ade20k_pretrained       @ 640_resnet_v1_101_beta-->new_deeplab

<a id="32___ade20k_pretrained_640_resnet_v1_101_beta_"></a>
### 32       @ ade20k_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_train.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size="640,640" train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip tf_initial_checkpoint=log/ade20k/resnet_v1_101_beta/model.ckpt-249443 train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31_ade20k dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_31_640_640_64_256_rot_15_345_4_flip num_clones=1  initialize_last_layer=0

<a id="vis___32_ade20k_pretrained_640_resnet_v1_101_bet_a_"></a>
#### vis       @ 32/ade20k_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31_ade20k/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=10 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31_ade20k/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31_ade20k/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31_ade20k/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_31_ade20k/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1

<a id="4___ade20k_pretrained_640_resnet_v1_101_beta_"></a>
### 4       @ ade20k_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=1 python36 new_deeplab_train.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size="640,640" train_batch_size=2 dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip tf_initial_checkpoint=log/ade20k/resnet_v1_101_beta/model.ckpt-249443 train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_ade20k dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord train_split=training_0_3_640_640_64_256_rot_15_345_4_flip num_clones=1  initialize_last_layer=0

<a id="vis___4_ade20k_pretrained_640_resnet_v1_101_beta_"></a>
#### vis       @ 4/ade20k_pretrained/640_resnet_v1_101_beta-->new_deeplab

CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="resnet_v1_101_beta" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_ade20k vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_ade20k/training_32_49_640_640_640_640 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord vis_split=training_32_49_640_640_640_640 vis_batch_size=10 also_save_vis_predictions=0 max_number_of_iterations=1 eval_interval_secs=0

python36 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_ade20k/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_ade20k/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python36 ../visDataset.py images_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_ade20k/training_32_49/raw save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/resnet_v1_101_0_3_ade20k/training_32_49/vis n_classes=3 start_id=0 end_id=-1 normalize_labels=1




