<!-- MarkdownTOC -->

- [ctc](#ctc_)
   - [640       @ ctc](#640___ct_c_)
- [320x640](#320x640_)
- [256](#256_)
   - [all       @ 256](#all___25_6_)
      - [ppt       @ all/256](#ppt___all_25_6_)
      - [rotation_and_flipping       @ all/256](#rotation_and_flipping___all_25_6_)
      - [merging       @ all/256](#merging___all_25_6_)
   - [0-31       @ 256](#0_31___25_6_)
      - [merging       @ 0-31/256](#merging___0_31_256_)
   - [32-49       @ 256](#32_49___25_6_)
      - [merging       @ 32-49/256](#merging___32_49_25_6_)
   - [batch_all       @ 256](#batch_all___25_6_)
   - [validation       @ 256](#validation___25_6_)
      - [stitching       @ validation/256](#stitching___validation_256_)
   - [videos       @ 256](#videos___25_6_)
      - [stitching       @ videos/256](#stitching___videos_256_)
- [384](#384_)
   - [40/160       @ 384](#40_160___38_4_)
   - [25/100       @ 384](#25_100___38_4_)
   - [validation       @ 384](#validation___38_4_)
      - [stitching       @ validation/384](#stitching___validation_384_)
   - [videos       @ 384](#videos___38_4_)
      - [stitching       @ videos/384](#stitching___videos_384_)
   - [vis       @ 384](#vis___38_4_)
      - [unet       @ vis/384](#unet___vis_38_4_)
         - [hml       @ unet/vis/384](#hml___unet_vis_384_)
         - [weird       @ unet/vis/384](#weird___unet_vis_384_)
- [512](#512_)
   - [40/160       @ 512](#40_160___51_2_)
   - [25/100       @ 512](#25_100___51_2_)
   - [validation       @ 512](#validation___51_2_)
      - [stitching       @ validation/512](#stitching___validation_512_)
   - [videos       @ 512](#videos___51_2_)
      - [stitching       @ videos/512](#stitching___videos_512_)
- [640](#640_)
   - [64/256       @ 640](#64_256___64_0_)
   - [non_aug       @ 640](#non_aug___64_0_)
      - [0__3       @ non_aug/640](#0_3___non_aug_64_0_)
         - [sel-2       @ 0__3/non_aug/640](#sel_2___0_3_non_aug_64_0_)
         - [sel-10       @ 0__3/non_aug/640](#sel_10___0_3_non_aug_64_0_)
         - [sel-100       @ 0__3/non_aug/640](#sel_100___0_3_non_aug_64_0_)
         - [sel-1000       @ 0__3/non_aug/640](#sel_1000___0_3_non_aug_64_0_)
         - [sel-5000       @ 0__3/non_aug/640](#sel_5000___0_3_non_aug_64_0_)
      - [32_-49       @ non_aug/640](#32_49___non_aug_64_0_)
      - [0__49       @ non_aug/640](#0_49___non_aug_64_0_)
      - [4__49       @ non_aug/640](#4_49___non_aug_64_0_)
      - [entire_image       @ non_aug/640](#entire_image___non_aug_64_0_)
         - [0-3       @ entire_image/non_aug/640](#0_3___entire_image_non_aug_640_)
         - [0-7       @ entire_image/non_aug/640](#0_7___entire_image_non_aug_640_)
         - [0-15       @ entire_image/non_aug/640](#0_15___entire_image_non_aug_640_)
         - [0-23       @ entire_image/non_aug/640](#0_23___entire_image_non_aug_640_)
         - [0-31       @ entire_image/non_aug/640](#0_31___entire_image_non_aug_640_)
         - [32-49       @ entire_image/non_aug/640](#32_49___entire_image_non_aug_640_)
         - [4-49       @ entire_image/non_aug/640](#4_49___entire_image_non_aug_640_)
      - [ablation       @ non_aug/640](#ablation___non_aug_64_0_)
         - [0__3       @ ablation/non_aug/640](#0_3___ablation_non_aug_640_)
         - [sel-2       @ ablation/non_aug/640](#sel_2___ablation_non_aug_640_)
         - [sel-2       @ ablation/non_aug/640](#sel_2___ablation_non_aug_640__1)
         - [sel-10       @ ablation/non_aug/640](#sel_10___ablation_non_aug_640_)
         - [sel-100       @ ablation/non_aug/640](#sel_100___ablation_non_aug_640_)
         - [sel-1000       @ ablation/non_aug/640](#sel_1000___ablation_non_aug_640_)
         - [sel-5000       @ ablation/non_aug/640](#sel_5000___ablation_non_aug_640_)
   - [25/100       @ 640](#25_100___64_0_)
   - [validation       @ 640](#validation___64_0_)
      - [stitching       @ validation/640](#stitching___validation_640_)
   - [videos       @ 640](#videos___64_0_)
      - [stitching       @ videos/640](#stitching___videos_640_)
- [800](#800_)
   - [80/320       @ 800](#80_320___80_0_)
   - [non_aug       @ 800](#non_aug___80_0_)
      - [0__3       @ non_aug/800](#0_3___non_aug_80_0_)
      - [32__49       @ non_aug/800](#32_49___non_aug_80_0_)
      - [0__49       @ non_aug/800](#0_49___non_aug_80_0_)
      - [4__49       @ non_aug/800](#4_49___non_aug_80_0_)
      - [entire_image       @ non_aug/800](#entire_image___non_aug_80_0_)
         - [32-49       @ entire_image/non_aug/800](#32_49___entire_image_non_aug_800_)
         - [4-49       @ entire_image/non_aug/800](#4_49___entire_image_non_aug_800_)
      - [ablation       @ non_aug/800](#ablation___non_aug_80_0_)
   - [25/100       @ 800](#25_100___80_0_)
   - [video       @ 800](#video___80_0_)
- [1000       @ sub_patch](#1000___sub_patc_h_)
   - [100/400       @ 1000](#100_400___1000_)
   - [video       @ 1000](#video___1000_)
      - [1920x1080       @ video/1000](#1920x1080___video_1000_)

<!-- /MarkdownTOC -->


<a id="ctc_"></a>
# ctc
<a id="640___ct_c_"></a>
## 640       @ ctc-->sub_patch
python subPatchBatch.py cfg=ctc:fluo-r:size-640:smin-64:smax-256:rmin-15:rmax-345:rnum-4:flip:log:seq-1

python3 subPatchDataset.py db_root_dir=/data/CTC seq_name=Fluo-C2DL-Huh7_01 img_ext=jpg labels_ext=png out_ext=png patch_height=640 patch_width=640 min_stride=64 max_stride=256 enable_flip=1 start_id=0 end_id=29 n_frames=30 show_img=0 out_seq_name=Fluo-C2DL-Huh7_01_0_29_640_640_64_256_rot_15_345_4_flip src_path=/data/CTC/Images/Fluo-C2DL-Huh7_01 labels_path=/data/CTC/Labels_PNG/Fluo-C2DL-Huh7_01 enable_rot=0

<a id="320x640_"></a>
# 320x640      
python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=320 patch_width=640 min_stride=25 max_stride=100

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=320 patch_width=640 min_stride=100 max_stride=200

<a id="256_"></a>
# 256
<a id="all___25_6_"></a>
## all       @ 256-->sub_patch
python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 show_img=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=100 max_stride=200

<a id="ppt___all_25_6_"></a>
### ppt       @ all/256-->sub_patch
python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 show_img=1 start_id=2 end_id=2

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 show_img=1 start_id=2 end_id=2 enable_rot=1 min_rot=15 max_rot=90

<a id="rotation_and_flipping___all_25_6_"></a>
### rotation_and_flipping       @ all/256-->sub_patch
python3 subPatchDataset.py db_root_dir=/home/abhineet/N/Datasets/617 seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=20 max_rot=75

python3 subPatchDataset.py db_root_dir=/home/abhineet/N/Datasets/617 seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=20 max_rot=90 enable_flip=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=90 enable_flip=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=90 enable_flip=1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=100 max_stride=200 enable_rot=1 min_rot=90 max_rot=180 enable_flip=1

<a id="merging___all_25_6_"></a>
### merging       @ all/256-->sub_patch
python3 mergeDatasets.py training_256_256_100_200_flip training_256_256_100_200_rot_90_180_flip 

<a id="0_31___25_6_"></a>
## 0-31       @ 256-->sub_patch

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=0 enable_flip=1 start_id=0 end_id=31

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=125 enable_flip=1 start_id=0 end_id=31

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=126 max_rot=235 enable_flip=1 start_id=0 end_id=31

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=236 max_rot=345 enable_flip=1 start_id=0 end_id=31

<a id="merging___0_31_256_"></a>
### merging       @ 0-31/256-->sub_patch
python3 mergeDatasets.py training_0_31_256_256_25_100_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

python3 mergeDatasets.py training_0_31_256_256_25_100_rot_126_235_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

python3 mergeDatasets.py training_0_31_256_256_25_100_rot_15_125_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

python3 mergeDatasets.py training_0_31_256_256_25_100_rot_236_345_flip training_0_31_256_256_25_100_rot_15_125_235_345_flip start_id=0 end_id=31

<a id="32_49___25_6_"></a>
## 32-49       @ 256-->sub_patch
python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=0 enable_flip=1 start_id=32 end_id=49

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=15 max_rot=125 enable_flip=1 start_id=32 end_id=49

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=126 max_rot=235 enable_flip=1 start_id=32 end_id=49

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 enable_rot=1 min_rot=236 max_rot=345 enable_flip=1 start_id=32 end_id=49

<a id="merging___32_49_25_6_"></a>
### merging       @ 32-49/256-->sub_patch
python3 mergeDatasets.py training_32_49_256_256_25_100_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

python3 mergeDatasets.py training_32_49_256_256_25_100_rot_126_235_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

python3 mergeDatasets.py training_32_49_256_256_25_100_rot_15_125_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

python3 mergeDatasets.py training_32_49_256_256_25_100_rot_236_345_flip training_32_49_256_256_25_100_rot_15_125_235_345_flip start_id=32 end_id=49

<a id="batch_all___25_6_"></a>
## batch_all       @ 256-->sub_patch
python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=256 patch_width=256 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

<a id="validation___25_6_"></a>
## validation       @ 256-->sub_patch
python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=256 patch_width=256 min_stride=256 max_stride=256 enable_rot=0 enable_flip=0 start_id=1 end_id=1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=256 patch_width=256 min_stride=256 max_stride=256 enable_rot=0 enable_flip=0 start_id=0 end_id=20 show_img=0

<a id="stitching___validation_256_"></a>
### stitching       @ validation/256-->sub_patch
python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_1_1_256_256_256_256 patch_height=256 patch_width=256 start_id=1 end_id=1 patch_seq_type=images show_img=1 stacked=1 method=1 resize_factor=0.5

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_1_1_256_256_256_256 patch_height=256 patch_width=256 start_id=1 end_id=1 patch_seq_type=labels_deeplab_xception show_img=1 stacked=1 method=1 resize_factor=0.5

<a id="videos___25_6_"></a>
## videos       @ 256-->sub_patch

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=256 patch_width=256 min_stride=256 max_stride=256 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

<a id="stitching___videos_256_"></a>
### stitching       @ videos/256-->sub_patch

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1 

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1

<a id="384_"></a>
# 384
<a id="40_160___38_4_"></a>
## 40/160       @ 384-->sub_patch
python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=40 max_stride=160 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=40 max_stride=160 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31


python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=40 max_stride=160 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="25_100___38_4_"></a>
## 25/100       @ 384-->sub_patch
python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=384 patch_width=384 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

<a id="validation___38_4_"></a>
## validation       @ 384-->sub_patch
python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=384 patch_width=384 min_stride=384 max_stride=384 enable_rot=0 enable_flip=0 start_id=0 end_id=-1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=384 patch_width=384 min_stride=384 max_stride=384 enable_rot=0 enable_flip=0 start_id=0 end_id=20


<a id="stitching___validation_384_"></a>
### stitching       @ validation/384-->sub_patch
python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_384_384_384_384 patch_height=384 patch_width=384 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

<a id="videos___38_4_"></a>
## videos       @ 384-->sub_patch
python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=384 patch_width=384 min_stride=384 max_stride=384 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

<a id="stitching___videos_384_"></a>
### stitching       @ videos/384-->sub_patch
python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_384_384_384_384 patch_height=384 patch_width=384 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1

<a id="vis___38_4_"></a>
## vis       @ 384-->sub_patch
<a id="unet___vis_38_4_"></a>
### unet       @ vis/384-->sub_patch
<a id="hml___unet_vis_384_"></a>
#### hml       @ unet/vis/384-->sub_patch
python3 visDataset.py --images_path=/data/617/images/training_256_256_100_200_rot_90_180_flip/images --labels_path=/data/617/images/training_256_256_100_200_rot_90_180_flip/labels --seg_path=/data/617/images/vgg_unet2_max_val_acc_training_256_256_100_200_rot_90_180_flip/predictions/raw --save_path=/data/617/images/vgg_unet2_max_val_acc_training_256_256_100_200_rot_90_180_flip/vis --n_classes=3 --start_id=0 --end_id=-1

python3 visDataset.py --images_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --seg_path=/data/617/images/vgg_unet2_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/vgg_unet2_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="weird___unet_vis_384_"></a>
#### weird       @ unet/vis/384-->sub_patch

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_256_256_256_256 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip img_ext=jpg show_img=0 method=1


python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_50_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=50 patch_seq_type=images show_img=1 method=1

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=vgg_unet2_max_val_acc_validation_0_563_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0 method=1
python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=fcn32_max_mean_acc_validation_0_563_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0 method=1

<a id="512_"></a>
# 512
<a id="40_160___51_2_"></a>
## 40/160       @ 512-->sub_patch
python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=50 max_stride=200 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=50 max_stride=200 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=50 max_stride=200 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="25_100___51_2_"></a>
## 25/100       @ 512-->sub_patch


python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=512 patch_width=512 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="validation___51_2_"></a>
## validation       @ 512-->sub_patch

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=-1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=20

<a id="stitching___validation_512_"></a>
### stitching       @ validation/512-->sub_patch

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_512_512_512_512 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

<a id="videos___51_2_"></a>
## videos       @ 512-->sub_patch

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=512 patch_width=512 min_stride=512 max_stride=512 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

<a id="stitching___videos_512_"></a>
### stitching       @ videos/512-->sub_patch
python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_512_512_512_512 patch_height=512 patch_width=512 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1

<a id="640_"></a>
# 640
<a id="64_256___64_0_"></a>
## 64/256       @ 640-->sub_patch
python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="non_aug___64_0_"></a>
## non_aug       @ 640-->sub_patch

<a id="0_3___non_aug_64_0_"></a>
### 0__3       @ non_aug/640-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=3 img_ext=tif

<a id="sel_2___0_3_non_aug_64_0_"></a>
#### sel-2       @ 0__3/non_aug/640-->sub_patch

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=2

<a id="sel_10___0_3_non_aug_64_0_"></a>
#### sel-10       @ 0__3/non_aug/640-->sub_patch

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=10

<a id="sel_100___0_3_non_aug_64_0_"></a>
#### sel-100       @ 0__3/non_aug/640-->sub_patch

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=100

<a id="sel_1000___0_3_non_aug_64_0_"></a>
#### sel-1000       @ 0__3/non_aug/640-->sub_patch

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=1000

<a id="sel_5000___0_3_non_aug_64_0_"></a>
#### sel-5000       @ 0__3/non_aug/640-->sub_patch

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_640_640 n_indices=5000

<a id="32_49___non_aug_64_0_"></a>
### 32_-49       @ non_aug/640-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

<a id="0_49___non_aug_64_0_"></a>
### 0__49       @ non_aug/640-->sub_patch

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=49 img_ext=tif

<a id="4_49___non_aug_64_0_"></a>
### 4__49       @ non_aug/640-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

<a id="entire_image___non_aug_64_0_"></a>
### entire_image       @ non_aug/640-->sub_patch

<a id="0_3___entire_image_non_aug_640_"></a>
#### 0-3       @ entire_image/non_aug/640-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=3 img_ext=tif

<a id="0_7___entire_image_non_aug_640_"></a>
#### 0-7       @ entire_image/non_aug/640-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=7 img_ext=tif

<a id="0_15___entire_image_non_aug_640_"></a>
#### 0-15       @ entire_image/non_aug/640-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=15 img_ext=tif

<a id="0_23___entire_image_non_aug_640_"></a>
#### 0-23       @ entire_image/non_aug/640-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=23 img_ext=tif

<a id="0_31___entire_image_non_aug_640_"></a>
#### 0-31       @ entire_image/non_aug/640-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=31 img_ext=tif

<a id="32_49___entire_image_non_aug_640_"></a>
#### 32-49       @ entire_image/non_aug/640-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

<a id="4_49___entire_image_non_aug_640_"></a>
#### 4-49       @ entire_image/non_aug/640-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

<a id="ablation___non_aug_64_0_"></a>
### ablation       @ non_aug/640-->sub_patch

<a id="0_3___ablation_non_aug_640_"></a>
#### 0__3       @ ablation/non_aug/640-->sub_patch

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=3

<a id="sel_2___ablation_non_aug_640_"></a>
#### sel-2       @ ablation/non_aug/640-->sub_patch

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=2

<a id="sel_2___ablation_non_aug_640__1"></a>
#### sel-2       @ ablation/non_aug/640-->sub_patch

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=2

<a id="sel_10___ablation_non_aug_640_"></a>
#### sel-10       @ ablation/non_aug/640-->sub_patch

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=10

<a id="sel_100___ablation_non_aug_640_"></a>
#### sel-100       @ ablation/non_aug/640-->sub_patch

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=100

<a id="sel_1000___ablation_non_aug_640_"></a>
#### sel-1000       @ ablation/non_aug/640-->sub_patch

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=1000

<a id="sel_5000___ablation_non_aug_640_"></a>
#### sel-5000       @ ablation/non_aug/640-->sub_patch

python3 selectiveDataset.py db_root_dir=/data/617/images src_dir=training_0_3_640_640_64_256_rot_15_345_4_flip n_indices=5000

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=7

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=15

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=23

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=64 max_stride=256 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

<a id="25_100___64_0_"></a>
## 25/100       @ 640-->sub_patch

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49


python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=640 patch_width=640 min_stride=100 max_stride=200 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0
python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

<a id="validation___64_0_"></a>
## validation       @ 640-->sub_patch

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=20

<a id="stitching___validation_640_"></a>
### stitching       @ validation/640-->sub_patch

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_563_640_640_640_640 patch_height=640 patch_width=640 start_id=0 end_id=-1 patch_seq_type=predictions show_img=0

<a id="videos___64_0_"></a>
## videos       @ 640-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001_3600 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00002_700_2500 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00020_2000_3800 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20161203_Deployment_1_YUN00001_900_2700 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20161203_Deployment_1_YUN00002_1800 patch_height=640 patch_width=640 min_stride=640 max_stride=640 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg


<a id="stitching___videos_640_"></a>
### stitching       @ videos/640-->sub_patch

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images/ seq_name=YUN00001 patch_seq_name=YUN00001_0_239_640_640_640_640 patch_height=640 patch_width=640 start_id=0 end_id=-1 patch_seq_type=images img_ext=jpg show_img=0 method=1


<a id="800_"></a>
# 800
python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=100 max_stride=200 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=20 show_img=0

<a id="80_320___80_0_"></a>
## 80/320       @ 800-->sub_patch
python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="non_aug___80_0_"></a>
## non_aug       @ 800-->sub_patch

<a id="0_3___non_aug_80_0_"></a>
### 0__3       @ non_aug/800-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=3 img_ext=tif

<a id="32_49___non_aug_80_0_"></a>
### 32__49       @ non_aug/800-->sub_patch

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

<a id="0_49___non_aug_80_0_"></a>
### 0__49       @ non_aug/800-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=49 img_ext=tif

<a id="4_49___non_aug_80_0_"></a>
### 4__49       @ non_aug/800-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

<a id="entire_image___non_aug_80_0_"></a>
### entire_image       @ non_aug/800-->sub_patch

<a id="32_49___entire_image_non_aug_800_"></a>
#### 32-49       @ entire_image/non_aug/800-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=32 end_id=49 img_ext=tif

<a id="4_49___entire_image_non_aug_800_"></a>
#### 4-49       @ entire_image/non_aug/800-->sub_patch

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=0 patch_width=0 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=4 end_id=49 img_ext=tif

<a id="ablation___non_aug_80_0_"></a>
### ablation       @ non_aug/800-->sub_patch

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=3

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=7

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=15

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=80 max_stride=320 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=23

<a id="25_100___80_0_"></a>
## 25/100       @ 800-->sub_patch

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=800 patch_width=800 min_stride=25 max_stride=100 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="video___80_0_"></a>
## video       @ 800-->sub_patch

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001_3600 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg


python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00002_700_2500 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00020_2000_3800 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20161203_Deployment_1_YUN00001_900_2700 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ../subPatchDataset.py db_root_dir=/data/617/images seq_name=20161203_Deployment_1_YUN00002_1800 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg




python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00002_2000 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00002_700 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20161201_YUN00002 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20170114_YUN00005 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

<a id="1000___sub_patc_h_"></a>
# 1000       @ sub_patch-->river_ice_segm

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=200 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 show_img=0

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=20 show_img=0


<a id="100_400___1000_"></a>
## 100/400       @ 1000-->sub_patch

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=400 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=49

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=400 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=0 end_id=31

python3 subPatchBatch.py db_root_dir=/data/617/images seq_name=training patch_height=1000 patch_width=1000 min_stride=100 max_stride=400 min_rot=15 max_rot=345 n_rot=4 enable_flip=1 start_id=32 end_id=49

<a id="video___1000_"></a>
## video       @ 1000-->sub_patch

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00002_2000 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20160122_YUN00002_700 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20161201_YUN00002 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20170114_YUN00005 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

<a id="1920x1080___video_1000_"></a>
### 1920x1080       @ video/1000-->sub_patch

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00001_1920x1080 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

python3 subPatchDataset.py db_root_dir=/data/617/images seq_name=YUN00002_1920x1080 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg









