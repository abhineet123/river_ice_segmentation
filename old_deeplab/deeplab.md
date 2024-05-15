<!-- MarkdownTOC -->

- [setup](#setup_)
- [256](#256_)
   - [build_data       @ 256](#build_data___25_6_)
   - [50       @ 256](#50___25_6_)
      - [output_stride_16       @ 50/256](#output_stride_16___50_256_)
         - [eval       @ output_stride_16/50/256](#eval___output_stride_16_50_25_6_)
         - [vis       @ output_stride_16/50/256](#vis___output_stride_16_50_25_6_)
         - [validation       @ output_stride_16/50/256](#validation___output_stride_16_50_25_6_)
         - [stitching       @ output_stride_16/50/256](#stitching___output_stride_16_50_25_6_)
      - [output_stride_8       @ 50/256](#output_stride_8___50_256_)
         - [eval       @ output_stride_8/50/256](#eval___output_stride_8_50_256_)
         - [vis       @ output_stride_8/50/256](#vis___output_stride_8_50_256_)
         - [validation       @ output_stride_8/50/256](#validation___output_stride_8_50_256_)
   - [32       @ 256](#32___25_6_)
      - [eval       @ 32/256](#eval___32_256_)
      - [vis       @ 32/256](#vis___32_256_)
      - [validation       @ 32/256](#validation___32_256_)
- [384](#384_)
   - [build_data       @ 384](#build_data___38_4_)
   - [50       @ 384](#50___38_4_)
      - [batch_size_6       @ 50/384](#batch_size_6___50_384_)
         - [eval       @ batch_size_6/50/384](#eval___batch_size_6_50_38_4_)
         - [vis       @ batch_size_6/50/384](#vis___batch_size_6_50_38_4_)
         - [validation       @ batch_size_6/50/384](#validation___batch_size_6_50_38_4_)
      - [batch_size_8       @ 50/384](#batch_size_8___50_384_)
         - [eval       @ batch_size_8/50/384](#eval___batch_size_8_50_38_4_)
         - [vis       @ batch_size_8/50/384](#vis___batch_size_8_50_38_4_)
   - [stride_8       @ 384](#stride_8___38_4_)
   - [32       @ 384](#32___38_4_)
      - [eval       @ 32/384](#eval___32_384_)
      - [vis       @ 32/384](#vis___32_384_)
      - [validation       @ 32/384](#validation___32_384_)
- [512](#512_)
   - [build_data       @ 512](#build_data___51_2_)
   - [50       @ 512](#50___51_2_)
      - [batch_size_6       @ 50/512](#batch_size_6___50_512_)
         - [eval       @ batch_size_6/50/512](#eval___batch_size_6_50_51_2_)
         - [vis       @ batch_size_6/50/512](#vis___batch_size_6_50_51_2_)
         - [validation       @ batch_size_6/50/512](#validation___batch_size_6_50_51_2_)
      - [batch_size_2       @ 50/512](#batch_size_2___50_512_)
         - [vis       @ batch_size_2/50/512](#vis___batch_size_2_50_51_2_)
         - [validation       @ batch_size_2/50/512](#validation___batch_size_2_50_51_2_)
   - [32       @ 512](#32___51_2_)
      - [eval       @ 32/512](#eval___32_512_)
      - [vis       @ 32/512](#vis___32_512_)
      - [validation       @ 32/512](#validation___32_512_)
- [640](#640_)
   - [build_data       @ 640](#build_data___64_0_)
      - [0_3       @ build_data/640](#0_3___build_data_640_)
      - [0_3_non_aug       @ build_data/640](#0_3_non_aug___build_data_640_)
         - [all       @ 0_3_non_aug/build_data/640](#all___0_3_non_aug_build_data_640_)
         - [sel_2       @ 0_3_non_aug/build_data/640](#sel_2___0_3_non_aug_build_data_640_)
         - [sel_10       @ 0_3_non_aug/build_data/640](#sel_10___0_3_non_aug_build_data_640_)
         - [sel_100       @ 0_3_non_aug/build_data/640](#sel_100___0_3_non_aug_build_data_640_)
         - [sel_1000       @ 0_3_non_aug/build_data/640](#sel_1000___0_3_non_aug_build_data_640_)
      - [0_7       @ build_data/640](#0_7___build_data_640_)
      - [0_15       @ build_data/640](#0_15___build_data_640_)
      - [0_23       @ build_data/640](#0_23___build_data_640_)
      - [0_31       @ build_data/640](#0_31___build_data_640_)
      - [0_49       @ build_data/640](#0_49___build_data_640_)
      - [32_49       @ build_data/640](#32_49___build_data_640_)
      - [4_49_no_aug       @ build_data/640](#4_49_no_aug___build_data_640_)
      - [32_49_no_aug       @ build_data/640](#32_49_no_aug___build_data_640_)
      - [validation_0_20       @ build_data/640](#validation_0_20___build_data_640_)
      - [YUN00001_0_239       @ build_data/640](#yun00001_0_239___build_data_640_)
      - [YUN00001_3600       @ build_data/640](#yun00001_3600___build_data_640_)
      - [YUN00001_0_8999       @ build_data/640](#yun00001_0_8999___build_data_640_)
      - [20160122_YUN00002_700_2500       @ build_data/640](#20160122_yun00002_700_2500___build_data_640_)
      - [20160122_YUN00020_2000_3800       @ build_data/640](#20160122_yun00020_2000_3800___build_data_640_)
      - [20161203_Deployment_1_YUN00001_900_2700       @ build_data/640](#20161203_deployment_1_yun00001_900_2700___build_data_640_)
      - [20161203_Deployment_1_YUN00002_1800       @ build_data/640](#20161203_deployment_1_yun00002_1800___build_data_640_)
   - [50       @ 640](#50___64_0_)
      - [eval       @ 50/640](#eval___50_640_)
      - [vis       @ 50/640](#vis___50_640_)
         - [stitching       @ vis/50/640](#stitching___vis_50_640_)
      - [validation       @ 50/640](#validation___50_640_)
         - [stitching       @ validation/50/640](#stitching___validation_50_64_0_)
         - [vis       @ validation/50/640](#vis___validation_50_64_0_)
         - [zip       @ validation/50/640](#zip___validation_50_64_0_)
      - [video       @ 50/640](#video___50_640_)
         - [stitching       @ video/50/640](#stitching___video_50_640_)
   - [4       @ 640](#4___64_0_)
      - [continue_40787       @ 4/640](#continue_40787___4_64_0_)
      - [vis       @ 4/640](#vis___4_64_0_)
      - [no_aug       @ 4/640](#no_aug___4_64_0_)
         - [stitched       @ no_aug/4/640](#stitched___no_aug_4_640_)
      - [no_aug_4_49       @ 4/640](#no_aug_4_49___4_64_0_)
         - [stitched       @ no_aug_4_49/4/640](#stitched___no_aug_4_49_4_64_0_)
   - [8       @ 640](#8___64_0_)
      - [vis       @ 8/640](#vis___8_64_0_)
      - [no_aug       @ 8/640](#no_aug___8_64_0_)
         - [stitched       @ no_aug/8/640](#stitched___no_aug_8_640_)
   - [16       @ 640](#16___64_0_)
      - [vis       @ 16/640](#vis___16_640_)
      - [no_aug       @ 16/640](#no_aug___16_640_)
         - [stitched       @ no_aug/16/640](#stitched___no_aug_16_64_0_)
   - [16_rt       @ 640](#16_rt___64_0_)
      - [no_aug       @ 16_rt/640](#no_aug___16_rt_64_0_)
         - [stitched       @ no_aug/16_rt/640](#stitched___no_aug_16_rt_640_)
   - [16_rt_3       @ 640](#16_rt_3___64_0_)
      - [no_aug       @ 16_rt_3/640](#no_aug___16_rt_3_64_0_)
         - [stitched       @ no_aug/16_rt_3/640](#stitched___no_aug_16_rt_3_640_)
   - [24       @ 640](#24___64_0_)
      - [vis       @ 24/640](#vis___24_640_)
      - [no_aug       @ 24/640](#no_aug___24_640_)
         - [stitched       @ no_aug/24/640](#stitched___no_aug_24_64_0_)
   - [32_orig       @ 640](#32_orig___64_0_)
      - [vis_png       @ 32_orig/640](#vis_png___32_orig_64_0_)
         - [20160122_YUN00002_700_2500       @ vis_png/32_orig/640](#20160122_yun00002_700_2500___vis_png_32_orig_64_0_)
         - [20160122_YUN00020_2000_3800       @ vis_png/32_orig/640](#20160122_yun00020_2000_3800___vis_png_32_orig_64_0_)
         - [20161203_Deployment_1_YUN00001_900_2700       @ vis_png/32_orig/640](#20161203_deployment_1_yun00001_900_2700___vis_png_32_orig_64_0_)
         - [20161203_Deployment_1_YUN00002_1800       @ vis_png/32_orig/640](#20161203_deployment_1_yun00002_1800___vis_png_32_orig_64_0_)
         - [YUN00001_3600       @ vis_png/32_orig/640](#yun00001_3600___vis_png_32_orig_64_0_)
   - [32       @ 640](#32___64_0_)
      - [vis       @ 32/640](#vis___32_640_)
      - [no_aug       @ 32/640](#no_aug___32_640_)
         - [stitched       @ no_aug/32/640](#stitched___no_aug_32_64_0_)
      - [YUN00001       @ 32/640](#yun00001___32_640_)
      - [YUN00001_3600       @ 32/640](#yun00001_3600___32_640_)
   - [4__non_aug       @ 640](#4_non_aug___64_0_)
      - [sel_2       @ 4__non_aug/640](#sel_2___4_non_aug_64_0_)
      - [sel_10       @ 4__non_aug/640](#sel_10___4_non_aug_64_0_)
      - [sel_100       @ 4__non_aug/640](#sel_100___4_non_aug_64_0_)
      - [sel_1000       @ 4__non_aug/640](#sel_1000___4_non_aug_64_0_)
         - [rt       @ sel_1000/4__non_aug/640](#rt___sel_1000_4_non_aug_64_0_)
         - [rt2       @ sel_1000/4__non_aug/640](#rt2___sel_1000_4_non_aug_64_0_)
- [800](#800_)
   - [build_data       @ 800](#build_data___80_0_)
      - [50       @ build_data/800](#50___build_data_800_)
      - [32       @ build_data/800](#32___build_data_800_)
      - [18_-_test       @ build_data/800](#18_test___build_data_800_)
      - [validation_0_20_800_800_800_800       @ build_data/800](#validation_0_20_800_800_800_800___build_data_800_)
      - [YUN00001_0_239_800_800_800_800       @ build_data/800](#yun00001_0_239_800_800_800_800___build_data_800_)
      - [4       @ build_data/800](#4___build_data_800_)
   - [50       @ 800](#50___80_0_)
      - [eval       @ 50/800](#eval___50_800_)
      - [vis       @ 50/800](#vis___50_800_)
         - [stitching       @ vis/50/800](#stitching___vis_50_800_)
      - [validation       @ 50/800](#validation___50_800_)
         - [stitching       @ validation/50/800](#stitching___validation_50_80_0_)
         - [vis       @ validation/50/800](#vis___validation_50_80_0_)
   - [4       @ 800](#4___80_0_)

<!-- /MarkdownTOC -->

<a id="setup_"></a>
# setup 

pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp35-cp35m-linux_x86_64.whl
pip2 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp27-none-linux_x86_64.whl
python remove_gt_colormap.py --original_gt_folder="/home/abhineet/N/Datasets/617/images/training_256_256_25_100_rot_15_90_flip/labels" --output_dir="/home/abhineet/N/Datasets/617/images/training_256_256_25_100_rot_15_90_flip/labels_raw"

lftp -e "cls -1 > _list; exit" "http://download.tensorflow.org/models"

<a id="256_"></a>
# 256

<a id="build_data___25_6_"></a>
## build_data       @ 256

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_49_256_256_25_100_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_31_256_256_25_100_rot_15_125_235_345_flip --image_format=png --label_format=png --output_dir=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_32_49_256_256_25_100_rot_15_125_235_345_flip --image_format=png --label_format=png --output_dir=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=validation_0_20_256_256_256_256 --image_format=png --label_format=png --output_dir=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=YUN00001_0_239_256_256_256_256 --image_format=png --label_format=png --output_dir=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip --create_dummy_labels=1


<a id="50___25_6_"></a>
## 50       @ 256

<a id="output_stride_16___50_256_"></a>
### output_stride_16       @ 50/256

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=256 --train_crop_size=256 --train_batch_size=10 --dataset=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49 --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --train_split=training_0_49_256_256_25_100_rot_15_345_4_flip --num_clones=2


<a id="eval___output_stride_16_50_25_6_"></a>
#### eval       @ output_stride_16/50/256

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=256 --eval_crop_size=256 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49 --eval_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49/eval --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --eval_split=training_32_49_256_256_25_100_rot_15_125_235_345_flip

miou_1.0[0.802874804]


<a id="vis___output_stride_16_50_25_6_"></a>
#### vis       @ output_stride_16/50/256

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49 --vis_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49/vis/training_32_49_256_256_25_100_rot_15_125_235_345_flip --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --vis_split=training_32_49_256_256_25_100_rot_15_125_235_345_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images --labels_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels --seg_path=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49/vis/training_32_49_256_256_25_100_rot_15_125_235_345_flip/raw --save_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation___output_stride_16_50_25_6_"></a>
#### validation       @ output_stride_16/50/256


CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=256 --vis_crop_size=256 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49 --vis_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49/vis --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --vis_split=validation_0_20_256_256_256_256 --vis_batch_size=10 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="stitching___output_stride_16_50_25_6_"></a>
#### stitching       @ output_stride_16/50/256

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_20_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=10 patch_seq_type=labels_deeplab_xception show_img=0 stacked=1 method=1

python3 stitchSubPatchDataset.py db_root_dir=/data/617/images seq_name=validation patch_seq_name=validation_0_20_256_256_256_256 patch_height=256 patch_width=256 start_id=0 end_id=10 patch_seq_type=images show_img=0 stacked=1 method=1


<a id="output_stride_8___50_256_"></a>
### output_stride_8       @ 50/256

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=12 --atrous_rates=24 --atrous_rates=36 --output_stride=8 --decoder_output_stride=4 --train_crop_size=256 --train_crop_size=256 --train_batch_size=10 --dataset=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8 --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --train_split=training_0_49_256_256_25_100_rot_15_345_4_flip --num_clones=2

<a id="eval___output_stride_8_50_256_"></a>
#### eval       @ output_stride_8/50/256

CUDA_VISIBLE_DEVICES=0 python3 eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=12 --atrous_rates=24 --atrous_rates=36 --output_stride=8 --decoder_output_stride=4 --eval_crop_size=256 --eval_crop_size=256 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8 --eval_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8/eval --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --eval_split=training_32_49_256_256_25_100_rot_15_125_235_345_flip

miou_1.0[0.799446404]
miou_1.0[0.799446404]

<a id="vis___output_stride_8_50_256_"></a>
#### vis       @ output_stride_8/50/256

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8 --vis_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8/vis/training_32_49_256_256_25_100_rot_15_125_235_345_flip --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --vis_split=training_32_49_256_256_25_100_rot_15_125_235_345_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

zr vis_xception_0_49_stride8_training_32_49_256_256_25_100_rot_15_125_235_345_flip log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8/vis/training_32_49_256_256_25_100_rot_15_125_235_345_flip

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images --labels_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels --seg_path=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8/vis/training_32_49_256_256_25_100_rot_15_125_235_345_flip/raw --save_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/vis_xception_0_49_stride8 --n_classes=3 --start_id=0 --end_id=-1


<a id="validation___output_stride_8_50_256_"></a>
#### validation       @ output_stride_8/50/256

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=12 --atrous_rates=24 --atrous_rates=36 --output_stride=8 --decoder_output_stride=4 --vis_crop_size=256 --vis_crop_size=256 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8 --vis_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49_stride8/vis --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --vis_split=validation_0_20_256_256_256_256 --vis_batch_size=10 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="32___25_6_"></a>
## 32       @ 256

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=256 --train_crop_size=256 --train_batch_size=10 --dataset=training_0_31_49_256_256_25_100_rot_15_125_235_345_flip --tf_initial_checkpoint=pre_trained/xception_0_31/model.ckpt --train_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31 --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --train_split=training_0_31_256_256_25_100_rot_15_125_235_345_flip


<a id="eval___32_256_"></a>
### eval       @ 32/256

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=256 --eval_crop_size=256 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31 --eval_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31/eval --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --eval_split=training_32_49_256_256_25_100_rot_15_125_235_345_flip


<a id="vis___32_256_"></a>
### vis       @ 32/256

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31 --vis_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31/vis/training_32_49_256_256_25_100_rot_15_125_235_345_flip --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --vis_split=training_32_49_256_256_25_100_rot_15_125_235_345_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images --labels_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels --seg_path=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31/vis/training_32_49_256_256_25_100_rot_15_125_235_345_flip/raw --save_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/vis_xception_0_31 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation___32_256_"></a>
### validation       @ 32/256

python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=256 --vis_crop_size=256 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31 --vis_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_31/vis --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --vis_split=validation_0_20_256_256_256_256 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

miou_1.0[0.735878468]

<a id="384_"></a>
# 384

<a id="build_data___38_4_"></a>
## build_data       @ 384

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_49_384_384_25_100_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_384_384_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_31_384_384_25_100_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_384_384_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_32_49_384_384_25_100_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_384_384_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=validation_0_20_384_384_384_384 --image_format=png --label_format=png --output_dir=training_0_31_49_384_384_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=YUN00001_0_239_384_384_384_384 --image_format=png --label_format=png --output_dir=training_0_31_49_384_384_25_100_rot_15_345_4_flip --create_dummy_labels=1


<a id="50___38_4_"></a>
## 50       @ 384

<a id="batch_size_6___50_384_"></a>
### batch_size_6       @ 50/384

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=384 --train_crop_size=384 --train_batch_size=6 --dataset=training_0_31_49_384_384_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49 --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_49_384_384_25_100_rot_15_345_4_flip --num_clones=2

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=384 --train_crop_size=384 --train_batch_size=6 --dataset=training_0_31_49_384_384_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_attempt_2 --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_49_384_384_25_100_rot_15_345_4_flip --num_clones=2


<a id="eval___batch_size_6_50_38_4_"></a>
#### eval       @ batch_size_6/50/384

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=384 --eval_crop_size=384 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49 --eval_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49/eval --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_384_384_25_100_rot_15_345_4_flip --eval_batch_size=5


<a id="vis___batch_size_6_50_38_4_"></a>
#### vis       @ batch_size_6/50/384

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49/vis/training_32_49_384_384_25_100_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_384_384_25_100_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49/vis/training_32_49_384_384_25_100_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation___batch_size_6_50_38_4_"></a>
#### validation       @ batch_size_6/50/384

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=384 --vis_crop_size=384 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49/vis --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_384_384_384_384 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="batch_size_8___50_384_"></a>
### batch_size_8       @ 50/384

CUDA_VISIBLE_DEVICES=0 python3 deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=384 --train_crop_size=384 --train_batch_size=8 --dataset=training_0_31_49_384_384_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_batch_8 --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_49_384_384_25_100_rot_15_345_4_flip --num_clones=2

<a id="eval___batch_size_8_50_38_4_"></a>
#### eval       @ batch_size_8/50/384

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=384 --eval_crop_size=384 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_batch_8 --eval_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_batch_8/eval --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_384_384_25_100_rot_15_345_4_flip --eval_batch_size=5


<a id="vis___batch_size_8_50_38_4_"></a>
#### vis       @ batch_size_8/50/384

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_batch_8 --vis_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_batch_8/vis/training_32_49_384_384_25_100_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_384_384_25_100_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

zrq xception_0_49_batch_8_training_32_49_384_384_25_100_rot_15_345_4_flip log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_batch_8

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_batch_8/vis/training_32_49_384_384_25_100_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/vis_xception_0_49_batch_8 --n_classes=3 --start_id=0 --end_id=-1

<a id="stride_8___38_4_"></a>
## stride_8       @ 384

CUDA_VISIBLE_DEVICES=1 python3 deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=12 --atrous_rates=24 --atrous_rates=36 --output_stride=8 --decoder_output_stride=4 --train_crop_size=384 --train_crop_size=384 --train_batch_size=6 --dataset=training_0_31_49_384_384_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49_stride_8 --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_49_384_384_25_100_rot_15_345_4_flip --num_clones=2

<a id="32___38_4_"></a>
## 32       @ 384

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=384 --train_crop_size=384 --train_batch_size=6 --dataset=training_0_31_49_384_384_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31 --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_31_384_384_25_100_rot_15_345_4_flip --num_clones=2

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=384 --train_crop_size=384 --train_batch_size=6 --dataset=training_0_31_49_384_384_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31_attempt2 --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_31_384_384_25_100_rot_15_345_4_flip --num_clones=2

<a id="eval___32_384_"></a>
### eval       @ 32/384

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=384 --eval_crop_size=384 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31 --eval_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31/eval --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_384_384_25_100_rot_15_345_4_flip --eval_batch_size=5


<a id="vis___32_384_"></a>
### vis       @ 32/384

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31/vis/training_32_49_384_384_25_100_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_384_384_25_100_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31/vis/training_32_49_384_384_25_100_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/vis_xception_0_31 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation___32_384_"></a>
### validation       @ 32/384

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=384 --vis_crop_size=384 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_31/vis --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_384_384_384_384 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="512_"></a>
# 512

<a id="build_data___51_2_"></a>
## build_data       @ 512

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_49_512_512_25_100_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_512_512_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_31_512_512_25_100_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_512_512_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_32_49_512_512_25_100_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_512_512_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=validation_0_20_512_512_512_512 --image_format=png --label_format=png --output_dir=training_0_31_49_512_512_25_100_rot_15_345_4_flip

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=YUN00001_0_239_512_512_512_512 --image_format=png --label_format=png --output_dir=training_0_31_49_512_512_25_100_rot_15_345_4_flip --create_dummy_labels=1


<a id="50___51_2_"></a>
## 50       @ 512

<a id="batch_size_6___50_512_"></a>
### batch_size_6       @ 50/512

CUDA_VISIBLE_DEVICES=1 python3 deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=512 --train_crop_size=512 --train_batch_size=6 --dataset=training_0_31_49_512_512_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49 --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_49_512_512_25_100_rot_15_345_4_flip --num_clones=1


<a id="eval___batch_size_6_50_51_2_"></a>
#### eval       @ batch_size_6/50/512

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=512 --eval_crop_size=512 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49 --eval_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49/eval --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_512_512_25_100_rot_15_345_4_flip --eval_batch_size=5

miou_1.0[0.723538] 


<a id="vis___batch_size_6_50_51_2_"></a>
#### vis       @ batch_size_6/50/512

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49/vis/training_32_49_512_512_25_100_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_512_512_25_100_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49/vis/training_32_49_512_512_25_100_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation___batch_size_6_50_51_2_"></a>
#### validation       @ batch_size_6/50/512

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=512 --vis_crop_size=512 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49/vis --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_512_512_512_512 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="batch_size_2___50_512_"></a>
### batch_size_2       @ 50/512

CUDA_VISIBLE_DEVICES=1 python3 deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=512 --train_crop_size=512 --train_batch_size=2 --dataset=training_0_31_49_512_512_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49_batch_2 --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_49_512_512_25_100_rot_15_345_4_flip --num_clones=1

<a id="vis___batch_size_2_50_51_2_"></a>
#### vis       @ batch_size_2/50/512

CUDA_VISIBLE_DEVICES=1 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49_batch_2 --vis_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49_batch_2/vis/training_32_49_512_512_25_100_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_512_512_25_100_rot_15_345_4_flip --vis_batch_size=10 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49_batch_2/vis/training_32_49_512_512_25_100_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/vis_xception_0_49_batch_2 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation___batch_size_2_50_51_2_"></a>
#### validation       @ batch_size_2/50/512

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=512 --vis_crop_size=512 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49_batch_2 --vis_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49_batch_2/vis --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_512_512_512_512 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0


<a id="32___51_2_"></a>
## 32       @ 512

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=512 --train_crop_size=512 --train_batch_size=6 --dataset=training_0_31_49_512_512_25_100_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31 --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --train_split=training_0_31_512_512_25_100_rot_15_345_4_flip --num_clones=2

<a id="eval___32_512_"></a>
### eval       @ 32/512

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=512 --eval_crop_size=512 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31 --eval_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31/eval --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_512_512_25_100_rot_15_345_4_flip --eval_batch_size=5

<a id="vis___32_512_"></a>
### vis       @ 32/512

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31/vis/training_32_49_512_512_25_100_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_512_512_25_100_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31/vis/training_32_49_512_512_25_100_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/vis_xception_0_31 --n_classes=3 --start_id=0 --end_id=-1

<a id="validation___32_512_"></a>
### validation       @ 32/512

python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=512 --vis_crop_size=512 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_31/vis --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_512_512_512_512 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="640_"></a>
# 640

<a id="build_data___64_0_"></a>
## build_data       @ 640

<a id="0_3___build_data_640_"></a>
### 0_3       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_3_640_640_64_256_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="0_3_non_aug___build_data_640_"></a>
### 0_3_non_aug       @ build_data/640

<a id="all___0_3_non_aug_build_data_640_"></a>
#### all       @ 0_3_non_aug/build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_3_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="sel_2___0_3_non_aug_build_data_640_"></a>
#### sel_2       @ 0_3_non_aug/build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_3_640_640_640_640_sel_2 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="sel_10___0_3_non_aug_build_data_640_"></a>
#### sel_10       @ 0_3_non_aug/build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_3_640_640_640_640_sel_10 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="sel_100___0_3_non_aug_build_data_640_"></a>
#### sel_100       @ 0_3_non_aug/build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_3_640_640_640_640_sel_100 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="sel_1000___0_3_non_aug_build_data_640_"></a>
#### sel_1000       @ 0_3_non_aug/build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_3_640_640_640_640_sel_1000 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="0_7___build_data_640_"></a>
### 0_7       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_7_640_640_64_256_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="0_15___build_data_640_"></a>
### 0_15       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_15_640_640_64_256_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="0_23___build_data_640_"></a>
### 0_23       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_23_640_640_64_256_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="0_31___build_data_640_"></a>
### 0_31       @ build_data/640

CUDA_VISIBLE_DEVICES=1 python2 datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_31_640_640_64_256_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="0_49___build_data_640_"></a>
### 0_49       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_49_640_640_64_256_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="32_49___build_data_640_"></a>
### 32_49       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_32_49_640_640_64_256_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="4_49_no_aug___build_data_640_"></a>
### 4_49_no_aug       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_4_49_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="32_49_no_aug___build_data_640_"></a>
### 32_49_no_aug       @ build_data/640

CUDA_VISIBLE_DEVICES=0 python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_32_49_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="validation_0_20___build_data_640_"></a>
### validation_0_20       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=validation_0_20_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip

<a id="yun00001_0_239___build_data_640_"></a>
### YUN00001_0_239       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=YUN00001_0_239_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip --create_dummy_labels=1

<a id="yun00001_3600___build_data_640_"></a>
### YUN00001_3600       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=YUN00001_3600_0_3599_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip --create_dummy_labels=0

<a id="yun00001_0_8999___build_data_640_"></a>
### YUN00001_0_8999       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=YUN00001_0_8999_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip --create_dummy_labels=0

<a id="20160122_yun00002_700_2500___build_data_640_"></a>
### 20160122_YUN00002_700_2500       @ build_data/640

python datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=20160122_YUN00002_700_2500_0_1799_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip --create_dummy_labels=0

<a id="20160122_yun00020_2000_3800___build_data_640_"></a>
### 20160122_YUN00020_2000_3800       @ build_data/640

CUDA_VISIBLE_DEVICES= python3 datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=20160122_YUN00020_2000_3800_0_1799_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip --create_dummy_labels=0

<a id="20161203_deployment_1_yun00001_900_2700___build_data_640_"></a>
### 20161203_Deployment_1_YUN00001_900_2700       @ build_data/640

CUDA_VISIBLE_DEVICES= python3 datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=20161203_Deployment_1_YUN00001_900_2700_0_1799_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip --create_dummy_labels=0

<a id="20161203_deployment_1_yun00002_1800___build_data_640_"></a>
### 20161203_Deployment_1_YUN00002_1800       @ build_data/640

CUDA_VISIBLE_DEVICES= python3 datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=20161203_Deployment_1_YUN00002_1800_0_1799_640_640_640_640 --image_format=png --label_format=png --output_dir=training_0_31_49_640_640_64_256_rot_15_345_4_flip --create_dummy_labels=0



<a id="50___64_0_"></a>
## 50       @ 640

CUDA_VISIBLE_DEVICES=1 python3 deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_49_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="eval___50_640_"></a>
### eval       @ 50/640

CUDA_VISIBLE_DEVICES=1 python3 eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=640 --eval_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49 --eval_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/eval --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_640_640_64_256_rot_15_345_4_flip --eval_batch_size=5

miou_1.0[0.739234447]

<a id="vis___50_640_"></a>
### vis       @ 50/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/training_32_49_640_640_64_256_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_64_256_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1

<a id="stitching___vis_50_640_"></a>
#### stitching       @ vis/50/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/training_32_49_640_640_64_256_rot_15_345_4_flip/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/training_32_49_640_640_64_256_rot_15_345_4_flip patch_height=640 patch_width=640 start_id=32 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=1

<a id="validation___50_640_"></a>
### validation       @ 50/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_20_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="stitching___validation_50_64_0_"></a>
#### stitching       @ validation/50/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_563_640_640_640_640/segmentation_results stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/validation_0_20_640_640_640_640 patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_563_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/validation_0_20_640_640_640_640/raw patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1 normalize_patches=1

<a id="vis___validation_50_64_0_"></a>
#### vis       @ validation/50/640

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1

<a id="zip___validation_50_64_0_"></a>
#### zip       @ validation/50/640

zrb training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_49_validation_0_20_640_640_640_640_1_25 log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_20_640_640_640_640/segmentation_results/img_XXX_* 1 25

training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_49_validation_0_20_640_640_640_640_1_25_grs_201804221134.zip

zrb training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_49_validation_0_20_640_640_640_640_raw_1_25 log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_20_640_640_640_640/raw/img_XXX_* 1 25

training_0_31_49_640_640_64_256_rot_15_345_4_flip_xception_0_49_validation_0_20_640_640_640_640_raw_1_25_grs_201804221136.zip 

zrb nazio deeplab/results/log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_20_640_640_640_640/segmentation_results/img_XXX_* 1 10 -j

<a id="video___50_640_"></a>
### video       @ 50/640

CUDA_VISIBLE_DEVICES=1 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/YUN00001_0_239_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=YUN00001_0_239_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="stitching___video_50_640_"></a>
#### stitching       @ video/50/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/YUN00001_0_239_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/YUN00001_0_239_640_640_640_640/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=1


python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_563_640_640_640_640/segmentation_results stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/validation_0_20_640_640_640_640 patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1


<a id="4___64_0_"></a>
## 4       @ 640

CUDA_VISIBLE_DEVICES=2 python3 deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="continue_40787___4_64_0_"></a>
### continue_40787       @ 4/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/model.ckpt-40787 --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="vis___4_64_0_"></a>
### vis       @ 4/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_64_256_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_64_256_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug___4_64_0_"></a>
### no_aug       @ 4/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched___no_aug_4_640_"></a>
#### stitched       @ no_aug/4/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug_4_49___4_64_0_"></a>
### no_aug_4_49       @ 4/640

CUDA_VISIBLE_DEVICES=1 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_4_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_4_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_4_49_640_640_640_640/images --labels_path=/data/617/images/training_4_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_4_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_4_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched___no_aug_4_49_4_64_0_"></a>
#### stitched       @ no_aug_4_49/4/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_4_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_4_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_4_49/images --labels_path=/data/617/images/training_4_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_4_49/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_4_49/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="8___64_0_"></a>
## 8       @ 640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_7_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="vis___8_64_0_"></a>
### vis       @ 8/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_64_256_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_64_256_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug___8_64_0_"></a>
### no_aug       @ 8/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched___no_aug_8_640_"></a>
#### stitched       @ no_aug/8/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png --labels_path=/data/617/images/training_32_49/labels

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="16___64_0_"></a>
## 16       @ 640

CUDA_VISIBLE_DEVICES=2 python3 deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_15_640_640_64_256_rot_15_345_4_flip --num_clones=1


<a id="vis___16_640_"></a>
### vis       @ 16/640

CUDA_VISIBLE_DEVICES=2 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_64_256_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_64_256_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug___16_640_"></a>
### no_aug       @ 16/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched___no_aug_16_64_0_"></a>
#### stitched       @ no_aug/16/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png labels_path=/data/617/images/training_32_49/labels

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="16_rt___64_0_"></a>
## 16_rt       @ 640

CUDA_VISIBLE_DEVICES=2 python3 deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=3 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_15_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="no_aug___16_rt_64_0_"></a>
### no_aug       @ 16_rt/640

CUDA_VISIBLE_DEVICES=1 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched___no_aug_16_rt_640_"></a>
#### stitched       @ no_aug/16_rt/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png labels_path=/data/617/images/training_32_49/labels

<a id="16_rt_3___64_0_"></a>
## 16_rt_3       @ 640

CUDA_VISIBLE_DEVICES=2 python3 deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=3 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt_3 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_15_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="no_aug___16_rt_3_64_0_"></a>
### no_aug       @ 16_rt_3/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt_3 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt_3/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt_3/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt_3/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched___no_aug_16_rt_3_640_"></a>
#### stitched       @ no_aug/16_rt_3/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images labels_path=/data/617/images/training_32_49/labels img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt_3/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15_rt_3/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png labels_path=/data/617/images/training_32_49/labels

<a id="24___64_0_"></a>
## 24       @ 640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_23_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="vis___24_640_"></a>
### vis       @ 24/640

CUDA_VISIBLE_DEVICES=2 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_64_256_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_64_256_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug___24_640_"></a>
### no_aug       @ 24/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched___no_aug_24_64_0_"></a>
#### stitched       @ no_aug/24/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="32_orig___64_0_"></a>
## 32_orig       @ 640

python deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=6 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_31_640_640_64_256_rot_15_345_4_flip --num_clones=2

python eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=640 --eval_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --eval_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/eval --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_640_640_64_256_rot_15_345_4_flip --eval_batch_size=5

python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/vis --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="vis_png___32_orig_64_0_"></a>
### vis_png       @ 32_orig/640

<a id="20160122_yun00002_700_2500___vis_png_32_orig_64_0_"></a>
#### 20160122_YUN00002_700_2500       @ vis_png/32_orig/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00002_700_2500/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20160122_YUN00002_700_2500_0_1799_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20160122_YUN00002_700_2500 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=png


<a id="20160122_yun00020_2000_3800___vis_png_32_orig_64_0_"></a>
#### 20160122_YUN00020_2000_3800       @ vis_png/32_orig/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00020_2000_3800/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20160122_YUN00020_2000_3800_0_1799_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20160122_YUN00020_2000_3800 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=png


<a id="20161203_deployment_1_yun00001_900_2700___vis_png_32_orig_64_0_"></a>
#### 20161203_Deployment_1_YUN00001_900_2700       @ vis_png/32_orig/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20161203_Deployment_1_YUN00001_900_2700/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20161203_Deployment_1_YUN00001_900_2700_0_1799_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20161203_Deployment_1_YUN00001_900_2700 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=png


<a id="20161203_deployment_1_yun00002_1800___vis_png_32_orig_64_0_"></a>
#### 20161203_Deployment_1_YUN00002_1800       @ vis_png/32_orig/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20161203_Deployment_1_YUN00002_1800/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20161203_Deployment_1_YUN00002_1800_0_1799_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/20161203_Deployment_1_YUN00002_1800 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=png

<a id="yun00001_3600___vis_png_32_orig_64_0_"></a>
#### YUN00001_3600       @ vis_png/32_orig/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001_3600_0_3599_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001_3600 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=png


<a id="32___64_0_"></a>
## 32       @ 640

CUDA_VISIBLE_DEVICES=1 python3 deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_31_640_640_64_256_rot_15_345_4_flip --num_clones=1

<a id="vis___32_640_"></a>
### vis       @ 32/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49_640_640_64_256_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_64_256_rot_15_345_4_flip --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug___32_640_"></a>
### no_aug       @ 32/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49_640_640_640_640/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched___no_aug_32_64_0_"></a>
#### stitched       @ no_aug/32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49/raw --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/training_32_49/vis --n_classes=3 --start_id=0 --end_id=-1 --normalize_labels=1

<a id="yun00001___32_640_"></a>
### YUN00001       @ 32/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001_0_8999_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=YUN00001_0_8999_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001_0_8999_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png out_ext=mkv width=1920 height=1080

<a id="yun00001_3600___32_640_"></a>
### YUN00001_3600       @ 32/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001_3600_0_3599_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=YUN00001_3600_0_3599_640_640_640_640 --vis_batch_size=25 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001_3600_0_3599_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_31/YUN00001_3600 patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=mkv width=1920 height=1080

<a id="4_non_aug___64_0_"></a>
## 4__non_aug       @ 640

<a id="sel_2___4_non_aug_64_0_"></a>
### sel_2       @ 4__non_aug/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_training_0_3_640_640_640_640_sel_2 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_640_640_sel_2 --num_clones=1

<a id="sel_10___4_non_aug_64_0_"></a>
### sel_10       @ 4__non_aug/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_training_0_3_640_640_640_640_sel_10 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_640_640_sel_10 --num_clones=1

<a id="sel_100___4_non_aug_64_0_"></a>
### sel_100       @ 4__non_aug/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_training_0_3_640_640_640_640_sel_100 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_640_640_sel_100 --num_clones=1

<a id="sel_1000___4_non_aug_64_0_"></a>
### sel_1000       @ 4__non_aug/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_training_0_3_640_640_640_640_sel_1000 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_640_640_sel_1000 --num_clones=1

<a id="rt___sel_1000_4_non_aug_64_0_"></a>
#### rt       @ sel_1000/4__non_aug/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_training_0_3_640_640_640_640_sel_1000_rt --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_640_640_sel_1000 --num_clones=1

<a id="rt2___sel_1000_4_non_aug_64_0_"></a>
#### rt2       @ sel_1000/4__non_aug/640

CUDA_VISIBLE_DEVICES=0 python3 deeplab_train.py --logtostderr --training_number_of_steps=1000000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=640 --train_crop_size=640 --train_batch_size=2 --dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_training_0_3_640_640_640_640_sel_1000_rt2 --dataset_dir=data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --train_split=training_0_3_640_640_640_640_sel_1000 --num_clones=1

<a id="800_"></a>
# 800

<a id="build_data___80_0_"></a>
## build_data       @ 800

<a id="50___build_data_800_"></a>
### 50       @ build_data/800

CUDA_VISIBLE_DEVICES=0 python3 datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_49_800_800_80_320_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_800_800_80_320_rot_15_345_4_flip

<a id="32___build_data_800_"></a>
### 32       @ build_data/800

CUDA_VISIBLE_DEVICES=0 python3 datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_31_800_800_80_320_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_800_800_80_320_rot_15_345_4_flip

<a id="18_test___build_data_800_"></a>
### 18_-_test       @ build_data/800

CUDA_VISIBLE_DEVICES=0 python3 datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_32_49_800_800_80_320_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_800_800_80_320_rot_15_345_4_flip

<a id="validation_0_20_800_800_800_800___build_data_800_"></a>
### validation_0_20_800_800_800_800       @ build_data/800

CUDA_VISIBLE_DEVICES=0 python3 datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=validation_0_20_800_800_800_800 --image_format=png --label_format=png --output_dir=training_0_31_49_800_800_80_320_rot_15_345_4_flip

<a id="yun00001_0_239_800_800_800_800___build_data_800_"></a>
### YUN00001_0_239_800_800_800_800       @ build_data/800

CUDA_VISIBLE_DEVICES=0 python3 datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=YUN00001_0_239_800_800_800_800 --image_format=png --label_format=png --output_dir=training_0_31_49_800_800_80_320_rot_15_345_4_flip --create_dummy_labels=1

<a id="4___build_data_800_"></a>
### 4       @ build_data/800

CUDA_VISIBLE_DEVICES=0 python3 datasets/build_617_data.py --db_root_dir=/data/617/images/ --db_dir=training_0_3_800_800_80_320_rot_15_345_4_flip --image_format=png --label_format=png --output_dir=training_0_31_49_800_800_80_320_rot_15_345_4_flip


<a id="50___80_0_"></a>
## 50       @ 800

CUDA_VISIBLE_DEVICES=1 python3 deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=800 --train_crop_size=800 --train_batch_size=2 --dataset=training_0_31_49_800_800_80_320_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_800_800_80_320_rot_15_345_4_flip/xception_0_49 --dataset_dir=/data/617/images/training_0_31_49_800_800_80_320_rot_15_345_4_flip/tfrecord --train_split=training_0_49_800_800_80_320_rot_15_345_4_flip --num_clones=1


<a id="eval___50_800_"></a>
### eval       @ 50/800

CUDA_VISIBLE_DEVICES=1 python3 eval.py --logtostderr  --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=800 --eval_crop_size=800 --dataset="training_0_31_49_800_800_80_320_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_800_800_80_320_rot_15_345_4_flip/xception_0_49 --eval_logdir=log/training_0_31_49_800_800_80_320_rot_15_345_4_flip/xception_0_49/eval --dataset_dir=/data/617/images/training_0_31_49_800_800_80_320_rot_15_345_4_flip/tfrecord --eval_split=training_32_49_800_800_80_320_rot_15_345_4_flip --eval_batch_size=5

<a id="vis___50_800_"></a>
### vis       @ 50/800

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=800 --vis_crop_size=800 --dataset="training_0_31_49_800_800_80_320_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_800_800_80_320_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_800_800_80_320_rot_15_345_4_flip/xception_0_49/vis/training_32_49_800_800_80_320_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_800_800_80_320_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_800_800_80_320_rot_15_345_4_flip --vis_batch_size=10 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_800_800_80_320_rot_15_345_4_flip/xception_0_49/vis/training_32_49_800_800_80_320_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_800_800_80_320_rot_15_345_4_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1

<a id="stitching___vis_50_800_"></a>
#### stitching       @ vis/50/800

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/training_32_49_640_640_64_256_rot_15_345_4_flip/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/training_32_49_640_640_64_256_rot_15_345_4_flip patch_height=640 patch_width=640 start_id=32 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=1

<a id="validation___50_800_"></a>
### validation       @ 50/800

CUDA_VISIBLE_DEVICES=0 python3 deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_20_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=validation_0_20_640_640_640_640 --vis_batch_size=50 --also_save_vis_predictions=0 --max_number_of_iterations=1 --eval_interval_secs=0

<a id="stitching___validation_50_80_0_"></a>
#### stitching       @ validation/50/800

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_563_640_640_640_640/segmentation_results stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/validation_0_20_640_640_640_640 patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/validation_0_563_640_640_640_640/raw stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/stitched/validation_0_20_640_640_640_640/raw patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1 normalize_patches=1

<a id="vis___validation_50_80_0_"></a>
#### vis       @ validation/50/800

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_49/vis/training_32_49_640_640_64_256_rot_15_345_4_flip/raw --save_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/vis_xception_0_49 --n_classes=3 --start_id=0 --end_id=-1

<a id="4___80_0_"></a>
## 4       @ 800

CUDA_VISIBLE_DEVICES=0 python3 deeplab_train.py --logtostderr --training_number_of_steps=100000 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=800 --train_crop_size=800 --train_batch_size=2 --dataset=training_0_31_49_800_800_80_320_rot_15_345_4_flip --tf_initial_checkpoint=pre_trained/xception/model.ckpt --train_logdir=log/training_0_31_49_800_800_80_320_rot_15_345_4_flip/xception_0_49 --dataset_dir=/data/617/images/training_0_31_49_800_800_80_320_rot_15_345_4_flip/tfrecord --train_split=training_0_3_800_800_80_320_rot_15_345_4_flip --num_clones=1



