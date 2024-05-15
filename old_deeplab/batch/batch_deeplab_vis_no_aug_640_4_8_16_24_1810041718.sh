set -x 

### 4

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_raw_predictions=1 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_640_640/raw_segmentation_results --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/training_32_49_640_640_640_640/raw_segmentation_results stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/stitched/training_32_49_640_640_640_640/raw_segmentation_results patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/stitched/training_32_49_640_640_640_640/raw_segmentation_results --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_3/stitched/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1


### 8

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_raw_predictions=1 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_640_640/raw_segmentation_results --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/training_32_49_640_640_640_640/raw_segmentation_results stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/stitched/training_32_49_640_640_640_640/raw_segmentation_results patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/stitched/training_32_49_640_640_640_640/raw_segmentation_results --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_7/stitched/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1



### 16

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_raw_predictions=1 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_640_640/raw_segmentation_results --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_640_640/raw_segmentation_results stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/stitched/training_32_49_640_640_640_640/raw_segmentation_results patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/stitched/training_32_49_640_640_640_640/raw_segmentation_results --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/stitched/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1


### 24

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_640_640 --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_640_640 --vis_batch_size=50 --also_save_raw_predictions=1 --max_number_of_iterations=1 --eval_interval_secs=0

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_640_640/raw_segmentation_results --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_640_640/raw_segmentation_results stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/stitched/training_32_49_640_640_640_640/raw_segmentation_results patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/stitched/training_32_49_640_640_640_640/raw_segmentation_results --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/stitched/training_32_49_640_640_640_640/vis --n_classes=3 --start_id=0 --end_id=-1


