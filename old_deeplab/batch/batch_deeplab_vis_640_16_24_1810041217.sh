set -x

## 16

### evaluation 

CUDA_VISIBLE_DEVICES=2 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_64_256_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_64_256_rot_15_345_4_flip --vis_batch_size=50 --also_save_raw_predictions=1 --max_number_of_iterations=1

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_64_256_rot_15_345_4_flip/raw_segmentation_results --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_15/training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

## 24

### evaluation 

CUDA_VISIBLE_DEVICES=2 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23 --vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_64_256_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_640_640_64_256_rot_15_345_4_flip --vis_batch_size=50 --also_save_raw_predictions=1 --max_number_of_iterations=1

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_64_256_rot_15_345_4_flip/raw_segmentation_results --save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/xception_0_23/training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1