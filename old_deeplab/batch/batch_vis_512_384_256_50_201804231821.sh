set -x

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_256_256_25_100_rot_15_125_235_345_flip" --checkpoint_dir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49 --vis_logdir=log/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/xception_0_49/vis/training_32_49_256_256_25_100_rot_15_125_235_345_flip --dataset_dir=/data/617/images/training_0_31_49_256_256_25_100_rot_15_125_235_345_flip/tfrecord --vis_split=training_32_49_256_256_25_100_rot_15_125_235_345_flip --vis_batch_size=50 --also_save_raw_predictions=1 --max_number_of_iterations=1

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_384_384_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_384_384_25_100_rot_15_345_4_flip/xception_0_49/vis/training_32_49_384_384_25_100_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_384_384_25_100_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_384_384_25_100_rot_15_345_4_flip --vis_batch_size=50 --also_save_raw_predictions=1 --max_number_of_iterations=1

CUDA_VISIBLE_DEVICES=0 python deeplab_vis.py --logtostderr --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=640 --vis_crop_size=640 --dataset="training_0_31_49_512_512_25_100_rot_15_345_4_flip" --checkpoint_dir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49 --vis_logdir=log/training_0_31_49_512_512_25_100_rot_15_345_4_flip/xception_0_49/vis/training_32_49_512_512_25_100_rot_15_345_4_flip --dataset_dir=/data/617/images/training_0_31_49_512_512_25_100_rot_15_345_4_flip/tfrecord --vis_split=training_32_49_512_512_25_100_rot_15_345_4_flip --vis_batch_size=50 --also_save_raw_predictions=1 --max_number_of_iterations=1





