# raw_vis
CUDA_VISIBLE_DEVICES=0 python36 new_deeplab_train.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 train_crop_size=640,640 train_batch_size=2 

dataset=training_0_31_49_640_640_64_256_rot_15_345_4_flip
train_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31
dataset_dir=/data/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord
train_split=training_0_31_640_640_64_256_rot_15_345_4_flip

# raw_vis
CUDA_VISIBLE_DEVICES=2 python36 new_deeplab_vis.py model_variant="nas_hnasnet" atrous_rates=6 atrous_rates=12 atrous_rates=18 output_stride=16 decoder_output_stride=4 vis_crop_size=640,640

 dataset="training_0_31_49_640_640_64_256_rot_15_345_4_flip"
 checkpoint_dir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31
 vis_logdir=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49_640_640_640_640
 dataset_dir=/data/617/images/training_0_31_49_640_640_64_256_rot_15_345_4_flip/tfrecord
 vis_split=training_32_49_640_640_640_640

# stitch
python36 ../stitchSubPatchDataset.py
img_ext=jpg

seq_name = 'Training'
db_root_dir = /home/abhineet/N/Datasets/617/
src_path=/data/617/images/training_32_49/images
patch_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49_640_640_640_640/raw
stitched_seq_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49/raw

patch_height=640 patch_width=640 start_id=0 end_id=-1 show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

# vis
python36 ../visDataset.py

images_path=/data/617/images/training_32_49/images
labels_path=/data/617/images/training_32_49/labels
seg_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49/raw
save_path=log/training_0_31_49_640_640_64_256_rot_15_345_4_flip/nas_hnasnet_0_31/training_32_49/vis

n_classes=3 start_id=0 end_id=-1 normalize_labels=1