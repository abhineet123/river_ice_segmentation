set -x

# max_val_acc

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/YUN00001_0_8999_640_640_640_640/images/" --output_path="log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_0_8999_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/YUN00001_0_8999_640_640_640_640/images --seg_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_0_8999_640_640_640_640_max_val_acc/raw --save_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_0_8999_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_0_8999_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png out_ext=mkv width=1920 height=1080

# min_loss


CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/weights_min_loss --test_images="/data/617/images/YUN00001_0_8999_640_640_640_640/images/" --output_path="log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_0_8999_640_640_640_640_min_loss/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/YUN00001_0_8999_640_640_640_640/images --seg_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_0_8999_640_640_640_640_min_loss/raw --save_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_0_8999_640_640_640_640_min_loss/vis --n_classes=3 --start_id=0 --end_id=-1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_0_8999_640_640_640_640_min_loss/raw stitched_seq_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_min_loss/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png out_ext=mkv width=1920 height=1080


# max_acc

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_acc --test_images="/data/617/images/YUN00001_0_8999_640_640_640_640/images/" --output_path="log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_0_8999_640_640_640_640_max_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/YUN00001_0_8999_640_640_640_640/images --seg_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_0_8999_640_640_640_640_max_acc/raw --save_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_0_8999_640_640_640_640_max_acc/vis --n_classes=3 --start_id=0 --end_id=-1

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_0_8999_640_640_640_640_max_acc/raw stitched_seq_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_max_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=png out_ext=mkv width=1920 height=1080

