set -x

## 8

### evaluation 

CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda1,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0


## 16

### evaluation 

CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda1,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0


## 24

### evaluation 

CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda1,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0