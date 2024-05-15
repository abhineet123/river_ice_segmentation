# 2

## max_val_acc

### vis

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0


### no_aug 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_max_val_acc/raw  --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0


## min_loss

### vis

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/weights_min_loss --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_64_256_rot_15_345_4_flip_min_loss/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_64_256_rot_15_345_4_flip_min_loss/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_64_256_rot_15_345_4_flip_min_loss/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0


### no_aug 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/weights_min_loss --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_640_640_min_loss/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_640_640_min_loss/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_640_640_min_loss/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_640_640_min_loss/raw stitched_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_min_loss/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_min_loss/raw  --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_min_loss/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0


## max_acc

### vis

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/weights_max_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_64_256_rot_15_345_4_flip_max_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_64_256_rot_15_345_4_flip_max_acc/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_64_256_rot_15_345_4_flip_max_acc/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0


### no_aug 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/weights_max_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_640_640_max_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_640_640_max_acc/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_640_640_max_acc/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_640_640_640_640_max_acc/raw stitched_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_max_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_max_acc/raw  --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2/training_32_49_max_acc/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0


# 5

## max_val_acc

### vis

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0


### no_aug 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_max_val_acc/raw  --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0


## min_loss

### vis

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/weights_min_loss --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_64_256_rot_15_345_4_flip_min_loss/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_64_256_rot_15_345_4_flip_min_loss/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_64_256_rot_15_345_4_flip_min_loss/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0


### no_aug 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/weights_min_loss --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_640_640_min_loss/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_640_640_min_loss/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_640_640_min_loss/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_640_640_min_loss/raw stitched_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_min_loss/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_min_loss/raw  --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_min_loss/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0


## max_acc

### vis

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/weights_max_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_64_256_rot_15_345_4_flip_max_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_64_256_rot_15_345_4_flip_max_acc/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_64_256_rot_15_345_4_flip_max_acc/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0


### no_aug 

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/weights_max_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_640_640_max_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_640_640_max_acc/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_640_640_max_acc/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0

#### stitched

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_640_640_640_640_max_acc/raw stitched_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_max_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_max_acc/raw  --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5/training_32_49_max_acc/vis --n_classes=3 --start_id=0 --end_id=-1 --out_ext=jpg --stitch=0 --save_stitched=0


