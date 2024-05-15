<!-- MarkdownTOC -->

- [512](#512)
    - [hml       @ 512](#hml__512)
    - [evaluation       @ 512](#evaluation__512)
        - [hml       @ evaluation/512](#hml__evaluation512)
    - [vis       @ 512](#vis__512)
            - [hml       @ vis/512](#hml__vis512)
    - [validation       @ 512](#validation__512)
            - [stitching       @ validation/512](#stitching__validation512)
    - [videos       @ 512](#videos__512)
- [640](#640)
        - [hml       @ 640/](#hml__640)
        - [evaluation       @ 640/](#evaluation__640)
            - [hml       @ evaluation/640/](#hml__evaluation640)
        - [vis       @ 640/](#vis__640)
        - [validation       @ 640/](#validation__640)
            - [hml       @ validation/640/](#hml__validation640)
        - [stitching       @ 640/](#stitching__640)
        - [4       @ 640/](#4__640)
        - [evaluation       @ 640/](#evaluation__640-1)
        - [no_aug       @ 640/](#no_aug__640)
            - [stitched       @ no_aug/640/](#stitched__no_aug640)
        - [no_aug_4_49       @ 640/](#no_aug449__640)
        - [8       @ 640/](#8__640)
        - [evaluation       @ 640/](#evaluation__640-2)
        - [no_aug       @ 640/](#no_aug__640-1)
            - [stitched       @ no_aug/640/](#stitched__no_aug640-1)
        - [16       @ 640/](#16__640)
        - [evaluation       @ 640/](#evaluation__640-3)
        - [no_aug       @ 640/](#no_aug__640-2)
            - [stitched       @ no_aug/640/](#stitched__no_aug640-2)
        - [24       @ 640/](#24__640)
        - [evaluation       @ 640/](#evaluation__640-4)
        - [no_aug       @ 640/](#no_aug__640-3)
            - [stitched       @ no_aug/640/](#stitched__no_aug640-3)
    - [32       @ 640](#32__640)
        - [evaluation       @ 32/640](#evaluation__32640)
        - [vis       @ 32/640](#vis__32640)
        - [no_aug       @ 32/640](#no_aug__32640)
            - [stitched       @ no_aug/32/640](#stitched__no_aug32640)
        - [validation       @ 32/640](#validation__32640)
        - [stitching       @ 32/640](#stitching__32640)
- [640_selective](#640_selective)
    - [4_non_aug       @ 640_selective](#4_non_aug__640_selective)
        - [2       @ 4_non_aug/640_selective](#2__4_non_aug640_selective)
        - [2_0_14       @ 4_non_aug/640_selective](#2014__4_non_aug640_selective)
        - [10       @ 4_non_aug/640_selective](#10__4_non_aug640_selective)
        - [10_0_14       @ 4_non_aug/640_selective](#10014__4_non_aug640_selective)
        - [100       @ 4_non_aug/640_selective](#100__4_non_aug640_selective)
        - [100_0_14       @ 4_non_aug/640_selective](#100014__4_non_aug640_selective)
        - [1000       @ 4_non_aug/640_selective](#1000__4_non_aug640_selective)
        - [1000_0_14       @ 4_non_aug/640_selective](#1000014__4_non_aug640_selective)
    - [4       @ 640_selective](#4__640_selective)
        - [2       @ 4/640_selective](#2__4640_selective)
        - [5       @ 4/640_selective](#5__4640_selective)
        - [10       @ 4/640_selective](#10__4640_selective)
        - [100       @ 4/640_selective](#100__4640_selective)
        - [vis       @ 4/640_selective](#vis__4640_selective)
        - [no_aug       @ 4/640_selective](#no_aug__4640_selective)
            - [stitched       @ no_aug/4/640_selective](#stitched__no_aug4640_selective)
        - [1K       @ 4/640_selective](#1k__4640_selective)
        - [5K       @ 4/640_selective](#5k__4640_selective)
    - [16       @ 640_selective](#16__640_selective)
        - [100       @ 16/640_selective](#100__16640_selective)
    - [32       @ 640_selective](#32__640_selective)
        - [100       @ 32/640_selective](#100__32640_selective)

<!-- /MarkdownTOC -->

<a id="512"></a>
# 512
CUDA_VISIBLE_DEVICES=1 KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_31_512_512_25_100_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_segnet" --epochs=1000 

<a id="hml__512"></a>
## hml       @ 512

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_train.py --save_weights_path=weights/vgg_segnet_0_31_512_512_25_100_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_segnet" --epochs=1000 

<a id="evaluation__512"></a>
## evaluation       @ 512

<a id="hml__evaluation512"></a>
### hml       @ evaluation/512

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_512_512_25_100_rot_15_345_4_flip/weights_max_val_acc_352.h5 --test_images="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_training_32_49_512_512_25_100_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_segnet" 

<a id="vis__512"></a>
## vis       @ 512

<a id="hml__vis512"></a>
#### hml       @ vis/512

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels --seg_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_512_512_25_100_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_512_512_25_100_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1 --save_stitched=0

<a id="validation__512"></a>
## validation       @ 512

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_512_512_25_100_rot_15_345_4_flip/weights_max_val_acc_352.h5 --test_images="/data/617/images/validation_0_20_512_512_512_512/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_validation_0_20_512_512_512_512/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/validation_0_20_512_512_512_512/images  --seg_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_512_512_512_512/predictions/raw --save_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_512_512_25_100_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitching__validation512"></a>
#### stitching       @ validation/512

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_512_512_512_512/predictions/raw stitched_seq_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_512_512_512_512/predictions/stitched patch_height=512 patch_width=512 start_id=0 end_id=20  show_img=0 stacked=1 method=1 normalize_patches=1

CUDA_VISIBLE_DEVICES=1 KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_32_18_512_512_25_100_rot_15_125_235_345_flip/weights_685.h5 --test_images="/data/617/images/validation_0_20_512_512_512_512/images/" --output_path="/data/617/images/validation_0_20_512_512_512_512/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_segnet" 

<a id="videos__512"></a>
## videos       @ 512

CUDA_VISIBLE_DEVICES=1 KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_32_18_512_512_25_100_rot_15_125_235_345_flip/weights_685.h5 --test_images="/data/617/images/YUN00001_0_239_512_512_512_512/images/" --output_path="/data/617/images/YUN00001_0_239_512_512_512_512/vgg_segnet_32_18_512_512_25_100_rot_15_125_235_345_flip/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_segnet" 

<a id="640"></a>
# 640

<a id="hml__640"></a>
### hml       @ 640/

CUDA_VISIBLE_DEVICES=0 KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_49_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_49_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 

<a id="evaluation__640"></a>
### evaluation       @ 640/

<a id="hml__evaluation640"></a>
#### hml       @ evaluation/640/

CUDA_VISIBLE_DEVICES=0 KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc_305.h5 --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

<a id="vis__640"></a>
### vis       @ 640/

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1 --save_stitched=0

<a id="validation__640"></a>
### validation       @ 640/

<a id="hml__validation640"></a>
#### hml       @ validation/640/

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc_305.h5 --test_images="/data/617/images/validation_0_20_640_640_640_640/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/validation_0_20_640_640_640_640/images  --seg_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/raw --save_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitching__640"></a>
### stitching       @ 640/

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/raw stitched_seq_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/stitched patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1 normalize_patches=1

zr vgg_segnet_max_val_acc_validation_0_20_640_640_640_640_stitched /data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/stitched

<a id="4__640"></a>
### 4       @ 640/

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --load_weights=1

<a id="evaluation__640-1"></a>
### evaluation       @ 640/

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug__640"></a>
### no_aug       @ 640/

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug640"></a>
#### stitched       @ no_aug/640/

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="no_aug449__640"></a>
### no_aug_4_49       @ 640/

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/weights_max_acc --test_images="/data/617/images/training_4_49_640_640_640_640/images/" --output_path="log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_4_49_640_640_640_640_max_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels img_ext=jpg  patch_seq_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_4_49_640_640_640_640_max_acc/raw stitched_seq_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip/training_4_49_max_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png




<a id="8__640"></a>
### 8       @ 640/

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_7_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_7_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --load_weights=1

<a id="evaluation__640-2"></a>
### evaluation       @ 640/

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="no_aug__640-1"></a>
### no_aug       @ 640/

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug640-1"></a>
#### stitched       @ no_aug/640/

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/vgg_segnet_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="16__640"></a>
### 16       @ 640/

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --load_weights=1


<a id="evaluation__640-3"></a>
### evaluation       @ 640/

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1



<a id="no_aug__640-2"></a>
### no_aug       @ 640/

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug640-2"></a>
#### stitched       @ no_aug/640/

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="24__640"></a>
### 24       @ 640/

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --load_weights=1

<a id="evaluation__640-4"></a>
### evaluation       @ 640/

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="no_aug__640-3"></a>
### no_aug       @ 640/

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug640-3"></a>
#### stitched       @ no_aug/640/

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/vgg_segnet_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="32__640"></a>
## 32       @ 640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000

<a id="evaluation__32640"></a>
### evaluation       @ 32/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 CUDNN_PATH="/usr/local/cuda-8.0/cuda/lib64/libcudnn.so.5" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc_189.h5 --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

<a id="vis__32640"></a>
### vis       @ 32/640

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="no_aug__32640"></a>
### no_aug       @ 32/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug32640"></a>
#### stitched       @ no_aug/32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="validation__32640"></a>
### validation       @ 32/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 CUDNN_PATH="/usr/local/cuda-8.0/cuda/lib64/libcudnn.so.5" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python2 img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc_189.h5 --test_images="/data/617/images/validation_0_20_640_640_640_640/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 


<a id="stitching__32640"></a>
### stitching       @ 32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/raw stitched_seq_path=/data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/stitched patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1 normalize_patches=1

zr vgg_segnet_max_val_acc_validation_0_20_640_640_640_640_stitched /data/617/images/vgg_segnet_max_val_acc_validation_0_20_640_640_640_640/predictions/stitched

<a id="640_selective"></a>
# 640_selective

<a id="4_non_aug__640_selective"></a>
## 4_non_aug       @ 640_selective

<a id="2__4_non_aug640_selective"></a>
### 2       @ 4_non_aug/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_2_rt2 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=2 --load_weights=1

<a id="2014__4_non_aug640_selective"></a>
### 2_0_14       @ 4_non_aug/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_0_14_2 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=2 --load_weights=1 --start_id=0 --end_id=14


<a id="10__4_non_aug640_selective"></a>
### 10       @ 4_non_aug/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_10_rt2 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=10 --load_weights=0

<a id="10014__4_non_aug640_selective"></a>
### 10_0_14       @ 4_non_aug/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_0_14_10 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=10 --load_weights=1 --start_id=0 --end_id=14


<a id="100__4_non_aug640_selective"></a>
### 100       @ 4_non_aug/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_100_rt --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=100 --load_weights=1

<a id="100014__4_non_aug640_selective"></a>
### 100_0_14       @ 4_non_aug/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_0_14_100 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=100 --load_weights=1 --start_id=0 --end_id=14

<a id="1000__4_non_aug640_selective"></a>
### 1000       @ 4_non_aug/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_1000 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=1000 --load_weights=1

<a id="1000014__4_non_aug640_selective"></a>
### 1000_0_14       @ 4_non_aug/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_0_14_1000 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=1000 --load_weights=1 --start_id=0 --end_id=14

<a id="4__640_selective"></a>
## 4       @ 640_selective

<a id="2__4640_selective"></a>
### 2       @ 4/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_2 --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=2 --load_weights=1

<a id="5__4640_selective"></a>
### 5       @ 4/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_5 --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=5 --load_weights=1

<a id="10__4640_selective"></a>
### 10       @ 4/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_10 --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=10 --load_weights=1

<a id="100__4640_selective"></a>
### 100       @ 4/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100 --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=100 --load_weights=1

<a id="vis__4640_selective"></a>
### vis       @ 4/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="no_aug__4640_selective"></a>
### no_aug       @ 4/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug4640_selective"></a>
#### stitched       @ no_aug/4/640_selective

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_max_val_acc/raw  --save_path=log/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="1k__4640_selective"></a>
### 1K       @ 4/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_1K --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=1000 --load_weights=1

<a id="5k__4640_selective"></a>
### 5K       @ 4/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_64_256_rot_15_345_4_flip_5K --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=5000 --load_weights=1

<a id="16__640_selective"></a>
## 16       @ 640_selective

<a id="100__16640_selective"></a>
### 100       @ 16/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_15_640_640_64_256_rot_15_345_4_flip_100 --train_images="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=100

<a id="32__640_selective"></a>
## 32       @ 640_selective

<a id="100__32640_selective"></a>
### 100       @ 32/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_31_640_640_64_256_rot_15_345_4_flip_100 --train_images="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=100
