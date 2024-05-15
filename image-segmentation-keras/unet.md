<!-- MarkdownTOC -->

- [640x320 / misc](#640x320__misc)
- [256](#256)
   - [evaluation       @ 256](#evaluation__256)
      - [vis       @ evaluation/256](#vis__evaluation256)
      - [hml       @ evaluation/256](#hml__evaluation256)
   - [vis       @ 256](#vis__256)
         - [hml       @ vis/256](#hml__vis256)
   - [validation       @ 256](#validation__256)
   - [videos       @ 256](#videos__256)
   - [new       @ 256](#new__256)
- [384](#384)
   - [evaluation       @ 384](#evaluation__384)
      - [vis       @ evaluation/384](#vis__evaluation384)
      - [hml       @ evaluation/384](#hml__evaluation384)
   - [validation       @ 384](#validation__384)
   - [videos       @ 384](#videos__384)
- [512](#512)
   - [evaluation       @ 512](#evaluation__512)
      - [hml       @ evaluation/512](#hml__evaluation512)
   - [vis       @ 512](#vis__512)
         - [hml       @ vis/512](#hml__vis512)
   - [validation       @ 512](#validation__512)
         - [stitching       @ validation/512](#stitching__validation512)
   - [videos       @ 512](#videos__512)
- [640](#640)
   - [4__non_aug       @ 640](#4__non_aug__640)
   - [4       @ 640](#4__640)
      - [evaluation       @ 4/640](#evaluation__4640)
      - [4_49       @ 4/640](#4_49__4640)
      - [no_aug       @ 4/640](#no_aug__4640)
         - [stitched       @ no_aug/4/640](#stitched__no_aug4640)
      - [no_aug__4_49       @ 4/640](#no_aug_449__4640)
   - [8       @ 640](#8__640)
      - [evaluation       @ 8/640](#evaluation__8640)
      - [no_aug       @ 8/640](#no_aug__8640)
         - [stitched       @ no_aug/8/640](#stitched__no_aug8640)
   - [16       @ 640](#16__640)
      - [continue_133       @ 16/640](#continue_133__16640)
      - [continue_latest       @ 16/640](#continue_latest__16640)
      - [evaluation       @ 16/640](#evaluation__16640)
      - [no_aug       @ 16/640](#no_aug__16640)
         - [stitched       @ no_aug/16/640](#stitched__no_aug16640)
   - [24       @ 640](#24__640)
      - [continue_171       @ 24/640](#continue_171__24640)
      - [continue_latest       @ 24/640](#continue_latest__24640)
      - [evaluation       @ 24/640](#evaluation__24640)
      - [no_aug       @ 24/640](#no_aug__24640)
         - [stitched       @ no_aug/24/640](#stitched__no_aug24640)
   - [32       @ 640](#32__640)
      - [evaluation       @ 32/640](#evaluation__32640)
      - [vis       @ 32/640](#vis__32640)
      - [no_aug       @ 32/640](#no_aug__32640)
         - [stitched       @ no_aug/32/640](#stitched__no_aug32640)
      - [validation       @ 32/640](#validation__32640)
      - [stitching       @ 32/640](#stitching__32640)
      - [YUN00001       @ 32/640](#yun00001__32640)
      - [YUN00001_3600       @ 32/640](#yun00001_3600__32640)
      - [vis_png       @ 32/640](#vis_png__32640)
         - [20160122_YUN00002_700_2500       @ vis_png/32/640](#20160122_yun00002_700_2500__vis_png32640)
         - [20160122_YUN00020_2000_3800       @ vis_png/32/640](#20160122_yun00020_2000_3800__vis_png32640)
         - [YUN00001_3600       @ vis_png/32/640](#yun00001_3600__vis_png32640)
- [640_selective](#640_selective)
   - [4__non_aug       @ 640_selective](#4__non_aug__640_selective)
      - [2       @ 4__non_aug/640_selective](#2__4__non_aug640_selective)
      - [2__0_14       @ 4__non_aug/640_selective](#2_014__4__non_aug640_selective)
      - [10       @ 4__non_aug/640_selective](#10__4__non_aug640_selective)
      - [10__rt3       @ 4__non_aug/640_selective](#10__rt3__4__non_aug640_selective)
      - [10__0_14       @ 4__non_aug/640_selective](#10_014__4__non_aug640_selective)
      - [100       @ 4__non_aug/640_selective](#100__4__non_aug640_selective)
      - [100__0_14       @ 4__non_aug/640_selective](#100_014__4__non_aug640_selective)
      - [1000       @ 4__non_aug/640_selective](#1000__4__non_aug640_selective)
      - [1000__0_14       @ 4__non_aug/640_selective](#1000_014__4__non_aug640_selective)
   - [4       @ 640_selective](#4__640_selective)
      - [2       @ 4/640_selective](#2__4640_selective)
      - [5       @ 4/640_selective](#5__4640_selective)
      - [10       @ 4/640_selective](#10__4640_selective)
         - [rt       @ 10/4/640_selective](#rt__104640_selective)
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



python visualizeDataset.py \
 --images="/home/abhineet/N/Datasets/617/Training_256_25_100/images/" \
 --annotations="/home/abhineet/N/Datasets/617/Training_256_25_100/labels/" \
 --n_classes=3 
 

<a id="640x320__misc"></a>
# 640x320 / misc
 
KERAS_BACKEND=theano THEANO_FLAGS=device=gpu,floatX=float32  python  img_keras_train.py --save_weights_path=weights/ex1 --train_images="/data/617/images/training_320_640_25_100/images/" --train_annotations="/data/617/images/training_320_640_25_100/labels/" --val_images="/data/617/images/training_320_640_100_200/images/" --val_annotations="/data/617/images/training_320_640_100_200/labels/" --n_classes=3 --input_height=320 --input_width=640 --model_name="vgg_unet2" 
 
 --model_name="fcn32" 
 --model_name="vgg_segnet"  
 --model_name="vgg_unet2" 
 
KERAS_BACKEND=theano THEANO_FLAGS=device=gpu,floatX=float32  python  img_keras_train.py --save_weights_path=weights/ex1 --train_images="/data/617/images/dataset1/images_prepped_train/" --train_annotations="/data/617/images/dataset1/annotations_prepped_train/" --val_images="/data/617/images/dataset1/images_prepped_test/" --val_annotations="/data/617/images/dataset1/annotations_prepped_test/" --n_classes=10 --input_height=320 --input_width=640 --model_name="vgg_segnet" 
 
 change tf.concat(0, [[self._batch_size], [num_dim, -1], [input_shape[2]]]) to tf.concat([[self._batch_size], [num_dim, -1], [input_shape[2]]], 0) and change another tf.concat similarly (also within this file). You should be fine.
 
<a id="256"></a>
# 256

KERAS_BACKEND=theano THEANO_FLAGS<!-- MarkdownTOC -->

="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_unet2_256_25_100_rot_15_90_flip --train_images="/data/617/images/training_256_256_25_100_rot_15_90_flip/images/" --train_annotations="/data/617/images/training_256_256_25_100_rot_15_90_flip/labels/" --val_images="/data/617/images/training_256_256_100_200_rot_90_180_flip/images/" --val_annotations="/data/617/images/training_256_256_100_200_rot_90_180_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_unet2" --epochs=1000

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/fcn32_256_25_100_rot_15_90_flip --train_images="/data/617/images/training_256_256_25_100_rot_15_90_flip/images/" --train_annotations="/data/617/images/training_256_256_25_100_rot_15_90_flip/labels/" --val_images="/data/617/images/training_256_256_100_200_rot_90_180_flip/images/" --val_annotations="/data/617/images/training_256_256_100_200_rot_90_180_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="fcn32" --epochs=1000  

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_256_25_100_rot_15_90_flip --train_images="/data/617/images/training_256_256_25_100_rot_15_90_flip/images/" --train_annotations="/data/617/images/training_256_256_25_100_rot_15_90_flip/labels/" --val_images="/data/617/images/training_256_256_100_200_flip/images/" --val_annotations="/data/617/images/training_256_256_100_200_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_segnet" --epochs=1000 

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/fcn8_256_25_100_rot_15_90_flip --train_images="/data/617/images/training_256_256_25_100_rot_15_90_flip/images/" --train_annotations="/data/617/images/training_256_256_25_100_rot_15_90_flip/labels/" --val_images="/data/617/images/training_256_256_100_200_flip/images/" --val_annotations="/data/617/images/training_256_256_100_200_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="fcn8" --epochs=1000 

<a id="evaluation__256"></a>
## evaluation       @ 256

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python2 img_keras_predict.py --save_weights_path=weights/vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/training_256_256_100_200_rot_90_180_flip/images/" --output_path="/data/617/images/vgg_unet2_max_val_acc_training_256_256_100_200_rot_90_180_flip/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_unet2" 

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python2 img_keras_predict.py --save_weights_path=weights/vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_266.h5 --test_images="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images/" --output_path="/data/617/images/vgg_segnet_max_mean_acc_training_32_49_256_256_25_100_rot_15_125_235_345_flip/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_segnet" 

<a id="vis__evaluation256"></a>
### vis       @ evaluation/256

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images --labels_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels --seg_path=/data/617/images/vgg_segnet_max_mean_acc_training_32_49_256_256_25_100_rot_15_125_235_345_flip/predictions/raw --save_path=/data/617/images/vgg_segnet_max_mean_acc_training_32_49_256_256_25_100_rot_15_125_235_345_flip/vis --n_classes=3 --start_id=0 --end_id=-1 --save_stitched=0 --normalize_labels=1



<a id="hml__evaluation256"></a>
### hml       @ evaluation/256

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-9.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/training_256_256_100_200_rot_90_180_flip/images/" --output_path="/data/617/images/vgg_unet2_max_val_acc_training_256_256_100_200_rot_90_180_flip/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_unet2" 

zr vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip_weights_490 weights/vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5
zr img_seg_keras_data data

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-9.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images/" --output_path="/data/617/images/vgg_unet2_max_val_acc_training_32_49_256_256_25_100_rot_15_125_235_345_flip/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_unet2" 

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-9.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_training_32_49_256_256_25_100_rot_15_125_235_345_flip/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_segnet" 

<a id="vis__256"></a>
## vis       @ 256

<a id="hml__vis256"></a>
#### hml       @ vis/256

python3 visDataset.py --images_path=/data/617/images/training_256_256_100_200_rot_90_180_flip/images --labels_path=/data/617/images/training_256_256_100_200_rot_90_180_flip/labels --seg_path=/data/617/images/vgg_unet2_max_val_acc_training_256_256_100_200_rot_90_180_flip/predictions/raw --save_path=/data/617/images/vgg_unet2_max_val_acc_training_256_256_100_200_rot_90_180_flip/vis --n_classes=3 --start_id=0 --end_id=-1

python3 visDataset.py --images_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images --labels_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels --seg_path=/data/617/images/vgg_unet2_max_val_acc_training_32_49_256_256_25_100_rot_15_125_235_345_flip/predictions/raw --save_path=/data/617/images/vgg_unet2_max_val_acc_training_32_49_256_256_25_100_rot_15_125_235_345_flip/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="validation__256"></a>
## validation       @ 256

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/validation_0_20_256_256_256_256/images/" --output_path="/data/617/images/vgg_unet2_max_val_acc_validation_0_20_256_256_256_256/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_unet2" 

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/fcn32_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_174.h5 --test_images="/data/617/images/validation_0_20_256_256_256_256/images/" --output_path="/data/617/images/fcn32_max_mean_acc_validation_0_20_256_256_256_256/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="fcn32" 


KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_685.h5 --test_images="/data/617/images/validation_0_20_256_256_256_256/images/" --output_path="/data/617/images/validation_0_20_256_256_256_256/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_segnet" 

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/fcn8_256_25_100_rot_15_90_flip/weights_685.h5 --test_images="/data/617/images/validation_0_20_256_256_256_256/images/" --output_path="/data/617/images/validation_0_20_256_256_256_256/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="fcn8" 

<a id="videos__256"></a>
## videos       @ 256

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/YUN00001_0_239_256_256_256_256/images/" --output_path="/data/617/images/YUN00001_0_239_256_256_256_256/vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_unet2" 

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_685.h5 --test_images="/data/617/images/YUN00001_0_239_256_256_256_256/images/" --output_path="/data/617/images/YUN00001_0_239_256_256_256_256/vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_segnet" 

<a id="new__256"></a>
## new       @ 256

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip --train_images="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/images/" --train_annotations="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/labels/" --val_images="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images/" --val_annotations="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_unet2" --epochs=1000

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/fcn32_32_18_256_256_25_100_rot_15_125_235_345_flip --train_images="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/images/" --train_annotations="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/labels/" --val_images="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images/" --val_annotations="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="fcn32" --epochs=1000

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip --train_images="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/images/" --train_annotations="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/labels/" --val_images="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images/" --val_annotations="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_segnet" --epochs=1000

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/fcn8_32_18_256_256_25_100_rot_15_125_235_345_flip --train_images="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/images/" --train_annotations="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/labels/" --val_images="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images/" --val_annotations="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="fcn8" --epochs=1000


<a id="384"></a>
# 384

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_31_384_384_25_100_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_384_384_25_100_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_384_384_25_100_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=384 --input_width=384 --model_name="vgg_unet2" --epochs=1000

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/fcn32_0_31_384_384_25_100_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_384_384_25_100_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_384_384_25_100_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=384 --input_width=384 --model_name="fcn32" --epochs=1000  

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_31_384_384_25_100_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_384_384_25_100_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_384_384_25_100_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=384 --input_width=384 --model_name="vgg_segnet" --epochs=1000 


<a id="evaluation__384"></a>
## evaluation       @ 384

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python2 img_keras_predict.py --save_weights_path=weights/vgg_unet2_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images/" --output_path="/data/617/images/vgg_unet2_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_unet2" 

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python2 img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_384_384_25_100_rot_15_345_4_flip/weights_583.h5 --test_images="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=384 --input_width=384 --model_name="vgg_segnet" 


<a id="vis__evaluation384"></a>
### vis       @ evaluation/384

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --seg_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1 --save_stitched=0 --normalize_labels=1


<a id="hml__evaluation384"></a>
### hml       @ evaluation/384

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-9.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_31_384_384_25_100_rot_15_345_4_flip/weights_497.h5 --test_images="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images/" --output_path="/data/617/images/vgg_unet2_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=384 --input_width=384 --model_name="vgg_unet2" 


<a id="validation__384"></a>
## validation       @ 384

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_32_18_384_384_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/validation_0_20_384_384_384_384/images/" --output_path="/data/617/images/vgg_unet2_max_val_acc_validation_0_20_384_384_384_384/predictions/" --n_classes=3 --input_height=384 --input_width=384 --model_name="vgg_unet2" 

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/fcn32_32_18_384_384_25_100_rot_15_125_235_345_flip/weights_174.h5 --test_images="/data/617/images/validation_0_20_384_384_384_384/images/" --output_path="/data/617/images/fcn32_max_mean_acc_validation_0_20_384_384_384_384/predictions/" --n_classes=3 --input_height=384 --input_width=384 --model_name="fcn32" 


KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_32_18_384_384_25_100_rot_15_125_235_345_flip/weights_685.h5 --test_images="/data/617/images/validation_0_20_384_384_384_384/images/" --output_path="/data/617/images/validation_0_20_384_384_384_384/predictions/" --n_classes=3 --input_height=384 --input_width=384 --model_name="vgg_segnet" 

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/fcn8_384_25_100_rot_15_90_flip/weights_685.h5 --test_images="/data/617/images/validation_0_20_384_384_384_384/images/" --output_path="/data/617/images/validation_0_20_384_384_384_384/predictions/" --n_classes=3 --input_height=384 --input_width=384 --model_name="fcn8" 

<a id="videos__384"></a>
## videos       @ 384

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_32_18_384_384_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/YUN00001_0_239_384_384_384_384/images/" --output_path="/data/617/images/YUN00001_0_239_384_384_384_384/vgg_unet2_32_18_384_384_25_100_rot_15_125_235_345_flip/" --n_classes=3 --input_height=384 --input_width=384 --model_name="vgg_unet2" 

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_32_18_384_384_25_100_rot_15_125_235_345_flip/weights_685.h5 --test_images="/data/617/images/YUN00001_0_239_384_384_384_384/images/" --output_path="/data/617/images/YUN00001_0_239_384_384_384_384/vgg_segnet_32_18_384_384_25_100_rot_15_125_235_345_flip/" --n_classes=3 --input_height=384 --input_width=384 --model_name="vgg_segnet" 


<a id="512"></a>
# 512

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_31_512_512_25_100_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_unet2" --epochs=1000

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/fcn32_0_31_512_512_25_100_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=512 --input_width=512 --model_name="fcn32" --epochs=1000  


<a id="evaluation__512"></a>
## evaluation       @ 512

<a id="hml__evaluation512"></a>
### hml       @ evaluation/512

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-9.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_31_512_512_25_100_rot_15_345_4_flip/weights_max_val_acc_140.h5 --test_images="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images/" --output_path="/data/617/images/vgg_unet2_max_val_acc_training_32_49_512_512_25_100_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_unet2" 

<a id="vis__512"></a>
## vis       @ 512

<a id="hml__vis512"></a>
#### hml       @ vis/512

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels --seg_path=/data/617/images/vgg_unet2_max_val_acc_training_32_49_512_512_25_100_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/vgg_unet2_max_val_acc_training_32_49_512_512_25_100_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="validation__512"></a>
## validation       @ 512

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-9.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_31_512_512_25_100_rot_15_345_4_flip/weights_max_val_acc_140.h5 --test_images="/data/617/images/validation_0_20_512_512_512_512/images/" --output_path="/data/617/images/vgg_unet2_max_val_acc_validation_0_20_512_512_512_512/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_unet2" 

<a id="stitching__validation512"></a>
#### stitching       @ validation/512

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=/data/617/images/vgg_unet2_max_val_acc_validation_0_20_512_512_512_512/predictions/raw stitched_seq_path=/data/617/images/vgg_unet2_max_val_acc_validation_0_20_512_512_512_512/predictions/stitched patch_height=512 patch_width=512 start_id=0 end_id=20  show_img=0 stacked=1 method=1 normalize_patches=1


KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_32_18_512_512_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/validation_0_20_512_512_512_512/images/" --output_path="/data/617/images/vgg_unet2_max_val_acc_validation_0_20_512_512_512_512/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_unet2" 

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/fcn32_32_18_512_512_25_100_rot_15_125_235_345_flip/weights_174.h5 --test_images="/data/617/images/validation_0_20_512_512_512_512/images/" --output_path="/data/617/images/fcn32_max_mean_acc_validation_0_20_512_512_512_512/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="fcn32" 



KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/fcn8_512_25_100_rot_15_90_flip/weights_685.h5 --test_images="/data/617/images/validation_0_20_512_512_512_512/images/" --output_path="/data/617/images/validation_0_20_512_512_512_512/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="fcn8" 

<a id="videos__512"></a>
## videos       @ 512

KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/cuda/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_32_18_512_512_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/YUN00001_0_239_512_512_512_512/images/" --output_path="/data/617/images/YUN00001_0_239_512_512_512_512/vgg_unet2_32_18_512_512_25_100_rot_15_125_235_345_flip/" --n_classes=3 --input_height=512 --input_width=512 --model_name="vgg_unet2" 



<a id="640"></a>
# 640

<a id="4__non_aug__640"></a>
## 4__non_aug       @ 640

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000

<a id="4__640"></a>
## 4       @ 640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000

<a id="evaluation__4640"></a>
### evaluation       @ 4/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="4_49__4640"></a>
### 4_49       @ 4/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="no_aug__4640"></a>
### no_aug       @ 4/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_4_49_640_640_640_640/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_4_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_4_49_640_640_640_640/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_4_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug4640"></a>
#### stitched       @ no_aug/4/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug_449__4640"></a>
### no_aug__4_49       @ 4/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_4_49_640_640_640_640/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_4_49/images labels_path=/data/617/images/training_4_49/labels img_ext=jpg  patch_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_4_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip/training_4_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png


<a id="8__640"></a>
## 8       @ 640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_7_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_7_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --load_weights=1

<a id="evaluation__8640"></a>
### evaluation       @ 8/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug__8640"></a>
### no_aug       @ 8/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug8640"></a>
#### stitched       @ no_aug/8/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/vgg_unet2_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="16__640"></a>
## 16       @ 640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000

<a id="continue_133__16640"></a>
### continue_133       @ 16/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --load_weights=weights/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/weights_max_acc_133.h5

<a id="continue_latest__16640"></a>
### continue_latest       @ 16/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --load_weights=1

<a id="evaluation__16640"></a>
### evaluation       @ 16/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug__16640"></a>
### no_aug       @ 16/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug16640"></a>
#### stitched       @ no_aug/16/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="24__640"></a>
## 24       @ 640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000

<a id="continue_171__24640"></a>
### continue_171       @ 24/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --load_weights=weights/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/weights_max_acc_171.h5

<a id="continue_latest__24640"></a>
### continue_latest       @ 24/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --load_weights=1

<a id="evaluation__24640"></a>
### evaluation       @ 24/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="no_aug__24640"></a>
### no_aug       @ 24/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug24640"></a>
#### stitched       @ no_aug/24/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/vgg_unet2_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="32__640"></a>
## 32       @ 640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000

<a id="evaluation__32640"></a>
### evaluation       @ 32/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc_189.h5 --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="/data/617/images/vgg_unet2_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

<a id="vis__32640"></a>
### vis       @ 32/640

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=/data/617/images/vgg_unet2_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/vgg_unet2_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="no_aug__32640"></a>
### no_aug       @ 32/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug32640"></a>
#### stitched       @ no_aug/32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="validation__32640"></a>
### validation       @ 32/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python2 img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc_189.h5 --test_images="/data/617/images/validation_0_20_640_640_640_640/images/" --output_path="/data/617/images/vgg_unet2_max_val_acc_validation_0_20_640_640_640_640/predictions/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 


<a id="stitching__32640"></a>
### stitching       @ 32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=/data/617/images/vgg_unet2_max_val_acc_validation_0_20_640_640_640_640/predictions/raw stitched_seq_path=/data/617/images/vgg_unet2_max_val_acc_validation_0_20_640_640_640_640/predictions/stitched patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1 normalize_patches=1

zr vgg_unet2_max_val_acc_validation_0_20_640_640_640_640_stitched /data/617/images/vgg_unet2_max_val_acc_validation_0_20_640_640_640_640/predictions/stitched

<a id="yun00001__32640"></a>
### YUN00001       @ 32/640

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/YUN00001_0_8999_640_640_640_640/images/" --output_path="log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001/images patch_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_0_8999_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=jpg out_ext=mkv width=1920 height=1080

<a id="yun00001_3600__32640"></a>
### YUN00001_3600       @ 32/640

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/YUN00001_3600_0_3599_640_640_640_640/images/" --output_path="log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_3600_0_3599_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 


python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images patch_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_3600_0_3599_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_3600_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 img_ext=jpg out_ext=mkv width=1920 height=1080

<a id="vis_png__32640"></a>
### vis_png       @ 32/640


<a id="20160122_yun00002_700_2500__vis_png32640"></a>
#### 20160122_YUN00002_700_2500       @ vis_png/32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00002_700_2500/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/20160122_YUN00002_700_2500_0_1799_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/20160122_YUN00002_700_2500_max_val_acc patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=png

<a id="20160122_yun00020_2000_3800__vis_png32640"></a>
#### 20160122_YUN00020_2000_3800       @ vis_png/32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00020_2000_3800/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/20160122_YUN00020_2000_3800_0_1799_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/20160122_YUN00020_2000_3800_max_val_acc patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=png

<a id="yun00001_3600__vis_png32640"></a>
#### YUN00001_3600       @ vis_png/32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_3600_0_3599_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip/YUN00001_3600_max_val_acc patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=png


<a id="640_selective"></a>
# 640_selective

<a id="4__non_aug__640_selective"></a>
## 4__non_aug       @ 640_selective

<a id="2__4__non_aug640_selective"></a>
### 2       @ 4__non_aug/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640_2_rt2 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=2 --load_weights=1

<a id="2_014__4__non_aug640_selective"></a>
### 2__0_14       @ 4__non_aug/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640_0_14_2 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=2 --load_weights=1 --start_id=0 --end_id=14


<a id="10__4__non_aug640_selective"></a>
### 10       @ 4__non_aug/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640_10_rt2 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=10 --load_weights=0

<a id="10__rt3__4__non_aug640_selective"></a>
### 10__rt3       @ 4__non_aug/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640_10_rt3 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=10 --load_weights=0

<a id="10_014__4__non_aug640_selective"></a>
### 10__0_14       @ 4__non_aug/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640_0_14_10 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=10 --load_weights=1 --start_id=0 --end_id=14


<a id="100__4__non_aug640_selective"></a>
### 100       @ 4__non_aug/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640_100_rt --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=100 --load_weights=1

<a id="100_014__4__non_aug640_selective"></a>
### 100__0_14       @ 4__non_aug/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640_0_14_100 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=100 --load_weights=1 --start_id=0 --end_id=14

<a id="1000__4__non_aug640_selective"></a>
### 1000       @ 4__non_aug/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640_1000 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=1000 --load_weights=1

<a id="1000_014__4__non_aug640_selective"></a>
### 1000__0_14       @ 4__non_aug/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640_0_14_1000 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=1000 --load_weights=1 --start_id=0 --end_id=14

<a id="4__640_selective"></a>
## 4       @ 640_selective

<a id="2__4640_selective"></a>
### 2       @ 4/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_2 --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=2 --load_weights=1

<a id="5__4640_selective"></a>
### 5       @ 4/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5 --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=5 --load_weights=1

<a id="10__4640_selective"></a>
### 10       @ 4/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_10 --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=10 --load_weights=1

<a id="rt__104640_selective"></a>
#### rt       @ 10/4/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_10_rt --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=10 --load_weights=1

<a id="100__4640_selective"></a>
### 100       @ 4/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_100 --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=100 --load_weights=1

<a id="vis__4640_selective"></a>
### vis       @ 4/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_100/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


<a id="no_aug__4640_selective"></a>
### no_aug       @ 4/640_selective

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_100/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="stitched__no_aug4640_selective"></a>
#### stitched       @ no_aug/4/640_selective

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_max_val_acc/raw  --save_path=log/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_100/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

<a id="1k__4640_selective"></a>
### 1K       @ 4/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_1K --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=1000 --load_weights=1

<a id="5k__4640_selective"></a>
### 5K       @ 4/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_3_640_640_64_256_rot_15_345_4_flip_5K --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=5000 --load_weights=1

<a id="16__640_selective"></a>
## 16       @ 640_selective

<a id="100__16640_selective"></a>
### 100       @ 16/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_15_640_640_64_256_rot_15_345_4_flip_100 --train_images="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=100

<a id="32__640_selective"></a>
## 32       @ 640_selective

<a id="100__32640_selective"></a>
### 100       @ 32/640_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python  img_keras_train.py --save_weights_path=weights/vgg_unet2_0_31_640_640_64_256_rot_15_345_4_flip_100 --train_images="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" --epochs=1000 --selective_loss=100







