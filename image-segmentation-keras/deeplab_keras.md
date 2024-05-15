<!--ts-->
   * [640x320 / misc](#640x320--misc)
   * [256](#256)
      * [evaluation](#evaluation)
         * [vis](#vis)
         * [hml](#hml)
      * [vis](#vis-1)
            * [hml](#hml-1)
      * [validation](#validation)
      * [videos](#videos)
      * [new](#new)
   * [384](#384)
      * [evaluation](#evaluation-1)
         * [vis](#vis-2)
         * [hml](#hml-2)
      * [validation](#validation-1)
      * [videos](#videos-1)
   * [512](#512)
      * [evaluation](#evaluation-2)
         * [hml](#hml-3)
      * [vis](#vis-3)
            * [hml](#hml-4)
      * [validation](#validation-2)
            * [stitching](#stitching)
      * [videos](#videos-2)
   * [640](#640)
      * [32](#32)
         * [evaluation](#evaluation-3)
         * [vis](#vis-4)
         * [validation](#validation-3)
         * [stitching](#stitching-1)
      * [4](#4)
         * [evaluation](#evaluation-4)
      * [8](#8)
         * [evaluation](#evaluation-5)
      * [16](#16)
         * [continue 133](#continue-133)
         * [continue latest](#continue-latest)
         * [evaluation](#evaluation-6)
      * [24](#24)
         * [continue 171](#continue-171)
         * [continue latest](#continue-latest-1)
         * [evaluation](#evaluation-7)

<!-- Added by: Tommy, at: 2018-10-05T12:51-06:00 -->

<!--te-->

python visualizeDataset.py \
 --images="/home/abhineet/N/Datasets/617/Training_256_25_100/images/" \
 --annotations="/home/abhineet/N/Datasets/617/Training_256_25_100/labels/" \
 --n_classes=3 
 

# 640x320 / misc
 
KERAS_BACKEND=tensorflow THEANO_FLAGS=device=gpu,floatX=float32  python  img_keras_train.py --save_weights_path=weights/ex1 --train_images="/data/617/images/training_320_640_25_100/images/" --train_annotations="/data/617/images/training_320_640_25_100/labels/" --val_images="/data/617/images/training_320_640_100_200/images/" --val_annotations="/data/617/images/training_320_640_100_200/labels/" --n_classes=3 --input_height=320 --input_width=640 --model_name="deeplab" 
 
 --model_name="fcn32" 
 --model_name="vgg_segnet"  
 --model_name="deeplab" 
 
KERAS_BACKEND=tensorflow THEANO_FLAGS=device=gpu,floatX=float32  python  img_keras_train.py --save_weights_path=weights/ex1 --train_images="/data/617/images/dataset1/images_prepped_train/" --train_annotations="/data/617/images/dataset1/annotations_prepped_train/" --val_images="/data/617/images/dataset1/images_prepped_test/" --val_annotations="/data/617/images/dataset1/annotations_prepped_test/" --n_classes=10 --input_height=320 --input_width=640 --model_name="vgg_segnet" 
 
 change tf.concat(0, [[self._batch_size], [num_dim, -1], [input_shape[2]]]) to tf.concat([[self._batch_size], [num_dim, -1], [input_shape[2]]], 0) and change another tf.concat similarly (also within this file). You should be fine.
 
# 256

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/"  python  img_keras_train.py --save_weights_path=weights/deeplab_256_25_100_rot_15_90_flip --train_images="/data/617/images/training_256_256_25_100_rot_15_90_flip/images/" --train_annotations="/data/617/images/training_256_256_25_100_rot_15_90_flip/labels/" --val_images="/data/617/images/training_256_256_100_200_rot_90_180_flip/images/" --val_annotations="/data/617/images/training_256_256_100_200_rot_90_180_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="deeplab" --epochs=1000

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/"  python  img_keras_train.py --save_weights_path=weights/fcn32_256_25_100_rot_15_90_flip --train_images="/data/617/images/training_256_256_25_100_rot_15_90_flip/images/" --train_annotations="/data/617/images/training_256_256_25_100_rot_15_90_flip/labels/" --val_images="/data/617/images/training_256_256_100_200_rot_90_180_flip/images/" --val_annotations="/data/617/images/training_256_256_100_200_rot_90_180_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="fcn32" --epochs=1000  

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_256_25_100_rot_15_90_flip --train_images="/data/617/images/training_256_256_25_100_rot_15_90_flip/images/" --train_annotations="/data/617/images/training_256_256_25_100_rot_15_90_flip/labels/" --val_images="/data/617/images/training_256_256_100_200_flip/images/" --val_annotations="/data/617/images/training_256_256_100_200_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_segnet" --epochs=1000 

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/fcn8_256_25_100_rot_15_90_flip --train_images="/data/617/images/training_256_256_25_100_rot_15_90_flip/images/" --train_annotations="/data/617/images/training_256_256_25_100_rot_15_90_flip/labels/" --val_images="/data/617/images/training_256_256_100_200_flip/images/" --val_annotations="/data/617/images/training_256_256_100_200_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="fcn8" --epochs=1000 

## evaluation       @ 256

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python2 img_keras_predict.py --save_weights_path=weights/deeplab_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/training_256_256_100_200_rot_90_180_flip/images/" --output_path="/data/617/images/deeplab_max_val_acc_training_256_256_100_200_rot_90_180_flip/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="deeplab" 

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python2 img_keras_predict.py --save_weights_path=weights/vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_266.h5 --test_images="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images/" --output_path="/data/617/images/vgg_segnet_max_mean_acc_training_32_49_256_256_25_100_rot_15_125_235_345_flip/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_segnet" 

### vis       @ evaluation/256

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images --labels_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels --seg_path=/data/617/images/vgg_segnet_max_mean_acc_training_32_49_256_256_25_100_rot_15_125_235_345_flip/predictions/raw --save_path=/data/617/images/vgg_segnet_max_mean_acc_training_32_49_256_256_25_100_rot_15_125_235_345_flip/vis --n_classes=3 --start_id=0 --end_id=-1 --save_stitched=0 --normalize_labels=1



### hml       @ evaluation/256

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/deeplab_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/training_256_256_100_200_rot_90_180_flip/images/" --output_path="/data/617/images/deeplab_max_val_acc_training_256_256_100_200_rot_90_180_flip/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="deeplab" 

zr deeplab_32_18_256_256_25_100_rot_15_125_235_345_flip_weights_490 weights/deeplab_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5
zr img_seg_keras_data data

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/deeplab_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images/" --output_path="/data/617/images/deeplab_max_val_acc_training_32_49_256_256_25_100_rot_15_125_235_345_flip/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="deeplab" 

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/deeplab_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_training_32_49_256_256_25_100_rot_15_125_235_345_flip/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_segnet" 

## vis       @ 256

#### hml       @ vis/256

python3 visDataset.py --images_path=/data/617/images/training_256_256_100_200_rot_90_180_flip/images --labels_path=/data/617/images/training_256_256_100_200_rot_90_180_flip/labels --seg_path=/data/617/images/deeplab_max_val_acc_training_256_256_100_200_rot_90_180_flip/predictions/raw --save_path=/data/617/images/deeplab_max_val_acc_training_256_256_100_200_rot_90_180_flip/vis --n_classes=3 --start_id=0 --end_id=-1

python3 visDataset.py --images_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images --labels_path=/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels --seg_path=/data/617/images/deeplab_max_val_acc_training_32_49_256_256_25_100_rot_15_125_235_345_flip/predictions/raw --save_path=/data/617/images/deeplab_max_val_acc_training_32_49_256_256_25_100_rot_15_125_235_345_flip/vis --n_classes=3 --start_id=0 --end_id=-1


## validation       @ 256

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/validation_0_20_256_256_256_256/images/" --output_path="/data/617/images/deeplab_max_val_acc_validation_0_20_256_256_256_256/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="deeplab" 

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python img_keras_predict.py --save_weights_path=weights/fcn32_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_174.h5 --test_images="/data/617/images/validation_0_20_256_256_256_256/images/" --output_path="/data/617/images/fcn32_max_mean_acc_validation_0_20_256_256_256_256/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="fcn32" 


KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_685.h5 --test_images="/data/617/images/validation_0_20_256_256_256_256/images/" --output_path="/data/617/images/validation_0_20_256_256_256_256/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_segnet" 

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/" python img_keras_predict.py --save_weights_path=weights/fcn8_256_25_100_rot_15_90_flip/weights_685.h5 --test_images="/data/617/images/validation_0_20_256_256_256_256/images/" --output_path="/data/617/images/validation_0_20_256_256_256_256/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="fcn8" 

## videos       @ 256

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/YUN00001_0_239_256_256_256_256/images/" --output_path="/data/617/images/YUN00001_0_239_256_256_256_256/deeplab_32_18_256_256_25_100_rot_15_125_235_345_flip/" --n_classes=3 --input_height=256 --input_width=256 --model_name="deeplab" 

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_685.h5 --test_images="/data/617/images/YUN00001_0_239_256_256_256_256/images/" --output_path="/data/617/images/YUN00001_0_239_256_256_256_256/vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_segnet" 

## new       @ 256

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/"  python  img_keras_train.py --save_weights_path=weights/deeplab_32_18_256_256_25_100_rot_15_125_235_345_flip --train_images="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/images/" --train_annotations="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/labels/" --val_images="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images/" --val_annotations="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="deeplab" --epochs=1000

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/"  python  img_keras_train.py --save_weights_path=weights/fcn32_32_18_256_256_25_100_rot_15_125_235_345_flip --train_images="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/images/" --train_annotations="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/labels/" --val_images="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images/" --val_annotations="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="fcn32" --epochs=1000

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_32_18_256_256_25_100_rot_15_125_235_345_flip --train_images="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/images/" --train_annotations="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/labels/" --val_images="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images/" --val_annotations="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="vgg_segnet" --epochs=1000

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/fcn8_32_18_256_256_25_100_rot_15_125_235_345_flip --train_images="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/images/" --train_annotations="/data/617/images/training_0_31_256_256_25_100_rot_15_125_235_345_flip/labels/" --val_images="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/images/" --val_annotations="/data/617/images/training_32_49_256_256_25_100_rot_15_125_235_345_flip/labels/" --n_classes=3 --input_height=256 --input_width=256 --model_name="fcn8" --epochs=1000


# 384

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/"  python  img_keras_train.py --save_weights_path=weights/deeplab_0_31_384_384_25_100_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_384_384_25_100_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_384_384_25_100_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=384 --input_width=384 --model_name="deeplab" --epochs=1000

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/"  python  img_keras_train.py --save_weights_path=weights/fcn32_0_31_384_384_25_100_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_384_384_25_100_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_384_384_25_100_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=384 --input_width=384 --model_name="fcn32" --epochs=1000  

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_31_384_384_25_100_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_384_384_25_100_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_384_384_25_100_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=384 --input_width=384 --model_name="vgg_segnet" --epochs=1000 


## evaluation       @ 384

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python2 img_keras_predict.py --save_weights_path=weights/deeplab_32_18_256_256_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images/" --output_path="/data/617/images/deeplab_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=256 --input_width=256 --model_name="deeplab" 

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python2 img_keras_predict.py --save_weights_path=weights/vgg_segnet_0_31_384_384_25_100_rot_15_345_4_flip/weights_583.h5 --test_images="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images/" --output_path="/data/617/images/vgg_segnet_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=384 --input_width=384 --model_name="vgg_segnet" 


### vis       @ evaluation/384

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/labels --seg_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/vgg_segnet_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1 --save_stitched=0 --normalize_labels=1


### hml       @ evaluation/384

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/deeplab_0_31_384_384_25_100_rot_15_345_4_flip/weights_497.h5 --test_images="/data/617/images/training_32_49_384_384_25_100_rot_15_345_4_flip/images/" --output_path="/data/617/images/deeplab_max_val_acc_training_32_49_384_384_25_100_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=384 --input_width=384 --model_name="deeplab" 


## validation       @ 384

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_32_18_384_384_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/validation_0_20_384_384_384_384/images/" --output_path="/data/617/images/deeplab_max_val_acc_validation_0_20_384_384_384_384/predictions/" --n_classes=3 --input_height=384 --input_width=384 --model_name="deeplab" 

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python img_keras_predict.py --save_weights_path=weights/fcn32_32_18_384_384_25_100_rot_15_125_235_345_flip/weights_174.h5 --test_images="/data/617/images/validation_0_20_384_384_384_384/images/" --output_path="/data/617/images/fcn32_max_mean_acc_validation_0_20_384_384_384_384/predictions/" --n_classes=3 --input_height=384 --input_width=384 --model_name="fcn32" 


KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_32_18_384_384_25_100_rot_15_125_235_345_flip/weights_685.h5 --test_images="/data/617/images/validation_0_20_384_384_384_384/images/" --output_path="/data/617/images/validation_0_20_384_384_384_384/predictions/" --n_classes=3 --input_height=384 --input_width=384 --model_name="vgg_segnet" 

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/" python img_keras_predict.py --save_weights_path=weights/fcn8_384_25_100_rot_15_90_flip/weights_685.h5 --test_images="/data/617/images/validation_0_20_384_384_384_384/images/" --output_path="/data/617/images/validation_0_20_384_384_384_384/predictions/" --n_classes=3 --input_height=384 --input_width=384 --model_name="fcn8" 

## videos       @ 384

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_32_18_384_384_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/YUN00001_0_239_384_384_384_384/images/" --output_path="/data/617/images/YUN00001_0_239_384_384_384_384/deeplab_32_18_384_384_25_100_rot_15_125_235_345_flip/" --n_classes=3 --input_height=384 --input_width=384 --model_name="deeplab" 

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_segnet_32_18_384_384_25_100_rot_15_125_235_345_flip/weights_685.h5 --test_images="/data/617/images/YUN00001_0_239_384_384_384_384/images/" --output_path="/data/617/images/YUN00001_0_239_384_384_384_384/vgg_segnet_32_18_384_384_25_100_rot_15_125_235_345_flip/" --n_classes=3 --input_height=384 --input_width=384 --model_name="vgg_segnet" 


# 512

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 CUDNN_PATH="/usr/local/cuda-8.0/cuda/lib64/libcudnn.so.5" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/"  python  img_keras_train.py --save_weights_path=weights/deeplab_0_31_512_512_25_100_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=512 --input_width=512 --model_name="deeplab" --epochs=1000

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/"  python  img_keras_train.py --save_weights_path=weights/fcn32_0_31_512_512_25_100_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_512_512_25_100_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=512 --input_width=512 --model_name="fcn32" --epochs=1000  


## evaluation       @ 512

### hml       @ evaluation/512

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/deeplab_0_31_512_512_25_100_rot_15_345_4_flip/weights_max_val_acc_140.h5 --test_images="/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images/" --output_path="/data/617/images/deeplab_max_val_acc_training_32_49_512_512_25_100_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="deeplab" 

## vis       @ 512

#### hml       @ vis/512

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_512_512_25_100_rot_15_345_4_flip/labels --seg_path=/data/617/images/deeplab_max_val_acc_training_32_49_512_512_25_100_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/deeplab_max_val_acc_training_32_49_512_512_25_100_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1


## validation       @ 512

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/targets/x86_64-linux/include/,dnn.library_path=/usr/local/cuda-8.0/targets/x86_64-linux/lib/" python2 img_keras_predict.py --save_weights_path=weights/deeplab_0_31_512_512_25_100_rot_15_345_4_flip/weights_max_val_acc_140.h5 --test_images="/data/617/images/validation_0_20_512_512_512_512/images/" --output_path="/data/617/images/deeplab_max_val_acc_validation_0_20_512_512_512_512/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="deeplab" 

#### stitching       @ validation/512

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=/data/617/images/deeplab_max_val_acc_validation_0_20_512_512_512_512/predictions/raw stitched_seq_path=/data/617/images/deeplab_max_val_acc_validation_0_20_512_512_512_512/predictions/stitched patch_height=512 patch_width=512 start_id=0 end_id=20  show_img=0 stacked=1 method=1 normalize_patches=1


KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_32_18_512_512_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/validation_0_20_512_512_512_512/images/" --output_path="/data/617/images/deeplab_max_val_acc_validation_0_20_512_512_512_512/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="deeplab" 

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python img_keras_predict.py --save_weights_path=weights/fcn32_32_18_512_512_25_100_rot_15_125_235_345_flip/weights_174.h5 --test_images="/data/617/images/validation_0_20_512_512_512_512/images/" --output_path="/data/617/images/fcn32_max_mean_acc_validation_0_20_512_512_512_512/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="fcn32" 



KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/lib64/" python img_keras_predict.py --save_weights_path=weights/fcn8_512_25_100_rot_15_90_flip/weights_685.h5 --test_images="/data/617/images/validation_0_20_512_512_512_512/images/" --output_path="/data/617/images/validation_0_20_512_512_512_512/predictions/" --n_classes=3 --input_height=512 --input_width=512 --model_name="fcn8" 

## videos       @ 512

KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/cuda/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_32_18_512_512_25_100_rot_15_125_235_345_flip/weights_490.h5 --test_images="/data/617/images/YUN00001_0_239_512_512_512_512/images/" --output_path="/data/617/images/YUN00001_0_239_512_512_512_512/deeplab_32_18_512_512_25_100_rot_15_125_235_345_flip/" --n_classes=3 --input_height=512 --input_width=512 --model_name="deeplab" 



# 640

## 4       @ 640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/deeplab_0_3_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" --epochs=1000

### evaluation       @ 4/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_0_3_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/deeplab_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/deeplab_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/deeplab_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


### no_aug       @ 4/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_0_3_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/deeplab_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/deeplab_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/deeplab_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitched       @ no_aug/4/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/deeplab_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/deeplab_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/deeplab_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/deeplab_0_3_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

## 8       @ 640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/deeplab_0_7_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_7_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_7_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" --epochs=1000 --load_weights=1

### evaluation       @ 8/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_0_7_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/deeplab_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/deeplab_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/deeplab_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

### no_aug       @ 8/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_0_7_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/deeplab_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/deeplab_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/deeplab_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitched       @ no_aug/8/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/deeplab_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/deeplab_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/deeplab_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/deeplab_0_7_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


## 16       @ 640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/deeplab_0_15_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" --epochs=1000

### continue_133       @ 16/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/deeplab_0_15_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" --epochs=1000 --load_weights=weights/deeplab_0_15_640_640_64_256_rot_15_345_4_flip/weights_max_acc_133.h5

### continue_latest       @ 16/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python  img_keras_train.py --save_weights_path=weights/deeplab_0_15_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_15_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" --epochs=1000 --load_weights=1

### evaluation       @ 16/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_0_15_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/deeplab_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/deeplab_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/deeplab_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

### no_aug       @ 16/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_0_15_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/deeplab_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/deeplab_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/deeplab_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitched       @ no_aug/16/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/deeplab_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/deeplab_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/deeplab_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/deeplab_0_15_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1


## 24       @ 640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/deeplab_0_23_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" --epochs=1000

### continue_171       @ 24/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/deeplab_0_23_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" --epochs=1000 --load_weights=weights/deeplab_0_23_640_640_64_256_rot_15_345_4_flip/weights_max_acc_171.h5

### continue_latest       @ 24/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/deeplab_0_23_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_23_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" --epochs=1000 --load_weights=1

### evaluation       @ 24/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_0_23_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="log/deeplab_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=log/deeplab_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/raw --save_path=log/deeplab_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_64_256_rot_15_345_4_flip_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

### no_aug       @ 24/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_0_23_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/deeplab_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/deeplab_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/deeplab_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitched       @ no_aug/24/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/deeplab_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/deeplab_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/deeplab_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/deeplab_0_23_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

## 32       @ 640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/cuda/lib64/libcudnn.so.5" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/cuda/lib64/"  python  img_keras_train.py --save_weights_path=weights/deeplab_0_31_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" --epochs=1000

### evaluation       @ 32/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 CUDNN_PATH="/usr/local/cuda-8.0/cuda/lib64/libcudnn.so.5" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc_189.h5 --test_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --output_path="/data/617/images/deeplab_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/predictions/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" 

### vis       @ 32/640

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images --labels_path=/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels --seg_path=/data/617/images/deeplab_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/predictions/raw --save_path=/data/617/images/deeplab_max_val_acc_training_32_49_640_640_64_256_rot_15_345_4_flip/vis --n_classes=3 --start_id=0 --end_id=-1


### no_aug       @ 32/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/deeplab_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc --test_images="/data/617/images/training_32_49_640_640_640_640/images/" --output_path="log/deeplab_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" 

python3 ../visDataset.py --images_path=/data/617/images/training_32_49_640_640_640_640/images --labels_path=/data/617/images/training_32_49_640_640_640_640/labels --seg_path=log/deeplab_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw --save_path=log/deeplab_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

#### stitched       @ no_aug/32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/training_32_49/images img_ext=jpg  patch_seq_path=log/deeplab_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_640_640_640_640_max_val_acc/raw stitched_seq_path=log/deeplab_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=png

python3 ../visDataset.py --images_path=/data/617/images/training_32_49/images --labels_path=/data/617/images/training_32_49/labels --seg_path=log/deeplab_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/raw  --save_path=log/deeplab_0_31_640_640_64_256_rot_15_345_4_flip/training_32_49_max_val_acc/vis --n_classes=3 --start_id=0 --end_id=-1

### validation       @ 32/640

CUDA_VISIBLE_DEVICES=1 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 CUDNN_PATH="/usr/local/cuda-8.0/cuda/lib64/libcudnn.so.5" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-8.0/include,dnn.library_path=/usr/local/cuda-8.0/cuda/lib64/" python2 img_keras_predict.py --save_weights_path=weights/deeplab_0_31_640_640_64_256_rot_15_345_4_flip/weights_max_val_acc_189.h5 --test_images="/data/617/images/validation_0_20_640_640_640_640/images/" --output_path="/data/617/images/deeplab_max_val_acc_validation_0_20_640_640_640_640/predictions/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" 


### stitching       @ 32/640

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/validation/images  patch_seq_path=/data/617/images/deeplab_max_val_acc_validation_0_20_640_640_640_640/predictions/raw stitched_seq_path=/data/617/images/deeplab_max_val_acc_validation_0_20_640_640_640_640/predictions/stitched patch_height=640 patch_width=640 start_id=0 end_id=20  show_img=0 stacked=1 method=1 normalize_patches=1

zr deeplab_max_val_acc_validation_0_20_640_640_640_640_stitched /data/617/images/deeplab_max_val_acc_validation_0_20_640_640_640_640/predictions/stitched

# 640 - selective

## 4       @ 640_-_selective

### 1K       @ 4/640_-_selective

CUDA_VISIBLE_DEVICES=2 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/deeplab_0_3_640_640_64_256_rot_15_345_4_flip_1K --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" --epochs=1000 --selective_loss=1000

### 5K       @ 4/640_-_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/deeplab_0_3_640_640_64_256_rot_15_345_4_flip_5K --train_images="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_3_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" --epochs=1000 --selective_loss=5000



## 32       @ 640_-_selective

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/cuda/lib64/libcudnn.so.5" KERAS_BACKEND=tensorflow THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/cuda/lib64/"  python  img_keras_train.py --save_weights_path=weights/deeplab_0_31_640_640_64_256_rot_15_345_4_flip --train_images="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/images/" --train_annotations="/data/617/images/training_0_31_640_640_64_256_rot_15_345_4_flip/labels/" --val_images="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/images/" --val_annotations="/data/617/images/training_32_49_640_640_64_256_rot_15_345_4_flip/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="deeplab" --epochs=1000 --selective_loss=1000







