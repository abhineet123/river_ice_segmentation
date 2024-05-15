
## 4 - non_aug

### 2

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_2 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=2 --load_weights=0


### 10

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_10 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=10 --load_weights=0

### 100

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_100 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=100 --load_weights=0

### 1000

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/"  python2  img_keras_train.py --save_weights_path=weights/vgg_segnet_0_3_640_640_640_640_1000 --train_images="/data/617/images/training_0_3_640_640_640_640/images/" --train_annotations="/data/617/images/training_0_3_640_640_640_640/labels/" --val_images="/data/617/images/training_32_49_640_640_640_640/images/" --val_annotations="/data/617/images/training_32_49_640_640_640_640/labels/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_segnet" --epochs=1000 --selective_loss=1000 --load_weights=0

