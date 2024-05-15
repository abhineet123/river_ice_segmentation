set -x

# 2

## YUN00001_3600


CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640_2/weights_min_loss --test_images="/data/617/images/YUN00001_3600_0_3599_640_640_640_640/images/" --output_path="log/vgg_unet2_0_3_640_640_640_640_2/YUN00001_3600_0_3599_640_640_640_640_min_loss/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 


python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_3_640_640_640_640_2/YUN00001_3600_0_3599_640_640_640_640_min_loss/raw stitched_seq_path=log/vgg_unet2_0_3_640_640_640_640_2/YUN00001_3600_min_loss patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=mkv width=1920 height=1080

## 20160122_YUN00020_2000_3800


CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640_2/weights_min_loss --test_images="/data/617/images/20160122_YUN00020_2000_3800_0_1799_640_640_640_640/images/" --output_path="log/vgg_unet2_0_3_640_640_640_640_2/20160122_YUN00020_2000_3800_0_1799_640_640_640_640_min_loss/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00020_2000_3800/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_3_640_640_640_640_2/20160122_YUN00020_2000_3800_0_1799_640_640_640_640_min_loss/raw stitched_seq_path=log/vgg_unet2_0_3_640_640_640_640_2/20160122_YUN00020_2000_3800_min_loss patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=mkv width=1920 height=1080

# 100

## YUN00001_3600


CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640_100/weights_min_loss --test_images="/data/617/images/YUN00001_3600_0_3599_640_640_640_640/images/" --output_path="log/vgg_unet2_0_3_640_640_640_640_100/YUN00001_3600_0_3599_640_640_640_640_min_loss/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 


python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_3_640_640_640_640_100/YUN00001_3600_0_3599_640_640_640_640_min_loss/raw stitched_seq_path=log/vgg_unet2_0_3_640_640_640_640_100/YUN00001_3600_min_loss patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=mkv width=1920 height=1080

## 20160122_YUN00020_2000_3800


CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640_100/weights_min_loss --test_images="/data/617/images/20160122_YUN00020_2000_3800_0_1799_640_640_640_640/images/" --output_path="log/vgg_unet2_0_3_640_640_640_640_100/20160122_YUN00020_2000_3800_0_1799_640_640_640_640_min_loss/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00020_2000_3800/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_3_640_640_640_640_100/20160122_YUN00020_2000_3800_0_1799_640_640_640_640_min_loss/raw stitched_seq_path=log/vgg_unet2_0_3_640_640_640_640_100/20160122_YUN00020_2000_3800_min_loss patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=mkv width=1920 height=1080

# 1000

## YUN00001_3600


CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640_1000/weights_min_loss --test_images="/data/617/images/YUN00001_3600_0_3599_640_640_640_640/images/" --output_path="log/vgg_unet2_0_3_640_640_640_640_1000/YUN00001_3600_0_3599_640_640_640_640_min_loss/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 


python3 ../stitchSubPatchDataset.py src_path=/data/617/images/YUN00001_3600/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_3_640_640_640_640_1000/YUN00001_3600_0_3599_640_640_640_640_min_loss/raw stitched_seq_path=log/vgg_unet2_0_3_640_640_640_640_1000/YUN00001_3600_min_loss patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=mkv width=1920 height=1080

## 20160122_YUN00020_2000_3800

CUDA_VISIBLE_DEVICES=0 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 CUDNN_PATH="/usr/local/cuda-9.0/lib64/libcudnn.so.7" KERAS_BACKEND=theano THEANO_FLAGS="device=cuda0,floatX=float32,dnn.include_path=/usr/local/cuda-9.0/include,dnn.library_path=/usr/local/cuda-9.0/lib64/" python img_keras_predict.py --save_weights_path=weights/vgg_unet2_0_3_640_640_640_640_1000/weights_min_loss --test_images="/data/617/images/20160122_YUN00020_2000_3800_0_1799_640_640_640_640/images/" --output_path="log/vgg_unet2_0_3_640_640_640_640_1000/20160122_YUN00020_2000_3800_0_1799_640_640_640_640_min_loss/" --n_classes=3 --input_height=640 --input_width=640 --model_name="vgg_unet2" 

python3 ../stitchSubPatchDataset.py src_path=/data/617/images/20160122_YUN00020_2000_3800/images img_ext=jpg  patch_seq_path=log/vgg_unet2_0_3_640_640_640_640_1000/20160122_YUN00020_2000_3800_0_1799_640_640_640_640_min_loss/raw stitched_seq_path=log/vgg_unet2_0_3_640_640_640_640_1000/20160122_YUN00020_2000_3800_min_loss patch_height=640 patch_width=640 start_id=0 end_id=-1  show_img=0 stacked=0 method=1 normalize_patches=0 img_ext=jpg out_ext=mkv width=1920 height=1080
