set -x

# 20161201_YUN00002

# python3 ~/PTF/videoToImgSeq.py db_root_dir=/data/617/videos actor=20161201 seq_name=YUN00002 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=0 dst_dir=/data/617/images/20161201_YUN00002/images 

## 1000

# python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20161201_YUN00002 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

# CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/20161201_YUN00002_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_401_4/weights/ --save_path=log/50_10000_10000_1000_random_401_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0 --start_id=2500

# python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20161201_YUN00002/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_401_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_401_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000_stitched patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=0.5

python3 ~/PTF/imgSeqToVideo.py src_path=log/50_10000_10000_1000_random_401_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000_stitched save_path=log/50_10000_10000_1000_random_401_4/predict/20161201_YUN00002_0_1799_1000_1000_1000_1000_stitched.mkv width=1920 height=1080 show_img=0


## 800

# python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20161201_YUN00002 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/20161201_YUN00002_0_1799_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_800_random_200_4/weights/ --save_path=log/50_10000_10000_800_random_200_4/predict/20161201_YUN00002_0_1799_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20161201_YUN00002/images img_ext=jpg  patch_seq_path=log/50_10000_10000_800_random_200_4/predict/20161201_YUN00002_0_1799_800_800_800_800 stitched_seq_path=log/50_10000_10000_800_random_200_4/predict/20161201_YUN00002_0_1799_800_800_800_8000_stitched patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 patch_ext=jpg out_ext=jpg resize_factor=0.5 patch_ext=jpg 

python3 ~/PTF/imgSeqToVideo.py src_path=log/50_10000_10000_1000_random_401_4/predict/20161201_YUN00002_0_1799_800_800_800_8000_stitched save_path=log/50_10000_10000_1000_random_401_4/predict/20161201_YUN00002_0_1799_800_800_800_8000_stitched.mkv width=1920 height=1080 show_img=0

# 20170114_YUN00005

# python3 ~/PTF/videoToImgSeq.py db_root_dir=/data/617/videos actor=20170114 seq_name=YUN00005 vid_fmt=mp4 n_frames=1800 resize_factor=1.0 start_id=0 dst_dir=/data/617/images/20170114_YUN00005/images 

## 1000

# python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20170114_YUN00005 patch_height=1000 patch_width=1000 min_stride=1000 max_stride=1000 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/20170114_YUN00005_0_1799_1000_1000_1000_1000/images --height=1000 --width=1000 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_1000_random_401_4/weights/ --save_path=log/50_10000_10000_1000_random_401_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20170114_YUN00005/images img_ext=jpg  patch_seq_path=log/50_10000_10000_1000_random_401_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000 stitched_seq_path=log/50_10000_10000_1000_random_401_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000_stitched patch_height=1000 patch_width=1000 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=0.5 patch_ext=jpg

python3 ~/PTF/imgSeqToVideo.py src_path=log/50_10000_10000_1000_random_401_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000_stitched save_path=log/50_10000_10000_1000_random_401_4/predict/20170114_YUN00005_0_1799_1000_1000_1000_1000_stitched.mkv width=1920 height=1080 show_img=0

## 800

# python3 ~/617_w18/Project/code/subPatchDataset.py db_root_dir=/data/617/images seq_name=20170114_YUN00005 patch_height=800 patch_width=800 min_stride=800 max_stride=800 enable_rot=0 enable_flip=0 start_id=0 end_id=-1 img_ext=jpg

# CUDA_VISIBLE_DEVICES=1 python3 il_predict.py --images_path=/data/617/images/20170114_YUN00005_0_1799_800_800_800_800/images --height=800 --width=800 --n_classes=3 --start_id=0 --end_id=-1 --weights_path=log/50_10000_10000_800_random_200_4/weights/ --save_path=log/50_10000_10000_800_random_200_4/predict/20170114_YUN00005_0_1799_800_800_800_800 --save_stitched=1 --gpu_id=2 --loss_type=4 --out_ext=jpg --save_raw=0

python3 ~/617_w18/Project/code/stitchSubPatchDataset.py src_path=/data/617/images/20170114_YUN00005/images img_ext=jpg  patch_seq_path=log/50_10000_10000_800_random_200_4/predict/20170114_YUN00005_0_1799_800_800_800_800 stitched_seq_path=log/50_10000_10000_800_random_200_4/predict/20170114_YUN00005_0_1799_800_800_800_8000_stitched patch_height=800 patch_width=800 start_id=0 end_id=-1  show_img=0 stacked=1 method=1 normalize_patches=0 out_ext=jpg resize_factor=0.5 patch_ext=jpg

python3 ~/PTF/imgSeqToVideo.py src_path=log/50_10000_10000_1000_random_401_4/predict/20170114_YUN00005_0_1799_800_800_800_8000_stitched save_path=log/50_10000_10000_1000_random_401_4/predict/20170114_YUN00005_0_1799_800_800_800_8000_stitched.mkv width=1920 height=1080 show_img=0


